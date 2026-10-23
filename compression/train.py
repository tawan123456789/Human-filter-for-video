"""Train/recover a compressed RVM Large model on local CUDA GPU.

Example: python -m compression.train --experiment SLIM50 --epochs 8
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .core import (ROOT,append_csv,atomic_json,atomic_torch,benchmark,common_parser,
                   device_for,evaluate,load_rvm,loaders,param_count,set_seed,sha256)
from .operators import apply_masks,compress,sparsity_info


def spec_from_profile(experiment,profile):
    cfg = json.loads(Path(profile).read_text(encoding='utf-8'))
    for spec in cfg['runs']:
        if spec['id']==experiment:
            return spec
    raise KeyError(f'Experiment {experiment!r} is not in {profile}. Available: {[s["id"] for s in cfg["runs"]]}')


def build_from_spec(args,spec):
    model=load_rvm(args.checkpoint,args.rvm_source,segmentation_only=True).cpu()
    masks,info=compress(model,spec.get('ops',[]))
    return model,masks,info


def segmentation_loss(z,y):
    bce=F.binary_cross_entropy_with_logits(z.float(),y.float())
    probs=z.float().sigmoid()
    intersection=(probs*y).sum((1,2,3))
    dice=1-(2*intersection+1)/(probs.sum((1,2,3))+y.sum((1,2,3))+1)
    return 0.8*bce+0.2*dice.mean()


def optimizer_for(model,args):
    backbone=[p for n,p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]
    rest=[p for n,p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    groups=[]
    if backbone:groups.append({'params':backbone,'lr':args.lr_backbone})
    if rest:groups.append({'params':rest,'lr':args.lr_other})
    return torch.optim.AdamW(groups,weight_decay=args.weight_decay)


def experiment(args,spec,context=None):
    """Persist last.pt every epoch; determine feasibility ONLY from validation."""
    if context is None:
        set_seed(args.seed)
        device=device_for(args.device)
        dataloaders,manifest=loaders(args,args.output_dir)
        context={'device':device,'loaders':dataloaders,'manifest':manifest}
    device=context['device']; dl=context['loaders']; out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    run_id=spec['id']
    run_dir=out/'runs'/run_id
    run_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_hash=sha256(args.checkpoint)
    split_hash=sha256(context['manifest'])
    baseline_path=out/'baseline.json'
    if baseline_path.is_file():
        reference=json.loads(baseline_path.read_text(encoding='utf-8'))
        if (reference.get('checkpoint_sha256')!=checkpoint_hash or reference.get('split_sha256')!=split_hash):
            raise RuntimeError('baseline.json was calculated from another checkpoint/split; use another --output-dir')
    else:
        baseline=load_rvm(args.checkpoint,args.rvm_source,True).to(device)
        val=evaluate(baseline,dl['val'],device)
        original_size=save_model_state(baseline,out/'exports'/'C0_segment_fp32_state.pth')
        reference={'checkpoint_sha256':checkpoint_hash,'split_sha256':split_hash,
                   'validation':val,'baseline_bytes':original_size,'baseline_params':param_count(baseline),
                   'note':'New split baseline: NOT necessarily comparable to README 93.54% historical test IoU'}
        atomic_json(reference,baseline_path)
        del baseline
        if device.type=='cuda':torch.cuda.empty_cache()
    threshold=max(0.,reference['validation']['iou']-args.max_iou_drop/100)
    run_cfg={'spec':spec,'source_sha256':checkpoint_hash,'split_sha256':split_hash,
             'epochs':args.epochs,'batch_size':args.batch_size,'lr_backbone':args.lr_backbone,
             'lr_other':args.lr_other,'seed':args.seed,'weight_decay':args.weight_decay,
             'kd_weight':args.kd_weight,'temperature':args.temperature,'size':args.size,
             'device':args.device,'amp':args.amp,'patience':args.patience,
             'max_iou_drop':args.max_iou_drop,'skip_benchmark':args.skip_benchmark}
    result_file=run_dir/'result.json'
    if result_file.is_file() and not args.force:
        result=json.loads(result_file.read_text(encoding='utf-8'))
        saved_config_file=run_dir/'config.json'
        saved_cfg=json.loads(saved_config_file.read_text(encoding='utf-8')).get('run') if saved_config_file.is_file() else None
        if (result['checkpoint_sha256'],result['split_sha256'],result['spec'])!=(checkpoint_hash,split_hash,spec) or saved_cfg!=run_cfg:
            raise RuntimeError(f'Run {run_id} exists with different source/training config; change run-id, output-dir, or pass --force.')
        if not (run_dir/'best.pt').is_file() or not (run_dir/'deploy_fp32_state.pth').is_file():
            raise RuntimeError(f'Run {run_id} reports completion but its artifacts are missing.')
        print('Already completed:',run_id)
        return result
    if args.force:
        for name in ('last.pt','best.pt','result.json','history.csv'):
            (run_dir/name).unlink(missing_ok=True)
    model,masks,metadata=build_from_spec(args,spec)
    model=model.to(device)
    val_before=evaluate(model,dl['val'],device)
    atomic_json({'run':run_cfg,'transforms':metadata,'validation_before_recovery':val_before},run_dir/'config.json')
    optimizer=optimizer_for(model,args)
    amp=args.amp and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    teacher=None
    if spec.get('kd',False):
        teacher=load_rvm(args.checkpoint,args.rvm_source,True).eval().to(device)
        for p in teacher.parameters():p.requires_grad_(False)
    last_file=run_dir/'last.pt'
    best_file=run_dir/'best.pt'
    start_epoch=0;best_iou=-1.;bad=0;history=[]
    if last_file.is_file():
        saved=torch.load(last_file,map_location='cpu',weights_only=True)
        if saved['run_cfg']!=run_cfg:
            raise RuntimeError(f'Cannot resume {run_id}: configuration changed; use --force for a fresh run')
        model.load_state_dict(saved['model_state'])
        masks=saved['masks']
        if masks:apply_masks(model,masks)
        optimizer.load_state_dict(saved['optimizer_state'])
        scaler.load_state_dict(saved['scaler_state'])
        start_epoch=saved['epoch']+1
        best_iou=saved['best_iou'];bad=saved['bad'];history=saved['history']
        print(f'Resuming {run_id} at epoch {start_epoch+1}')
    for epoch in range(start_epoch,args.epochs):
        model.train();batch_losses=[]
        for images,labels,_ in dl['train']:
            images=images.to(device,non_blocking=True)
            labels=labels.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=amp):
                z,*_=model(images,segmentation_pass=True)
                loss=segmentation_loss(z,labels)
                if teacher is not None:
                    with torch.no_grad():
                        zt,*_=teacher(images,segmentation_pass=True)
                    T=args.temperature
                    kd=F.binary_cross_entropy_with_logits(z.float()/T, torch.sigmoid(zt.float()/T))*T*T
                    loss=(1-args.kd_weight)*loss+args.kd_weight*kd
            scaler.scale(loss).backward()
            # Unscale BEFORE clipping/masking; avoid regrowth of pruned weights.
            scaler.unscale_(optimizer)
            if masks:apply_masks(model,masks,grads=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer);scaler.update()
            if masks:apply_masks(model,masks)
            batch_losses.append(float(loss.detach().item()))
        val=evaluate(model,dl['val'],device)
        row={'epoch':epoch+1,'train_loss':float(np.mean(batch_losses)),
             'val_iou':val['iou'],'val_dice':val['dice']}
        history.append(row)
        print(f'{run_id} epoch {epoch+1:02d}: loss={row["train_loss"]:.4f} val IoU={val["iou"]:.4f}')
        state_cpu={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if val['iou']>best_iou+1e-6:
            best_iou=val['iou'];bad=0
            atomic_torch({'run_cfg':run_cfg,'model_state':state_cpu,
                          'masks':masks,'validation':val,'epoch':epoch},best_file)
        else:
            bad+=1
        atomic_torch({'run_cfg':run_cfg,'model_state':state_cpu,
                      'masks':masks,'optimizer_state':optimizer.state_dict(),
                      'scaler_state':scaler.state_dict(),'best_iou':best_iou,
                      'bad':bad,'history':history,'epoch':epoch},last_file)
        # Rewrite full history atomically to avoid partially mixed runs.
        history_csv=run_dir/'history.csv'
        tmp=run_dir/'history_temp.csv'
        import csv
        with tmp.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['epoch','train_loss','val_iou','val_dice'])
            w.writeheader();w.writerows(history)
        tmp.replace(history_csv)
        if bad>=args.patience:
            print('Early stopping:',run_id)
            break
    if not best_file.is_file():
        raise RuntimeError(f'No best checkpoint in {run_dir}')
    best=torch.load(best_file,map_location='cpu',weights_only=True)
    model.load_state_dict(best['model_state'],strict=True)
    masks=best['masks']
    if masks:apply_masks(model,masks)
    measured=evaluate(model,dl['val'],device)
    bytes_fp32=save_model_state(model,run_dir/'deploy_fp32_state.pth')
    # Stateful steady-state inference benchmark, including recurrent tensor I/O.
    perf=benchmark(model,device,args.size,args.warmup,args.iters) if not args.skip_benchmark else {}
    result={'run_id':run_id,'spec':spec,'checkpoint_sha256':checkpoint_hash,
            'split_sha256':split_hash,'baseline_val_iou':reference['validation']['iou'],
            'val_before_iou':val_before['iou'],'val_iou':measured['iou'],
            'val_dice':measured['dice'],'min_val_iou':threshold,'valid':measured['iou']>=threshold,
            'bytes_fp32':bytes_fp32,'baseline_bytes':reference['baseline_bytes'],
            'params':param_count(model),'baseline_params':reference['baseline_params'],
            'best_epoch':best['epoch']+1,**sparsity_info(masks),**perf}
    atomic_json(result,result_file)
    print('COMPLETED',run_id,'valid',result['valid'],'IoU',round(result['val_iou'],4),'bytes',bytes_fp32)
    del model,teacher
    if device.type=='cuda':torch.cuda.empty_cache()
    return result


def save_model_state(model,path):
    atomic_torch({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},path)
    return Path(path).stat().st_size


def parser_factory():
    p=common_parser(argparse.ArgumentParser(description='Fine-tune compressed RVM-V3-Large using local GPU'))
    p.add_argument('--experiment',default='SLIM50')
    p.add_argument('--profile',type=Path,default=ROOT/'compression'/'configs'/'quick.json')
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--lr-backbone',type=float,default=1e-5)
    p.add_argument('--lr-other',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=0.01)
    p.add_argument('--kd-weight',type=float,default=0.2)
    p.add_argument('--temperature',type=float,default=2.)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--iters',type=int,default=50)
    p.add_argument('--skip-benchmark',action='store_true')
    p.add_argument('--force',action='store_true',help='Discard previous run and restart')
    return p


def main():
    args=parser_factory().parse_args()
    if not 0<=args.kd_weight<=1:raise ValueError('kd-weight must be 0..1')
    if args.epochs<1:raise ValueError('epochs must be at least 1')
    spec=spec_from_profile(args.experiment,args.profile)
    experiment(args,spec)

if __name__=='__main__':main()
