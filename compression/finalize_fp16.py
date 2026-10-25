"""Package final FP16 run metadata and result-derived comparison artifacts."""
import csv,json,shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from .core import ROOT,atomic_json,discover_pairs,PersonDataset,device_for
from .quantization import load_variant


def main():
    root=ROOT/'compression'/'outputs'/'final_fp16'
    runs=root/'runs';exports=root/'exports'
    baseline=json.loads((root/'baseline.json').read_text(encoding='utf-8'))
    mapping=[('SVD50','model_a_svd50','Model A: SVD50 + KD'),
             ('SLIM50_SVD50','model_b_slim50_svd50','Model B: SLIM50 + SVD50 + KD')]
    baseline_dir=root/'baseline';baseline_dir.mkdir(exist_ok=True)
    shutil.copy2(exports/'C0'/'baseline_fp16.onnx',baseline_dir/'baseline_fp16.onnx')
    base_cpu=json.loads((exports/'C0'/'cpu_smoke.json').read_text(encoding='utf-8'))
    atomic_json({'checkpoint_sha256':baseline['checkpoint_sha256'],
      'validation':baseline['validation'],'parameters':baseline['baseline_params'],
      'checkpoint_bytes':baseline['baseline_bytes'],'test_metrics':'NOT VERIFIED',
      'onnx_fp16':json.loads((exports/'C0'/'onnx_fp16.json').read_text(encoding='utf-8')),
      'onnx_fp16_cpu_smoke':base_cpu},baseline_dir/'metrics.json')
    records=[{'variant':'Original RVM-MobilenetV3-Large','run_id':'C0',
      'val_iou':baseline['validation']['iou'],'val_dice':baseline['validation']['dice'],
      'params':baseline['baseline_params'],'structural_reduction_pct':0.0,
      'checkpoint_bytes':baseline['baseline_bytes'],
      'onnx_fp16_bytes':(baseline_dir/'baseline_fp16.onnx').stat().st_size,
      'onnx_cpu_val_iou':base_cpu['validation_metrics']['iou'],
      'onnx_cpu_val_dice':base_cpu['validation_metrics']['dice'],
      'cpu_parity_max_abs':base_cpu['stateful_parity']['max_abs'],
      'cpu_parity_mean_abs':base_cpu['stateful_parity']['mean_abs'],
      'test_status':'NOT VERIFIED','ort_cuda_status':'BLOCKED'}]
    for run_id,folder,label in mapping:
        src=runs/run_id;dst=root/folder;dst.mkdir(exist_ok=True)
        result=json.loads((src/'result.json').read_text(encoding='utf-8'))
        exp=json.loads((exports/run_id/'onnx_fp16.json').read_text(encoding='utf-8'))
        cpu=json.loads((exports/run_id/'cpu_smoke.json').read_text(encoding='utf-8'))
        shutil.copy2(src/'best.pt',dst/'best.pt')
        shutil.copy2(src/'config.json',dst/'train_config.json')
        shutil.copy2(src/'history.csv',dst/'training_history.csv')
        shutil.copy2(exports/run_id/f'{run_id.lower()}_fp16.onnx',dst/f'{folder}_fp16.onnx')
        atomic_json({'result':result,'fp16_export':exp,'cpu_smoke':cpu},dst/'export_validation.json')
        atomic_json({'runtime_status':'BLOCKED','reason':exp.get('blocker'),
          'providers_available':exp.get('providers_available'),
          'pytorch_recurrent_benchmark':{k:result.get(k) for k in ('latency_ms_p50','latency_ms_p95','fps','iters')},
          'timing_note':'PyTorch CUDA reference only; not ONNX Runtime benchmark.'},dst/'benchmark.json')
        records.append({'variant':label,'run_id':run_id,'val_iou':result['val_iou'],
          'val_dice':result['val_dice'],'params':result['params'],
          'structural_reduction_pct':100*(1-result['params']/result['baseline_params']),
          'checkpoint_bytes':result['bytes_fp32'],'onnx_fp16_bytes':(dst/f'{folder}_fp16.onnx').stat().st_size,
          'onnx_cpu_val_iou':cpu['validation_metrics']['iou'],'onnx_cpu_val_dice':cpu['validation_metrics']['dice'],
          'cpu_parity_max_abs':cpu['stateful_parity']['max_abs'],'cpu_parity_mean_abs':cpu['stateful_parity']['mean_abs'],
          'test_status':'NOT VERIFIED','ort_cuda_status':'BLOCKED'})
    comp=root/'comparison';fig=comp/'figures';fig.mkdir(parents=True,exist_ok=True)
    (comp/'qualitative_samples').mkdir(parents=True,exist_ok=True)
    atomic_json(records,comp/'results.json')
    with (comp/'results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    try:
        import matplotlib.pyplot as plt
        names=[r['variant'] for r in records]
        sizes=[r['onnx_fp16_bytes']/1e6 for r in records]
        ious=[r['val_iou'] for r in records]
        def bar(values,title,ylabel,path,fmt=None):
            fig_,ax=plt.subplots(figsize=(8,4.5));bars=ax.bar(names,values,color=['#537895','#46a078','#dd9b48'])
            ax.set_title(title);ax.set_ylabel(ylabel);ax.tick_params(axis='x',rotation=12);ax.grid(axis='y',alpha=.25)
            if fmt:
                for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,b.get_height(),fmt.format(v),ha='center',va='bottom')
            fig_.tight_layout();fig_.savefig(path,dpi=160);plt.close(fig_)
        bar(sizes,'ONNX FP16 artifact size','MB',fig/'model_size.png','{:.2f}')
        bar(ious,'Validation IoU','IoU',fig/'validation_iou.png','{:.4f}')
        fig_,ax=plt.subplots(figsize=(6.5,4.5))
        for r in records:ax.scatter(r['onnx_fp16_bytes']/1e6,r['val_iou'],s=80,label=r['variant'])
        ax.set_xlabel('ONNX FP16 artifact (MB)');ax.set_ylabel('Validation IoU');ax.set_title('Validation accuracy vs model size');ax.grid(alpha=.25);ax.legend(fontsize=8);fig_.tight_layout();fig_.savefig(fig/'accuracy_vs_size.png',dpi=160);plt.close(fig_)
        for run_id,folder,label in mapping:
            rows=list(csv.DictReader((root/folder/'training_history.csv').open(encoding='utf-8')))
            fig_,ax=plt.subplots(figsize=(6.5,4));ax.plot([int(x['epoch']) for x in rows],[float(x['train_loss']) for x in rows],marker='o',label='Train loss')
            ax.set_xlabel('Epoch');ax.set_ylabel('Loss');ax.set_title(f'{label}: training loss');ax.grid(alpha=.25);ax.legend();fig_.tight_layout();fig_.savefig(fig/f'{run_id.lower()}_training_loss.png',dpi=160);plt.close(fig_)
            fig_,ax=plt.subplots(figsize=(6.5,4));ax.plot([int(x['epoch']) for x in rows],[float(x['val_iou']) for x in rows],marker='o',label='Validation IoU')
            ax.set_xlabel('Epoch');ax.set_ylabel('IoU');ax.set_title(f'{label}: validation IoU');ax.grid(alpha=.25);ax.legend();fig_.tight_layout();fig_.savefig(fig/f'{run_id.lower()}_validation_iou.png',dpi=160);plt.close(fig_)
        # Identical validation images, GT and predictions from all three actual
        # FP32 checkpoints; these are qualitative segmentation examples only.
        import argparse
        from .core import DEFAULT_CHECKPOINT
        manifest=json.loads((root/'midterm_split_manifest.json').read_text(encoding='utf-8'))
        pairs=discover_pairs(ROOT/'data'/'supervisely')
        ids=manifest['val'][:3]
        samples=[PersonDataset(pairs,ids,256)[i] for i in range(len(ids))]
        dev=device_for('cuda')
        args=SimpleNamespace(checkpoint=DEFAULT_CHECKPOINT,rvm_source=ROOT/'third_party'/'RobustVideoMatting',output_dir=root)
        pred_by_model={}
        for run_id,label in [('C0','Original'),('SVD50','SVD50 + KD'),('SLIM50_SVD50','SLIM50 SVD50 + KD')]:
            model=load_variant(args,run_id).to(dev).eval();pred=[]
            with torch.inference_mode():
                for x,_,_ in samples:
                    logits,*_=model(x.unsqueeze(0).to(dev),segmentation_pass=True)
                    pred.append((logits.sigmoid()[0,0].cpu().numpy()>=.5).astype(np.uint8))
            pred_by_model[label]=pred;del model
            if dev.type=='cuda':torch.cuda.empty_cache()
        fig_,axs=plt.subplots(len(samples),5,figsize=(14,3*len(samples)))
        if len(samples)==1:axs=np.expand_dims(axs,0)
        headers=['Input','Ground truth','Original','SVD50 + KD','SLIM50 SVD50 + KD']
        for j,h in enumerate(headers):axs[0,j].set_title(h)
        for i,(x,y,_) in enumerate(samples):
            axs[i,0].imshow(np.clip(x.numpy().transpose(1,2,0),0,1));axs[i,1].imshow(y[0].numpy(),cmap='gray',vmin=0,vmax=1)
            for j,label in enumerate(headers[2:],2):axs[i,j].imshow(pred_by_model[label][i],cmap='gray',vmin=0,vmax=1)
            for ax in axs[i]:ax.axis('off')
        fig_.suptitle('Validation qualitative examples · identical image IDs');fig_.tight_layout();fig_.savefig(comp/'qualitative_samples'/'validation_examples.png',dpi=160);plt.close(fig_)
    except ImportError:
        print('matplotlib unavailable; comparison figures not created')
    print(f'Packaged {len(records)} variants at {root}')


if __name__=='__main__':main()
