"""Freeze selection using VALIDATION only, then audit Test ONCE.

Select the minimum deployable file size among valid candidates. Quantized ONNX
is evaluated on test with same still-image metrics; stateful video quality is
an additional separate task.
"""
import argparse
import json
from pathlib import Path

import torch

from .core import atomic_json,common_parser,device_for,evaluate,loaders,load_rvm,sha256
from .quantization import load_variant,ort_dataset_iou


def gather(output,base):
    candidates=[]
    export=output/'exports'/'C0_segment_fp32_state.pth'
    if export.is_file():
        candidates.append({'id':'C0','format':'pytorch_fp32','bytes':export.stat().st_size,
                           'val_iou':base['validation']['iou'],'path':str(export)})
    for path in sorted((output/'runs').glob('*/result.json')):
        obj=json.loads(path.read_text(encoding='utf-8'))
        if obj.get('valid'):
            ck=path.parent/'deploy_fp32_state.pth'
            if ck.is_file():candidates.append({'id':obj['run_id'],'format':'pytorch_fp32',
                                               'bytes':ck.stat().st_size,'val_iou':obj['val_iou'],'path':str(ck)})
    for path in sorted((output/'exports').glob('*/*.json')):
        obj=json.loads(path.read_text(encoding='utf-8'))
        if obj.get('valid') and obj.get('path') and Path(obj['path']).exists():
            candidates.append({'id':obj.get('run_id','C0'),
                'format':obj.get('format',obj.get('type')),'bytes':Path(obj['path']).stat().st_size,
                'val_iou':obj['val_iou'],'path':obj['path']})
    return candidates


def main():
    p=common_parser(argparse.ArgumentParser(description='Validation selection and one-time final Test audit'))
    p.add_argument('--confirm-final',action='store_true',help='Required; Test must not be used during parameter search')
    args=p.parse_args()
    if not args.confirm_final:raise SystemExit('Pass --confirm-final ONLY when you are ready to freeze model selection and evaluate Test.')
    output=Path(args.output_dir)
    target=output/'reports'/'final_test.json'
    if target.exists():raise RuntimeError('Final Test already evaluated. File exists: '+str(target))
    baseline=json.loads((output/'baseline.json').read_text(encoding='utf-8'))
    loaders_,manifest=loaders(args,output)
    if baseline['checkpoint_sha256']!=sha256(args.checkpoint) or baseline['split_sha256']!=sha256(manifest):
        raise RuntimeError('Baseline versus current data/checkpoint mismatch')
    feasible=[c for c in gather(output,baseline)
              if c['val_iou']>=max(0.,baseline['validation']['iou']-args.max_iou_drop/100)]
    if not feasible:raise RuntimeError('No valid deployable candidate found')
    selected=min(feasible,key=lambda x:x['bytes'])
    atomic_json({'selected_from_validation':selected,'feasible_candidates':feasible},
                output/'reports'/'frozen_selection.json')
    print('Frozen selection (validation only):',selected)
    fmt=selected['format']
    if fmt in ('onnx_int8_qdq','onnx_fp32'):
        import onnxruntime as ort
        sess=ort.InferenceSession(str(selected['path']),providers=['CPUExecutionProvider'])
        measured=ort_dataset_iou(sess,loaders_['test'])
    elif fmt in ('pytorch_fp32','pytorch_fp16'):
        model=load_variant(args,selected['id'])
        device=device_for(args.device)
        if fmt=='pytorch_fp16':
            from .quantization import eval_half
            measured=eval_half(model,loaders_['test'],device)
        else:
            measured=evaluate(model.to(device),loaders_['test'],device)
    else:
        raise RuntimeError(f'Unsupported final format {fmt}. Check selection JSON.')
    test_baseline=evaluate(load_rvm(args.checkpoint,args.rvm_source,True).to(device_for(args.device)),
                           loaders_['test'],device_for(args.device))
    result={'frozen_selection':selected,'test':measured,'baseline_test':test_baseline,
            'test_iou_drop_pp':(test_baseline['iou']-measured['iou'])*100,
            'passes_test_constraint':(test_baseline['iou']-measured['iou'])*100<=args.max_iou_drop,
            'test_set_touched_once':True,'note':'Do not select a different model based on this test result.'}
    atomic_json(result,target)
    print(json.dumps(result,indent=2),'\nSaved:',target)

if __name__=='__main__':main()
