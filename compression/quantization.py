"""FP16 and stateful ONNX validation for RVM compression.

ONNX export includes four recurrent state inputs and four state outputs. PTQ
requires optional `onnx`/`onnxruntime` dependencies and is not guaranteed to
succeed for all RVM/ORT combinations. Failed exports are never marked valid.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .core import (ROOT,atomic_json,benchmark,common_parser,device_for,evaluate,loaders,
                   load_rvm,param_count,sha256)
from .operators import apply_masks,compress
from .train import save_model_state


def load_variant(args,run_id):
    if run_id in ('C0','BASELINE'):
        model=load_rvm(args.checkpoint,args.rvm_source,True)
        return model
    rd=Path(args.output_dir)/'runs'/run_id
    info=rd/'result.json'
    best=rd/'best.pt'
    if not info.is_file() or not best.is_file():
        raise FileNotFoundError(f'Run {run_id} missing result.json or best.pt; train it first.')
    result=json.loads(info.read_text(encoding='utf-8'))
    if result['checkpoint_sha256']!=sha256(args.checkpoint):
        raise RuntimeError('Checkpoint changed after compression run.')
    model=load_rvm(args.checkpoint,args.rvm_source,True).cpu()
    masks,meta=compress(model,result['spec'].get('ops',[]))
    checkpoint=torch.load(best,map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model_state'],strict=True)
    if masks:apply_masks(model,checkpoint['masks'])
    return model


def _metric_accumulate(records):
    x=np.asarray(records,dtype=np.float64)
    return {'iou':float(x[:,0].mean()),'dice':float(x[:,1].mean()),'count':int(x.shape[0])}


@torch.inference_mode()
def eval_half(model,loader,device):
    model.half().to(device).eval()
    scores=[]
    for x,y,_ in loader:
        x=x.to(device).half(); y=y.to(device)>=.5
        out,*_=model(x,segmentation_pass=True)
        pred=out.float().sigmoid()>=.5
        tp=(pred&y).sum((1,2,3)).float()
        fp=(pred&~y).sum((1,2,3)).float()
        fn=(~pred&y).sum((1,2,3)).float()
        union=tp+fp+fn
        iou=torch.where(union>0,tp/union.clamp_min(1),torch.ones_like(union))
        dice=torch.where(2*tp+fp+fn>0,2*tp/(2*tp+fp+fn).clamp_min(1),torch.ones_like(tp))
        scores.extend(zip(iou.cpu().tolist(),dice.cpu().tolist()))
    return _metric_accumulate(scores)


@torch.inference_mode()
def benchmark_half(model,device,size=256,warmup=10,iters=50):
    """Benchmark recurrent inference with FP16 inputs and state tensors."""
    model.eval().half().to(device)
    x=torch.rand((1,3,size,size),device=device,dtype=torch.float16)
    states=[None]*4
    for _ in range(warmup):
        _,*states=model(x,*states,segmentation_pass=True)
    if device.type=='cuda':torch.cuda.synchronize()
    durations=[]
    for _ in range(iters):
        if device.type=='cuda':torch.cuda.synchronize()
        start=time.perf_counter()
        _,*states=model(x,*states,segmentation_pass=True)
        if device.type=='cuda':torch.cuda.synchronize()
        durations.append((time.perf_counter()-start)*1000)
    arr=np.asarray(durations)
    return {'latency_ms_p50':float(np.median(arr)),
            'latency_ms_p95':float(np.percentile(arr,95)),
            'fps':float(1000/arr.mean()),'iters':iters}


class StatefulRVM(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model=model
    def forward(self,image,r1,r2,r3,r4):
        return tuple(self.model(image,r1,r2,r3,r4,1.0,True))


def make_input(model,resolution):
    x=torch.rand(1,3,resolution,resolution,dtype=torch.float32)
    with torch.inference_mode():
        _,*states=model(x,segmentation_pass=True)
    return (x,*[torch.zeros_like(t) for t in states])


def _run_model_session(sess,x,states):
    inputs={'image':x}
    inputs.update({f'r{i+1}':states[i] for i in range(4)})
    return sess.run(None,inputs)


def export_stateful_onnx(model,path,resolution=256,opset_version=17):
    import onnx
    model=model.cpu().float().eval()
    wrap=StatefulRVM(model).eval()
    inputs=make_input(model,resolution)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(wrap,inputs,str(path),opset_version=opset_version,dynamo=False,
        input_names=['image','r1','r2','r3','r4'],
        output_names=['logits','r1o','r2o','r3o','r4o'],
        do_constant_folding=True)
    onnx.checker.check_model(str(path))
    return path


def convert_onnx_fp16(source, destination):
    """Convert graph weights/computation to FP16, keeping public I/O in FP32."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    graph=onnx.load(str(source))
    converted=convert_float_to_float16(graph,keep_io_types=True)
    # ORT's converter appends Cast nodes after the original nodes; restore a
    # valid stable topological order before checking/saving the converted graph.
    nodes=list(converted.graph.node)
    producer={name:i for i,node in enumerate(nodes) for name in node.output if name}
    ordered=[];remaining=set(range(len(nodes)));available={x.name for x in converted.graph.input}
    available.update(x.name for x in converted.graph.initializer)
    while remaining:
        ready=[i for i in sorted(remaining) if all(not name or name not in producer or producer[name]==i or name in available for name in nodes[i].input)]
        if not ready:
            # Some values can be optional empty-string inputs; unresolved edges
            # indicate a malformed graph and should not be hidden.
            raise RuntimeError('FP16 conversion generated cyclic or unresolved ONNX dependencies')
        for i in ready:
            ordered.append(nodes[i]);remaining.remove(i)
            available.update(name for name in nodes[i].output if name)
    del converted.graph.node[:]
    converted.graph.node.extend(ordered)
    onnx.checker.check_model(converted)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(converted,str(destination))
    # A nominal FP16 export must contain actual FP16 weights and/or computation.
    fp16_initializers=sum(x.data_type==onnx.TensorProto.FLOAT16 for x in converted.graph.initializer)
    fp16_nodes=sum(1 for n in converted.graph.node if n.op_type in {'Cast'} and any(
        a.name=='to' and a.i==onnx.TensorProto.FLOAT16 for a in n.attribute))
    if not fp16_initializers and not fp16_nodes:
        raise RuntimeError('FP16 conversion yielded no FP16 initializers or compute casts')
    return {'fp16_initializers':fp16_initializers,'fp16_cast_nodes':fp16_nodes,
            'initializer_count':len(converted.graph.initializer)}


def create_ort_session(path, require_cuda=True):
    import onnxruntime as ort
    available=ort.get_available_providers()
    if require_cuda and 'CUDAExecutionProvider' not in available:
        raise RuntimeError(f'CUDAExecutionProvider unavailable; available providers: {available}')
    providers=['CUDAExecutionProvider','CPUExecutionProvider'] if require_cuda else ['CPUExecutionProvider']
    session=ort.InferenceSession(str(path),providers=providers)
    active=session.get_providers()
    if require_cuda and (not active or active[0]!='CUDAExecutionProvider'):
        raise RuntimeError(f'CUDA provider was not activated; active providers: {active}')
    return session


def validate_fp16_export(model,fp16_path,loader,resolution,warmup,iters):
    """Graph checks, 5-frame zero/propagated state parity, metrics and CUDA timing."""
    import onnx, onnxruntime as ort
    graph=onnx.load(str(fp16_path));onnx.checker.check_model(graph)
    meta={'input_names':[x.name for x in graph.graph.input],
          'output_names':[x.name for x in graph.graph.output],
          'input_types':[x.type.tensor_type.elem_type for x in graph.graph.input],
          'output_types':[x.type.tensor_type.elem_type for x in graph.graph.output],
          'fp16_initializers':sum(x.data_type==onnx.TensorProto.FLOAT16 for x in graph.graph.initializer),
          'providers_available':ort.get_available_providers()}
    expected_in=['image','r1','r2','r3','r4']; expected_out=['logits','r1o','r2o','r3o','r4o']
    if meta['input_names']!=expected_in or meta['output_names']!=expected_out:
        raise RuntimeError(f'Invalid recurrent interface: {meta}')
    sess=create_ort_session(fp16_path,require_cuda=True)
    meta['providers_active']=sess.get_providers()
    # Feed graph-specific zero states and carry outputs for multiple frames.
    x=torch.rand(1,3,resolution,resolution).numpy().astype(np.float32)
    states=[np.zeros(tuple(int(v) for v in t.shape),np.float32) for t in sess.get_inputs()[1:]]
    model=model.cpu().float().eval(); errors=[]
    with torch.inference_mode():
        for _ in range(5):
            po=StatefulRVM(model)(torch.from_numpy(x),*[torch.from_numpy(s) for s in states])
            oo=_run_model_session(sess,x,states)
            for a,b in zip(po,oo):
                if a.shape!=b.shape or not np.isfinite(b).all():raise RuntimeError('Bad output dimensions or non-finite ONNX value')
                errors.append(float(np.max(np.abs(a.numpy()-b))))
            states=oo[1:]
    meta['five_frame_max_abs_error']=max(errors)
    meta['five_frame_mean_abs_error']=float(np.mean(errors))
    meta['state_propagation_checked']=True
    if meta['five_frame_max_abs_error']>0.08:raise RuntimeError('ONNX FP16 stateful parity exceeded 0.08')
    meta['validation_metrics']=ort_dataset_iou(sess,loader)
    meta['benchmark']=ort_benchmark(sess,resolution,warmup,iters)
    meta['artifact_bytes']=Path(fp16_path).stat().st_size
    return meta


def check_onnx_parity(model,sess,shape,atol=0.035):
    model.cpu().float().eval()
    torch.manual_seed(20261024)
    inp=make_input(model,shape)
    x=inp[0].numpy();states=[v.numpy() for v in inp[1:]]
    maxdelta=0; deltas=[]; per_output=[0.0]*5
    with torch.inference_mode():
        for _ in range(5):
            py_out=StatefulRVM(model)(torch.from_numpy(x),*[torch.from_numpy(s) for s in states])
            ox=_run_model_session(sess,x,states)
            if len(ox)!=5:raise RuntimeError('ONNX does not return all 5 outputs (mask + 4 states)')
            for index,(a,b) in enumerate(zip(py_out,ox)):
                if a.shape!=b.shape:raise RuntimeError(f'ONNX state shape mismatch: {a.shape} vs {b.shape}')
                delta=float(np.max(np.abs(a.numpy()-b)))
                maxdelta=max(maxdelta,delta)
                per_output[index]=max(per_output[index],delta)
                deltas.append(float(np.mean(np.abs(a.numpy()-b))))
                if not np.isfinite(delta) or (atol is not None and delta>atol):
                    raise RuntimeError(f'ONNX/PyTorch stateful parity failure: max absolute error {delta:.6f} > {atol}')
            states=ox[1:]
    return {'max_abs':maxdelta,'mean_abs':float(np.mean(deltas)),'max_abs_by_output':dict(zip(['logits','r1o','r2o','r3o','r4o'],per_output)),'frames':5,'outputs_checked':5}


def ort_dataset_iou(sess,loader):
    scores=[]
    state_specs=[v.shape for v in sess.get_inputs()[1:]]
    for x,y,_ in loader:
        for xi,yi in zip(x,y):
            arr=xi.unsqueeze(0).numpy().astype(np.float32)
            states=[np.zeros([int(d) for d in dims],dtype=np.float32) for dims in state_specs]
            logits,*_=_run_model_session(sess,arr,states)
            pred=1/(1+np.exp(-np.clip(logits.astype(np.float64),-50,50)))>=.5
            target=yi.unsqueeze(0).numpy()>=.5
            tp=int(np.logical_and(pred,target).sum())
            fp=int(np.logical_and(pred,~target).sum())
            fn=int(np.logical_and(~pred,target).sum())
            u=tp+fp+fn
            scores.append((tp/u if u else 1.,2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 1.))
    return _metric_accumulate(scores)


def ort_benchmark(sess,resolution,warmup=10,iters=50):
    shapes=[i.shape for i in sess.get_inputs()[1:]]
    x=np.random.default_rng(123).random((1,3,resolution,resolution)).astype(np.float32)
    states=[np.zeros(tuple(int(v) for v in shape),dtype=np.float32) for shape in shapes]
    for _ in range(warmup):
        _,*states=_run_model_session(sess,x,states)
    samples=[]
    for _ in range(iters):
        start=time.perf_counter()
        _,*states=_run_model_session(sess,x,states)
        samples.append((time.perf_counter()-start)*1000)
    return {'cpu_latency_p50_ms':float(np.median(samples)),
            'cpu_latency_p95_ms':float(np.percentile(samples,95)),
            'cpu_fps':1000/float(np.mean(samples))}


def quantize(args):
    dl,manifest=loaders(args,args.output_dir)
    baseline_path=Path(args.output_dir)/'baseline.json'
    if not baseline_path.is_file():
        raise FileNotFoundError('Run the baseline/training first to create baseline.json')
    baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
    if baseline['checkpoint_sha256']!=sha256(args.checkpoint) or baseline['split_sha256']!=sha256(manifest):
        raise RuntimeError('Baseline checkpoint or data split changed')
    threshold=max(0.,baseline['validation']['iou']-args.max_iou_drop/100)
    model=load_variant(args,args.run_id).cpu().eval()
    out=Path(args.output_dir)/'exports'/args.run_id
    out.mkdir(parents=True,exist_ok=True)
    if args.format in ('fp16','both'):
        device=device_for('cuda')
        half=eval_half(model,dl['val'],device)
        path=out/'state_fp16.pth'
        bytes_=save_model_state(model,path)
        metrics=benchmark_half(model,device,args.size,args.warmup,args.iters)
        result={'run_id':args.run_id,'type':'pytorch_fp16','val_iou':half['iou'],
                'val_dice':half['dice'],'threshold':threshold,'valid':half['iou']>=threshold,
                'deploy_bytes':bytes_,'path':str(path),**metrics}
        atomic_json(result,out/'fp16.json')
        print(result)
        # Reload clean FP32 model for ONNX export.
        model=load_variant(args,args.run_id).cpu().eval()
    if args.format in ('onnx','onnx-fp16'):
        try:
            import onnx
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError('Install optional ONNX export packages: pip install -r compression/requirements-quant.txt') from exc
        p32=out/f'stateful_fp32_opset{args.opset}.onnx'
        export_stateful_onnx(model,p32,args.size,args.opset)
        if args.format=='onnx':
            cpu32=create_ort_session(p32,require_cuda=False)
            parity=check_onnx_parity(model,cpu32,args.size)
            metrics32=ort_dataset_iou(cpu32,dl['val'])
            result={'format':'onnx_fp32','opset':args.opset,'run_id':args.run_id,
                    'val_iou':metrics32['iou'],'valid':metrics32['iou']>=threshold,
                    'parity':parity,'deploy_bytes':p32.stat().st_size,'path':str(p32),
                    'providers_active':cpu32.get_providers()}
            atomic_json(result,out/f'onnx_fp32_opset{args.opset}.json');print(result)
        else:
            p16=out/(('baseline_fp16.onnx' if args.run_id in ('C0','BASELINE') else args.run_id.lower()+'_fp16.onnx'))
            conversion=convert_onnx_fp16(p32,p16)
            try:
                result=validate_fp16_export(model,p16,dl['val'],args.size,args.warmup,args.iters)
                result.update({'format':'onnx_fp16','run_id':args.run_id,'fp32_intermediate':str(p32),
                               'conversion':conversion,'threshold':threshold,
                               'valid':result['validation_metrics']['iou']>=threshold,'path':str(p16),
                               'runtime_status':'validated'})
            except RuntimeError as exc:
                result={'format':'onnx_fp16','run_id':args.run_id,'path':str(p16),
                        'fp32_intermediate':str(p32),'conversion':conversion,
                        'artifact_bytes':p16.stat().st_size,'threshold':threshold,
                        'runtime_status':'BLOCKED','blocker':str(exc),
                        'providers_available':ort.get_available_providers(),
                        'valid':False}
            atomic_json(result,out/'onnx_fp16.json');print(result)
    if args.format in ('int8','both'):
        try:
            import onnx
            import onnxruntime as ort
            from onnxruntime.quantization import CalibrationDataReader,QuantFormat,QuantType,quantize_static
        except ImportError as exc:
            raise RuntimeError('Install optional CPU quantization packages: pip install -r compression/requirements-quant.txt') from exc
        p32=out/'stateful_fp32.onnx';p8=out/'stateful_int8.onnx'
        export_stateful_onnx(model,p32,args.size,args.opset)
        cpu32=ort.InferenceSession(str(p32),providers=['CPUExecutionProvider'])
        parity=check_onnx_parity(model,cpu32,args.size)
        metrics32=ort_dataset_iou(cpu32,dl['val'])
        float_performance=ort_benchmark(cpu32,args.size,args.warmup,args.iters)
        atomic_json({'format':'onnx_fp32','opset':args.opset,'run_id':args.run_id,'val_iou':metrics32['iou'],
                     'valid':metrics32['iou']>=threshold,'parity':parity,
                     'deploy_bytes':p32.stat().st_size,'path':str(p32),**float_performance},out/'onnx_fp32.json')
        # Calibrate both zero state and one propagated state per training image.
        # All calibration samples come exclusively from TRAIN, never val/test.
        class Reader(CalibrationDataReader):
            def __init__(self,train_loader,limit,reference_sess):
                self.samples=[]
                state_specs=[i.shape for i in reference_sess.get_inputs()[1:]]
                for images,_,_ in train_loader:
                    for image in images:
                        x=image.unsqueeze(0).numpy().astype(np.float32)
                        z=[np.zeros([int(a) for a in shape],dtype=np.float32) for shape in state_specs]
                        self.samples.append({'image':x,**{f'r{j+1}':z[j] for j in range(4)}})
                        _,*s=_run_model_session(reference_sess,x,z)
                        self.samples.append({'image':x,**{f'r{j+1}':s[j] for j in range(4)}})
                        if len(self.samples)>=limit*2:break
                    if len(self.samples)>=limit*2:break
                self.rewind()
            def get_next(self):return next(self._iter,None)
            def rewind(self):self._iter=iter(self.samples)
        reader=Reader(dl['train'],args.calibration_samples,cpu32)
        if len(reader.samples)<2:raise RuntimeError('Calibration dataset empty')
        quantize_static(str(p32),str(p8),reader,quant_format=QuantFormat.QDQ,
                        per_channel=args.per_channel,activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8,op_types_to_quantize=['Conv'])
        graph=onnx.load(str(p8))
        q_count=sum(x.op_type=='QuantizeLinear' for x in graph.graph.node)
        if q_count==0:raise RuntimeError('Quantizer produced no QuantizeLinear nodes; no valid INT8 conversion')
        cpu8=ort.InferenceSession(str(p8),providers=['CPUExecutionProvider'])
        x=make_input(model,args.size)
        _run_model_session(cpu8,x[0].numpy(),[v.numpy() for v in x[1:]])
        metrics8=ort_dataset_iou(cpu8,dl['val'])
        perf8=ort_benchmark(cpu8,args.size,args.warmup,args.iters)
        result={'format':'onnx_int8_qdq','run_id':args.run_id,
            'calibration_images':args.calibration_samples,'qdq_quantize_nodes':q_count,
            'val_iou':metrics8['iou'],'val_dice':metrics8['dice'],
            'threshold':threshold,'valid':metrics8['iou']>=threshold,
            'deploy_bytes':p8.stat().st_size,'onnx_fp32_bytes':p32.stat().st_size,
            'ratio_vs_onnx_fp32':p32.stat().st_size/max(1,p8.stat().st_size),
            'path':str(p8),'note':'stateful recurrent I/O; still-image IoU, not video temporal quality',**perf8}
        atomic_json(result,out/'int8.json')
        print(result)


def main():
    p=common_parser(argparse.ArgumentParser(description='FP16 and stateful ONNX INT8 PTQ'))
    p.add_argument('--run-id',default='C0')
    p.add_argument('--format',choices=['fp16','onnx','onnx-fp16','int8','both'],default='both')
    p.add_argument('--opset',type=int,default=17,help='ONNX opset version')
    p.add_argument('--calibration-samples',type=int,default=128)
    p.add_argument('--per-channel',action='store_true')
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--iters',type=int,default=50)
    args=p.parse_args()
    quantize(args)

if __name__=='__main__':main()
