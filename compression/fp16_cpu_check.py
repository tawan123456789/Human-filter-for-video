"""CPU-only ONNX FP16 smoke/parity check when CUDA EP is unavailable."""
import argparse,json
from pathlib import Path
from .core import ROOT,atomic_json,common_parser,loaders,sha256
from .quantization import create_ort_session,check_onnx_parity,ort_dataset_iou,load_variant


def main():
    p=common_parser(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--run-id',default='C0')
    a=p.parse_args()
    dl,manifest=loaders(a,a.output_dir)
    model=load_variant(a,a.run_id).cpu().eval()
    outdir=Path(a.output_dir)/'exports'/a.run_id
    path=outdir/('baseline_fp16.onnx' if a.run_id in ('C0','BASELINE') else a.run_id.lower()+'_fp16.onnx')
    sess=create_ort_session(path,require_cuda=False)
    parity_error=None
    try:parity=check_onnx_parity(model,sess,a.size,atol=None)
    except RuntimeError as exc:parity=None;parity_error=str(exc)
    metrics=ort_dataset_iou(sess,dl['val'])
    result={'runtime':'ONNX Runtime CPU smoke test only','providers_active':sess.get_providers(),
      'stateful_parity':parity,'stateful_parity_status':'MEASURED; no numeric pass threshold specified by project',
      'stateful_parity_error':parity_error,'validation_metrics':metrics,'split_sha256':sha256(manifest),
      'artifact_bytes':path.stat().st_size,
      'cuda_execution_status':'BLOCKED: CUDAExecutionProvider unavailable; CPU smoke is not a GPU benchmark'}
    atomic_json(result,outdir/'cpu_smoke.json');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
