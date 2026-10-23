"""Evaluate baseline/one selected run on Validation, with optional GPU benchmark."""
import argparse
import json
from pathlib import Path

from .core import atomic_json,benchmark,common_parser,device_for,evaluate,loaders,load_rvm,sha256
from .quantization import load_variant


def main():
    p=common_parser(argparse.ArgumentParser(description='Evaluate one RVM compression candidate'))
    p.add_argument('--run-id',default='C0')
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--iters',type=int,default=50)
    args=p.parse_args()
    device=device_for(args.device)
    dl,manifest=loaders(args,args.output_dir)
    m=load_variant(args,args.run_id).to(device)
    measured=evaluate(m,dl['val'],device)
    measured.update(benchmark(m,device,args.size,args.warmup,args.iters))
    measured['run_id']=args.run_id
    measured['split_sha256']=sha256(manifest)
    out=Path(args.output_dir)/'reports'/f'validate_{args.run_id}.json'
    atomic_json(measured,out)
    print(json.dumps(measured,indent=2), '\nSaved:',out)

if __name__=='__main__':main()
