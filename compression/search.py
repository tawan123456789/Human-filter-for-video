"""Launch a resumable suite; failed runs are logged and never misreported as complete."""
import csv
import json
import traceback
from pathlib import Path

import torch

from .core import device_for,loaders,set_seed
from .train import experiment,parser_factory


def main():
    parser=parser_factory()
    parser.description='Search compression configurations from one fine-tuned RVM-Large checkpoint'
    args=parser.parse_args()
    profile=json.loads(args.profile.read_text(encoding='utf-8'))
    set_seed(args.seed)
    ctx={'device':device_for(args.device)}
    ctx['loaders'],ctx['manifest']=loaders(args,args.output_dir)
    results=[]
    for spec in profile['runs']:
        try:
            results.append(experiment(args,spec,ctx))
        except Exception:
            d=Path(args.output_dir)/'logs'
            d.mkdir(parents=True,exist_ok=True)
            (d/(spec['id']+'_error.txt')).write_text(traceback.format_exc(),encoding='utf-8')
            print('FAILED:',spec['id'],'see logs')
            if ctx['device'].type=='cuda':torch.cuda.empty_cache()
    report=Path(args.output_dir)/'search_results.csv'
    report.parent.mkdir(parents=True,exist_ok=True)
    if results:
        fieldnames=sorted(set().union(*(r.keys() for r in results)))
        # Persist all intermediate and failed-as-record results independently.
        with report.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore')
            w.writeheader();w.writerows({k:(json.dumps(v) if isinstance(v,(list,dict)) else v) for k,v in r.items()} for r in results)
        feasible=[r for r in results if r['valid']]
        if feasible:
            best=min(feasible,key=lambda r:r['bytes_fp32'])
            print('SMALLEST VALID FP32 checkpoint:',best['run_id'],best['bytes_fp32'],'bytes; Val IoU',best['val_iou'])
        else:print('No candidate currently satisfies the validation constraint.')
    print('Saved:',report if results else 'No results (all runs failed); inspect outputs/logs/')
    if not results:
        raise SystemExit('All compression runs failed; see logged tracebacks.')

if __name__=='__main__':main()
