"""Validate local GPU/paths, data splits, checkpoint and RVM segmentation parity."""
import argparse
import json
from pathlib import Path

import torch

from .core import common_parser,device_for,discover_pairs,fixed_splits,load_rvm,param_count,sha256,atomic_json


def main():
    p=common_parser(argparse.ArgumentParser(description='Check local prerequisites'))
    a=p.parse_args()
    print('PyTorch:',torch.__version__,'CUDA runtime:',torch.version.cuda)
    if torch.cuda.is_available():print('GPU:',torch.cuda.get_device_name(0))
    device=device_for(a.device)
    print('Checkpoint:',a.checkpoint, 'SHA256:',sha256(a.checkpoint))
    pairs=discover_pairs(a.data_dir,a.images_dir,a.masks_dir)
    split,manifest=fixed_splits(pairs,a.output_dir,a.split_manifest,a.seed)
    print('Paired images:',len(pairs), 'split:',{p:len(v) for p,v in split.items()}, 'manifest:',manifest)
    full=load_rvm(a.checkpoint,a.rvm_source,segmentation_only=False).to(device).eval()
    seg=load_rvm(a.checkpoint,a.rvm_source,segmentation_only=True).to(device).eval()
    with torch.inference_mode():
        x=torch.rand(1,3,a.size,a.size,device=device)
        f,*fr=full(x,segmentation_pass=True)
        s,*sr=seg(x,segmentation_pass=True)
        maxdelta=max([(f-s).abs().max().item()]+[(u-v).abs().max().item() for u,v in zip(fr,sr)])
        assert len(fr)==len(sr)==4
        assert maxdelta<=1e-6, f'Segmentation trim changes outputs/states: {maxdelta}'
    print('Model trim parity max delta:',maxdelta,
          'full parameters:',param_count(full),'seg-only parameters:',param_count(seg))
    print('PASS: local environment and segmentation-only equivalence. No training was run.')

if __name__=='__main__':main()
