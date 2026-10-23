"""Unit tests independent of internet/GPU/upstream 3rd-party code."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from compression.core import fixed_splits,load_checkpoint_state,evaluate,PersonDataset
from compression.operators import apply_masks,compress,decompose_svd,make_global_masks,sparsity_info


def test_nonzero_global_pruning_is_not_structural():
    module=nn.Sequential(nn.Conv2d(32,32,3,padding=1),nn.Conv2d(32,32,1))
    initial=sum(p.numel() for p in module.parameters())
    masks=make_global_masks(module,.4)
    info=sparsity_info(masks)
    assert abs(info['eligible_sparsity']-.4)<.001
    assert sum(p.numel() for p in module.parameters())==initial
    loss=module(torch.rand(2,32,8,8)).sum()
    loss.backward()
    apply_masks(module,masks,grads=True)
    for name,p in module.named_parameters():
        if name in masks:assert torch.all(p.grad[~masks[name]]==0)


def test_low_rank_reduces_params_and_preserves_tensor_shape():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder=nn.Sequential(nn.Conv2d(32,64,3,padding=1))
    m=Dummy().eval()
    x=torch.randn(2,32,8,8)
    old=sum(p.numel() for p in m.parameters())
    meta=decompose_svd(m,.5)
    assert sum(p.numel() for p in m.parameters())<old
    assert m.decoder(x).shape==(2,64,8,8)
    assert meta['layers']


def test_split_is_sticky_and_disjoint(tmp_path):
    pairs={str(i):(Path(str(i)),Path(str(i))) for i in range(15)}
    a,p=fixed_splits(pairs,tmp_path,seed=42)
    b,_=fixed_splits(pairs,tmp_path,seed=314)
    assert a==b
    all_keys=a['train']+a['val']+a['test']
    assert set(all_keys)==set(pairs) and len(all_keys)==len(set(all_keys))


def test_missing_checkpoint_is_error(tmp_path):
    with pytest.raises(FileNotFoundError):load_checkpoint_state(tmp_path/'missing.pth')


def test_mask_and_rgb_size(tmp_path):
    a=tmp_path/'image.png';b=tmp_path/'mask.png'
    Image.fromarray(np.full((14,16,3),255,dtype=np.uint8)).save(a)
    Image.fromarray(np.full((14,16),255,dtype=np.uint8)).save(b)
    ds=PersonDataset({'a':(a,b)},['a'],size=32)
    x,y,_=ds[0]
    assert x.shape==(3,32,32) and y.shape==(1,32,32)
    assert float(x.min())==1 and float(y.max())==1
