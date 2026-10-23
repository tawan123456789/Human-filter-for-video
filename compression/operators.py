"""Compression operators for the original segmentation-only RVM-Large.

All operations modify an independent model reconstructed from the same baseline.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from torch import nn


def child_replace(root, dotted, new):
    node = root
    names = dotted.split('.')
    for n in names[:-1]:
        node=node._modules[n]
    node._modules[names[-1]]=new


def _overlap_load(dest, src):
    old=src.state_dict()
    new=dest.state_dict()
    with torch.no_grad():
        for key, target in new.items():
            if key not in old or old[key].ndim!=target.ndim:
                continue
            slices=tuple(slice(0,min(a,b)) for a,b in zip(target.shape,old[key].shape))
            target[slices].copy_(old[key][slices])
    dest.load_state_dict(new, strict=True)


def slim_decoder(model,ratio):
    """Real, structurally smaller RVM decoder; ConvGRU state channel sizes also change.

    Does NOT prune the backbone or ASPP. Uses overlap initialization, then recovery training.
    """
    if not 0<ratio<1:
        raise ValueError('decoder slim ratio must be between 0 and 1')
    # RVM Large original RecurrentDecoder([16,24,40,128],[80,40,32,16]).
    from model.decoder import RecurrentDecoder,Projection
    before=copy.deepcopy(model)
    widths=[max(4,2*round(w*ratio/2)) for w in [80,40,32,16]]
    model.decoder=RecurrentDecoder([16,24,40,128],widths)
    model.project_seg=Projection(widths[-1],1)
    _overlap_load(model,before)
    return {'kind':'slim','ratio':ratio,'widths':widths}


def decompose_svd(model, rank_ratio, min_params=2000):
    """Replace supported dense decoder/aspp Conv2d by spatial Conv + 1x1 Conv.

    Only touches layers whose factorization strictly reduces parameter count.
    Rebuilding requires applying the same op to the same original checkpoint.
    """
    if not 0<rank_ratio<1:raise ValueError('rank_ratio must be between 0 and 1')
    changes=[]
    for name, layer in list(model.named_modules()):
        if not isinstance(layer,nn.Conv2d):continue
        if not (name.startswith('decoder.') or name.startswith('aspp.')):continue
        if layer.groups!=1 or layer.weight.numel()<min_params:continue
        co,ci,h,w=layer.weight.shape
        max_rank=min(co,ci*h*w)
        rank=max(1, math.floor(max_rank*rank_ratio))
        orig=layer.weight.numel()+(co if layer.bias is not None else 0)
        reduced=rank*(ci*h*w+co)+(co if layer.bias is not None else 0)
        if reduced>=orig:continue
        U,S,V=torch.linalg.svd(layer.weight.detach().float().reshape(co,-1).cpu(), full_matrices=False)
        left=nn.Conv2d(ci,rank,layer.kernel_size,stride=layer.stride,padding=layer.padding,dilation=layer.dilation,bias=False)
        right=nn.Conv2d(rank,co,1,bias=layer.bias is not None)
        with torch.no_grad():
            left.weight.copy_(V[:rank].reshape(rank,ci,h,w))
            right.weight.copy_((U[:,:rank]*S[:rank]).reshape(co,rank,1,1))
            if layer.bias is not None:right.bias.copy_(layer.bias.detach())
        child_replace(model,name,nn.Sequential(left,right))
        changes.append({'layer':name,'rank':rank,'old_params':orig,'new_params':reduced})
    if not changes:
        raise RuntimeError('No eligible SVD layers that reduce parameters')
    return {'kind':'svd','ratio':rank_ratio,'layers':changes}


def make_global_masks(model, sparsity):
    """Global magnitude masks; this is SPARSITY ONLY, not structural file reduction."""
    if not 0<=sparsity<1:
        raise ValueError('sparsity must be in [0,1)')
    eligible=[(name,p) for name,p in model.named_parameters()
              if name.endswith('weight') and p.ndim==4 and p.numel()>=1000
              and not name.startswith('project_seg.')]
    if not eligible:raise RuntimeError('No eligible convolution weights to prune')
    flat=torch.cat([p.detach().abs().flatten().cpu() for _,p in eligible])
    count=math.floor(sparsity*flat.numel())
    # top-k on each weight globally, ties handled by exactly selecting the desired count.
    order=torch.argsort(flat,stable=True)
    allmask=torch.ones(flat.numel(),dtype=torch.bool)
    allmask[order[:count]]=False
    masks={}; offset=0
    for name,p in eligible:
        n=p.numel()
        mask=allmask[offset:offset+n].reshape(p.shape).clone()
        masks[name]=mask
        offset+=n
    apply_masks(model,masks)
    return masks


def apply_masks(model, masks, grads=False):
    lookup=dict(model.named_parameters())
    with torch.no_grad():
        for name,mask in masks.items():
            if name not in lookup:
                raise KeyError(f'Pruning mask references missing parameter: {name}')
            p=lookup[name]
            m=mask.to(device=p.device)
            if p.shape!=m.shape:
                raise RuntimeError(f'Mask shape mismatch for {name}: {m.shape} versus {p.shape}')
            p.mul_(m)
            if grads and p.grad is not None:p.grad.mul_(m)


def compress(model, operations):
    """Apply operators in order. Masks are kept separately for optimizer-safe recovery."""
    metadata=[]; masks={}
    for op in operations:
        kind=op['kind']
        if kind=='slim':
            if masks:raise ValueError('Decoder slimming after unstructured pruning is unsupported: reorder ops')
            metadata.append(slim_decoder(model,float(op['ratio'])))
        elif kind=='svd':
            if masks:raise ValueError('SVD after unstructured pruning is unsupported: reorder ops')
            metadata.append(decompose_svd(model,float(op['ratio'])))
        elif kind=='prune':
            if masks:raise ValueError('Only one global prune operation per run is supported')
            masks=make_global_masks(model,float(op['sparsity']))
            metadata.append({'kind':'prune','sparsity':float(op['sparsity'])})
        else:raise ValueError(f'Unsupported compression op: {kind}')
    return masks,metadata


def sparsity_info(masks):
    if not masks:return {'pruned_weight_count':0,'eligible_weight_count':0,'eligible_sparsity':0.0}
    zeros=sum((~v).sum().item() for v in masks.values())
    total=sum(v.numel() for v in masks.values())
    return {'pruned_weight_count':int(zeros),'eligible_weight_count':int(total),'eligible_sparsity':zeros/max(total,1)}
