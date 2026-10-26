"""Dataset, model, checkpoint, metrics and reproducible run utilities.

No pretrained or random fallback: a trained checkpoint is mandatory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'compression' / 'outputs'
DEFAULT_CHECKPOINT = ROOT / 'outputs' / 'checkpoints' / 'rvm_mobilenetv3_supervisely_best.pth'


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_for(name='cuda'):
    if str(name).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install a CUDA-enabled PyTorch build and verify nvidia-smi.')
    return torch.device(name)


def atomic_torch(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name, suffix='.tmp')
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def load_checkpoint_state(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Trained RVM checkpoint missing: {path}. Never fall back to an untrained model.')
    # Trained checkpoint should be weights-only-friendly; do not load untrusted files.
    data = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(data, dict):
        for key in ('model_state_dict', 'state_dict', 'model_state', 'model', 'net'):
            if isinstance(data.get(key), dict):
                data = data[key]
                break
    if not isinstance(data, dict) or not data:
        raise ValueError('Checkpoint must be a state_dict or contain model_state_dict.')
    for prefix in ('module.', 'model.', 'net.'):
        if sum(k.startswith(prefix) for k in data) > len(data)/2:
            data = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in data.items()}
    return data


def load_rvm(checkpoint, rvm_source=None, segmentation_only=True):
    if rvm_source:
        source = Path(rvm_source).resolve()
        if not (source / 'model' / 'model.py').exists():
            raise FileNotFoundError(f'Expected upstream RVM checkout with model/model.py: {source}')
        # Prefer verified local source for offline repeatability.
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from model import MattingNetwork
        model = MattingNetwork('mobilenetv3', pretrained_backbone=False)
    else:
        try:
            model = torch.hub.load('PeterL1n/RobustVideoMatting', 'mobilenetv3', pretrained=False, trust_repo=True)
        except Exception as exc:
            raise RuntimeError('Cannot load RVM source through TorchHub. Use --rvm-source with a local PeterL1n/RobustVideoMatting checkout.') from exc
    weights = load_checkpoint_state(checkpoint)
    incompatible = model.load_state_dict(weights, strict=False)
    needed = ('backbone.', 'aspp.', 'decoder.', 'project_seg.')
    missing = [k for k in incompatible.missing_keys if k.startswith(needed)]
    unexpected = [k for k in incompatible.unexpected_keys if k.startswith(needed)]
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint/core model mismatch: missing={missing[:12]}, unexpected={unexpected[:12]}')
    if segmentation_only:
        model.project_mat = nn.Identity()
        model.refiner = nn.Identity()
    return model


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def discover_pairs(data_dir, image_dir=None, mask_dir=None):
    root = Path(data_dir).resolve()
    if image_dir or mask_dir:
        if not (image_dir and mask_dir):
            raise ValueError('Supply both --images-dir and --masks-dir')
        images, masks = Path(image_dir).resolve(), Path(mask_dir).resolve()
    else:
        matches = []
        for base in (root, root/'supervisely', root/'data'/'supervisely'):
            for i_name, m_name in [('images', 'masks'), ('images', 'labels'), ('img', 'mask'), ('Images', 'Masks')]:
                if (base/i_name).is_dir() and (base/m_name).is_dir():
                    matches.append((base/i_name, base/m_name))
        if not matches:
            raise FileNotFoundError(f'Could not find images/ and masks/ under {root}; use --images-dir and --masks-dir')
        images, masks = matches[0]
    supported = {'.jpg','.jpeg','.png','.bmp','.webp'}
    def collect(p):
        out = {}
        for f in sorted(p.rglob('*')):
            if f.suffix.lower() not in supported or not f.is_file():
                continue
            name = f.stem.lower()
            for suffix in ('_mask','-mask','_label','-label','_seg','-seg','_gt'):
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            if name in out:
                raise ValueError(f'Duplicate file stem {name}: {out[name]} and {f}')
            out[name] = f
        return out
    a,b = collect(images),collect(masks)
    if not a or not b:
        raise RuntimeError(f'Empty dataset: {images}, {masks}')
    if a.keys() != b.keys():
        raise RuntimeError(f'Image/mask mismatch: only images={sorted(a.keys()-b.keys())[:5]}, only masks={sorted(b.keys()-a.keys())[:5]}')
    return {k: (a[k],b[k]) for k in sorted(a)}


def fixed_splits(pairs, output_dir, manifest=None, seed=42):
    """Persist a split ONCE. Never regenerate if one exists; do not tune on test."""
    default = Path(output_dir)/'split_manifest.json'
    path = Path(manifest).resolve() if manifest else default
    keys = sorted(pairs)
    if path.is_file():
        d = json.loads(path.read_text(encoding='utf-8'))
    else:
        if manifest:
            raise FileNotFoundError(f'Requested split manifest not found: {path}')
        order = np.random.default_rng(seed).permutation(keys).tolist()
        n = len(order)
        n_train, n_val = int(n*0.8), int(n*0.1)
        d={'train':order[:n_train], 'val':order[n_train:n_train+n_val], 'test':order[n_train+n_val:]}
        atomic_json(d,path)
        print('WARNING: NEW split created. Do not compare its metrics directly to README historical 93.54% IoU.')
    flat = d.get('train',[])+d.get('val',[])+d.get('test',[])
    if len(flat)!=len(set(flat)) or set(flat)!=set(keys):
        raise ValueError('Split manifest does not match dataset exactly, or has overlapping IDs.')
    if not all(d.get(k) for k in ('train','val','test')):
        raise ValueError('Split contains an empty partition.')
    return d, path


class PersonDataset(Dataset):
    def __init__(self, pairs, keys, size=256, augment=False):
        self.pairs, self.ids, self.size, self.augment = pairs, list(keys), size, augment
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        img,mask = self.pairs[self.ids[idx]]
        with Image.open(img) as f:
            image = ImageOps.exif_transpose(f).convert('RGB').resize((self.size,self.size),Image.Resampling.BILINEAR)
        if self.augment and random.random() < 0.25:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
        x = np.asarray(image, dtype=np.float32).copy().transpose(2,0,1)/255.0
        if self.augment:
            x = torch.from_numpy(x)
            if random.random() < 0.35:
                x = (x + torch.randn_like(x) * random.uniform(0.01, 0.05)).clamp_(0.0, 1.0)
        else:
            x = torch.from_numpy(x)
        with Image.open(mask) as f:
            y = np.asarray(f.convert('L').resize((self.size,self.size),Image.Resampling.NEAREST),dtype=np.uint8)
        y = (y > (0 if y.max()<=1 else 127)).astype(np.float32)[None]
        return x, torch.from_numpy(y), self.ids[idx]


def loaders(args, output_dir):
    pairs=discover_pairs(args.data_dir,args.images_dir,args.masks_dir)
    split, manifest=fixed_splits(pairs,output_dir,args.split_manifest,args.seed)
    def make(part, shuffle=False):
        return DataLoader(PersonDataset(pairs,split[part],args.size,augment=(part=='train')),batch_size=args.batch_size,
                          shuffle=shuffle,num_workers=args.workers,pin_memory=str(args.device).startswith('cuda'))
    return {part:make(part,part=='train') for part in ('train','val','test')},manifest


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    tp_total=fp_total=fn_total=0.
    values=[]
    for x,y,_ in loader:
        x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
        logits,*_ = model(x,segmentation_pass=True)
        if logits.shape != y.shape:
            raise RuntimeError(f'Output {logits.shape} != masks {y.shape}')
        pred=logits.float().sigmoid()>=.5
        gt=y>=.5
        tp=(pred & gt).sum(dim=(1,2,3)).float()
        fp=(pred & ~gt).sum(dim=(1,2,3)).float()
        fn=(~pred & gt).sum(dim=(1,2,3)).float()
        union=tp+fp+fn
        iou=torch.where(union>0, tp/union.clamp_min(1),torch.ones_like(union))
        dice=torch.where(2*tp+fp+fn>0,2*tp/(2*tp+fp+fn).clamp_min(1),torch.ones_like(tp))
        values.extend(zip(iou.cpu().tolist(),dice.cpu().tolist()))
        tp_total+=tp.sum().item(); fp_total+=fp.sum().item();fn_total+=fn.sum().item()
    if not values:
        raise RuntimeError('Empty evaluation dataset')
    arr=np.asarray(values)
    return {'iou':float(arr[:,0].mean()),'dice':float(arr[:,1].mean()),
            'global_iou':tp_total/max(1,tp_total+fp_total+fn_total),'count':len(values)}


@torch.inference_mode()
def benchmark(model,device,size=256,warmup=10,iters=50):
    model.eval().to(device)
    x=torch.rand((1,3,size,size),device=device)
    states=[None]*4
    for _ in range(warmup):
        _,*states=model(x,*states,segmentation_pass=True)
    if device.type=='cuda':torch.cuda.synchronize()
    durations=[]
    for _ in range(iters):
        if device.type=='cuda': torch.cuda.synchronize()
        start=time.perf_counter()
        _,*states=model(x,*states,segmentation_pass=True)
        if device.type=='cuda': torch.cuda.synchronize()
        durations.append((time.perf_counter()-start)*1000)
    arr=np.asarray(durations)
    return {'latency_ms_p50':float(np.median(arr)),'latency_ms_p95':float(np.percentile(arr,95)),
            'fps':float(1000/arr.mean()),'iters':iters}


def common_parser(parser):
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--rvm-source',type=Path,help='Local upstream RobustVideoMatting repo; avoids TorchHub download')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data'/'supervisely')
    parser.add_argument('--images-dir',type=Path)
    parser.add_argument('--masks-dir',type=Path)
    parser.add_argument('--split-manifest',type=Path)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--size',type=int,default=256)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--workers',type=int,default=0 if os.name=='nt' else 2)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max-iou-drop',type=float,default=10., help='percentage points, NOT relative percent')
    return parser


def append_csv(path,row):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    exists=path.exists()
    with path.open('a',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if not exists: writer.writeheader()
        writer.writerow(row)
