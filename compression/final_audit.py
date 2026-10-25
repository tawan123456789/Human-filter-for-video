"""Reconstruct the Midterm split and write the final-project leakage audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .core import (DEFAULT_CHECKPOINT, ROOT, atomic_json, discover_pairs, sha256)


def canonical_hash(ids):
    return hashlib.sha256(("\n".join(ids)+"\n").encode("utf-8")).hexdigest()


def audit(output_dir=None):
    out=Path(output_dir or ROOT/'compression'/'outputs'/'final_fp16')
    audit_dir=out/'audit';audit_dir.mkdir(parents=True,exist_ok=True)
    pairs=discover_pairs(ROOT/'data'/'supervisely')
    ids=sorted(pairs)
    generator=torch.Generator();generator.manual_seed(42)
    perm=torch.randperm(len(ids),generator=generator).tolist()
    n=len(ids);nt=int(.8*n);nv=int(.1*n)
    midterm={'train':[ids[i] for i in perm[:nt]],
             'val':[ids[i] for i in perm[nt:nt+nv]],
             'test':[ids[i] for i in perm[nt+nv:]]}
    manifest=out/'midterm_split_manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text(encoding='utf-8'))
        if saved!=midterm:raise RuntimeError('Existing final split differs from reconstructed Midterm split')
    else:atomic_json(midterm,manifest)

    old_path=ROOT/'compression'/'outputs'/'split_manifest.json'
    old=json.loads(old_path.read_text(encoding='utf-8')) if old_path.exists() else None
    overlaps={}
    if old:
        overlaps={f'midterm_{a}_vs_compression_{b}':len(set(midterm[a])&set(old[b]))
                  for a in midterm for b in old}
    # Existing compression checkpoints consumed the NumPy split, so any Midterm
    # test ID in its train/validation is contaminated and those runs are excluded.
    prior_test_in_compression_train=(len(set(midterm['test'])&set(old['train'])) if old else None)
    notebook=ROOT/'notebooks'/'rvm_finetune.ipynb'
    report={
      'status':'AUDITED; original Midterm split reconstructable from notebook source; prior compression runs use a distinct split',
      'source_dataset':'Local Supervisely person image/mask pairs in data/supervisely',
      'sample_count':n,'input_resolution':[256,256],
      'allocation':{k:len(v) for k,v in midterm.items()},
      'split_algorithm':'sorted matched stems; torch.Generator().manual_seed(42); torch.randperm; floor 80%, floor 10%, remainder test',
      'split_manifest':str(manifest.relative_to(ROOT)),'split_file_sha256':sha256(manifest),
      'split_id_hashes':{k:canonical_hash(v) for k,v in midterm.items()},
      'notebook_path':str(notebook.relative_to(ROOT)),'notebook_sha256':sha256(notebook),
      'checkpoint_path':str(DEFAULT_CHECKPOINT.relative_to(ROOT)),
      'checkpoint_exists':DEFAULT_CHECKPOINT.is_file(),
      'checkpoint_sha256':sha256(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.is_file() else None,
      'compression_manifest_path':str(old_path.relative_to(ROOT)) if old else None,
      'compression_manifest_sha256':sha256(old_path) if old else None,
      'overlap_counts':overlaps,
      'midterm_test_ids_in_prior_compression_train':prior_test_in_compression_train,
      'prior_compression_run_reuse':'REJECTED: trained on different NumPy split; model selection/data provenance does not match reconstructed Midterm split',
      'teacher_test_leakage':'No overlap by construction if notebook training actually used the documented train_ids; notebook source shows training/validation/test assignments. Checkpoint itself does not embed ID manifest, so attribution is source-reconstructed, not cryptographically recorded.',
      'test_metrics_policy':'NOT VERIFIED until source notebook execution provenance is corroborated; report no test metrics. Final students must use this split manifest and validation only.',
      'mask_preprocessing':'RGB bilinear resize with antialias; grayscale mask nearest resize and >127 binary threshold (matching Midterm notebook).'
    }
    atomic_json(report,audit_dir/'dataset_split_audit.json')
    atomic_json({'checkpoint':str(DEFAULT_CHECKPOINT.relative_to(ROOT)),
      'exists':DEFAULT_CHECKPOINT.is_file(),
      'sha256':report['checkpoint_sha256'],
      'checkpoint_format':'weights-only state dict; model architecture inferred as RVM mobilenetv3 from notebook/repository',
      'training_provenance':'notebook source defines seed-42 PyTorch split and fine-tuning; checkpoint contains no embedded split ID manifest, so lineage cannot be proven from checkpoint bytes alone',
      'reuse_status':'teacher initialization eligible; prior compression student checkpoints rejected for final evaluation due split mismatch'},audit_dir/'checkpoint_provenance.json')
    return report


if __name__=='__main__':
    r=audit();print(json.dumps(r,indent=2,ensure_ascii=False))
