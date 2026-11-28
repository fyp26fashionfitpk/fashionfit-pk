"""Prepare truthful smoke inputs from the existing frozen evaluation artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

def prepare_smoke_inputs(root: Path, out: Path):
    def read(p):
        with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    case=next(c for c in read(root/'05_SPLITS/cases.csv') if c['scenario']=='adult_to_adult')
    index={r['image_id']:r for r in read(root/'05_SPLITS/frozen.csv')}
    original=json.loads((root/'10_RUNS/benchmark_baseline_fashn_vton_1_5/protocol.json').read_text())
    expected=next(c for c in original['cases'] if c['case_id']==case['case_id'])
    if any(case[k]!=expected[k] for k in ['source_image_id','target_image_id','scenario','mode']):raise ValueError('Smoke case changed from evaluated pairing')
    for key in ['source_image_id','target_image_id']:
        r=index[case[key]]
        if r['split']!='test' or hashlib.sha256(Path(r['image_path']).read_bytes()).hexdigest()!=original['inputs'][case[key]]['image']:raise ValueError('Smoke image differs from frozen evaluation')
    donor=index[case['source_image_id']];person=Path(index[case['target_image_id']]['image_path']);mask=Path(donor['mask_path'])
    mask_sha=hashlib.sha256(mask.read_bytes()).hexdigest()
    if mask_sha!=donor['mask_sha256']:raise ValueError('Reviewed donor mask changed')
    with Image.open(donor['image_path']) as im,Image.open(mask) as lab:
        rgb=np.asarray(im.convert('RGB'));labels=np.asarray(lab)
        if labels.shape!=rgb.shape[:2] or not np.any(labels==1):raise ValueError('Invalid donor kurta mask')
        cutout=Image.fromarray(np.where((labels==1)[...,None],rgb,255).astype(np.uint8))
    out.mkdir(parents=True,exist_ok=True);cutout_path=out/'source_visible_kurta_pseudo_cutout.png';cutout.save(cutout_path)
    info=dict(case=case,person_path=str(person),donor_model_photo=str(donor['image_path']),catvton_reference=str(cutout_path),
        catvton_reference_role='reviewed_visible_fabric_pseudo_cutout; not a real flat-lay and not reconstructed hidden fabric',
        source_mask_sha256=mask_sha,catvton_reference_sha256=hashlib.sha256(cutout_path.read_bytes()).hexdigest())
    (out/'input_provenance.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    return info
