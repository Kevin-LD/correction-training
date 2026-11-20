"""Validate downloaded SiT assets without network access or a GPU."""
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import torch
from safetensors import safe_open
from src.datasets.codec import build_codec
from src.utils.config import read_config

records = json.loads((ROOT / 'environment/assets.json').read_text())
for item in records:
    path = ROOT / item['path']
    assert path.stat().st_size == item['size'], path
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    assert digest == item['sha256'], path
    if item.get('expected_sha256'):
        assert digest == item['expected_sha256'], path
    print('SHA256 OK:', item['path'], flush=True)

base = ROOT / 'checkpoints/pretrained/sit/pretrained_models/SiT-XL-2-256x256.safetensors'
with safe_open(base, framework='pt', device='cpu') as weights:
    keys = list(weights.keys())
    assert keys
    print('SiT safetensors:', len(keys), 'tensors', flush=True)

torch.set_num_threads(4)
codec = build_codec(read_config(ROOT / 'configs/train/sit.yml'), device='cpu')
with torch.no_grad():
    z = codec.encode(torch.zeros(1, 3, 32, 32))
    image = codec.decode(z)
    assert z.shape == (1, 4, 4, 4) and image.shape == (1, 3, 32, 32)
    assert z.isfinite().all() and image.isfinite().all()
print('Pinned VAE offline CPU encode/decode passed', flush=True)

reference = ROOT / 'evaluation-assets/VIRTUAL_imagenet256_labeled.npz'
with zipfile.ZipFile(reference) as archive:
    print('Reference ZIP entries:', archive.namelist(), flush=True)
with np.load(reference, allow_pickle=False) as stats:
    assert stats['mu'].shape == (2048,)
    assert stats['sigma'].shape == (2048, 2048)
    assert np.isfinite(stats['mu']).all() and np.isfinite(stats['sigma']).all()
print('FID reference statistics passed', flush=True)
