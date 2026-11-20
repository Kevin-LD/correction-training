"""Download pinned SiT training assets and public ADM evaluation references."""
import hashlib
import json
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')

def training():
    from huggingface_hub import hf_hub_download
    from src.utils.pretrained import PRETRAINED
    from src.utils.config import read_config
    source = PRETRAINED['sit']
    path = hf_hub_download(**source, local_dir=ROOT / 'checkpoints/pretrained/sit')
    records = [dict(kind='sit', path=str(Path(path).relative_to(ROOT)),
                    expected_sha256='b48cca85a5bf402c0c5039984959dc99ab4d27d59145a4c4cd3b49e1cc0ab660', **source)]
    recipe = read_config(ROOT / 'configs/train/sit.yml')
    codec = recipe['codec']
    for filename in ('config.json', 'diffusion_pytorch_model.bin'):
        path = hf_hub_download(repo_id=codec['pretrained'], revision=codec['revision'], filename=filename)
        records.append(dict(kind='vae', path=str(Path(path).relative_to(ROOT)),
                            repo_id=codec['pretrained'], revision=codec['revision'], filename=filename))
    return records

def evaluation():
    import subprocess
    from src.tools.prepare_evaluation import ASSETS
    directory = ROOT / 'evaluation-assets'
    directory.mkdir(exist_ok=True)
    records = []
    for name, url in ASSETS.items():
        path = directory / name
        if os.environ.get('CORRECTION_EVAL_HF') == '1' and name != 'adm_evaluator.py':
            from huggingface_hub import hf_hub_download
            mirrors = {
                'VIRTUAL_imagenet256_labeled.npz': ('microsoft/distilled_decoding',
                    '7392028d7820e88590f6a75cd52fffca6b4b938f', 'ImageNet-256_FID_Statistic/VIRTUAL_imagenet256_labeled.npz',
                    'b32732719497e42660a9affb4a966068cba0855ac449b82015e34ec376d20758'),
                'classify_image_graph_def.pb': ('molinamarc/syntheva',
                    'e98076431c59e6875c29bde7f19cc0c78b01b526', 'classify_image_graph_def.pb',
                    '009d6814d1bc560d4e7b236e170e9b2d5ca6f4b57bd8037f6db05776204415c6'),
            }
            repo, revision, filename, expected = mirrors[name]
            if not path.exists():
                cached = Path(hf_hub_download(repo_id=repo, revision=revision, filename=filename))
                with cached.open('rb') as f:
                    actual = hashlib.file_digest(f, 'sha256').hexdigest()
                if actual != expected:
                    raise ValueError(f'Checksum mismatch: {name}')
                os.link(cached.resolve(), path)
            records.append(dict(kind='evaluation', path=str(path.relative_to(ROOT)),
                                url=url, mirror_repo=repo, mirror_revision=revision,
                                expected_sha256=expected))
            continue
        if not path.exists():
            partial = path.with_name(name + '.partial')
            command = ['curl', '-fL', '--retry', '2', '--connect-timeout', '15',
                            '--speed-limit', '1024', '--speed-time', '90',
                            '-C', '-', '-o', str(partial), url]
            result = subprocess.run(command)
            if result.returncode == 33:
                # Some proxies do not honor resume requests. Keep the fragment.
                partial.replace(partial.with_suffix('.previous-partial'))
                result = subprocess.run(command)
            result.check_returncode()
            partial.replace(path)
        records.append(dict(kind='evaluation', path=str(path.relative_to(ROOT)), url=url))
    return records

def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(training), pool.submit(evaluation)]
        records = [r for job in jobs for r in job.result()]
    for record in records:
        path = ROOT / record['path']
        with path.open('rb') as f:
            record['sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
        record['size'] = path.stat().st_size
        print(record['kind'], record['path'], record['size'], flush=True)
    target = ROOT / 'environment/assets.json'
    target.write_text(json.dumps(records, indent=2) + '\n')

if __name__ == '__main__':
    main()
