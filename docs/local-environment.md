# Shared experiment environment

The CPU and GPU containers use the same absolute project path and the same
Conda prefix, `envs/correction`. The manager is
`/inspire/hdd/global_user/ky26466/.tools/miniconda3`.
System Python and the base Conda environment are not modified.

## Install on the CPU container

```bash
bash scripts/setup_env.sh
bash scripts/check_cpu_env.sh
```

The installer uses the internal Conda and pip repositories, a project-local
package cache, Python 3.11, and the repository's latent training dependencies.
The intended PyTorch build is 2.5.1 with CUDA 12.4; the CPU check verifies it.
`environment/requirements.lock.txt` and `environment/conda-linux-64.explicit.txt`
record the successfully installed package versions. The latter records the
Conda portion only; the pip lock records pip-managed packages as well.
The install script uses the repository requirements; it does not automatically
reinstall from the generated locks.

An alternative pip index can be selected explicitly with
`CORRECTION_PIP_INDEX_URL`. Neither script changes proxy or traffic forwarding
settings. Package installation does not download pretrained weights or data.

During initial setup on 2026-11-18, the internal Conda source worked. The
internal pip source timed out on package files, Tsinghua returned HTTP 403,
and direct PyPI downloads were slow. Huawei's mirror successfully downloaded
NumPy at about 8 MB/s and was selected for the training dependencies:

```bash
CORRECTION_PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/ \
  bash scripts/setup_env.sh
```

## Activate

```bash
source scripts/activate_env.sh cpu
# In the GPU container:
source scripts/activate_env.sh gpu
```

GPU mode enables Hugging Face offline mode. It does not make arbitrary
downloaders offline: training must use existing local base weights, and all VAE
files for the configured revision must be present in `.cache/huggingface`.
Weights and ImageNet preparation are a separate step.
CPU checks use one computation thread; this limit is not imposed on GPU runs.

## Verify in the GPU container

```bash
nvidia-smi
bash scripts/check_gpu_env.sh
bash scripts/check_gpu_env.sh --distributed
```

The first script checks eight visible GPUs and FP32/BF16 forward and backward
operations on each. The distributed check starts eight ranks and validates an
NCCL all-reduce, with a 120-second process-group timeout.
These checks do not download assets. Actual training, memory capacity, and
checkpoint/resume must subsequently be checked with a short training run.

Both CPU and GPU check scripts automatically save stdout and stderr while
displaying them in the terminal. Each run creates a unique UTC-timestamped log
under `logs/setup/`: `cpu-check-*.log`, `gpu-check-*.log`, or
`gpu-distributed-*.log`. A failed check retains a nonzero exit status.
The CPU container can read these files directly after a GPU-container run.

CPU test logs and install logs are stored in `logs/setup/`. The CPU container
has a four-CPU quota and a 16 GiB memory limit, despite larger host totals.
TensorFlow/ADM and other evaluation-only dependencies are deferred.

## Verified installation (2026-11-18)

- Python 3.11.18; PyTorch 2.5.1+cu124, CUDA runtime 12.4.
- diffusers 0.35.2, huggingface_hub 1.33.0, accelerate 1.15.0.
- `pip check` passed; CPU forward/backward passed.
- All 74 repository CPU tests passed (13.580 seconds).
- Download tests require permission to open a loopback socket. The sandbox
  blocked these tests; the successful full run was outside the sandbox with
  Hugging Face offline mode enabled. Test downloads use mocks/local toy files.
- Environment size: about 6.7 GiB; pip cache: 2.9 GiB; Conda cache: 311 MiB.
- User-provided GPU-container output confirms eight NVIDIA H200 GPUs, each
  exposing 139.8 GiB, passed FP32/BF16 forward/backward checks. All eight ranks
  passed the NCCL all-reduce check with PyTorch 2.5.1+cu124. These checks were
  executed by the user in the GPU container, not by Codex in the CPU container.
- See the 2026-11-19 asset preparation record below for weights and references.

Logs: `logs/setup/install-huawei.log` (package installation),
`logs/setup/finalize.log` (successful finalization), and
`logs/setup/cpu-check.log` (successful CPU checks).

## SiT training and evaluation assets (2026-11-19)

ImageNet is available read-only at
`/inspire/hdd/global_public/public_datas/ILSVRC/Data/CLS-LOC/train` (1,000 classes,
1,281,167 images). No public dataset files were changed or copied.

Downloaded assets:

- SiT: `checkpoints/pretrained/sit/pretrained_models/SiT-XL-2-256x256.safetensors`,
  2,700,473,968 bytes, pinned to the training code's HF revision.
- VAE: `stabilityai/sd-vae-ft-ema`, revision
  `f04b2c4b98319346dad8c65879f680b1997b204a`, in `.cache/huggingface`.
  Configuration and `diffusion_pytorch_model.bin` (334,707,217 bytes) are cached.
- ADM reference: `evaluation-assets/VIRTUAL_imagenet256_labeled.npz`,
  2,037,122,530 bytes.
- Inception: `evaluation-assets/classify_image_graph_def.pb`, 95,673,916 bytes.
- Pinned ADM evaluator: `evaluation-assets/adm_evaluator.py`, 25,057 bytes.

`environment/assets.json` records versions, sources, sizes, and SHA256 digests.
The Azure reference source was too slow and had TLS failures through the PC
proxy. The reference file was obtained from `microsoft/distilled_decoding`,
and Inception from `molinamarc/syntheva`; their published SHA256 digests were
verified. Original URLs and actual mirror revisions are retained in the manifest.

The user authorized the PC proxy for this download only. The wrapper
`scripts/with_pc_proxy.sh` sets proxy variables for its child process tree;
no global proxy or forwarding settings were changed. Future proxy use should
follow the user's authorization for that future download.

ADM uses the separate Conda prefix `envs/correction-adm`, Python 3.11 and
TensorFlow 2.18.1 with its declared CUDA dependencies. Training dependencies
remain in `envs/correction`. Installed ADM versions are saved in
`environment/requirements-adm.lock.txt`.

Verified on CPU, with output saved under `logs/setup/`:

```bash
bash scripts/check_assets.sh
bash scripts/check_adm_env.sh
```

Asset digests, SiT safetensors structure, pinned VAE offline encode/decode,
and FID reference statistics passed. ADM's local Inception extracted finite
2048-dimensional pooled features and 2023-dimensional spatial features.
TensorFlow GPU execution has not been checked.

For offline SiT metric evaluation after generating images, run from this root:

```bash
envs/correction-adm/bin/python src/metrics.py --model sit --samples runs/sit-50k
```

This requires actual generated samples; no training or 50K-image evaluation
was started during asset preparation. The optional VAE moments cache was not
generated.
