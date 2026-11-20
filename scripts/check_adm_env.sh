#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p logs/setup
{
  "$ROOT/envs/correction-adm/bin/python" -m pip check
  "$ROOT/envs/correction-adm/bin/python" -u - <<'PY'
import importlib.util
from pathlib import Path
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
root = Path.cwd()
graph = root / 'evaluation-assets/classify_image_graph_def.pb'
assert graph.is_file(), graph
spec = importlib.util.spec_from_file_location('adm', root / 'evaluation-assets/adm_evaluator.py')
adm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adm)
adm.INCEPTION_V3_PATH = str(graph)
config = tf.ConfigProto(device_count={'GPU': 0}, intra_op_parallelism_threads=4,
                       inter_op_parallelism_threads=1)
with tf.Session(config=config) as session:
    evaluator = adm.Evaluator(session)
    pool, spatial = evaluator.compute_activations([np.zeros((1, 256, 256, 3), dtype=np.uint8)])
    assert pool.shape == (1, 2048), pool.shape
    assert np.isfinite(pool).all() and np.isfinite(spatial).all()
print('TensorFlow:', tf.__version__)
print('ADM local Inception CPU feature extraction passed:', pool.shape, spatial.shape)
PY
} 2>&1 | tee "$ROOT/logs/setup/adm-check-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
