"""Multiprocess DDP acceptance. Each scenario uses a fresh torchrun process tree."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from torch_like.training import DistributedContext, DistributedEvalSampler, DistributedSession

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = [
    "basic",
    "equivalence",
    "resume",
    "empty_validation",
    "domains",
    "early_stop",
    "lifecycle",
    "fail_nan",
    "fail_bad_adapter",
    "fail_bad_validation",
    "fail_config_mismatch",
    "fail_wrong_train_sampler",
    "fail_wrong_val_sampler",
    "fail_buffered_metric",
    "fail_write_error",
    "fail_validation_empty",
    "resume_guards",
    "cosine_resume",
    "unused_parameters",
]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_real_two_rank_gloo(scenario, tmp_path):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("This PyTorch build lacks Gloo")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--max-restarts=0",
        str(ROOT / "tests/_workers/ddp_runtime_worker.py"),
        scenario,
        str(tmp_path),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=(os.name != "nt"),
    )
    try:
        output, _ = process.communicate(timeout=100)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        output, _ = process.communicate()
        pytest.fail(f"torchrun timeout in {scenario}:\n{output}")
    assert process.returncode == 0, output
    for rank in (0, 1):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert result["clean_exit"], result


@pytest.mark.parametrize("size,world", [(0, 2), (1, 2), (7, 3), (8, 2), (11, 4)])
def test_eval_sampler_exact_cover(size, world):
    dataset = TensorDataset(torch.arange(size))
    shards = [list(DistributedEvalSampler(dataset, DistributedContext(rank=r, world_size=world))) for r in range(world)]
    assert sorted(i for shard in shards for i in shard) == list(range(size))
    assert sum(map(len, shards)) == size


def test_session_single_process_no_group():
    with DistributedSession(DistributedContext(), device="cpu") as session:
        assert not session.owns_process_group
    assert not torch.distributed.is_initialized()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_session_timeout(timeout):
    with pytest.raises(ValueError, match="timeout_seconds"):
        DistributedSession(device="cpu", timeout_seconds=timeout)


def test_invalid_backend_device_combination():
    with pytest.raises(ValueError, match="device and backend"):
        DistributedSession(device="cpu", backend="nccl")
