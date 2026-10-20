"""DDP owns native reducers; avoid Python cycles retaining them past Gloo teardown."""

import gc
import unittest
import weakref
from unittest import mock

import torch

from torch_like.training._ddp import _DDPExecution, _LossModule
from torch_like.training.distributed import DDPConfig, DistributedContext


class Owner:
    def __init__(self):
        self.device = torch.device("cpu")
        self.distributed_context = DistributedContext(rank=0, world_size=2)

    def compute_loss(self, batch):
        return batch.sum()


class DDPLifetimeTests(unittest.TestCase):
    def test_loss_module_does_not_keep_trainer_bound_method_owner_alive(self):
        owner = Owner()
        wrapped = _LossModule(torch.nn.Identity(), owner.compute_loss)
        self.assertEqual(float(wrapped(torch.ones(2))), 2.0)
        owner_ref = weakref.ref(owner)
        del owner
        gc.collect()
        self.assertIsNone(owner_ref())
        with self.assertRaisesRegex(RuntimeError, "owner is no longer alive"):
            wrapped(torch.ones(2))

    def test_execution_helper_does_not_keep_owning_trainer_alive(self):
        owner = Owner()
        with mock.patch("torch_like.training._ddp._require_process_group"), mock.patch(
            "torch_like.training._ddp.dist.get_backend", return_value="gloo"
        ):
            execution = _DDPExecution(owner, DDPConfig())
        owner_ref = weakref.ref(owner)
        del owner
        gc.collect()
        self.assertIsNone(owner_ref())
        with self.assertRaises(ReferenceError):
            _ = execution.trainer.device


if __name__ == "__main__":
    unittest.main()
