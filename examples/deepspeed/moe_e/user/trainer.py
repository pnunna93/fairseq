# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a network across multiple GPUs.
"""

import contextlib
import logging
import sys
import time
from argparse import Namespace
from itertools import chain
from typing import Any, Dict, List

import torch
from fairseq import checkpoint_utils, models, optim, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics
from fairseq.models.ema import build_ema
from fairseq.nan_detector import NanDetector
from fairseq.optim import lr_scheduler
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

from fairseq.trainer import Trainer

class DeepspeedETrainer(Trainer):
    """Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, cfg: FairseqConfig, task, model, criterion, quantizer=None):
        super().__init__(cfg, task, model, criterion, quantizer)

    @property
    def is_moe(self):
        return any(
            getattr(param, 'expert', False) for param in self.model.parameters()
        )

    @metrics.aggregate("train")
    def train_step(self, samples, raise_oom=False):
        """Do forward, backward and parameter update."""
        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        metrics.log_start_time("train_wall", priority=800, round=0)

        # If EMA is enabled through store_ema=True
        # and task.uses_ema is True, pass the EMA model as a keyword
        # argument to the task.
        extra_kwargs = {}
        if self.cfg.ema.store_ema and getattr(self.task, "uses_ema", False):
            extra_kwargs["ema_model"] = self.ema.get_model()

        # forward and backward pass
        logging_outputs, sample_size, ooms = [], 0, 0
        for i, sample in enumerate(samples):  # delayed update loop
            sample, is_dummy_batch = self._prepare_sample(sample)

            def maybe_no_sync():
                """
                Whenever *samples* contains more than one mini-batch, we
                want to accumulate gradients locally and only call
                all-reduce in the last backwards pass.
                """
                if (
                    self.data_parallel_world_size > 1
                    and hasattr(self.model, "no_sync")
                    and i < len(samples) - 1
                    # The no_sync context manager results in increased memory
                    # usage with FSDP, since full-size gradients will be
                    # accumulated on each GPU. It's typically a better tradeoff
                    # to do the extra communication with FSDP.
                    and not self.is_fsdp
                ):
                    return self.model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            try:
                with maybe_no_sync():
                    # forward and backward
                    # print("trainer.py ::", type(self.model))
                    # raise ValueError(type(self.task))
                    loss, sample_size_i, logging_output = self.task.train_step(
                        sample=sample,
                        model=self.model,
                        # ds_model=None,
                        # ds_model=self._deepspeed_model,
                        criterion=self.criterion,
                        optimizer=self.optimizer,
                        update_num=self.get_num_updates(),
                        ignore_grad=is_dummy_batch,
                        **extra_kwargs,
                    )
                    del loss

                logging_outputs.append(logging_output)
                sample_size += sample_size_i

                # emptying the CUDA cache after the first step can
                # reduce the chance of OOM
                if self.cuda and self.get_num_updates() == 0:
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if raise_oom:
                        raise e
                    logger.warning(
                        "attempting to recover from OOM in forward/backward pass"
                    )
                    ooms += 1
                    self.zero_grad()
                    if self.cuda:
                        torch.cuda.empty_cache()
                    if self.cfg.distributed_training.distributed_world_size == 1:
                        return None
                else:
                    raise e

            if self.tpu and i < len(samples) - 1:
                # tpu-comment: every XLA operation before marking step is
                # appended to the IR graph, and processing too many batches
                # before marking step can lead to OOM errors.
                # To handle gradient accumulation use case, we explicitly
                # mark step here for every forward pass without a backward pass
                self._xla_markstep_and_send_to_cpu()

        if is_dummy_batch:
            if torch.is_tensor(sample_size):
                sample_size.zero_()
            else:
                sample_size *= 0.0

        if torch.is_tensor(sample_size):
            sample_size = sample_size.float()
        else:
            sample_size = float(sample_size)

        # gather logging outputs from all replicas
        if self._sync_stats():
            train_time = self._local_cumulative_training_time()
            logging_outputs, (
                sample_size,
                ooms,
                total_train_time,
            ) = self._aggregate_logging_outputs(
                logging_outputs, sample_size, ooms, train_time, ignore=is_dummy_batch
            )
            self._cumulative_training_time = (
                total_train_time / self.data_parallel_world_size
            )

        overflow = False
        try:
            with torch.autograd.profiler.record_function("reduce-grads"):
                # reduce gradients across workers
                self.optimizer.all_reduce_grads(self.model)
                if utils.has_parameters(self.criterion):
                    self.optimizer.all_reduce_grads(self.criterion)

            with torch.autograd.profiler.record_function("multiply-grads"):
                # multiply gradients by (data_parallel_size / sample_size) since
                # DDP normalizes by the number of data parallel workers for
                # improved fp16 precision.
                # Thus we get (sum_of_gradients / sample_size) at the end.
                # In case of fp16, this step also undoes loss scaling.
                # (Debugging note: Some optimizers perform this scaling on the
                # fly, so inspecting model.parameters() or optimizer.params may
                # still show the original, unscaled gradients.)
                numer = (
                    self.data_parallel_world_size
                    if not self.cfg.optimization.use_bmuf or self._sync_stats()
                    else 1
                )
                self.optimizer.multiply_grads(numer / (sample_size or 1.0))
                # Note: (sample_size or 1.0) handles the case of a zero gradient, in a
                # way that avoids CPU/device transfers in case sample_size is a GPU or
                # TPU object. The assumption is that the gradient itself is also 0.

            with torch.autograd.profiler.record_function("clip-grads"):
                # clip grads
                grad_norm = self.clip_grad_norm(self.cfg.optimization.clip_norm)

            # check that grad norms are consistent across workers
            # on tpu check tensor is slow
            if not self.tpu:
                if (
                    not self.cfg.optimization.use_bmuf
                    and self.cfg.distributed_training.ddp_backend != "slow_mo"
                    and not self.is_moe
                ):
                    self._check_grad_norms(grad_norm)
                if not torch.isfinite(grad_norm).all():
                    # in case of AMP, if gradients are Nan/Inf then
                    # optimizer step is still required
                    if self.cfg.common.amp:
                        overflow = True
                    else:
                        # check local gradnorm single GPU case, trigger NanDetector
                        raise FloatingPointError("gradients are Nan/Inf")

            with torch.autograd.profiler.record_function("optimizer"):
                # take an optimization step
                self.task.optimizer_step(
                    self.optimizer,
                    model=self.model,
                    # model=self._deepspeed_model,
                    update_num=self.get_num_updates()
                )
                if self.cfg.common.amp and overflow:
                    if self._amp_retries == self.cfg.common.amp_batch_retries:
                        logger.info("AMP: skipping this batch.")
                        self._amp_retries = 0
                    else:
                        self._amp_retries += 1
                        return self.train_step(samples, raise_oom)  # recursion to feed in same batch

        except FloatingPointError:
            # re-run the forward and backward pass with hooks attached to print
            # out where it fails
            self.zero_grad()
            with NanDetector(self.get_model()):
                for _, sample in enumerate(samples):
                    sample, _ = self._prepare_sample(sample)
                    self.task.train_step(
                        sample,
                        self.model,
                        self.criterion,
                        self.optimizer,
                        self.get_num_updates(),
                        ignore_grad=False,
                        **extra_kwargs,
                    )
            raise
        except OverflowError as e:
            overflow = True
            logger.info(
                f"NOTE: gradient overflow detected, ignoring gradient, {str(e)}"
            )
            grad_norm = torch.tensor(0.0).cuda()
            self.zero_grad()
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._log_oom(e)
                logger.error("OOM during optimization, irrecoverable")
            raise e

        # Some distributed wrappers (e.g., SlowMo) need access to the optimizer
        # after the step
        if hasattr(self.model, "perform_additional_optimizer_actions"):
            if hasattr(self.optimizer, "fp32_params"):
                self.model.perform_additional_optimizer_actions(
                    self.optimizer.optimizer, self.optimizer.fp32_params
                )
            else:
                self.model.perform_additional_optimizer_actions(
                    self.optimizer.optimizer
                )

        logging_output = None
        if not overflow or self.cfg.distributed_training.ddp_backend == "slow_mo":
            self.set_num_updates(self.get_num_updates() + 1)

            if self.cfg.ema.store_ema:
                # Step EMA forward with new model.
                self.ema.step(
                    self.get_model(),
                    self.get_num_updates(),
                )
                metrics.log_scalar(
                    "ema_decay",
                    self.ema.get_decay(),
                    priority=10000,
                    round=5,
                    weight=0,
                )

            if self.tpu:
                import torch_xla.core.xla_model as xm

                # mark step on TPUs
                self._xla_markstep_and_send_to_cpu()

                # only log stats every log_interval steps
                # this causes wps to be misreported when log_interval > 1
                logging_output = {}
                if self.get_num_updates() % self.cfg.common.log_interval == 0:
                    # log memory usage
                    mem_info = xm.get_memory_info(self.device)
                    gb_free = mem_info["kb_free"] / 1024 / 1024
                    gb_total = mem_info["kb_total"] / 1024 / 1024
                    metrics.log_scalar(
                        "gb_free", gb_free, priority=1500, round=1, weight=0
                    )
                    metrics.log_scalar(
                        "gb_total", gb_total, priority=1600, round=1, weight=0
                    )
                    logging_outputs = self._xla_markstep_and_send_to_cpu(
                        logging_outputs
                    )
                    logging_output = self._reduce_and_log_stats(
                        logging_outputs, sample_size, grad_norm
                    )

                # log whenever there's an XLA compilation, since these
                # slow down training and may indicate opportunities for
                # optimization
                self._check_xla_compilation()
            else:
                if self.cuda and self.cuda_env is not None:
                    # log minimum free memory over the iteration
                    gb_used = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
                    torch.cuda.reset_peak_memory_stats()
                    gb_free = self.cuda_env.total_memory_in_GB - gb_used
                    metrics.log_scalar(
                        "gb_free", gb_free, priority=1500, round=1, weight=0
                    )

                # log stats
                logging_output = self._reduce_and_log_stats(
                    logging_outputs, sample_size, grad_norm
                )

                # clear CUDA cache to reduce memory fragmentation
                if (
                    self.cuda
                    and self.cfg.common.empty_cache_freq > 0
                    and (
                        (self.get_num_updates() + self.cfg.common.empty_cache_freq - 1)
                        % self.cfg.common.empty_cache_freq
                    )
                    == 0
                ):
                    torch.cuda.empty_cache()

        if self.cfg.common.fp16 or self.cfg.common.amp:
            metrics.log_scalar(
                "loss_scale",
                (
                    self.optimizer.scaler.loss_scale
                    if self.cfg.common.fp16
                    else self.optimizer.scaler.get_scale()
                ),
                priority=700,
                round=4,
                weight=0,
            )

        metrics.log_stop_time("train_wall")
        return logging_output
