#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.deepspeed_moe_utils")

import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators, data_utils
# from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig, CheckpointConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf

from user.trainer import DeepspeedETrainer

# from fairseq.pdb import set_trace
# set_trace()

# x = input('Debug? ')
# import debugpy
# debugpy.listen(5678)
# print('waiting for debugger...')
# debugpy.wait_for_client()
# debugpy.breakpoint()
# print('attached')


class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d

def init_deepspeed_env_(cfg: FairseqConfig):
    init_deepspeed_bool = cfg.model.deepspeed_moe and cfg.distributed_training.distributed_rank is not None
    if init_deepspeed_bool:
        import deepspeed
        # os.environ['LOCAL_RANK'] = str(cfg.distributed_training.distributed_rank % cfg.distributed_training.distributed_num_procs)
        os.environ['LOCAL_RANK'] = str(cfg.distributed_training.device_id)
        os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = str(cfg.distributed_training.device_id)

def prepare_deepspeed_config_dict(cfg: FairseqConfig):
    """
    Convert relevant settings in MainzTrain config to DeepSpeed config dictionary
    """
    ZERO_STAGE = 0
    FP16 = cfg.common.fp16
    GRAD_CLIPPING = cfg.optimization.clip_norm # self.opt.get('GRAD_CLIPPING', 0)
    WORLD_SIZE = cfg.distributed_training.distributed_world_size
    GRAD_ACCUM = int(cfg.optimization.update_freq[0])
    deepspeed_config_dict = {}
    deepspeed_config_dict['gradient_accumulation_steps'] = GRAD_ACCUM
    deepspeed_config_dict['steps_per_print'] = cfg.common.log_interval
    deepspeed_config_dict['wall_clock_breakdown'] = False
    INIT_FP16_SCALE_POWER = int(math.log2(cfg.common.fp16_init_scale))
    if ZERO_STAGE > 0:
        logger.info("DeepSpeed ZeRO is turned on. Using fp16 mode. fp16_opt_level is ignored.")
        deepspeed_config_dict['fp16'] = {'enabled': FP16, 'initial_scale_power': INIT_FP16_SCALE_POWER}
    else:
        # Deepspeed amp integration has unresolved bugs right now, so we stick to the fp16 mode
        # This PR resolves part of the bugs: https://github.com/microsoft/DeepSpeed/pull/290
        # setting amp to be enabled to avoid unnecessary model parameter broadcasting
        # deepspeed_config_dict['amp'] = {'enabled': FP16, 'opt_level': self.opt['FP16_OPT_LEVEL']}
        deepspeed_config_dict['amp'] = {'enabled': FP16}
        deepspeed_config_dict['fp16'] = {'enabled': FP16, 'initial_scale_power': INIT_FP16_SCALE_POWER}
    deepspeed_config_dict['zero_optimization'] = {'stage': ZERO_STAGE}
    deepspeed_config_dict['gradient_clipping'] = GRAD_CLIPPING
    #! 'train_batch_size' is a dummy value here. It is required by DeepSpeed, but We use
    # our own batch generator in Mainz, so it doesn't affect the actual training process.
    # It is only used in DeepSpeed for throughput calculation, which we don't look at.
    # The only requirement is that it is divisible by self.opt['world_size'] * self.grad_acc_steps.
    deepspeed_config_dict['train_batch_size'] = 1 * int(WORLD_SIZE) * int(GRAD_ACCUM)
    # deepspeed_config_dict['train_micro_batch_size_per_gpu'] = 1
    logger.info(
        f"{WORLD_SIZE}**, {GRAD_ACCUM}**, "
        f"{cfg.distributed_training.distributed_rank}**, {cfg.distributed_training.distributed_num_procs}**"
    )
    # exit(1)

    return deepspeed_config_dict

import deepspeed
def _init_model_(
    model,
    cfg: FairseqConfig,
    **kwargs,
):
    ds_args = ObjectView({})
    ds_args.deepspeed = True
    ds_args.deepspeed_config = None
    ds_args.local_rank = cfg.distributed_training.distributed_rank % cfg.distributed_training.distributed_num_procs
    tmp_module, _optim, _, _lr_sched = deepspeed.initialize(
        args=ds_args,
        model=model,
        dist_init_required=kwargs.pop('dist_init_required', False),
        config_params=prepare_deepspeed_config_dict(cfg),
        # config_params={'train_micro_batch_size_per_gpu': 1,
        #                 'fp16': {'enabled': cfg.common.fp16},
        #                 'amp': {'enabled': True}}
    ) # setting amp to be enabled to avoid unnecessary model parameter broadcasting
    # print([x for x,y in tmp_module.module.named_modules()][:10])
    return tmp_module

def _load_deepspeed_checkpoint(
        cfg: CheckpointConfig,
        ds_module: deepspeed.DeepSpeedEngine,
        checkpoint_path: str=None,
        **passthrough_args
):
    """
    Load a checkpoint and restore the training iterator.

    *passthrough_args* will be passed through to
    ``trainer.get_train_iterator``.
    """

    suffix = None
    # suffix = trainer.checkpoint_suffix
    if checkpoint_path is not None:
        checkpoint_path = os.path.join(
            os.path.dirname(checkpoint_path),
            "deepspeed_moe",
            os.path.basename(checkpoint_path),
        )
    elif (
        cfg.restore_file == "checkpoint_last.pt"
    ):  # default value of restore_file is 'checkpoint_last.pt'
        checkpoint_path = os.path.join(
            cfg.save_dir, "deepspeed_moe", "checkpoint_last{}.pt".format(suffix)
        )
        first_launch = not PathManager.exists(checkpoint_path)
        if cfg.finetune_from_model is not None and first_launch:
            # if there is no last checkpoint to restore, start the finetune from pretrained model
            # else just use usual logic to load checkpoint, e.g. restart from last checkpoint and etc.
            if PathManager.exists(cfg.finetune_from_model):
                mp = cfg.finetune_from_model
                checkpoint_path = os.path.join(
                    os.path.dirname(mp), 'deepspeed_moe', os.path.basename(mp)
                )
                logger.info(
                    f"Loading pretrained MoE model from {checkpoint_path}: "
                     "optimizer, lr scheduler, meters, dataloader have been reset."
                )
            else:
                raise ValueError(
                    f"--funetune-from-model {cfg.finetune_from_model} does not exist"
                )
    elif suffix is not None:
        checkpoint_path = cfg.restore_file.replace(".pt", suffix + ".pt")
    else:
        checkpoint_path = cfg.restore_file

    if cfg.restore_file != "checkpoint_last.pt" and cfg.finetune_from_model:
        raise ValueError(
            "--finetune-from-model and --restore-file (non-default value) "
            "can not be specified together: " + str(cfg)
        )

    # logger.critical(f"{checkpoint_path}**")
    _load_path, _client_states = ds_module.load_checkpoint(
        checkpoint_path,
        tag='default',
        load_module_strict=True,
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
        load_module_only=True,
    )

    return _load_path, _client_states
    # return extra_state, epoch_itr

def load_deepspeed_state_(
        cfg: FairseqConfig,
        model,
        weights_path: str=None,
        init_kwargs=None,
):
    ds_module: deepspeed.DeepSpeedEngine = _init_model_(
        model, cfg, **(init_kwargs or {}))
    if not weights_path:
        weights_path = f"{cfg.checkpoint.save_dir}/checkpoint_last.pt"
    if not os.path.lexists(weights_path):
        logger.warning(f"Couldn't find {weights_path}**; Skipping load...")
        return ds_module
    _load_path, _client_states = _load_deepspeed_checkpoint(
        cfg.checkpoint,
        ds_module=ds_module,
        checkpoint_path=weights_path,
    )
    if _load_path:
        logger.info(f"Loaded DeepSpeed weights from: ``{_load_path}''.")
    else:
        logger.critical(f"FAILED to load DeepSpeed weights from: ``{_load_path}''.")
        raise RuntimeError
    # print(ds_module)
    # print(ds_module.has_moe_layers)
    return ds_module
    # del tmp_module

def _save_deepspeed_checkpoint(
        cfg: CheckpointConfig,
        save_dir: str,
        trainer: DeepspeedETrainer,
        ds_module: deepspeed.DeepSpeedEngine,
        epoch_itr,
        val_loss,
):
    from fairseq import distributed_utils, meters
    from fairseq.checkpoint_utils import checkpoint_paths, collections

    # only one worker should attempt to create the required dir
    if trainer.data_parallel_rank == 0:
        os.makedirs(cfg.save_dir, exist_ok=True)

    prev_best = getattr(_save_deepspeed_checkpoint, "best", val_loss)
    if val_loss is not None:
        best_function = max if cfg.maximize_best_checkpoint_metric else min
        _save_deepspeed_checkpoint.best = best_function(val_loss, prev_best)

    if cfg.no_save:
        return

    # if not trainer.should_save_checkpoint_on_current_rank:
    #     if trainer.always_call_state_dict_during_save_checkpoint:
    #         trainer.state_dict()
    #     return

    write_timer = meters.StopwatchMeter()
    write_timer.start()

    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()
    logger.info(f"{trainer.get_num_updates()}**")

    def is_better(a, b):
        return a >= b if cfg.maximize_best_checkpoint_metric else a <= b

    suffix = trainer.checkpoint_suffix
    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds["checkpoint{}{}.pt".format(epoch, suffix)] = (
        end_of_epoch
        and not cfg.no_epoch_checkpoints
        and epoch % cfg.save_interval == 0
    )
    checkpoint_conds["checkpoint_{}_{}{}.pt".format(epoch, updates, suffix)] = (
        not end_of_epoch
        and cfg.save_interval_updates > 0
        and updates % cfg.save_interval_updates == 0
    )
    checkpoint_conds["checkpoint_best{}.pt".format(suffix)] = val_loss is not None and (
        not hasattr(_save_deepspeed_checkpoint, "best")
        or is_better(val_loss, _save_deepspeed_checkpoint.best)
    )
    if val_loss is not None and cfg.keep_best_checkpoints > 0:
        worst_best = getattr(_save_deepspeed_checkpoint, "best", None)
        chkpts = checkpoint_paths(
            save_dir,
            pattern=r"checkpoint\.best_{}_(\d+\.?\d*){}\.pt".format(
                cfg.best_checkpoint_metric, suffix
            ),
        )
        if len(chkpts) > 0:
            p = chkpts[-1] if cfg.maximize_best_checkpoint_metric else chkpts[0]
            worst_best = float(p.rsplit("_")[-1].replace("{}.pt".format(suffix), ""))
        # add random digits to resolve ties
        with data_utils.numpy_seed(epoch, updates, val_loss):
            rand_sfx = np.random.randint(0, cfg.keep_best_checkpoints)

        checkpoint_conds[
            "checkpoint.best_{}_{:.3f}{}{}.pt".format(
                cfg.best_checkpoint_metric,
                val_loss,
                rand_sfx,
                suffix,
            )
        ] = worst_best is None or is_better(val_loss, worst_best)
    checkpoint_conds[
        "checkpoint_last{}.pt".format(suffix)
    ] = not cfg.no_last_checkpoints

    # extra_state = {"train_iterator": epoch_itr.state_dict(), "val_loss": val_loss}
    # if hasattr(_save_deepspeed_checkpoint, "best"):
    #     extra_state.update({"best": _save_deepspeed_checkpoint.best})

    checkpoints = [
        os.path.join(save_dir, fn) for fn, cond in checkpoint_conds.items() if cond
    ]
    import shutil
    from torch import distributed
    if len(checkpoints) > 0:
        # trainer._save_deepspeed_checkpoint(checkpoints[0], extra_state)
        try:
            main_ckpt = checkpoints[0]
            ds_module.save_checkpoint(
                            main_ckpt,
                            tag='default',
                            save_latest=False)

            distributed.barrier()
            if trainer.data_parallel_rank == 0:
                for cp in checkpoints[1:]:
                    # PathManager.copy(checkpoints[0], cp, overwrite=True)
                    try:
                        shutil.copytree(main_ckpt, cp, dirs_exist_ok=True)
                    except TypeError as exc:
                        logger.error(f"Failed an overwriting copy to {cp}; will remove "
                                    f"existing checkpoint directory first, which might leave it corrupted.",
                                    exc_info=True)
                        shutil.rmtree(cp, ignore_errors=True)
                        shutil.copytree(main_ckpt, cp)

            #? Wait until RANK==0 copies the checkpoint.
            distributed.barrier()
        except:
            logger.critical(f"{checkpoints[0]}**; Failed copying to {cp}**")
            raise

        write_timer.stop()
        logger.info(
            f"[Rank {trainer.data_parallel_rank}] "
            "Saved deepspeed MoE checkpoint {} (epoch {} @ {} updates, score {}) (writing took {} seconds)".format(
                checkpoints[0], epoch, updates, val_loss, write_timer.sum
            )
        )

    #? If not RANK==0, skip the rest of the function. This's safe if all methods...
    #? used are and remain pure. Otherwise, RANK!=0 workers will have an inconsistent state.
    if trainer.data_parallel_rank == 0:
        pass
    else:
        distributed.barrier()

    if not end_of_epoch and cfg.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        if cfg.keep_interval_updates_pattern == -1:
            checkpoints = checkpoint_paths(
                save_dir, pattern=r"checkpoint_\d+_(\d+){}\.pt".format(suffix)
            )
        else:
            checkpoints = checkpoint_paths(
                save_dir,
                pattern=r"checkpoint_\d+_(\d+){}\.pt".format(suffix),
                keep_match=True,
            )
            checkpoints = [
                x[0]
                for x in checkpoints
                if x[1] % cfg.keep_interval_updates_pattern != 0
            ]

        for old_chk in checkpoints[cfg.keep_interval_updates :]:
            if os.path.lexists(old_chk):
                shutil.rmtree(old_chk, ignore_errors=True)

    if cfg.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(save_dir, pattern=r"checkpoint(\d+)\.pt")
        for old_chk in checkpoints[cfg.keep_last_epochs :]:
            if os.path.lexists(old_chk):
                shutil.rmtree(old_chk)

    if cfg.keep_best_checkpoints > 0:
        # only keep the best N checkpoints according to validation metric
        checkpoints = checkpoint_paths(
            save_dir, pattern=r"checkpoint\.best_{}_(\d+\.?\d*)\.pt".format(cfg.best_checkpoint_metric))
        if not cfg.maximize_best_checkpoint_metric:
            checkpoints = checkpoints[::-1]
        for old_chk in checkpoints[cfg.keep_best_checkpoints:]:
            if os.path.lexists(old_chk):
                shutil.rmtree(old_chk)

    if trainer.data_parallel_rank == 0:
        distributed.barrier()

def save_deepspeed_state_(
        cfg: FairseqConfig,
        model,
        trainer: DeepspeedETrainer,
        save_dir: str=None,
        ckpt_tag=None,
        ds_module: deepspeed.DeepSpeedEngine=None,
        *,
        epoch_itr,
        val_loss,
):
    assert not ckpt_tag
    if ds_module is None:
        ds_module = _init_model_(model, cfg)
    if not save_dir:
        save_dir = f"{cfg.checkpoint.save_dir}/deepspeed_moe"
    if ckpt_tag:
        save_dir = f"{save_dir}/{ckpt_tag}"
    ckpt_path = _save_deepspeed_checkpoint(
        cfg.checkpoint,
        save_dir,
        trainer=trainer,
        ds_module=ds_module,
        epoch_itr=epoch_itr,
        val_loss=val_loss,
    )
    return ckpt_path
    # return f"{save_dir}/default"
