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
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
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
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf

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
        os.environ['LOCAL_RANK'] = str(cfg.distributed_training.distributed_rank % cfg.distributed_training.distributed_num_procs)

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
        # deepspeed_config_dict['amp'] = {'enabled': FP16, 'opt_level': self.opt['FP16_OPT_LEVEL']}
        deepspeed_config_dict['fp16'] = {'enabled': FP16, 'initial_scale_power': INIT_FP16_SCALE_POWER}
    deepspeed_config_dict['zero_optimization'] = {'stage': ZERO_STAGE}
    deepspeed_config_dict['gradient_clipping'] = GRAD_CLIPPING
    #! 'train_batch_size' is a dummy value here. It is required by DeepSpeed, but We use
    # our own batch generator in Mainz, so it doesn't affect the actual training process.
    # It is only used in DeepSpeed for throughput calculation, which we don't look at.
    # The only requirement is that it is divisible by self.opt['world_size'] * self.grad_acc_steps.
    print(WORLD_SIZE, GRAD_ACCUM)
    # exit(1)
    deepspeed_config_dict['train_batch_size'] = 1 * int(WORLD_SIZE) * int(GRAD_ACCUM)

    # If there are advanced overriding deepspeed settings, apply them to the deepspeed config dictionary
    # if 'DEEPSPEED_CONFIG_OVERRIDES' in self.opt:
    #     for k, v in self.opt['DEEPSPEED_CONFIG_OVERRIDES'].items():
    #         deepspeed_config_dict[k] = v

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

def load_deepspeed_state_(
    model,
    cfg: FairseqConfig,
    weights_path: str=None,
    init=True,
    init_kwargs=None,
):
    tmp_module = _init_model_(model, cfg, **(init_kwargs or {}))
    if not weights_path:
        try:
            weights_path = f"{cfg.checkpoint.save_dir}/deepspeed_moe"
        except:
            pass
    if not weights_path:
        try:
            weights_path = f"{cfg.checkpoint.save_dir}/deepspeed_moe"
        except:
            pass
    tmp_module: deepspeed.DeepSpeedEngine
    _load_path, _client_states = tmp_module.load_checkpoint(
        weights_path,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
        load_module_only=True,
    )
    if _load_path:
        logger.info(f"Loaded DeepSpeed weights from: ``{_load_path}''.")
    # print(tmp_module)
    # print(tmp_module.has_moe_layers)
    return tmp_module
    # del tmp_module

def save_deepspeed_state_(
    model,
    cfg: FairseqConfig,
    weights_path: str=None,
    mod=None,
    init=True,
):
    if mod is not None:
        tmp_module = mod
    else:
        tmp_module = _init_model_(model, cfg)
    if not weights_path:
        weights_path = f"{cfg.checkpoint.save_dir}/deepspeed_moe"
    tmp_module: deepspeed.DeepSpeedEngine
    tmp_module.save_checkpoint(f"{cfg.checkpoint.save_dir}/deepspeed_moe",
                        # tag='',
                        save_latest=not cfg.checkpoint.no_last_checkpoints)
    del tmp_module
