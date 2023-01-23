# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Type, Tuple

import torch
from fairseq import metrics, utils
from fairseq import criterions
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


logger = logging.getLogger(__name__)


#region helper types
try:
    from typing import TypedDict
except:
    from typing_extensions import TypedDict


# Dict[str, List[Optional[torch.Tensor]]]
_ExtraNetOutput = TypedDict(
    "_ExtraNetOutput",
    {
        "attn": List[torch.Tensor],
        "inner_states": List[torch.Tensor],
    },
    # total=False,
)

# NetOutput = Tuple[torch.Tensor, Optional[_ExtraNetOutput]]
NetOutput = Tuple[torch.Tensor, _ExtraNetOutput]

Sample = TypedDict(
    "Sample",
    {
        "id": int, # id,
        "nsentences": int, # len(samples),
        "ntokens": int, # ntokens,
        "net_input": TypedDict(
            'NetInput',
            {
                "src_tokens": torch.Tensor,
                "src_lengths": torch.Tensor
            }),
        "target": torch.Tensor, # target,
    }
)
#endregion



@dataclass
class CombinedModelCriterionConfig(FairseqDataclass):
    base_criterion: str = field(
        default='',
        metadata={"help": "fairseq-recognized name of the non-model criterion"}
    )
    base_criterion_config: str = field(
        default='{}',
        metadata={"help": "base criterion config as a string of a json dict"}
    )
    # loss_weights: Dict[str, float] = field(
    loss_weights: str = field(
        default="",
        metadata={"help": "weights for the loss terms as a str of a json dict"},
    )
    # log_keys: List[str] = field(
    #     default_factory=list,
    #     metadata={"help": "additional output keys to log"},
    # )
    crit_world_size: int = II('distributed_training.distributed_world_size')


@register_criterion("model_and_base", dataclass=CombinedModelCriterionConfig)
class CombinedModelCriterion(FairseqCriterion):
    """
        This criterion relies on the model to supply losses.
        The losses should be a dictionary of name -> scalar returned by
        the model either by including it in the net_output dict or by
        implementing a get_losses(net_output, sample) method. The final loss is
        a scaled sum of all losses according to weights in loss_weights.
        If no weights are provided, then all losses are scaled by 1.0.

        The losses will be automatically logged. Additional keys from
        net_output dict can be logged via the log_keys parameter.
    """

    def __init__(
            self,
            task,
            cfg: CombinedModelCriterionConfig,
            # base_criterion='', base_criterion_config='{}',
            # loss_weights=None, log_keys=None,
    ):
        super().__init__(task)
        import json
        self.cfg = cfg
        self.loss_weights = json.loads(cfg.loss_weights) or {}
        # self.log_keys = cfg.log_keys
        self.log_keys = {}
        #*** Build the "main/base" criterion
        cls: Type[FairseqCriterion] = criterions.CRITERION_REGISTRY[cfg.base_criterion]
        conf_cls = criterions.CRITERION_DATACLASS_REGISTRY[cfg.base_criterion]
        bc_cfg = conf_cls(**json.loads(cfg.base_criterion_config))
        bc_cfg._name = cfg.base_criterion
        self.base_crit = cls.build_criterion(bc_cfg, task)
        # from fairseq.pdb import set_trace
        # set_trace()
        # self.base_crit: FairseqCriterion = criterions.build_criterion(bc_cfg, task)

    def base_crit_name(self):
        return self.base_crit.__class__.__name__

    def get_losses(
            self,
            model: FairseqEncoderDecoderModel,
            net_output: NetOutput,
            sample: Sample,
    ) -> Dict[str, torch.Tensor]:
        base_loss, __ = self.base_crit.compute_loss(model, net_output, sample, reduce=True)
        model_losses = {}
        if hasattr(model, 'get_losses'):
            model_losses = model.get_losses(net_output, sample)
        return {
            f"{self.base_crit_name()}": base_loss,
            **model_losses
        }

    def forward(self, model, sample, reduce=True):
        net_output: dict = model(**sample["net_input"])

        sample_size = net_output[1].get("sample_size", sample["ntokens"])
        scaled_losses = {}

        # if hasattr(model, "get_losses"):
        #     losses = model.get_losses(net_output, sample)
        # elif isinstance(net_output, dict) and "losses" in net_output:
        #     losses = net_output["losses"]
        # else:
        #     raise Exception("Could not retrieve losses")

        losses = self.get_losses(model, net_output, sample)

        for lk, lv in losses.items():
            coef = self.loss_weights.get(lk, 1.0)
            if coef == 1.0 and lk == self.base_crit_name():
                coef = self.loss_weights.get('base_crit', 1.0)
            if coef != 0 and lv is not None:
                scaled_losses[lk] = coef * lv.float()

        # raise ValueError(f"Yaay-2! {sample_size}; {losses} -> {scaled_losses}.")
        loss: torch.Tensor = sum(scaled_losses.values())
        # from fairseq.pdb import set_trace
        # set_trace()
        if reduce and loss.numel() > 1:
            loss = loss.sum()

        logging_output = {
            "loss": loss.data,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            "_world_size": self.cfg.crit_world_size,
        }

        for lk in (self.log_keys or []):
            if lk in net_output and net_output[lk] is not None:
                logging_output[lk] = float(net_output[lk])

        for lk, l in scaled_losses.items():
            logging_output[f"loss_{lk}"] = l.detach().item()

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)

        builtin_keys = {
            "loss",
            "ntokens",
            "nsentences",
            "sample_size",
            "_world_size",
        }

        world_size = utils.item(
            sum(log.get("_world_size", 0) for log in logging_outputs)
        )

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k, 0) for log in logging_outputs)
                if k.startswith("loss_"):
                    metrics.log_scalar(k, val / sample_size, sample_size, round=3)
                else:
                    metrics.log_scalar(k, val / world_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

