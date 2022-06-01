# Adapted from modules/transformer_layers.py

from typing import Dict, List, Tuple, Optional

import math
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor

from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)

from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder
)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


class TransformerEncoderLayer_MOE(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, index=-1):
        super().__init__()
        self.ep_group = f"ep_size_{args.ep_world_size}"
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8) or 8
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        export = getattr(args, "export", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu") or "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.experts = None
        if (index + 1) % 2 == 0:
            from deepspeed.moe.layer import MoE
            self.expert_counts = torch.zeros(1, args.num_experts, dtype=torch.int64).to('cpu')
            self.fc1 = self.build_fc1(
                self.embed_dim,
                args.encoder_ffn_embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.fc2 = self.build_fc2(
                args.encoder_ffn_embed_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.experts = MoE(
                self.embed_dim,
                expert=nn.Sequential(
                    self.fc1,
                    nn.ReLU(),
                    self.activation_dropout_module,
                    self.fc2,
                ),
                num_experts=args.num_experts,
                ep_size=args.ep_world_size,
                k=args.top_k,
                min_capacity=0, # force set to zero. args.min_capacity,
                noisy_gate_policy="Jitter", # args.noisy_gate_policy,
                capacity_factor=1.0,
                eval_capacity_factor=4.0,
            )
            # raise ValueError(self.experts.expert)
            for p in self.experts.parameters():
                p.expert = True
            self.fc1 = None
            self.fc2 = None
        else:
            self.fc1 = self.build_fc1(
                self.embed_dim,
                args.encoder_ffn_embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.fc2 = self.build_fc2(
                args.encoder_ffn_embed_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward_experts(self, x: torch.Tensor, encoder_padding_mask: torch.Tensor):
        experts_gate_loss = None
        # flatten and pad up to max
        s, b, d_model = x.shape
        num_tokens = s*b
        reshaped_x = x.reshape(num_tokens, 1, d_model)

        max_tensor = torch.tensor([s*b], dtype=torch.int32).to(x.device)
        import deepspeed
        a2a_group = deepspeed.utils.groups._get_expert_parallel_group(self.ep_group)
        torch.distributed.barrier(group=a2a_group)
        torch.distributed.all_reduce(max_tensor, torch.distributed.ReduceOp.MAX, group=a2a_group)

        pad_len = max_tensor - s*b
        assert pad_len >= 0
        used_token = encoder_padding_mask.eq(False).transpose(0, 1).reshape(num_tokens)
        if pad_len > 0:
            pad_tensor = torch.zeros(pad_len, 1, d_model, dtype=x.dtype).to(reshaped_x.device)
            reshaped_x = torch.cat((reshaped_x, pad_tensor), 0)
            pad_tensor2 = torch.zeros(pad_len, dtype=used_token.dtype).to(used_token.device)
            used_token = torch.cat((used_token, pad_tensor2), 0)
        x, experts_gate_loss, exp_counts = self.experts(reshaped_x, used_token)
        self.expert_counts = torch.add(self.expert_counts, exp_counts)
        balance_loss_ratio = 1.0
        experts_gate_loss = experts_gate_loss * balance_loss_ratio

        # remove padding and reshape-back
        if pad_len > 0:
            x = torch.split(x, s*b)[0]
        x = x.reshape(s, b, d_model)
        return x, experts_gate_loss

    def forward(
        self,
        x,
        encoder_padding_mask: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, __ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        experts_gate_loss = None
        if self.experts is not None:
            x, experts_gate_loss = self.forward_experts(x, encoder_padding_mask)
        else:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, experts_gate_loss


class TransformerDecoderLayer_MOE(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, index=-1
    ):
        super().__init__()
        self.ep_group = f"ep_size_{args.ep_world_size}"
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        export = getattr(args, "export", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.experts = None
        if (index + 1) % 2 == 0:
            from deepspeed.moe.layer import MoE
            self.expert_counts = torch.zeros(1, args.num_experts, dtype=torch.int64).to('cpu')
            self.fc1 = self.build_fc1(
                self.embed_dim,
                args.encoder_ffn_embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.fc2 = self.build_fc2(
                args.encoder_ffn_embed_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.experts = MoE(
                self.embed_dim,
                expert=nn.Sequential(
                    self.fc1,
                    nn.ReLU(),
                    self.activation_dropout_module,
                    self.fc2,
                ),
                num_experts=args.num_experts,
                ep_size=args.ep_world_size,
                k=args.top_k,
                min_capacity=0, # force set to zero. args.min_capacity,
                noisy_gate_policy="Jitter", # args.noisy_gate_policy,
                capacity_factor=1.0,
                eval_capacity_factor=4.0,
            )
            # raise ValueError(self.experts.expert)
            for p in self.experts.parameters():
                p.expert = True
            self.fc1 = None
            self.fc2 = None
        else:
            self.fc1 = self.build_fc1(
                self.embed_dim,
                args.decoder_ffn_embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.fc2 = self.build_fc2(
                args.decoder_ffn_embed_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward_experts(self, x: torch.Tensor, self_attn_padding_mask=None):
        experts_gate_loss = None
        # flatten and pad up to max
        s, b, d_model = x.shape
        num_tokens = s*b
        reshaped_x = x.reshape(num_tokens, 1, d_model)

        max_tensor = torch.tensor([s*b], dtype=torch.int32).to(x.device)
        import deepspeed
        a2a_group = deepspeed.utils.groups._get_expert_parallel_group(self.ep_group)
        torch.distributed.barrier(group=a2a_group)
        torch.distributed.all_reduce(max_tensor, torch.distributed.ReduceOp.MAX, group=a2a_group)

        pad_len = max_tensor - s*b
        assert pad_len >= 0
        if self_attn_padding_mask is None:
            used_token = torch.ones(num_tokens, dtype=torch.bool).to(x.device)
        else:
            used_token = self_attn_padding_mask.eq(torch.tensor(False)).transpose(0, 1).reshape(num_tokens)
        if pad_len > 0:
            pad_tensor = torch.zeros(pad_len, 1, d_model, dtype=x.dtype).to(reshaped_x.device)
            reshaped_x = torch.cat((reshaped_x, pad_tensor), 0)
            pad_tensor2 = torch.zeros(pad_len, dtype=used_token.dtype).to(used_token.device)
            used_token = torch.cat((used_token, pad_tensor2), 0)
        x, experts_gate_loss, exp_counts = self.experts(reshaped_x, used_token)
        self.expert_counts = torch.add(self.expert_counts, exp_counts)
        balance_loss_ratio = 1.0
        experts_gate_loss = experts_gate_loss * balance_loss_ratio

        # remove padding and reshape-back
        if pad_len > 0:
            x = torch.split(x, s*b)[0]
        x = x.reshape(s, b, d_model)
        return x, experts_gate_loss

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        experts_gate_loss = None
        if self.experts is not None:
            x, experts_gate_loss = self.forward_experts(x, self_attn_padding_mask)
        else:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None, experts_gate_loss

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn



def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


