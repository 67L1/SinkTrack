import math
import warnings
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging

import math
import warnings
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
import re

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
except:
    pass


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)
from modeling_minicpm import (
    MiniCPMAttention,
    MiniCPMDecoderLayer,
    MiniCPM3Model,
    MiniCPM3ForCausalLM,
    apply_rotary_pos_emb,
    rotate_half
)

class MiniCPMInjectionAttention(MiniCPMAttention):
    """
    MiniCPM Attention with cross-attention injection for the first token.
    Inherits from MiniCPMAttention to reuse its complex weight structure.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            global_prompt_embedding: Optional[torch.Tensor] = None,
            injection_layer_idx: Optional[int] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        is_injection_step = (
                injection_layer_idx is not None
                and self.layer_idx > 0
                and self.layer_idx % injection_layer_idx == 0
                and self.layer_idx <= 50
                and global_prompt_embedding is not None
                and q_len > 1
        )

        if is_injection_step:
            print(f"Injecting global prompt embedding at layer {self.layer_idx}...")
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
            q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
            q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

            compressed_kv_text = self.kv_a_proj_with_mqa(hidden_states)
            compressed_kv_text, k_pe_text = torch.split(compressed_kv_text, [self.kv_lora_rank, self.qk_rope_head_dim],
                                                        dim=-1)
            k_pe_text = k_pe_text.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
            kv_text = (
                self.kv_b_proj(self.kv_a_layernorm(compressed_kv_text))
                .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
                .transpose(1, 2)
            )
            k_nope_text, value_states_text = torch.split(kv_text, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            prompt_len = global_prompt_embedding.shape[1]
            compressed_kv_prompt = self.kv_a_proj_with_mqa(global_prompt_embedding)
            compressed_kv_prompt, k_pe_prompt_ignored = torch.split(compressed_kv_prompt,
                                                                    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            kv_prompt = (
                self.kv_b_proj(self.kv_a_layernorm(compressed_kv_prompt))
                .view(bsz, prompt_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
                .transpose(1, 2)
            )
            k_nope_prompt, value_states_prompt = torch.split(kv_prompt, [self.qk_nope_head_dim, self.v_head_dim],
                                                             dim=-1)
            k_pe_prompt = torch.zeros_like(k_pe_text[:, :, :prompt_len, :])

            kv_seq_len = value_states_text.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

            cos, sin = self.rotary_emb(value_states_text, seq_len=kv_seq_len)
            q_pe, k_pe_text = apply_rotary_pos_emb(q_pe, k_pe_text, cos, sin, position_ids)

            query_states = torch.cat([q_nope, q_pe], dim=-1)
            key_states_text = k_pe_text.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
            key_states_text[:, :, :, : self.qk_nope_head_dim] = k_nope_text
            key_states_text[:, :, :, self.qk_nope_head_dim:] = k_pe_text
            key_states_prompt = k_pe_prompt.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
            key_states_prompt[:, :, :, : self.qk_nope_head_dim] = k_nope_prompt
            key_states_prompt[:, :, :, self.qk_nope_head_dim:] = k_pe_prompt

            if past_key_value is not None:
                key_states_text, value_states_text = past_key_value.update(key_states_text, value_states_text,
                                                                           self.layer_idx, {"sin": sin, "cos": cos})

            query_first = query_states[:, :, :1, :]
            attn_weights_first = torch.matmul(query_first, key_states_prompt.transpose(2, 3)) * self.softmax_scale
            attn_weights_first = nn.functional.softmax(attn_weights_first, dim=-1, dtype=torch.float32).to(
                query_states.dtype)
            attn_output_first = torch.matmul(attn_weights_first, value_states_prompt)

            attn_output_others = None
            if q_len > 1:
                query_others = query_states[:, :, 1:, :]
                key_others = key_states_text
                value_others = value_states_text

                attn_weights_others = torch.matmul(query_others, key_others.transpose(2, 3)) * self.softmax_scale
                if attn_weights_others.size() != (bsz, self.num_heads, q_len - 1, kv_seq_len):
                    raise ValueError(
                        f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                        f" {attn_weights_others.size()}"
                    )

                if attention_mask is not None:
                    causal_mask_others = attention_mask[:, :, 1:, :]
                    attn_weights_others = attn_weights_others + causal_mask_others

                attn_weights_others = nn.functional.softmax(attn_weights_others, dim=-1, dtype=torch.float32).to(
                    query_states.dtype)


                attn_weights_others = nn.functional.dropout(attn_weights_others, p=self.attention_dropout,
                                                            training=self.training)
                attn_output_others = torch.matmul(attn_weights_others, value_others)

            if attn_output_others is not None:
                attn_output = torch.cat([attn_output_first, attn_output_others], dim=2)
            else:
                attn_output = attn_output_first

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
            attn_output = self.o_proj(attn_output)

            return attn_output, None, past_key_value

        else:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )



class MiniCPMInjectionDecoderLayer(MiniCPMDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MiniCPMInjectionAttention(config=config, layer_idx=layer_idx)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            global_prompt_embedding: Optional[torch.Tensor] = None,
            injection_layer_idx: Optional[int] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            global_prompt_embedding=global_prompt_embedding,
            injection_layer_idx=injection_layer_idx,
            **kwargs,
        )

        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MiniCPM3ModelWithPromptInjection(MiniCPM3Model):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MiniCPMInjectionDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            injection_layer_idx: Optional[int] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.config.scale_emb

        global_prompt_embedding = None
        if (
                injection_layer_idx is not None and
                inputs_embeds.shape[1] > 1
        ):

            global_prompt_embedding = inputs_embeds

        self._use_flash_attention_2 = False
        self._use_sdpa = False

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)


            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                global_prompt_embedding=global_prompt_embedding,
                injection_layer_idx=injection_layer_idx,
            )
            hidden_states = layer_outputs[0]

            if (
                    injection_layer_idx is not None and
                    hidden_states.shape[1] > 1 and
                    (idx + 1) % injection_layer_idx == 0 and
                    idx != (self.config.num_hidden_layers - 1)
            ):
                logger.info(f"Updating global prompt embedding after layer {idx}")
                global_prompt_embedding = hidden_states

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )



class MiniCPM3ForCausalLMWithPromptInjection(MiniCPM3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MiniCPM3ModelWithPromptInjection(config)
        self.post_init()

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        injection_layer_idx = kwargs.pop("injection_layer_idx", None)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs
        )

        model_inputs["injection_layer_idx"] = injection_layer_idx
        return model_inputs

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            injection_layer_idx: Optional[int] = None,  
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # global_prompt_embedding=global_prompt_embedding,
            injection_layer_idx=injection_layer_idx,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states / (self.config.hidden_size / self.config.dim_model_base))
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )