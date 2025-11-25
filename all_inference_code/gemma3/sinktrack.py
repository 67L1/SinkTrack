import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

# 导入所有需要的原始gemma3类和函数
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3DecoderLayer,
    Gemma3TextModel,
    Gemma3Model,
    Gemma3ForConditionalGeneration,
    Gemma3CausalLMOutputWithPast,
    Gemma3ModelOutputWithPast,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import logging
from transformers.utils import (
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torchdynamo_compiling,
    logging,
)
from transformers.masking_utils import create_causal_mask, create_masks_for_generate, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast

logger = logging.get_logger(__name__)


class Gemma3InjectionAttention(Gemma3Attention):
    """
    继承自 Gemma3Attention，增加了在指定层向第一个 token 注入全局图像信息的功能。
    注入通过对第一个 token 使用 Cross-Attention 实现，而其他 token 保持 Self-Attention。
    """

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            # 新增参数
            global_image_embedding: Optional[torch.Tensor] = None,
            injection_layer_idx: Optional[int] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        bsz, q_len, _ = hidden_states.shape

        # 判断是否执行注入操作
        is_injection_step = (
                injection_layer_idx is not None
                and self.layer_idx % injection_layer_idx == 0
                and self.layer_idx != 0
                and self.layer_idx <= 25
                and global_image_embedding is not None
                and q_len > 1
        )

        if is_injection_step:
            print(f"Injecting global image info in layer {self.layer_idx}...")
            # --- 注入逻辑 ---

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            # 1. 准备所有 Q, K, V
            # Q from text
            query_states = self.q_proj(hidden_states)
            # K, V from text (for self-attention part)
            key_states_text = self.k_proj(hidden_states)
            value_states_text = self.v_proj(hidden_states)

            # K, V from global image embedding (for cross-attention part)
            if global_image_embedding.dim() == 2:
                global_image_embedding = global_image_embedding.unsqueeze(1)  # Shape: (bsz, 1, hidden_size)

            # K, V 投影层需要 (bsz, seq_len, hidden_size) 形状的输入
            img_bsz, img_len, img_hidden = global_image_embedding.shape
            key_states_image = self.k_proj(global_image_embedding)
            value_states_image = self.v_proj(global_image_embedding)

            # 2. Reshape & Norm
            query_states = self.q_norm(query_states.view(hidden_shape).transpose(1, 2))
            key_states_text = self.k_norm(key_states_text.view(hidden_shape).transpose(1, 2))
            value_states_text = value_states_text.view(hidden_shape).transpose(1, 2)

            key_states_image = self.k_norm(key_states_image.view(bsz, img_len, -1, self.head_dim).transpose(1, 2))
            value_states_image = value_states_image.view(bsz, img_len, -1, self.head_dim).transpose(1, 2)

            # 3. RoPE (仅应用于文本部分的 Q 和 K)
            cos, sin = position_embeddings
            query_states, key_states_text = apply_rotary_pos_emb(query_states, key_states_text, cos, sin)

            # 4. KV 缓存更新 (使用文本自身的 K/V)
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states_text, value_states_text = past_key_value.update(key_states_text, value_states_text,
                                                                           self.layer_idx, cache_kwargs)

            # 5. 分离计算
            # 5.1 第一个 Token: Cross-Attention
            query_first = query_states[:, :, :1, :]  # (bsz, num_heads, 1, head_dim)

            key_image_repeated = repeat_kv(key_states_image, self.num_key_value_groups)
            value_image_repeated = repeat_kv(value_states_image, self.num_key_value_groups)

            # 使用 PyTorch 内置的高效 SDPA
            attn_output_first = torch.nn.functional.scaled_dot_product_attention(
                query_first,
                key_image_repeated,
                value_image_repeated,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            )

            # 5.2 其余 Tokens: Self-Attention
            query_others = query_states[:, :, 1:, :]  # (bsz, num_heads, q_len-1, head_dim)
            key_others = key_states_text
            value_others = value_states_text

            # 调整 attention mask
            # causal_mask has shape (bsz, 1, q_len, kv_len)
            causal_mask_others = attention_mask[:, :, 1:, :] if attention_mask is not None else None

            # Gemma3的eager attention实现细节较多，这里为了清晰，我们同样用SDPA
            # 注意：如果原始实现有更复杂的逻辑（如softcapping），这里需要相应适配
            attn_output_others = torch.nn.functional.scaled_dot_product_attention(
                query_others,
                repeat_kv(key_others, self.num_key_value_groups),
                repeat_kv(value_others, self.num_key_value_groups),
                attn_mask=causal_mask_others,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=causal_mask_others is None and query_others.shape[2] > 1,  # 仅在没有mask时才启用内置causal
                scale=self.scaling,
            )

            # 6. 合并结果
            attn_output = torch.cat([attn_output_first, attn_output_others], dim=2)
            attn_weights = None  # SDPA不直接返回权重，设为None
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)

            return attn_output, attn_weights

        else:
            # --- 标准自注意力逻辑 (直接调用父类) ---
            # 从 kwargs 中移除我们自定义的参数，以防父类方法报错
            # kwargs.pop("global_image_embedding", None)
            # kwargs.pop("injection_layer_idx", None)
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs,
            )

        # Reshape and project



class Gemma3InjectionDecoderLayer(Gemma3DecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # 替换为我们自定义的 Attention 模块
        self.self_attn = Gemma3InjectionAttention(config, layer_idx)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings_global: torch.Tensor,
            position_embeddings_local: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            # 新增参数
            global_image_embedding: Optional[torch.Tensor] = None,
            injection_layer_idx: Optional[int] = None,
            **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        # 调用自定义的 Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            # 传递新参数
            global_image_embedding=global_image_embedding,
            injection_layer_idx=injection_layer_idx,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Gemma3TextModelWithInjection(Gemma3TextModel):
    def __init__(self, config):
        super().__init__(config)
        # 替换为我们自定义的 Decoder Layer 列表
        self.layers = nn.ModuleList(
            [Gemma3InjectionDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            # 新增参数
            global_image_embedding: Optional[torch.Tensor] = None,
            injection_layer_idx: Optional[int] = None,
            st_ed_idx = None,
            **kwargs,
    ) -> Union[Tuple, Gemma3ModelOutputWithPast]:
        # 此处省略了与原始Gemma3TextModel.forward中完全相同的输入检查和mask准备代码
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=attention_mask[decoder_layer.attention_type],  # 注意gemma3的mask是dict
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                # 传递新参数
                global_image_embedding=global_image_embedding,
                injection_layer_idx=injection_layer_idx,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if global_image_embedding is not None and (idx) % injection_layer_idx == 0 and idx != 0:
                # global_image_embedding = hidden_states
                # if idx <= 13:
                #     global_image_embedding = global_image_embedding
                # else:
                global_image_embedding = hidden_states[:, st_ed_idx[0]:st_ed_idx[1] + 1, :]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Gemma3ModelWithInjection(Gemma3Model):
    def __init__(self, config):
        super().__init__(config)
        # 替换为我们自定义的 Text Model
        self.language_model = Gemma3TextModelWithInjection(config.text_config)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            # 新增参数
            injection_layer_idx: Optional[int] = None,
            **lm_kwargs,
    ) -> Union[tuple, Gemma3ModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Replace image id woth PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.image_token_id >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # Merge text and images
        global_image_embedding = None
        start_indices = None
        end_indices = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                special_image_mask = special_image_mask.all(-1)
            else:
                special_image_mask = input_ids == self.config.image_token_id

            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            # 计算全局图像嵌入 (B, NumPatches, Hidden) -> (B, Hidden)
            global_image_embedding = image_features
            if injection_layer_idx is not None:
                print(f"Preparing global image embedding for injection.")
                # 1. Create a boolean mask for the image tokens
                image_mask = (input_ids == self.config.image_token_id)

                # 2. Find the start indices
                # Cast the boolean mask to an integer tensor and find the index of the first occurrence of 1.
                start_indices = torch.argmax(image_mask.int(), dim=1)

                # 3. Find the end indices
                # Flip the mask along the sequence dimension
                flipped_mask = torch.flip(image_mask, dims=[1])
                # Find the index of the first 'True' in the flipped mask
                end_indices_rev = torch.argmax(flipped_mask.int(), dim=1)
                # Convert the end indices back to the original coordinate space
                sequence_length = input_ids.shape[1]
                end_indices = sequence_length - 1 - end_indices_rev
                print("Start Indices:", start_indices)
                print("End Indices:", end_indices)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config.get_text_config(),
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            if token_type_ids is not None and inputs_embeds.shape[1] != 1:
                # We need to pass an additional mask function to account for token type ids, and it needs to be an `or`

                # First find where a new image block starts: 1 if image and previous not image
                # The images cannot attend to future images, but can attend to all prev images and to itself bidirectionally
                is_image = (token_type_ids == 1).to(cache_position.device)
                new_image_start = is_image & ~nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
                image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
                image_group_ids = torch.where(is_image, image_group_ids, torch.full_like(token_type_ids, -1))
                mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                    token_type_ids.to(cache_position.device), image_group_ids, self.config.mm_tokens_per_image
                )

            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # 调用自定义的 language model，并传递新参数
        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            global_image_embedding=global_image_embedding,
            injection_layer_idx=injection_layer_idx,
            st_ed_idx=(start_indices, end_indices),
            **lm_kwargs,
        )

        return Gemma3ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class Gemma3ForConditionalGenerationWithInjection(Gemma3ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # 替换为我们自定义的 Model
        self.model = Gemma3ModelWithInjection(config)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            # 新增参数
            injection_layer_idx: Optional[int] = None,
            **lm_kwargs,
    ) -> Union[tuple, Gemma3CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            # 传递新参数
            injection_layer_idx=injection_layer_idx,
            **lm_kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1]:].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Gemma3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """确保自定义参数在 generate 循环中被传递"""
        # 弹出我们的自定义参数，以免父类方法报错
        injection_layer_idx = kwargs.pop("injection_layer_idx", None)

        # 调用父类方法获取标准输入
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)

        # 将自定义参数加回去
        model_inputs["injection_layer_idx"] = injection_layer_idx

        return model_inputs