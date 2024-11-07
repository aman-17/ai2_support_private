from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import OlmoConfig
from hf_olmo import OLMoConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, Sampler, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers)


class FlippedSiluAndMul(SiluAndMul):
    """OLMo is trained with SwiGLU with flipped halves."""
    
    def forward(self, x: torch.Tensor):
        a, b = x.chunk(2, dim=-1)
        flipped = torch.cat((b, a), dim=-1)
        return super().forward(flipped)


class OlmoAttention(nn.Module):
    def __init__(
        self,
        config: OlmoConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        self.total_num_heads = config.num_attention_heads
        
        assert self.hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size
        self.head_dim = self.hidden_size // self.total_num_heads
        if hasattr(self.config, "use_norm_reordering") and self.config.use_norm_reordering:
            self.max_position_embeddings = config.max_sequence_length
        else:
            self.max_position_embeddings = config.max_position_embeddings
        
        self.rope_theta = config.rope_theta
        self.clip_qkv = config.clip_qkv

        # self.use_norm_reordering = getattr(config, "use_norm_reordering", False)
        if hasattr(self.config, "use_norm_reordering") and self.config.use_norm_reordering:
            bias=config.include_bias
        else:
            bias=config.attention_bias

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            bias=bias,
            quant_config=quant_config,
        )
        

        if hasattr(self.config, "use_norm_reordering") and self.config.use_norm_reordering and config.attention_layer_norm:
            self.k_norm = RMSNorm(
                (config.d_model // config.n_heads) * config.effective_n_kv_heads,
                eps=config.layer_norm_eps,
            )
            self.q_norm = RMSNorm(
                config.hidden_size,
                eps=config.layer_norm_eps,
            )
            
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale=self.scaling,
            cache_config=cache_config,
            quant_config=quant_config
        )
        if hasattr(self.config, "use_norm_reordering") and self.config.use_norm_reordering:
            o_proj_bias = config.include_bias
        else:
            o_proj_bias = config.attention_bias

        self.o_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=o_proj_bias,
            quant_config=quant_config,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        if hasattr(self.config, "use_norm_reordering") and self.config.use_norm_reordering:
            q = self.q_norm.forward_native(q)
            k = self.k_norm.forward_native(k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class OlmoMLP(nn.Module):
    def __init__(
        self,
        config: OlmoConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        if hasattr(self.config, "use_norm_reordering") and self.config.use_norm_reordering:
            try:
                self.intermediate_size = config.intermediate_size
            except AttributeError:
                if config.mlp_hidden_size is not None:
                    self.intermediate_size = config.mlp_hidden_size // 2
                else:
                    self.intermediate_size = (config.d_model * config.mlp_ratio) // 2
        else:
            self.intermediate_size = config.intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
        )
        
        if hasattr(self.config, "use_norm_reordering") and self.config.use_norm_reordering:
            self.act_fn = FlippedSiluAndMul() 
        else:
            self.act_fn = SiluAndMul()
        
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

    def forward(
            self, 
            x: torch.Tensor
        ) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class OlmoDecoderLayer(nn.Module):
    def __init__(self,
                config: OlmoConfig,
                cache_config: Optional[CacheConfig] = None,
                quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        # Store config as instance variable
        self.config = config
        
        self.self_attn = OlmoAttention(config, cache_config, quant_config)
        self.mlp = OlmoMLP(config, quant_config)
        
        if hasattr(self.config, "use_norm_reordering") and self.config.use_norm_reordering:
            self.norm_after = config.norm_after
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.input_layernorm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, bias=False)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, bias=False)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        
        if hasattr(self.config, "use_norm_reordering") and self.config.use_norm_reordering:
            if self.norm_after:
                hidden_states = self.self_attn(positions, hidden_states, kv_cache,
                                            attn_metadata)
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states = self.input_layernorm(hidden_states)
                hidden_states = self.self_attn(positions, hidden_states, kv_cache,
                                            attn_metadata)
            hidden_states = hidden_states + residual

            residual = hidden_states
            if self.norm_after:
                hidden_states = self.mlp(hidden_states)
                hidden_states = self.post_attention_layernorm(hidden_states)
            else:
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states
        else:
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(positions, hidden_states, kv_cache,
                                        attn_metadata)
            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states


class OlmoModel(nn.Module):
    def __init__(self,
                 config: Union[OlmoConfig, OLMoConfig],
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.config = config

        if hasattr(self.config, "use_norm_reordering") and self.config.use_norm_reordering:
            self.embed_tokens = VocabParallelEmbedding(config.embedding_size,
                                                      config.hidden_size)
            self.layers = nn.ModuleList([
                OlmoDecoderLayer(config, cache_config, quant_config)
                for layer_idx in range(config.num_hidden_layers)
            ])
            self.norm = RMSNorm(
                config.hidden_size,
                eps=config.layer_norm_eps,
            )
        else:
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                      config.hidden_size)
            self.start_layer, self.end_layer, self.layers = make_layers(
                config.num_hidden_layers,
                lambda prefix: OlmoDecoderLayer(config, cache_config, quant_config),
                prefix=f"{prefix}.layers")
            self.norm = nn.LayerNorm(config.hidden_size,
                                   elementwise_affine=False,
                                   bias=False)
            self.make_empty_intermediate_tensors = (
                make_empty_intermediate_tensors_factory(["hidden_states"],
                                                      config.hidden_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        """
        if hasattr(self.config, "use_norm_reordering") and self.config.use_norm_reordering:
            inputs_embeds = self.embed_tokens(input_ids)
            hidden_states = inputs_embeds
            for layer_idx, decoder_layer in enumerate(self.layers):
                hidden_states = decoder_layer(
                    positions,
                    hidden_states,
                    kv_caches[layer_idx],
                    attn_metadata,
                )
            hidden_states = self.norm(hidden_states)
            return hidden_states
        else:
            if get_pp_group().is_first_rank:
                inputs_embeds = self.embed_tokens(input_ids)
                hidden_states = inputs_embeds
            else:
                assert intermediate_tensors is not None
                hidden_states = intermediate_tensors["hidden_states"]

            for i in range(self.start_layer, self.end_layer):
                hidden_states = self.layers[i](
                    positions,
                    hidden_states,
                    kv_caches[i - self.start_layer],
                    attn_metadata,
                )

            if not get_pp_group().is_last_rank:
                return IntermediateTensors({"hidden_states": hidden_states})

            hidden_states = self.norm(hidden_states)
            return hidden_states
        

class OlmoForCausalLM(nn.Module):
    """
    Extremely barebones HF model wrapper.
    """

    def __init__(self,
                 config: OlmoConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.model = OlmoModel(config, cache_config, quant_config)

        if hasattr(self.config, "use_norm_reordering") and self.config.use_norm_reordering:
            if config.weight_tying:
                self.lm_head = self.model.embed_tokens
            else:
                self.unpadded_vocab_size = config.vocab_size
                self.lm_head = ParallelLMHead(
                    config.embedding_size,
                    config.hidden_size,
                    org_num_embeddings=config.embedding_size,
                    quant_config=quant_config,
                )
            self.logits_processor = LogitsProcessor(config.embedding_size)
            self.sampler = Sampler()
        else:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.unpadded_vocab_size = config.vocab_size
                self.lm_head = ParallelLMHead(
                    self.unpadded_vocab_size,
                    config.hidden_size,
                    org_num_embeddings=config.vocab_size,
                    quant_config=quant_config,
                )
            self.logits_processor = LogitsProcessor(config.vocab_size)
            self.sampler = get_sampler()
            self.make_empty_intermediate_tensors = (
                self.model.make_empty_intermediate_tensors)


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if hasattr(self.config, "use_norm_reordering") and self.config.use_norm_reordering:
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
            )
        else:
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
                intermediate_tensors=intermediate_tensors,
            )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                     sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def _create_map(self):
        mapper = {}
        for layer_i in range(self.config.n_layers):
            mapper[f"model.transformer.blocks.{layer_i}.att_proj.weight"] = f"model.layers.{layer_i}.self_attn.qkv_proj.weight"
            mapper[f"model.transformer.blocks.{layer_i}.attn_out.weight"] = f"model.layers.{layer_i}.self_attn.o_proj.weight"
            mapper[f"model.transformer.blocks.{layer_i}.ff_proj.weight"] = f"model.layers.{layer_i}.mlp.gate_up_proj.weight"
            mapper[f"model.transformer.blocks.{layer_i}.ff_out.weight"] = f"model.layers.{layer_i}.mlp.down_proj.weight"

            mapper[f"model.transformer.blocks.{layer_i}.attn_norm.weight"] = f"model.layers.{layer_i}.input_layernorm.weight"
            mapper[f"model.transformer.blocks.{layer_i}.ff_norm.weight"] = f"model.layers.{layer_i}.post_attention_layernorm.weight"
            mapper[f"model.transformer.blocks.{layer_i}.k_norm.weight"] = f"model.layers.{layer_i}.self_attn.k_norm.weight"
            mapper[f"model.transformer.blocks.{layer_i}.q_norm.weight"] = f"model.layers.{layer_i}.self_attn.q_norm.weight"

        mapper["model.transformer.ln_f.weight"] = "model.norm.weight"
        mapper["model.transformer.wte.weight"] = "model.embed_tokens.weight"
        mapper["model.transformer.ff_out.weight"] = "lm_head.weight"
        return mapper

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        
        if hasattr(self.config, "use_norm_reordering") and self.config.use_norm_reordering:
            mapper = self._create_map()
            
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                continue
                
            if hasattr(self.config, "use_norm_reordering") and self.config.use_norm_reordering:
                if self.config.weight_tying and "lm_head.weight" in name:
                    continue
            else:
                if self.config.tie_word_embeddings and "lm_head.weight" in name:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if not (hasattr(self.config, "use_norm_reordering") and self.config.use_norm_reordering):
                    if is_pp_missing_parameter(name, self):
                        continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if not (hasattr(self.config, "use_norm_reordering") and self.config.use_norm_reordering):
                    if is_pp_missing_parameter(name, self):
                        continue
                if hasattr(self.config, "use_norm_reordering") and self.config.use_norm_reordering:
                    param = params_dict[mapper.get(name, name)]
                else:
                    param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
                weight_loader(param, loaded_weight)