from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Olmo1124Config
from vllm import ModelRegistry,LLM, SamplingParams
from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

class FlippedSiluAndMul(SiluAndMul):
    """OLMo is trained with SwiGLU with flipped halves."""

    def forward(self, x: torch.Tensor):
        a, b = x.chunk(2, dim=-1)
        flipped = torch.cat((b, a), dim=-1)
        return super().forward(flipped)

class OlmoAttention(nn.Module):
    """
    This is the attention block where the output is computed as
    ``Attention(LN(x))`` in ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(
        self,
        config: Olmo1124Config,
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

        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)
        self.head_dim = self.hidden_size // self.total_num_heads
        self.max_position_embeddings = config.max_sequence_length
        self.rope_theta = config.rope_theta
        self.clip_qkv = config.clip_qkv

        # Attention input projection. Projects x -> (q, k, v)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            bias=config.include_bias,
            quant_config=quant_config,
        )
        # attention_layer_norm = True
        if config.attention_layer_norm:
            # TODO: finish adding qk norm and norm_after
            self.k_norm = RMSNorm(
                (config.hidden_size // config.num_attention_heads) * config.num_key_value_heads,
                eps=config.rms_norm_eps,
                #elementwise_affine=config.attention_layer_norm_with_affine,
                #bias=False,
            )
            self.q_norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )

        # Rotary embeddings.
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.scaling = self.head_dim**-0.5
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              scale=self.scaling,
                              cache_config=cache_config,
                              quant_config=quant_config)

        # Attention output projection.
        self.o_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=config.include_bias,
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
        q = self.q_norm.forward_native(q)
        k = self.k_norm.forward_native(k)
        #q = self.q_norm(q) 
        #k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class OlmoMLP(nn.Module):
    """
    This is the MLP block where the output is computed as
    ``MLP(LN(x))`` in ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(
        self,
        config: Olmo1124Config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        try:
            self.intermediate_size = config.intermediate_size
        except AttributeError:
            if config.mlp_hidden_size is not None:
                self.intermediate_size = config.mlp_hidden_size // 2
            else:
                self.intermediate_size = (config.d_model * config.mlp_ratio) // 2

        # Feed-forward input projection.
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
        )

        # Activation function.
        self.act_fn = FlippedSiluAndMul()

        # Feed-forward output projection.
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class OlmoDecoderLayer(nn.Module):
    """
    This is a typical transformer block where the output is
    computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self,
                 config: Olmo1124Config,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        # Attention block.
        self.self_attn = OlmoAttention(config, cache_config, quant_config)

        # MLP block.
        self.mlp = OlmoMLP(config, quant_config)

        # LayerNorm

        self.norm_after = config.norm_after
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        """
        self.input_layernorm = nn.LayerNorm(config.hidden_size,
                                            elementwise_affine=False,
                                            bias=False)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size,
                                                     elementwise_affine=False,
                                                     bias=False)
        """

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention block.
        residual = hidden_states
        if self.norm_after:
            hidden_states = self.self_attn(positions, hidden_states, kv_cache,
                                           attn_metadata)
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.self_attn(positions, hidden_states, kv_cache,
                                           attn_metadata)
        hidden_states = hidden_states + residual

        # MLP block.
        residual = hidden_states
        if self.norm_after:
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.post_attention_layernorm(hidden_states)
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class OlmoModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        
        self.config = vllm_config.model_config.hf_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config
        self.prefix = prefix
        
        self.embed_tokens = VocabParallelEmbedding(
            self.config.embedding_size,
            self.config.hidden_size
        )
        
        self.layers = nn.ModuleList([
            OlmoDecoderLayer(
                vllm_config=vllm_config,
                prefix=f"{prefix}.layers.{layer_idx}"
            )
            for layer_idx in range(self.config.num_hidden_layers)
        ])
        
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.layer_norm_eps
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        """
        # Get embeddings of input
        # shape: (batch_size, seq_len, d_model)
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        
        # Apply transformer layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_caches[layer_idx],
                attn_metadata=attn_metadata,
            )
            
        # Apply final layer normalization
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Olmo1124ForCausalLM(nn.Module):
    """
    Extremely barebones HF model wrapper.
    """
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config
        self.lora_config = vllm_config.lora_config
        
        self.model = OlmoModel(
            vllm_config=self.config,
            prefix=prefix
        )
        
        if self.config.weight_tying:
            self.lm_head = self.model.embed_tokens
        else:
            self.unpadded_vocab_size = self.config.vocab_size
            self.lm_head = ParallelLMHead(
                config.embedding_size,
                config.hidden_size,
                org_num_embeddings=config.embedding_size,
                quant_config=self.quant_config,
            )
            
        self.logits_processor = LogitsProcessor(self.config.embedding_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
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
        mapper = self._create_map()
        print("mapper", mapper)
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            # With tie_word_embeddings, we can skip lm_head.weight
            # The weight might appear unnecessarily in the files if the model is
            # processed with quantization, LoRA, fine-tuning, etc.
            if self.config.weight_tying and "lm_head.weight" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[mapper.get(name, name)]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
