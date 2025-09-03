
import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Iterable

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed.parallel_state import (
    get_pp_group, get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size)
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.models.utils import PPMissingLayer, is_pp_missing_parameter, make_layers, maybe_prefix
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsV0Only, SupportsPP
from vllm.sequence import IntermediateTensors

from .gla_attention import GatedLinearAttention
from .configuration_gla_swa import GLAswaConfig
from .gla_cache import GLACacheManager, GLACacheParams

class GLAswaRotaryEmbedding(CustomOp):
    name = "GLAswaRotaryEmbedding"

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: float,
        is_neox_style: bool,
        cache_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position
        self.base = base
        self.is_neox_style = is_neox_style
        self.cache_dtype = cache_dtype
        cache = self._compute_cos_sin_cache().to(cache_dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm import _custom_ops as ops
        self.cos_sin_cache = self.cos_sin_cache.to(positions.device)
        query_cast = query.to(self.cache_dtype)
        key_cast = key.to(self.cache_dtype)
        ops.rotary_embedding(positions, query_cast, key_cast, self.head_size,
                             self.cos_sin_cache, self.is_neox_style)
        query = query_cast.to(query.dtype)
        key = key_cast.to(key.dtype)
        return query, key


class GLAswaGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        layer_idx: int = None,
        prefix: str = "mlp",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gate_up, _ = self.gate_up_proj(x) # 第二个返回的是 skip_bias_add=True 时的 bias
        x = self.act_fn(gate_up) # silu(gate) * up
        x, _ = self.down_proj(x)
        return x


class FlashAttention(nn.Module):

    def __init__(
        self,
        config: GLAswaConfig,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        rope_theta: float = 100000.0,
        sliding_window: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        layer_idx: int = None,
        prefix: str = "flash_attn",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads

        assert self.total_num_heads % tp_size == 0, \
            f"Total number of attention heads {self.total_num_heads} must be divisible by tensor model parallel size {tp_size}."
        self.tp_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0, \
                f"Total number of key-value heads {self.total_num_kv_heads} must be divisible by tensor model parallel size {tp_size}."
        else:
            assert tp_size % self.total_num_kv_heads == 0, \
                f"Tensor model parallel size {tp_size} must be divisible by total number of key value heads {self.total_num_kv_heads}."
        self.tp_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        
        self.head_dim = head_dim
        self.q_size = self.tp_heads * self.head_dim
        self.kv_size = self.tp_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.prefix = prefix

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.attn = Attention(
            self.tp_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.tp_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window+1,
            prefix=f"{prefix}.attn",
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs
    ) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        qkv, _ = self.qkv_proj(hidden_states) 
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = attn_metadata.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class GLAswaBlock(nn.Module):

    def __init__(
        self,
        config: GLAswaConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "decoder"
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.quant_config = quant_config
        self.cache_config = cache_config
        self.prefix = prefix

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.attn_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.norm_eps,
        )
        if self.layer_idx in config.attn_layers:
            self.attn = FlashAttention(
                config=config,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                head_dim=self.head_dim,
                num_kv_heads=config.num_key_value_heads,
                rope_theta=config.rope_theta,
                sliding_window=config.sliding_window,
                quant_config=quant_config,
                cache_config=cache_config,
                layer_idx=layer_idx,
                prefix=f"{prefix}.attn"
            )
        else:
            self.attn = GatedLinearAttention(
                config=config,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                gate_logit_normalizer=config.gate_logit_normalizer,
                gate_low_rank_dim=config.gate_low_rank_dim,
                layer_idx=layer_idx,
                quant_config=quant_config,
                prefix=f"{prefix}.attn"
            )
        self.mlp_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.norm_eps,
        )
        self.mlp = GLAswaGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            layer_idx=layer_idx,
            prefix=f"{prefix}.mlp"
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[GLACacheParams] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        residual = hidden_states
        
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(
            hidden_states=hidden_states,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata
        )
        
        hidden_states, residual = self.mlp_norm(hidden_states, residual=residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, residual


class GLAswaModel(nn.Module):
    
    def __init__(
        self,
        config: GLAswaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        scheduler_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embeddings = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=self.vocab_size,
            )
        else:
            self.embeddings = PPMissingLayer()

        def layer_fn(prefix: str):
            layer_idx = int(prefix.split(".")[-1])
            return GLAswaBlock(
                config=config, 
                layer_idx=layer_idx,
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=prefix
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            num_hidden_layers=config.num_hidden_layers,
            layer_fn=layer_fn,
            prefix=f"{prefix}.layers",
        )
        linear_layer_nums = sum(1 for i in range(config.num_hidden_layers)
                                if i not in config.attn_layers)
        # linear_layer_nums = config.num_hidden_layers - len(config.attn_layers)
        max_slots_number = scheduler_config.max_num_seqs # Maximum number of sequences to be processed in a single iteration.
        self.cache_shape = (linear_layer_nums, max_slots_number, 
                            config.num_attention_heads //
                            get_tensor_model_parallel_world_size(),
                            config.head_dim, config.head_dim)
        
        _dummy = torch.zeros(1)
        self._dtype = _dummy.dtype
        del _dummy

        self.gla_cache = GLACacheManager(
            dtype=torch.bfloat16,
            cache_shape=self.cache_shape,
        )

        rope_theta = getattr(config, "rope_theta", 10000)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        if hasattr(config, "max_model_len") and isinstance(
                config.max_model_len, int):
            max_position_embeddings = min(config.max_position_embeddings,
                                          config.max_model_len)
        
        self.rotary_emb = GLAswaRotaryEmbedding(
            head_size=head_dim,
            rotary_dim=config.rotary_dim if hasattr(config, "rotary_dim") else head_dim,
            max_position=max_position_embeddings,
            base=int(rope_theta),
            is_neox_style=True,
            cache_dtype=torch.bfloat16,
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.norm_eps
            )
        else:
            self.norm = PPMissingLayer()
        
    def _clear_prefill_cache(self, attn_metadata,
                            gla_cache_tensors: torch.Tensor, **kwargs):
        seq_to_slot_maps = {}
        seq_id_map = sum(list(kwargs["request_ids_to_seq_ids"].values()), [])
        for _, seq_to_slot_map in (
                self.gla_cache.cache_indices_mapping.items()):
            seq_to_slot_maps.update(seq_to_slot_map)

        slots_to_clear = []
        for _prefill_id in range(getattr(attn_metadata, "num_prefills", 0)):
            if _prefill_id >= len(seq_id_map):
                break
            seq_id = seq_id_map[_prefill_id]
            # context_len = 0 represents the prefill request is newly created
            # and has not been processed yet, so we can clear the cache for this seq_id
            if attn_metadata.context_lens_tensor[_prefill_id] == 0 \
                    and seq_id in seq_to_slot_maps:
                slots_to_clear.append(seq_to_slot_maps[seq_id])
        
        if slots_to_clear:
            slots_tensor = torch.tensor(slots_to_clear,
                                        device=gla_cache_tensors.device,
                                        dtype=torch.long)
            gla_cache_tensors[:, slots_tensor, ...] = 0

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(input_ids)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, IntermediateTensors]:
        
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return None
        
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if "request_ids_to_seq_ids" not in kwargs:
            kwargs["request_ids_to_seq_ids"] = {}
        if "finished_requests_ids" not in kwargs:
            kwargs["finished_requests_ids"] = []

        (
            gla_cache_tensors,
            state_indices_tensor,    
        ) = self.gla_cache.current_run_tensors(**kwargs)

        if getattr(attn_metadata, "num_prefills", 0) > 0:
            self._clear_prefill_cache(
                attn_metadata, gla_cache_tensors, **kwargs
            )
        gla_cache_params = GLACacheParams(
            gla_cache_tensors, state_indices_tensor
        )

        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                hidden_states = self.embeddings(input_ids)
            else:
                hidden_states = inputs_embeds
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors['hidden_states']
            residual = intermediate_tensors['residual']
        
        attn_metadata.rotary_emb = self.rotary_emb
        gla_cache_index = 0

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            _caches = None
            if i not in self.config.attn_layers:
                current_state_layer = gla_cache_index
                _caches = gla_cache_params.at_layer_idx(current_state_layer)
                gla_cache_index += 1

            hidden_states, residual = layer(
                hidden_states=hidden_states,
                positions=positions,
                kv_caches=_caches,
                attn_metadata=attn_metadata,
                # residual=residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": torch.zeros(hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device),
                # "residual": residual # None 就别传
                # File "/opt/conda/lib/python3.10/site-packages/vllm/worker/model_runner.py", line 2042, in forward
                # self.input_buffers[key].copy_(intermediate_tensors[key],
                # TypeError: copy_(): argument 'other' (position 1) must be Tensor, not NoneType
            })
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class GLAswaForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsV0Only, SupportsPP):
    
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    def __init__(self, *, 
                vllm_config: VllmConfig,
                prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self._lora_config = lora_config
        self.CONCAT_FFN = True

        self.unpadded_vocab_size = config.vocab_size
        if hasattr(vllm_config.model_config, "max_model_len"):
            self.config.max_model_len = vllm_config.model_config.max_model_len
        
        self.model = GLAswaModel(
            self.config,
            quant_config=quant_config,
            cache_config=vllm_config.cache_config,
            scheduler_config=vllm_config.scheduler_config,
            prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                self.config.hidden_size,
                org_num_embeddings=self.config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        
            self.logits_processor = LogitsProcessor(
                self.unpadded_vocab_size,
                self.config.vocab_size
            )
        else:
            self.lm_head = PPMissingLayer()
        
        self.lm_head.float()
        flash_layer_count = sum(1 for i in range(self.config.num_hidden_layers) if i in config.attn_layers)
        self.kv_cache = [torch.tensor([]) for _ in range(flash_layer_count)]
        return

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.model.gla_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs
        )
    
    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.model.gla_cache.get_seqlen_agnostic_capture_inputs(
            batch_size
        )

    def get_input_embeddings(
        self, 
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids) 
    
    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, intermediate_tensors, 
                                    inputs_embeds, **kwargs)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                        sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states.float(),
                                        sampling_metadata)
        return logits

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype,
        device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
            "residual":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
        })
    
    # https://zhuanlan.zhihu.com/p/1908151478557839879
    def load_weights(self, weights: 
            Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        
        def which_layer(name: str) -> int:
            if "layers" in name:
                after_layer = name.split("layers")[-1]
                return int(after_layer.split(".")[1])
            return None
        
        def is_densed_mlp_weight(name: str) -> bool:
            return ".mlp." in name

        def load_mlp_weight(name: str, loaded_weight: torch.Tensor,
                                   self) -> None:
            if not self.CONCAT_FFN: # CONCAT_FFN is True by default
                if "gate_proj" in name:
                    name = name.replace("gate_proj", "w1", 1)
                elif "up_proj" in name:
                    name = name.replace("up_proj", "w3", 1)
                elif "down_proj" in name:
                    name = name.replace("down_proj", "w2", 1)
            else:
                if "gate_proj" in name:
                    name = name.replace("gate_proj", "gate_up_proj", 1)
                    loaded_shard_id = 0
                elif "up_proj" in name:
                    name = name.replace("up_proj", "gate_up_proj", 1)
                    loaded_shard_id = 1
            if is_pp_missing_parameter(name, self):
                return
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            
            if not self.CONCAT_FFN:
                weight_loader(param, loaded_weight)
            else:
                if "gate_up_proj" in name:
                    weight_loader(param, loaded_weight, loaded_shard_id)
                elif "down_proj" in name:
                    weight_loader(param, loaded_weight)
                else:
                    raise AssertionError(
                        "MLP weight not in [gate_up_proj, down_proj]")
            
            loaded_params.add(name)
            return
        
        def is_attn_weight(name: str) -> bool:
            return "attn" in name

        def load_attn_weight(name: str, loaded_weight: torch.Tensor,
                                  self) -> None:
            flash_mha_params_mapping = [
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
            ]
            for (param_name, weight_name, shard_id) in flash_mha_params_mapping:
                if weight_name not in name or "gk_proj" in name:
                    continue
                if is_pp_missing_parameter(name, self):
                    return
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if is_pp_missing_parameter(name, self):
                    return
                if "gk_proj.0." in name:
                    name = name.replace("gk_proj.0.", "gk_proj.linear0.")
                elif "gk_proj.1" in name:
                    name = name.replace("gk_proj.1.", "gk_proj.linear1.")
                param = params_dict[name]
                # print(name, f"{param.shape=}, {loaded_weight.shape=}")
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

            return
        
        def is_layer_norm_weight(name: str) -> bool:
            return "norm" in name and not name.endswith(
                ".bias") and name in params_dict

        def load_basic_weight(name: str, loaded_weight: torch.Tensor,
                              self) -> None:
            if is_pp_missing_parameter(name, self):
                return
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            # print(name, f"{param.shape=}, {loaded_weight.shape=}")
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
            return
        
        # for key in params_dict.keys():
        #     print(key)

        for name, loaded_weight in weights:
            weight_at_layer = which_layer(name)
            if weight_at_layer is not None and weight_at_layer >= self.config.num_hidden_layers:
                continue
            
            if is_densed_mlp_weight(name):
                load_mlp_weight(name, loaded_weight, self)
                continue
            if is_layer_norm_weight(name):
                load_basic_weight(name, loaded_weight, self)
                continue
            if is_attn_weight(name):
                load_attn_weight(name, loaded_weight, self)
                continue
            
            load_basic_weight(name, loaded_weight, self)
        
        return loaded_params