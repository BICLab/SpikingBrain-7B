import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Dict, Any, List
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from spikingbrain.config import SpikingBrainConfig
from spikingbrain.core.layers import SpikingBrainDecoderLayer, SpikingRMSNorm
from spikingbrain.core.cache import SpikingStateCache


class SpikingBrain7BPreTrainedModel(PreTrainedModel):
    config_class = SpikingBrainConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class SpikingBrain7BModel(SpikingBrain7BPreTrainedModel):
    def __init__(self, config: SpikingBrainConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([
            SpikingBrainDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = SpikingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        state_cache: Optional[SpikingStateCache] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        hidden_states = self.embed_tokens(input_ids)
        all_spikes = []

        for layer in self.layers:
            hidden_states, layer_spikes = layer(hidden_states, state_cache=state_cache)
            all_spikes.append(layer_spikes)

        hidden_states = self.norm(hidden_states)
        return hidden_states, all_spikes


class SpikingBrain7BForCausalLM(SpikingBrain7BPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: SpikingBrainConfig):
        super().__init__(config)
        self.model = SpikingBrain7BModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        state_cache: Optional[SpikingStateCache] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states, all_spikes = self.model(input_ids, state_cache=state_cache)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits, state_cache, all_spikes)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=state_cache,
            hidden_states=hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[SpikingStateCache] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # During sequential decode steps, feed only the single newest token step (T=1)
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "state_cache": past_key_values,
        }
