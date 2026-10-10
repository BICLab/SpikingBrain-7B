import torch
from spikingbrain.config import SpikingBrainConfig
from spikingbrain.core.attention import LinearSpikeAttention
from spikingbrain.core.cache import SpikingStateCache


def test_linear_spike_attention_forward():
    config = SpikingBrainConfig(hidden_size=64, num_attention_heads=4, num_hidden_layers=1)
    attn = LinearSpikeAttention(config, layer_idx=0)
    
    batch_size, seq_len = 2, 8
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    context, q_spikes, k_spikes = attn(hidden_states)
    assert context.shape == (batch_size, seq_len, config.hidden_size)
    # Verify binary spike gating {0, 1}
    assert torch.all((q_spikes == 0.0) | (q_spikes == 1.0))
    assert torch.all((k_spikes == 0.0) | (k_spikes == 1.0))


def test_attention_cache_decode_step():
    config = SpikingBrainConfig(hidden_size=64, num_attention_heads=4, num_hidden_layers=1)
    attn = LinearSpikeAttention(config, layer_idx=0)
    cache = SpikingStateCache()
    
    # Prefill step
    prompt = torch.randn(1, 4, config.hidden_size)
    _ = attn(prompt, state_cache=cache)
    
    # Decode step (T=1)
    token = torch.randn(1, 1, config.hidden_size)
    context, _, _ = attn(token, state_cache=cache)
    assert context.shape == (1, 1, config.hidden_size)
