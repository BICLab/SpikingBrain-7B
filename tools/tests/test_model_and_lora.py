import torch
from spikingbrain.config import SpikingBrainConfig
from spikingbrain.core.modeling_spikingbrain import SpikingBrain7BForCausalLM
from spikingbrain.peft.spike_lora import inject_spike_lora, merge_spike_lora, SpikeLoRALinear


def test_model_forward_and_loss():
    config = SpikingBrainConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    model = SpikingBrain7BForCausalLM(config)
    
    input_ids = torch.randint(0, 100, (2, 8))
    labels = input_ids.clone()
    
    output = model(input_ids, labels=labels)
    assert output.loss is not None
    assert output.logits.shape == (2, 8, 100)
    assert output.loss.item() > 0.0


def test_spike_lora_injection_and_merge():
    config = SpikingBrainConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    model = SpikingBrain7BForCausalLM(config)
    
    # Inject Spike-LoRA
    count = inject_spike_lora(model, r=4, lora_alpha=8.0, target_modules=["q_proj", "v_proj"])
    assert count == 4  # 2 layers * 2 projections
    
    # Verify gradient flow into adapter parameters
    input_ids = torch.randint(0, 100, (1, 4))
    labels = input_ids.clone()
    loss = model(input_ids, labels=labels).loss
    loss.backward()
    
    adapter_found = False
    for module in model.modules():
        if isinstance(module, SpikeLoRALinear):
            adapter_found = True
            assert module.lora_A.grad is not None
    assert adapter_found

    # Test merge
    merge_spike_lora(model)
