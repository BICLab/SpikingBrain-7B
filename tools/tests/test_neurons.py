import torch
import pytest
from spikingbrain.core.neurons import LIFNeuron, ALIFNeuron, FastSigmoidSurrogate, ATanSurrogate


def test_surrogate_gradients():
    x = torch.tensor([-1.5, 0.0, 1.5], requires_grad=True)
    spikes = FastSigmoidSurrogate.apply(x, 2.0)
    assert torch.all(spikes == torch.tensor([0.0, 1.0, 1.0]))
    
    loss = spikes.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad[1] > x.grad[0]  # Gradient peaks near threshold boundary (x=0)


def test_lif_dynamics():
    neuron = LIFNeuron(v_threshold=1.0, leak_factor=0.5, reset_mode="hard")
    
    # Step 1: Sub-threshold current
    current_1 = torch.tensor([0.8])
    spikes_1, v_mem_1 = neuron(current_1)
    assert spikes_1.item() == 0.0
    assert pytest.approx(v_mem_1.item(), 1e-4) == 0.8
    
    # Step 2: Suprathreshold current causes spike & hard reset
    current_2 = torch.tensor([0.7])
    spikes_2, v_mem_2 = neuron(current_2, v_mem=v_mem_1)
    # V = 0.5 * 0.8 + 0.7 = 1.1 >= 1.0 -> Spike! Hard reset -> 0.0
    assert spikes_2.item() == 1.0
    assert pytest.approx(v_mem_2.item(), 1e-4) == 0.0


def test_alif_adaptation():
    neuron = ALIFNeuron(v_threshold=1.0, leak_factor=0.5, adaptation_scale=0.5)
    current = torch.tensor([1.5])
    
    spikes_1, v_1, adapt_1 = neuron(current)
    assert spikes_1.item() == 1.0
    assert adapt_1.item() == 1.0  # Adaptation increases upon firing
