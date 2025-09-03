import torch
import torch.nn as nn

try:
    from .neuron import spike_fake_quant, SpikeCountBitwiseNode
    spike_is_available = True
except Exception as e:
    print('need https://github.com/BICLab/Int2Spike repo to do fake int2spike, ', e)
    spike_is_available = False

def dynamic_spikes(x, k=3.0):
    vth = x.abs().mean([-1], keepdim=True).float() / k
    vth = vth.clamp(min=1e-5, max=1e4)
    spikes_int = (x / vth).round()

    if spike_is_available:
        spikes_int = spike_fake_quant(spikes_int, lif_quantizer=SpikeCountBitwiseNode(is_bidirectional=True))

    return spikes_int, vth

class QuantLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, w_group_size=128, dynamic_sfr=3.0):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)

        self.k = dynamic_sfr
        self.w_group_size = w_group_size
        self.weight_quantizer = Quantizer(in_features, out_features, w_group_size)

    def forward(self, x):
        # BLD
        assert not self.training
        # if self.w_group_size is not None:
        #     spikes_int, vth = dynamic_spikes(x, self.k)
        #     weight = self.weight_quantizer(self.weight).reshape(self.out_features, -1, self.w_group_size)
        #     spikes_int = spikes_int.reshape(*spikes_int.shape[:-1], 1, -1, self.w_group_size)
        #     o =  (spikes_int.float() * weight).sum(-1) # BLOG # group wise matmul.
        #     o = (o * vth.float()).sum(-1).to(self.weight) # BLO
        # else:
        #     spikes_int, vth = dynamic_spikes(x, self.k)
        #     weight = self.weight_quantizer(self.weight)
        #     o =  spikes_int.float() @ weight # BLO
        #     o = (o * vth.float()).to(self.weight)
        # return o

        spikes_int, vth = dynamic_spikes(x, self.k)
        x = (spikes_int * vth).to(x.dtype)
        weight = self.weight_quantizer(self.weight)
        out = torch.nn.functional.linear(x, weight, self.bias)
        return out

class Quantizer(nn.Module):
    def __init__(self, in_features: int, out_features: int, w_group_size=None):
        super().__init__()
        
        self.out_features = out_features
        self.in_features = in_features
        self.w_group_size = w_group_size
        
        if w_group_size is None:
            shape = (out_features, 1)
        else:
            shape = (out_features, in_features // w_group_size, 1)
        self.register_buffer('scales', torch.ones(shape))
        # using sym quant for simplicity
        self.register_buffer('zeros', None)

    def forward(self, weight):
        # BLD
        assert not self.training
        org_type = weight.dtype
        if self.w_group_size is not None:
            weight = weight.reshape(self.out_features, -1, self.w_group_size)
            weight = (weight / self.scales).round() * self.scales
            return weight.reshape(self.out_features, self.in_features).to(org_type)
        else:
            weight = (weight / self.scales).round() * self.scales
            return weight.to(org_type)
