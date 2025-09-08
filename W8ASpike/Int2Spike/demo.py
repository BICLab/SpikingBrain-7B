import torch
from neuron import (
    SpikeCountBinaryLIFNode,
    SpikeCountTernaryLIFNode,
    SpikeCountBitwiseNode
)
torch.manual_seed(42)
mean, std = 0, 1
size = (1024, 2048)
x = torch.normal(mean=mean, std=std, size=size)
xmin, xmax = x.min(), x.max()

# === Demo: Convert int4 tensor into 0/1 binary spike trains ===
qmin, qmax = 0, 15
x_scale = (xmax - xmin) / (qmax - qmin)
x_zp = qmin - x.min() / x_scale
x_q1 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

lif1 = SpikeCountBinaryLIFNode()
spike1 = lif1(x_q1.to(torch.float32)) 
rate1 = lif1.firing_rate()
print(f"int4 to 0/1: Time steps = {spike1.shape[0]}, firing rate = {rate1}")
lif1.visualize_spike(filename=f"Binary_Lif_T{spike1.shape[0]} _N20.png", title=f"BinaryLif(Tokens=20 TimeStep={spike1.shape[0]} FiringRate={rate1:.2f})")

# === Demo: Convert int4 tensor into -1/0/1 ternary spike trains ===
qmin, qmax = -8, 7
x_scale = (xmax - xmin) / (qmax - qmin)
x_zp = qmin - x.min() / x_scale
x_q2 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

lif2 = SpikeCountTernaryLIFNode()
spike2 = lif2(x_q2.to(torch.float32)) 
rate2 = lif2.firing_rate()
print(f"int4 to -1/0/1: Time steps = {spike2.shape[0]}, firing rate = {rate2}")
lif2.visualize_spike(filename=f"Ternary_Lif_T{spike2.shape[0]} _N20.png", title=f"Ternary_Lif(Tokens=20 TimeStep={spike2.shape[0]} FiringRate={rate2:.2f})")

# === Demo: Convert int4 tensor into bitwise spike trains ===
qmin, qmax = 0, 15
x_scale = (xmax - xmin) / (qmax - qmin)
x_zp = qmin - x.min() / x_scale
x_q1 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

lif3 = SpikeCountBitwiseNode()
spike3 = lif3(x_q1.to(torch.float32)) 
rate3 = lif3.firing_rate()
print(f"int4 to Bitwise: Time steps = {spike3.shape[0]}, firing rate = {rate3}")
lif3.visualize_spike(filename=f"Bitwise_Lif_T{spike3.shape[0]} _N20.png", title=f"Bitwise_Lif(Tokens=20 TimeStep={spike3.shape[0]} FiringRate={rate3:.2f})")

qmin, qmax = -8, 7
x_scale = (xmax - xmin) / (qmax - qmin)
x_zp = qmin - x.min() / x_scale
x_q2 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

lif4 = SpikeCountBitwiseNode(is_bidirectional=True)
spike4 = lif4(x_q2.to(torch.float32)) 
rate4 = lif4.firing_rate()
print(f"int4 to Bitwise: Time steps = {spike4.shape[0]}, firing rate = {rate4}")
lif4.visualize_spike(filename=f"Bitwise_Lif_T{spike4.shape[0]} _N20_bidirectional.png", title=f"Bitwise_Lif(Tokens=20 TimeStep={spike4.shape[0]} FiringRate={rate4:.2f} Bidirectional)")

qmin, qmax = -8, 7
x_scale = (xmax - xmin) / (qmax - qmin)
x_zp = qmin - x.min() / x_scale
x_q2 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

lif5 = SpikeCountBitwiseNode(is_bidirectional=True, is_two_complement=True)
spike5 = lif5(x_q2.to(torch.float32)) 
rate5 = lif5.firing_rate()
print(f"int4 to Bitwise: Time steps = {spike5.shape[0]}, firing rate = {rate5}")
lif5.visualize_spike(filename=f"Bitwise_Lif_T{spike5.shape[0]} _N20_complement.png", title=f"Bitwise_Lif(Tokens=20 TimeStep={spike5.shape[0]} FiringRate={rate5:.2f} Complement)")

# # === Demo: Convert int6 tensor into 0/1 binary spike trains ===
# qmin, qmax = 0, 63
# x_scale = (xmax - xmin) / (qmax - qmin)
# x_zp = qmin - x.min() / x_scale
# x_q1 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

# lif1 = SpikeCountBinaryLIFNode()
# spike1 = lif1(x_q1.to(torch.float32)) 
# rate1 = lif1.firing_rate()
# print(f"int6 to 0/1: Time steps = {spike1.shape[0]}, firing rate = {rate1}")

# # === Demo: Convert int6 tensor into -1/0/1 ternary spike trains ===
# qmin, qmax = -32, 31
# x_scale = (xmax - xmin) / (qmax - qmin)
# x_zp = qmin - x.min() / x_scale
# x_q2 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

# lif2 = SpikeCountTernaryLIFNode()
# spike2 = lif2(x_q2.to(torch.float32)) 
# rate2 = lif2.firing_rate()
# print(f"int6 to -1/0/1: Time steps = {spike2.shape[0]}, firing rate = {rate2}")

# # === Demo: Convert int6 tensor into bitwise spike trains ===
# qmin, qmax = 0, 63
# x_scale = (xmax - xmin) / (qmax - qmin)
# x_zp = qmin - x.min() / x_scale
# x_q1 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

# lif3 = SpikeCountBitwiseNode()
# spike3 = lif3(x_q1.to(torch.float32)) 
# rate3 = lif3.firing_rate()
# print(f"int6 to Bitwise: Time steps = {spike3.shape[0]}, firing rate = {rate3}")

# # === Demo: Convert int8 tensor into 0/1 binary spike trains ===
# qmin, qmax = 0, 255
# x_scale = (xmax - xmin) / (qmax - qmin)
# x_zp = qmin - x.min() / x_scale
# x_q1 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

# lif1 = SpikeCountBinaryLIFNode()
# spike1 = lif1(x_q1.to(torch.float32)) 
# rate1 = lif1.firing_rate()
# print(f"int8 to 0/1: Time steps = {spike1.shape[0]}, firing rate = {rate1}")

# # === Demo: Convert int8 tensor into -1/0/1 ternary spike trains ===
# qmin, qmax = -128, 127
# x_scale = (xmax - xmin) / (qmax - qmin)
# x_zp = qmin - x.min() / x_scale
# x_q2 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

# lif2 = SpikeCountTernaryLIFNode()
# spike2 = lif2(x_q2.to(torch.float32)) 
# rate2 = lif2.firing_rate()
# print(f"int8 to -1/0/1: Time steps = {spike2.shape[0]}, firing rate = {rate2}")

# # === Demo: Convert int8 tensor into bitwise spike trains ===
# qmin, qmax = 0, 255
# x_scale = (xmax - xmin) / (qmax - qmin)
# x_zp = qmin - x.min() / x_scale
# x_q1 = torch.round(x / x_scale + x_zp).clamp(qmin, qmax)

# lif3 = SpikeCountBitwiseNode()
# spike3 = lif3(x_q1.to(torch.float32)) 
# rate3 = lif3.firing_rate()
# print(f"int8 to Bitwise: Time steps = {spike3.shape[0]}, firing rate = {rate3}")
