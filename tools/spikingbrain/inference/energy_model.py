from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class HardwareEnergySpec:
    """
    Standard CMOS energy dissipation specifications in picoJoules (pJ).
    Default reference values: 45nm standard CMOS process node.
    """
    process_node: str = "45nm"
    mac_energy_pj: float = 4.6   # 32-bit Floating-Point Multiply-Accumulate
    ac_energy_pj: float = 0.9    # 32-bit Fixed/Floating-Point Accumulate (Addition only)


class NeuromorphicEnergyModel:
    """
    Evaluates dynamic energy consumption of SpikingBrain layers by converting
    binary spike multiplications into accumulate-only operations (ACs/SOPs).
    """
    def __init__(self, spec: HardwareEnergySpec = HardwareEnergySpec()):
        self.spec = spec

    def compute_energy(
        self,
        dense_flops: int,
        spike_rate: float,
    ) -> Dict[str, Any]:
        """
        Calculates energy comparison:
        dense_flops: baseline floating point operations if fully dense
        spike_rate: empirical spike density (0.0 to 1.0)
        """
        # Dense baseline energy: all operations are MACs
        dense_energy_j = (dense_flops * self.spec.mac_energy_pj) * 1e-12

        # SNN energy: inactive neurons consume ~0 pJ; active neurons execute ACs (SOPs)
        sops = int(dense_flops * spike_rate)
        snn_energy_j = (sops * self.spec.ac_energy_pj) * 1e-12

        energy_savings_ratio = (dense_energy_j / snn_energy_j) if snn_energy_j > 0 else (self.spec.mac_energy_pj / self.spec.ac_energy_pj)

        return {
            "dense_flops": dense_flops,
            "sops": sops,
            "spike_rate": spike_rate,
            "dense_energy_joules": dense_energy_j,
            "snn_energy_joules": snn_energy_j,
            "energy_efficiency_ratio": energy_savings_ratio,
        }
