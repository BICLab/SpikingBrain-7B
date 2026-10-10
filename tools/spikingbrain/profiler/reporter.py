import json
from typing import Dict, Any
from tabulate import tabulate


class SpikeProfilerReporter:
    """
    Visual and structured telemetry reporter formatting micro-level sparsity,
    synaptic operations (SOPs), and dynamic energy consumption metrics.
    """
    @staticmethod
    def print_terminal_report(telemetry_data: Dict[str, Any]):
        global_rate = telemetry_data.get("global_spike_rate", 0.0)
        global_sparsity = telemetry_data.get("global_sparsity_percent", 0.0)
        energy = telemetry_data.get("energy_metrics", {})
        layer_rates = telemetry_data.get("layer_spike_rates", {})

        print("\n" + "=" * 75)
        print("[*] SPIKINGBRAIN-7B NEUROMORPHIC SPARSITY & ENERGY PROFILE REPORT [*]")
        print("=" * 75)
        print(f"Global Spike Rate:         {global_rate:.4f} ({global_rate * 100:.2f}%)")
        print(f"Global Sparsity:           {global_sparsity:.2f}%")
        print(f"Dense Baseline FLOPs:      {energy.get('dense_flops', 0):,}")
        print(f"Synaptic Operations (SOPs):{energy.get('sops', 0):,}")
        print(f"Dense Baseline Energy:     {energy.get('dense_energy_joules', 0.0):.6e} J")
        print(f"SpikingBrain Energy:       {energy.get('snn_energy_joules', 0.0):.6e} J")
        print(f"Energy Efficiency Ratio:   {energy.get('energy_efficiency_ratio', 0.0):.2f}x savings")
        print("-" * 75)

        table_rows = []
        for layer_name, rate in sorted(layer_rates.items()):
            table_rows.append([layer_name, f"{rate:.4f}", f"{(1.0 - rate) * 100:.2f}%"])

        if table_rows:
            headers = ["Layer / Neuron Module", "Spike Density", "Sparsity (%)"]
            print(tabulate(table_rows, headers=headers, tablefmt="github"))
        print("=" * 75 + "\n")

    @staticmethod
    def export_json(telemetry_data: Dict[str, Any], filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(telemetry_data, f, indent=2)
