import gradio as gr
from spikingbrain.webui.dashboard import build_webui
from spikingbrain.config import SpikingBrainConfig


def test_build_webui():
    cfg = SpikingBrainConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    demo = build_webui(config=cfg)
    assert isinstance(demo, gr.Blocks)
