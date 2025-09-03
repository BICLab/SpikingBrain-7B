from vllm import ModelRegistry
from transformers import AutoConfig, AutoModel

def register_7B_model():
    from .modeling_gla_swa import GLAswaForCausalLM
    from .configuration_gla_swa import GLAswaConfig

    AutoConfig.register("gla_swa", GLAswaConfig)

    ModelRegistry.register_model(
        "GLAswaForCausalLM",
        "vllm_hymeta.model_for_7B.modeling_gla_swa:GLAswaForCausalLM",
    )