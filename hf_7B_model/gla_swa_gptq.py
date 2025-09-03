
#### This code should be placed at /usr/local/lib/python3.12/dist-packages/gptqmodel/models/definitions/gla_swa.py
#### refer to https://github.com/ModelCloud/GPTQModel/blob/main/gptqmodel/models/definitions/deepseek_v2.py
#### And you should add register this model at /usr/local/lib/python3.12/dist-packages/gptqmodel/models/auto.py

from ..base import BaseGPTQModel

class GLAswaGPTQ(BaseGPTQModel):
    # Strict=True -> all layer_modules must exists in model
    # Some models (deepseek2-lite) dynamically create lora modules based on config.rank
    layer_modules_strict = False # Cause some are linear attn and has gk_proj, some not

    base_modules = ["model.embeddings", "model.norm"]
    pre_lm_head_norm_module = "model.norm"

    layers_node = "model.layers"
    layer_type = "HybridBlock"
    layer_modules = [
        ["attn.q_proj", "attn.k_proj", "attn.v_proj"],
        ["attn.o_proj"], # ["attn.gk_proj"], 

        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]