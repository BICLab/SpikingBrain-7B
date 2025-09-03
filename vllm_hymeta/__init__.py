
def register():

    return "vllm_hymeta.platform.HymetaCudaPlatform"

def register_model():
    from .model_for_7B import register_7B_model
    register_7B_model()
