from vllm.platforms import PlatformEnum, _Backend

from vllm.logger import init_logger

logger = init_logger(__name__)

from vllm.platforms.cuda import NvmlCudaPlatform # 不用 CudaPlatform 避免在 ray 初始化前就初始化了 cuda 环境


class HymetaCudaPlatform(NvmlCudaPlatform):
    """
    Platform for HYMETA Cache on CUDA.
    """

    _enum = PlatformEnum.CUDA

    # def __init__(self, config: VllmConfig):
    #     super().__init__(config)

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1,
                             use_mla) -> str:
        # VLLM_ATTENTION_BACKEND="XFORMERS"
        if selected_backend == _Backend.XFORMERS:
            logger.info("Using FlashAttentionBackend for 7B model.")
            return "vllm.attention.backends.flash_attn.FlashAttentionBackend"
        logger.info("Using MetaAttentionBackend for 70B model. \n" \
                    "### Note: To use 7B model or *non-hymeta* model, please set VLLM_ATTENTION_BACKEND=\"XFORMERS\"")
        return "vllm_hymeta.attention.meta_attention.MetaAttentionBackend"