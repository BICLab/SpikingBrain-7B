from transformers import PretrainedConfig


class SpikingBrainConfig(PretrainedConfig):
    """
    Configuration class for the SpikingBrain-7B hybrid spiking transformer.
    """
    model_type = "spikingbrain"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        # Neuromorphic / Spiking Parameters
        v_threshold: float = 1.0,
        leak_factor: float = 0.9,
        surrogate_alpha: float = 2.0,
        surrogate_function: str = "fast_sigmoid",
        spike_reset_mode: str = "hard",  # 'hard' or 'soft'
        linear_attention_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        # SNN-specific hyper-parameters
        self.v_threshold = v_threshold
        self.leak_factor = leak_factor
        self.surrogate_alpha = surrogate_alpha
        self.surrogate_function = surrogate_function
        self.spike_reset_mode = spike_reset_mode
        self.linear_attention_eps = linear_attention_eps
