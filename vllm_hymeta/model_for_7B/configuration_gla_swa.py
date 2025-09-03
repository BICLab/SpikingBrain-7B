"""GLA model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class GLAswaConfig(PretrainedConfig):

    model_type = "gla_swa"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=3584,
        num_hidden_layers=28,
        attn_mode="chunk",
        num_attention_heads=28,
        num_key_value_heads=4,
        use_short_conv=False,
        conv_size=4,
        gate_logit_normalizer=16,
        gate_low_rank_dim=16,
        intermediate_size=18944,
        hidden_act="swish",
        max_position_embeddings=4096 * 32,
        sliding_window=4096,
        elementwise_affine=True,
        norm_eps=1e-6,
        rope_theta=1000000.0,
        attention_dropout=0.0,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=151643,
        eos_token_id=151643,
        tie_word_embeddings=False,
        initializer_range=0.02,
        fuse_cross_entropy=True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attn_mode = attn_mode
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.gate_logit_normalizer = gate_logit_normalizer
        self.gate_low_rank_dim = gate_low_rank_dim
        self.intermediate_size = intermediate_size
        self.sliding_window = sliding_window
        self.hidden_act = hidden_act
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.fuse_cross_entropy = fuse_cross_entropy
        self.attn_layers = list(range(1, num_hidden_layers, 2))

        self.attn_type_list = [1 if i in self.attn_layers else 0 for i in range(num_hidden_layers)]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )