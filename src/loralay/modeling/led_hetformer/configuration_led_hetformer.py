from typing import List, Union

from transformers.configuration_utils import PretrainedConfig

class LEDHeterformerConfig(PretrainedConfig):
    """
        Args:
            attention_window: list of attention window sizes of length = number of layers.
                window size = number of attention locations on each side.
                For an affective window size of 512, use `attention_window=[256]*num_layers`
                which is 256 on each side.
            attention_dilation: list of attention dilation of length = number of layers.
                attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of both sides
            attention_mode: 'n2' for regular n^2 self-attention, 'tvm' for TVM implemenation of Heterformer
                selfattention, 'sliding_chunks' for another implementation of Heterformer selfattention
    """
    model_type = "led_hetformer"
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "attention_probs_dropout_prob": "attention_dropout",
        "initializer_range": "init_std",
    }

    def __init__(
        self,
        vocab_size=50265,
        max_encoder_position_embeddings=4096,
        max_decoder_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=3072,
        encoder_attention_heads=12,
        decoder_layers=12,
        decoder_ffn_dim=3072,
        decoder_attention_heads=12,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=768,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        init_std=0.02,
        decoder_start_token_id=2,
        classifier_dropout=0.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        attention_window: Union[List[int], int] = 512,
        attention_dilation: Union[List[int], int] = 1,
        autoregressive: bool = False,
        attention_mode: str = 'sliding_chunks',
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_encoder_position_embeddings = max_encoder_position_embeddings
        self.max_decoder_position_embeddings = max_decoder_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.attention_window = attention_window

        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode
        assert self.attention_mode in ['tvm', 'sliding_chunks', 'n2', 'sliding_chunks_no_overlap']
        if isinstance(self.attention_dilation, int):
            self.attention_dilation = [self.attention_dilation] * self.encoder_layers

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )

        