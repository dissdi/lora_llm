from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05