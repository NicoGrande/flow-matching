from pydantic import BaseModel, Field
import torch
import yaml


class TrainingConfig(BaseModel):
    """Configuration for training the Diffusion Transformer model."""
    
    # Dataset
    dataset: str = Field(default="cifar10", description="Which dataset to use")

    # Model parameters
    embedding_dim: int = Field(..., description="Dimension of embeddings")
    num_heads: int = Field(..., description="Number of attention heads")
    num_transformer_blocks: int = Field(..., description="Number of transformer blocks")
    patch_size: int = Field(
        ..., description="Size of each patch (assumes square patches)"
    )
    sequence_len: int = Field(..., description="Sequence length (number of patches)")
    input_dim: int = Field(
        default=3, description="Number of input channels (3 for RGB)"
    )

    # Training parameters
    batch_size: int = Field(..., description="Batch size for training")
    learning_rate: float = Field(..., description="Learning rate")
    num_epochs: int = Field(..., description="Number of training epochs")
    optimizer: str = Field(
        default="adam", description="Optimizer type: 'adam' or 'sgd'"
    )
    weight_decay: float = Field(default=0.0, description="Weight decay for optimizer")

    # Device and data parameters
    device: str = Field(
        default="auto", description="Device to use: 'auto', 'cuda', or 'cpu'"
    )
    num_workers: int = Field(default=2, description="Number of data loader workers")
    data_root: str = Field(default="./data", description="Root directory for data")

    # Logging and checkpointing
    log_interval: int = Field(
        default=100, description="Log training metrics every N batches"
    )
    save_interval: int = Field(
        default=1000, description="Save checkpoint every N batches"
    )
    checkpoint_dir: str = Field(
        default="./checkpoints", description="Directory to save checkpoints"
    )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            TrainingConfig instance.
        """
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def get_device(self) -> torch.device:
        """Get the appropriate device for training.

        Returns:
            torch.device instance.
        """
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
