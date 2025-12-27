import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import logging
import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, Tuple
import os
import re
import glob

from models.diffusion_transformer import DiffusionTransformer


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrainingConfig(BaseModel):
    """Configuration for training the Diffusion Transformer model."""

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


def get_cifar10_dataloaders(config: TrainingConfig):
    """Get CIFAR10 data loaders.

    Args:
        config: Training configuration.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    # Define transformations (e.g., convert to tensor and normalize)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load training data
    trainset = torchvision.datasets.CIFAR10(
        root=config.data_root, train=True, download=True, transform=transform
    )
    trainloader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.get_device().type == "cuda" else False,
    )

    # Load test data
    testset = torchvision.datasets.CIFAR10(
        root=config.data_root, train=False, download=True, transform=transform
    )
    testloader = DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.get_device().type == "cuda" else False,
    )

    return trainloader, testloader


def initialize_config(yaml_path: Optional[str] = None) -> TrainingConfig:
    """Initialize configuration from YAML file or use defaults.

    Args:
        yaml_path: Optional path to YAML config file.

    Returns:
        TrainingConfig instance.
    """
    if yaml_path and Path(yaml_path).exists():
        logger.info(f"Loading configuration from {yaml_path}")
        return TrainingConfig.from_yaml(yaml_path)
    else:
        logger.warning("No YAML config provided, using default values")
        # Default config for CIFAR10
        # CIFAR10: 32x32 images, with patch_size=4: 8x8=64 patches
        return TrainingConfig(
            embedding_dim=384,
            num_heads=6,
            num_transformer_blocks=6,
            patch_size=4,
            sequence_len=64,  # (32/4)^2 = 64 patches for CIFAR10
            input_dim=3,
            batch_size=64,
            learning_rate=1e-4,
            num_epochs=100,
        )


def load_latest_checkpoint(
    checkpoint_dir: str, device: torch.device
) -> Optional[Tuple[dict, str]]:
    """Load the latest checkpoint from the checkpoint directory if one exists.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        device: Device to load the checkpoint on.

    Returns:
        Tuple of (checkpoint dict, checkpoint path) if found, None otherwise.
    """
    checkpoint_dir_path = Path(checkpoint_dir)
    if not checkpoint_dir_path.exists():
        return None

    # Look for checkpoint files matching pattern checkpoint_step_*.pt
    checkpoint_pattern = checkpoint_dir_path / "checkpoint_step_*.pt"
    checkpoint_files = glob.glob(str(checkpoint_pattern))

    latest_checkpoint = None
    latest_step = -1

    # Find the checkpoint with the highest step number
    for checkpoint_path in checkpoint_files:
        match = re.search(r"checkpoint_step_(\d+)\.pt", checkpoint_path)
        if match:
            step = int(match.group(1))
            if step > latest_step:
                latest_step = step
                latest_checkpoint = checkpoint_path

    # If no step-based checkpoint found, check for final_model.pt
    if latest_checkpoint is None:
        final_model_path = checkpoint_dir_path / "final_model.pt"
        if final_model_path.exists():
            latest_checkpoint = str(final_model_path)
            logger.info(f"Found final model checkpoint: {latest_checkpoint}")
        else:
            return None
    else:
        logger.info(f"Found latest checkpoint: {latest_checkpoint} (step {latest_step})")

    # Load the checkpoint
    try:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        logger.info(f"Successfully loaded checkpoint from {latest_checkpoint}")
        return checkpoint, latest_checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {latest_checkpoint}: {e}")
        return None


def train(yaml_path: Optional[str] = None):
    """Train the Diffusion Transformer model for flow matching.

    Args:
        yaml_path: Optional path to YAML configuration file.
    """
    config = initialize_config(yaml_path)
    device = config.get_device()

    logger.info(f"Using device: {device}")
    logger.info(f"Configuration: {config.model_dump_json(indent=2)}")

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Initialize model
    model = DiffusionTransformer(config)
    model = model.to(device)
    logger.info(
        f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Get data loaders
    train_dataloader, test_dataloader = get_cifar10_dataloaders(config)
    logger.info(f"Data loaders initialized. Training batches: {len(train_dataloader)}")

    # Loss function
    loss_fn = nn.MSELoss()

    # Optimizer
    if config.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    logger.info(f"Using optimizer: {config.optimizer}")

    # Try to load from checkpoint if one exists
    start_epoch = 0
    global_step = 0
    checkpoint_data = load_latest_checkpoint(config.checkpoint_dir, device)
    
    if checkpoint_data is not None:
        checkpoint, checkpoint_path = checkpoint_data
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        # Load model state
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Model state loaded from checkpoint")
        
        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Optimizer state loaded from checkpoint")
        
        # Resume from saved epoch and step
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1  # Start from next epoch
            logger.info(f"Resuming from epoch {start_epoch}")
        
        if "step" in checkpoint:
            global_step = checkpoint["step"]
            logger.info(f"Resuming from global step {global_step}")
        
        if "loss" in checkpoint:
            logger.info(f"Last checkpoint loss: {checkpoint['loss']:.6f}")
    else:
        logger.info("No checkpoint found, starting training from scratch")

    # Training loop
    model.train()

    for epoch in range(start_epoch, config.num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")

        for batch_idx, batch in enumerate(train_dataloader):
            # CIFAR10 returns (images, labels) tuple
            images, _ = batch
            images = images.to(device)
            batch_size = images.size(0)

            optimizer.zero_grad()

            # Get flow matching interpolated images
            # Sample random noise
            noise = torch.randn_like(images)

            # Sample random timesteps t in [0, 1]
            t_values = torch.rand(batch_size, device=device)

            # Create interpolated images: x_t = t * x_1 + (1 - t) * x_0
            # where x_1 is the data and x_0 is the noise
            interpolated = (
                t_values.view(-1, 1, 1, 1) * images
                + (1 - t_values.view(-1, 1, 1, 1)) * noise
            )

            # Predict vector field v_theta(x_t, t)
            v_pred = model(interpolated, t_values)

            # Compute flow matching objective: ||v_theta(x_t, t) - (x_1 - x_0)||^2
            # Target velocity field: v = x_1 - x_0 = images - noise
            target = images - noise
            loss = loss_fn(v_pred, target)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Logging
            if global_step % config.log_interval == 0:
                avg_loss = epoch_loss / num_batches
                logger.info(
                    f"Epoch {epoch + 1}/{config.num_epochs}, "
                    f"Batch {batch_idx + 1}/{len(train_dataloader)}, "
                    f"Step {global_step}, "
                    f"Loss: {loss.item():.6f}, "
                    f"Avg Loss: {avg_loss:.6f}"
                )

            # Save checkpoint
            if global_step % config.save_interval == 0:
                checkpoint_path = os.path.join(
                    config.checkpoint_dir, f"checkpoint_step_{global_step}.pt"
                )
                torch.save(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item(),
                        "config": config.model_dump(),
                    },
                    checkpoint_path,
                )
                logger.info(f"Checkpoint saved to {checkpoint_path}")

        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(
            f"Epoch {epoch + 1}/{config.num_epochs} completed. "
            f"Average loss: {avg_epoch_loss:.6f}"
        )

        test_loss = 0.0
        test_batch_count = 0
        for batch_idx, batch in enumerate(test_dataloader):
            with torch.no_grad():
                images, _ = batch
                images = images.to(device)
                batch_size = images.size(0)

                # Get flow matching interpolated images
                # Sample random noise
                noise = torch.randn_like(images)

                # Sample random timesteps t in [0, 1]
                t_values = torch.rand(batch_size, device=device)

                # Create interpolated images: x_t = t * x_1 + (1 - t) * x_0
                # where x_1 is the data and x_0 is the noise
                interpolated = (
                    t_values.view(-1, 1, 1, 1) * images
                    + (1 - t_values.view(-1, 1, 1, 1)) * noise
                )

                # Predict vector field v_theta(x_t, t)
                v_pred = model(interpolated, t_values)

                # Compute flow matching objective: ||v_theta(x_t, t) - (x_1 - x_0)||^2
                # Target velocity field: v = x_1 - x_0 = images - noise
                target = images - noise
                loss = loss_fn(v_pred, target)

                test_loss += loss
                test_batch_count += 1

        logger.info(
            f"Epoch {epoch + 1} average test loss: {test_loss / test_batch_count:.6f}"
        )

    # Save final model
    final_model_path = os.path.join(config.checkpoint_dir, "final_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.model_dump(),
        },
        final_model_path,
    )
    logger.info(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Diffusion Transformer for Flow Matching"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    train(yaml_path=args.config)
