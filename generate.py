import torch
import argparse
import logging
from pathlib import Path
import os
from typing import Optional
import torchvision.utils as vutils

from train import TrainingConfig
from models.diffusion_transformer import DiffusionTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> tuple[DiffusionTransformer, TrainingConfig]:
    """Load model and config from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model on.

    Returns:
        Tuple of (model, config).
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config from checkpoint or create from defaults
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        config = TrainingConfig(**config_dict)
    else:
        logger.warning("No config found in checkpoint, using defaults")
        config = TrainingConfig(
            embedding_dim=384,
            num_heads=6,
            num_transformer_blocks=6,
            patch_size=4,
            sequence_len=64,
            input_dim=3,
            batch_size=64,
            learning_rate=1e-4,
            num_epochs=100,
        )

    # Initialize model
    model = DiffusionTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully")
    return model, config


def generate_images(
    model: DiffusionTransformer,
    config: TrainingConfig,
    num_images: int = 16,
    num_steps: int = 50,
    device: torch.device = None,
) -> torch.Tensor:
    """Generate images using flow matching.

    Args:
        model: Trained DiffusionTransformer model.
        config: Training configuration.
        num_images: Number of images to generate.
        num_steps: Number of integration steps (more steps = better quality).
        device: Device to run generation on.

    Returns:
        Generated images tensor of shape (num_images, input_dim, height, width).
    """
    if device is None:
        device = config.get_device()

    model.eval()
    height = width = int(config.sequence_len**0.5) * config.patch_size

    # Start from random noise: x_0 ~ N(0, I)
    # Shape: (num_images, input_dim, height, width)
    x = torch.randn(num_images, config.input_dim, height, width, device=device)

    # Shape: (num_images, num_classes)
    labels = torch.randint(0, 10, (num_images,), device=device)

    # Time steps from 0 to 1
    dt = 1.0 / num_steps
    timesteps = torch.linspace(0, 1, num_steps + 1, device=device)

    logger.info(f"Generating {num_images} images with {num_steps} steps...")

    with torch.no_grad():
        for i, t in enumerate(timesteps[:-1]):
            # Current timestep
            t_current = t.expand(num_images)

            # Predict velocity field: v_theta(x_t, t)
            v_pred = model(x, t_current, labels)

            # Euler step: x_{t+dt} = x_t + dt * v_theta(x_t, t)
            x = x + dt * v_pred

            if (i + 1) % (num_steps // 10) == 0 or i == 0:
                logger.info(f"Step {i + 1}/{num_steps}, t = {t.item():.4f}")

    logger.info("Generation complete!")
    return x


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    """Denormalize images from [-1, 1] to [0, 1].

    Args:
        images: Images in range [-1, 1].

    Returns:
        Images in range [0, 1].
    """
    # Reverse the normalization: (x + 1) / 2
    return (images + 1.0) / 2.0


def save_images(
    images: torch.Tensor, output_path: str, nrow: int = 4, padding: int = 2
):
    """Save images to file.

    Args:
        images: Image tensor of shape (N, C, H, W) in range [0, 1].
        output_path: Path to save the image grid.
        nrow: Number of images per row in the grid.
        padding: Padding between images.
    """
    # Clamp to [0, 1] to ensure valid range
    images = torch.clamp(images, 0.0, 1.0)

    # Create grid and save
    grid = vutils.make_grid(images, nrow=nrow, padding=padding, normalize=False)
    vutils.save_image(grid, output_path)
    logger.info(f"Images saved to {output_path}")


def generate(
    checkpoint_path: str,
    output_dir: str = "./generated",
    num_images: int = 16,
    num_steps: int = 50,
    config_path: Optional[str] = None,
    device: Optional[str] = None,
):
    """Main generation function.

    Args:
        checkpoint_path: Path to model checkpoint.
        output_dir: Directory to save generated images.
        num_images: Number of images to generate.
        num_steps: Number of integration steps.
        config_path: Optional path to YAML config (for device settings).
        device: Optional device override ('auto', 'cuda', or 'cpu').
    """
    # Load config if provided
    if config_path and Path(config_path).exists():
        config = TrainingConfig.from_yaml(config_path)
        if device:
            config.device = device
    else:
        # Will load config from checkpoint
        config = None

    # Determine device
    if config:
        device = config.get_device()
    else:
        if device == "auto" or device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

    logger.info(f"Using device: {device}")

    # Load model
    model, loaded_config = load_model_from_checkpoint(checkpoint_path, device)
    if config is None:
        config = loaded_config

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate images
    generated_images = generate_images(
        model, config, num_images=num_images, num_steps=num_steps, device=device
    )

    # Denormalize from [-1, 1] to [0, 1]
    generated_images = denormalize_images(generated_images)

    # Save images
    output_path = os.path.join(output_dir, "generated_images.png")
    save_images(generated_images, output_path, nrow=int(num_images**0.5))

    # Also save individual images
    individual_dir = os.path.join(output_dir, "individual")
    os.makedirs(individual_dir, exist_ok=True)
    for i, img in enumerate(generated_images):
        img_path = os.path.join(individual_dir, f"image_{i:04d}.png")
        vutils.save_image(img, img_path)

    logger.info(f"All images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images using trained Flow Matching model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated",
        help="Directory to save generated images (default: ./generated)",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=16,
        help="Number of images to generate (default: 16)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of integration steps (default: 50, more steps = better quality)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to YAML config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'auto', 'cuda', or 'cpu' (default: auto)",
    )

    args = parser.parse_args()

    generate(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_images=args.num_images,
        num_steps=args.num_steps,
        config_path=args.config,
        device=args.device,
    )
