from PIL import Image
from pathlib import Path
import torch
import numpy as np
import random
from torchvision import transforms
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

def set_seed(seed: int):
    """
    Set the seed for reproducibility in random number generation.

    Args:
        seed (int): The seed value to set for random number generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)

def main(seed: int, data_path: str, sampling_timesteps: int=250):
    """
    Main function to initialize and train the diffusion model.

    Args:
        seed (int): The seed value for random number generation to ensure reproducibility.
        data_path (str): Path to the dataset directory.
        sampling_timesteps (int): Number of sampling timesteps during generation.
    
    Steps:
        1. Sets the seed for reproducibility.
        2. Initializes the Unet model.
        3. Sets up the GaussianDiffusion and Trainer.
        4. Loads the pre-trained model weights.
        5. Starts the training process.
    """
    # Set seed for reproducibility
    set_seed(seed)

    # Define the path to the dataset
    paths = Path(data_path)
    if not paths.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")
    print(f"Dataset path exists: {paths.exists()}")

    # Get list of image paths
    image_paths = list(paths.glob('*.png'))
    
    # Initialize the model
    model = Unet(
        32,
        dim_mults=(1, 2, 4, 4, 8),
        flash_attn=True
    )
    print(f'Number of parameters in the model is {sum(i.numel() for i in model.parameters())}')

    # Set up the GaussianDiffusion object
    diffusion = GaussianDiffusion(
        model,
        image_size=512,
        timesteps=1000,                # Total number of timesteps for diffusion
        sampling_timesteps=sampling_timesteps  # Sampling timesteps provided by the user
    )

    # Initialize the Trainer
    trainer = Trainer(
        diffusion,
        paths,
        train_batch_size=4,
        train_lr=5e-4,
        train_num_steps=60000,        
        gradient_accumulate_every=5,    
        ema_decay=0.995,                
        amp=True,                       
        calculate_fid=False,         
        save_and_sample_every=2000,
        num_samples=4,
        results_folder='./results',
    )

    # Load pre-trained model weights
    trainer.load(25)

    # Start training
    trainer.train()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train the diffusion model with specified random seed, dataset path, and sampling timesteps.")
    parser.add_argument("--seed", type=int, default=42, help="Seed value for random number generation.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the image dataset directory.")
    parser.add_argument("--sampling_timesteps", type=int, default=250, help="Number of sampling timesteps for the diffusion process.")
    args = parser.parse_args()

    main(args.seed, args.data_path, args.sampling_timesteps)
