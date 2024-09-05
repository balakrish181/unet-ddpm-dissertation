import argparse
import os
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import random
from torchvision import transforms
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

def set_seed(seed: int):
    """
    Set the seed for reproducibility in random number generation.

    Args:
        seed (int): The seed value to set for random number generators.
    
    This function ensures that results are reproducible by fixing the random seed 
    across NumPy, PyTorch (both CPU and GPU), and Python's built-in random module.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)
    
    # Ensures deterministic behavior in PyTorch's CUDA backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(num_samples: int, seed: int):
    """
    Main function to generate images using a pre-trained Denoising Diffusion model.

    Args:
        num_samples (int): The number of images to generate.
        seed (int): The seed value for reproducibility.

    This function loads the pre-trained diffusion model, generates the specified
    number of images, and saves them to a specified output folder.
    """
    # Set seed for reproducibility
    set_seed(seed)

    # Define the path to the dataset
    paths = Path('./train_data/')
    print(f"Dataset path exists: {paths.exists()}")

    # Initialize the model
    model = Unet(
        dim=32,
        dim_mults=(1, 2, 4, 4, 8),
        flash_attn=True
    )
    print(f'Number of parameters in the model is {sum(i.numel() for i in model.parameters())}')

    # Set up the GaussianDiffusion object
    diffusion = GaussianDiffusion(
        model=model,
        image_size=512,
        timesteps=1000,
        sampling_timesteps=1000
    )

    # Initialize the Trainer object
    trainer = Trainer(
        diffusion=diffusion,
        folder=paths,
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
    trainer.load(24)
    trainer.model.eval()

    # Define the folder where images will be saved
    output_folder = "./sampled_images/"
    os.makedirs(output_folder, exist_ok=True)

    # Fixed batch size for generation
    batch_size = 2

    # Calculate the number of full batches needed and the remainder
    full_batches = num_samples // batch_size
    remainder = num_samples % batch_size

    # Generate images in full batches
    for batch_num in range(full_batches):
        gen_image_loaded = diffusion.sample(batch_size=batch_size)
        save_images(gen_image_loaded, output_folder, batch_num * batch_size)

    # Generate remaining images if there are any
    if remainder > 0:
        gen_image_loaded = diffusion.sample(batch_size=remainder)
        save_images(gen_image_loaded, output_folder, full_batches * batch_size)

def save_images(gen_image_loaded: torch.Tensor, output_folder: str, start_idx: int):
    """
    Save the generated images to the specified folder.

    Args:
        gen_image_loaded (torch.Tensor): Tensor containing generated images.
        output_folder (str): Directory where images will be saved.
        start_idx (int): Starting index for naming the saved images.
    
    This function converts the generated image tensors into NumPy arrays, 
    normalizes them, and saves them as PNG files in the output folder.
    """
    for i in range(gen_image_loaded.size(0)):
        np_array = gen_image_loaded[i].cpu().numpy()

        # Normalize the NumPy array if it's in the range 0-1
        np_array = np_array * 255.0

        # Ensure the array is in uint8 format (common for image data)
        np_array = np_array.astype(np.uint8)

        # Convert to PIL Image (for RGB images)
        image = Image.fromarray(np_array.transpose(1, 2, 0))

        # Save each image with a unique filename in the specified folder
        image.save(os.path.join(output_folder, f"generated_image_{start_idx + i}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using the Denoising Diffusion model.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of images to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed value for reproducibility.")
    args = parser.parse_args()

    main(args.num_samples, args.seed)
