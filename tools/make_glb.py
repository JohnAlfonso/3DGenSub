import os
import argparse
from pathlib import Path
import requests
from typing import Optional

def process_images(
    input_folder: str,
    output_folder: str,
    api_url: str = "http://localhost:8000/generate",
    seed: int = -1
):
    """
    Process images from input folder and save GLB files to output folder.
    
    Args:
        input_folder: Path to folder containing input images
        output_folder: Path to folder where GLB files will be saved
        api_url: URL of the serve API endpoint
        seed: Random seed for generation (-1 for random)
    """
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    input_path = Path(input_folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for idx, image_file in enumerate(image_files, 1):
        output_file = output_path / f"{image_file.stem}.glb"
        if output_file.exists():
            print(f"Skipping {idx}/{len(image_files)}: {image_file.name} (exists: {output_file})")
            continue

        print(f"Processing {idx}/{len(image_files)}: {image_file.name}")

        try:
            # Open and read the image file
            with open(image_file, 'rb') as f:
                files = {'prompt_image_file': (image_file.name, f, 'image/*')}
                data = {'seed': seed}

                # Make POST request to the API
                response = requests.post(api_url, files=files, data=data, stream=True)
                response.raise_for_status()

                # Save the GLB file
                with open(output_file, 'wb') as out_f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            out_f.write(chunk)

                print(f"  ✓ Saved to {output_file}")

        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error processing {image_file.name}: {e}")
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate GLB files from images using the serve API"
    )
    parser.add_argument(
        '--input',
        '-i',
        required=True,
        help='Path to folder containing input images'
    )
    parser.add_argument(
        '--output',
        '-o',
        required=True,
        help='Path to folder where GLB files will be saved'
    )
    parser.add_argument(
        '--api-url',
        '-u',
        default='http://localhost:10006/generate',
        help='URL of the serve API endpoint (default: http://localhost:10006/generate)'
    )
    parser.add_argument(
        '--seed',
        '-s',
        type=int,
        default=42,
        help='Random seed for generation (default: -1 for random)'
    )
    
    args = parser.parse_args()
    
    process_images(
        input_folder=args.input,
        output_folder=args.output,
        api_url=args.api_url,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
