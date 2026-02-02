import os
import argparse
import time
from pathlib import Path
import requests
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from b2sdk.v2 import InMemoryAccountInfo, B2Api

# NEVER hardcode real keys in production
APPLICATION_KEY_ID = "0054c1bbe6bcfe7000000001a"
APPLICATION_KEY = "K005FlRPMrp14TclqSQhrLsYCUaarj8"
BUCKET_NAME = "404-gen"

def upload_glb_to_b2(glb_path: str, remote_file_name: Optional[str] = None, max_retries: int = 3):
    """
    Upload a GLB file to Backblaze B2 cloud storage with retry logic.
    
    Args:
        glb_path: Path to the local GLB file
        remote_file_name: Name to use for the file in B2 (defaults to the local file name)
        max_retries: Maximum number of retry attempts (default: 3)
    """
    if remote_file_name is None:
        remote_file_name = os.path.basename(glb_path)
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            info = InMemoryAccountInfo()
            b2_api = B2Api(info)

            b2_api.authorize_account("production", APPLICATION_KEY_ID, APPLICATION_KEY)

            bucket = b2_api.get_bucket_by_name(BUCKET_NAME)

            local_file_path = glb_path

            result = bucket.upload_local_file(
                local_file=local_file_path,
                file_name= "test/" + remote_file_name,
                content_type="model/gltf-binary"
            )

            print(f"Upload complete: {remote_file_name}")
            print(f"File URL: {bucket.get_download_url("test/" + remote_file_name)}")
            return result
            
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"Upload failed for {remote_file_name} (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Upload failed for {remote_file_name} after {max_retries} attempts: {e}")
    
    # If all retries failed, raise the last exception
    raise last_exception

def process_images(
    input_folder: str,
    output_folder: str,
    api_url: str = "http://localhost:8000/generate",
    seed: int = -1
):
    """
    Process images from input folder and save GLB files to output folder.
    Uploads to B2 in parallel while processing continues.
    
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
    
    # Create thread pool for parallel uploads (max 4 concurrent uploads)
    upload_executor = ThreadPoolExecutor(max_workers=4)
    upload_futures = []
    
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
                
                # Start upload in background immediately
                future = upload_executor.submit(
                    upload_glb_to_b2,
                    str(output_file),
                    output_file.name
                )
                upload_futures.append((future, output_file.name))
                print(f"  ⬆ Upload started for {output_file.name}")

        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error processing {image_file.name}: {e}")
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
    
    # Wait for all uploads to complete
    print(f"\nWaiting for {len(upload_futures)} uploads to complete...")
    for idx, (future, filename) in enumerate(upload_futures, 1):
        try:
            result = future.result()
            print(f"  ✓ Upload complete ({idx}/{len(upload_futures)}): {filename}")
        except Exception as e:
            print(f"  ✗ Upload failed ({idx}/{len(upload_futures)}): {filename} - {e}")
    
    upload_executor.shutdown(wait=True)
    print("\nAll processing and uploads complete!")

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
