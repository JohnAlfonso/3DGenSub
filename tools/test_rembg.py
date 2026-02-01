from io import BytesIO
from pathlib import Path

from PIL import Image

from rembg import new_session, remove


def remove_background(input_image: Image.Image) -> Image.Image:
    """
    Remove background from the input image using rembg library.
    
    Args:
        input_image: PIL Image object
    Returns:
        PIL Image object with background removed
    """
    session = new_session("isnet-general-use")
    
    output_image = remove(image, session=session)

    return output_image

if __name__ == "__main__":
    image = Image.open("special_prompt/3e743f77623f822d1f1f0264f823576aa8a20bd7344ba0dce92ef8ac63a82547.png")
    outimage = remove_background(image)
    outimage.save("test.png")