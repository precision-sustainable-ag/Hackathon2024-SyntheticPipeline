import shutil
from pathlib import Path
import cv2
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tempfile import NamedTemporaryFile

def get_image_format(file_path):
    """Detects the true format of the image using PIL."""
    try:
        with Image.open(file_path) as img:
            return img.format.lower()  # Returns 'jpeg', 'png', etc.
    except Exception as e:
        raise ValueError(f"Could not determine the format of {file_path}: {e}")

def safe_save_image(image, output_path):
    """Safely saves the image to avoid corruption."""
    with NamedTemporaryFile(delete=False, suffix=output_path.suffix) as temp_file:
        cv2.imwrite(temp_file.name, image)
    shutil.move(temp_file.name, output_path)

def resize_image(input_path, output_dir, width, height):
    """Resize the image and save it safely."""
    output_path = Path(output_dir) / input_path.name
    
    # Read the image from input path
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Failed to read image: {input_path}")
    
    # Check for alpha channel and convert if necessary
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    # Check if the format matches the file extension
    image_format = get_image_format(input_path)
    if image_format != input_path.suffix[1:]:
        print(f"Warning: {input_path.name} is {image_format.upper()} but named as {input_path.suffix}")

    # Skip resizing if the image already matches the target dimensions
    if image.shape[:2] == (height, width):
        print(f'Image {input_path} already has the specified dimensions')
        shutil.copy2(input_path, output_path)
    else:
        # Resize and save the image
        resized_image = cv2.resize(image, (width, height))
        safe_save_image(resized_image, output_path)

    return output_path

def process_image(input_path):
    height, width = 6368, 9560
    output_dir = Path("data/backgrounds_resized")
    resized_image_path = resize_image(input_path, output_dir, width, height)
    print(f'Resized image saved to: {resized_image_path}')

# Example usage
input_dir = Path("data/backgrounds")
output_dir = Path("data/backgrounds_resized")
output_dir.mkdir(parents=True, exist_ok=True)

image_paths = sorted([x for x in input_dir.glob("*") if x.suffix in [".jpg", ".JPG", "jpeg", "JPEG", ".png", ".PNG"]])

print(f"Found {len(image_paths)} images to process")

with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(process_image, input_path) for input_path in image_paths]
    for future in futures:
        future.result()  # Wait for all futures to complete
