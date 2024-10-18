from PIL import Image, ImageDraw
from pathlib import Path
import random
import cv2
import numpy as np
from tqdm import tqdm  


def remove_black_background_and_resize(cutout):
    """Remove black background and resize."""
    cutout = cutout.convert('RGBA')  # Ensure cutout is RGBA

    # Convert cutout to a numpy array
    data = np.array(cutout)
    h, w = data.shape[:2]
    resized_data = cv2.resize(data, (w // 10, h // 10), interpolation=cv2.INTER_AREA)

    # Identify black pixels (where R, G, B are all 0)
    black_mask = (resized_data[:, :, :3] == [0, 0, 0]).all(axis=-1)

    # Set alpha channel to 0 (transparent) for black pixels
    resized_data[black_mask] = [0, 0, 0, 0]

    # Convert back to an image
    return Image.fromarray(resized_data)

# Seed for reproducibility
random.seed(42)

# Paths to cutouts and background
background_path = Path('data/backgrounds_resized')
backgrounds = list(background_path.glob('*.jpg')) + list(background_path.glob('*.JPG'))
random.shuffle(backgrounds)
background_image_path = backgrounds[0]

# cutout_dir = Path('/home/mkutuga/SemiF-SyntheticPipeline/data/cutouts')
cutouts = [
    # "data/cutouts/MD_1688066688_7.png",
    # "data/cutouts/MD_Row-48_1656515996_10.png",
    # "data/cutouts/MD_1660068086_1.png",
    # "data/cutouts/MD_1660313909_1.png",
    # "data/cutouts/MD_1660320054_0.png",
    # "data/cutouts/MD_1660582048_0.png",
    # "data/cutouts/MD_1660744495_6.png",
    # "data/cutouts/MD_1660744597_4.png",
    # "data/cutouts/MD_1684770776_3.png",
    # "data/cutouts/MD_1688063001_1.png",
    "data/cutouts/MD_1688063286_2.png",
    "data/cutouts/MD_1688141689_0.png",
    "data/cutouts/MD_1692112215_1.png",
    "data/cutouts/MD_Row-4_1657027474_13.png",
    # "data/cutouts/MD_Row-12_1657037490_4.png",
    "data/cutouts/MD_Row-35_1656460046_1.png",
    # "data/cutouts/TX_1677089166_2.png",
    "data/cutouts/TX_1677087661_5.png",
    # "data/cutouts/TX_1677087141_1.png",
    "data/cutouts/NC_Row-1_1657546308_1.png",
    # "data/cutouts/NC_1668009255_0.png",
    "data/cutouts/TX_1678901329_0.png",

]

# Output GIF path
output_gif = 'cutout_overlay.gif'
final_bbox_image_path = 'final_bounding_boxes.png'

# Load and resize the background image
background = Image.open(background_image_path).convert('RGBA')
background_array = np.array(background)
h, w = background_array.shape[:2]
background_resized = cv2.resize(background_array, (w // 10, h // 10), interpolation=cv2.INTER_AREA)
background = Image.fromarray(background_resized)


# Get dimensions of the background
bg_width, bg_height = background.size

# Create a cumulative image to keep adding cutouts
cumulative_image = background.copy()

# Store frames for the GIF
frames = []
final_positions = []  # Track final positions for bounding boxes


# Animate each cutout moving from left to its final random position
for i, cutout_path in tqdm(enumerate(cutouts), total=len(cutouts), desc="Processing cutouts"):
    # Load and process the cutout
    cutout = Image.open(cutout_path)
    cutout = remove_black_background_and_resize(cutout)

    # Generate a random final position for the cutout
    max_x = bg_width - cutout.width
    max_y = bg_height - cutout.height
    final_position = (random.randint(0, max_x), random.randint(0, max_y))

    # Animate the cutout moving from the left side of the screen
    for x in range(-cutout.width, final_position[0] + 1, 10):
        # Create a new frame with the current cumulative state
        frame = cumulative_image.copy()

        # Paste the cutout at the current x position and final y position
        frame.paste(cutout, (x, final_position[1]), cutout)

        # Add the frame to the GIF frames list
        frames.append(frame)

    # Add the cutout to the cumulative image at its final position
    cumulative_image.paste(cutout, final_position, cutout)

    # Save the final position and dimensions for the bounding box
    bbox = (
        final_position[0],  # x1
        final_position[1],  # y1
        final_position[0] + cutout.width,  # x2
        final_position[1] + cutout.height  # y2
    )
    final_positions.append(bbox)

print(f"Processed {len(cutouts)} cutouts")
# Save all frames as an animated GIF
frames[0].save(
    output_gif, save_all=True, append_images=frames[1:], 
    duration=10, loop=1, optimize=True, quality=75
)


print(f"GIF saved as {output_gif}")


# Create a single image with all bounding boxes overlayed
bbox_image = cumulative_image.copy()
draw = ImageDraw.Draw(bbox_image)

# Draw all the bounding boxes (red outline)
for bbox in final_positions:
    draw.rectangle(bbox, outline="red", width=2)

# Save the final bounding box image
bbox_image.save(final_bbox_image_path)
print(f"Bounding box image saved as {final_bbox_image_path}")
