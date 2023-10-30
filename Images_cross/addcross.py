from PIL import Image, ImageDraw
import os

def draw_cross(draw, center, size, width):
    x, y = center
    draw.line((x - size, y, x + size, y), fill="white", width=width)
    draw.line((x, y - size, x, y + size), fill="white", width=width)

def add_black_image_with_cross(folder, size, cross_size, cross_width):
    # Create a new image with a black background
    new_image = Image.new('RGB', size, (0, 0, 0))
    draw = ImageDraw.Draw(new_image)
    center = (size[0] // 2, size[1] // 2)
    draw_cross(draw, center, cross_size, cross_width)

    # Save the new image as '0.jpg'
    new_image_path = os.path.join(folder, "0.jpg")
    new_image.save(new_image_path)

def shift_and_process_images(folder):
    # Shift the images by renaming them in reverse order
    for i in range(40, -1, -1):
        src = os.path.join(folder, f"{i}.jpg")
        dst = os.path.join(folder, f"{i + 1}.jpg")
        if os.path.exists(src):
            os.rename(src, dst)

    # Process images to add cross, starting from 1.jpg
    for i in range(1, 42):
        filepath = os.path.join(folder, f"{i}.jpg")
        if os.path.exists(filepath):
            with Image.open(filepath) as im:
                draw = ImageDraw.Draw(im)
                center = (im.width // 2, im.height // 2)
                size = min(im.width, im.height) // 18
                width = size // 5
                draw_cross(draw, center, size, width)
                im.save(filepath)

    # Add the black image with cross as 0.jpg
    if os.path.exists(os.path.join(folder, "1.jpg")):
        first_image = Image.open(os.path.join(folder, "1.jpg"))
        add_black_image_with_cross(folder, first_image.size, size, width)

# Base path where folders are located
base_path = r"C:\Users\tnlab\OneDrive\Documents\GitHub\AlphaFold\Neurofeedback-Based-BCI\Images_cross\Composite_Images"

# Process each folder
folders = [os.path.join(base_path, f"Block{i}") for i in range(1, 9)]
for folder in folders:
    shift_and_process_images(folder)
    print(f"Processed and updated images in folder: {folder}")

print("All images in all folders have been processed and updated.")
