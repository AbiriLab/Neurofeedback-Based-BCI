

# from PIL import Image, ImageDraw
# import os
# import random
# import shutil
# import pandas as pd

# def convert_to_bw_and_add_fixation(image_path, output_path):
#     """Convert an image to black and white, add a bold white cross fixation, and save it."""
#     image = Image.open(image_path).convert("L")
#     draw = ImageDraw.Draw(image)
#     center_x, center_y = image.width // 2, image.height // 2
#     cross_size = 20
#     cross_width = 4

#     draw.line((center_x, center_y - cross_size, center_x, center_y + cross_size), fill=255, width=cross_width)
#     draw.line((center_x - cross_size, center_y, center_x + cross_size, center_y), fill=255, width=cross_width)

#     image.save(output_path)

# def create_and_populate_blocks(scene_path, face_path, target_folder, face_proportions, images_per_block=40):
#     # Ensure the target folder is clean
#     if os.path.exists(target_folder):
#         shutil.rmtree(target_folder)
#     os.makedirs(target_folder)

#     # Get all image paths
#     scene_images = [os.path.join(scene_path, f) for f in os.listdir(scene_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
#     face_images = [os.path.join(face_path, f) for f in os.listdir(face_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

#     for block, face_prop in zip(range(1, len(face_proportions) + 1), face_proportions):
#         block_folder = os.path.join(target_folder, f'Block{block}')
#         os.makedirs(block_folder)

#         # Select face and scene images based on proportions
#         num_face_images = round(face_prop * images_per_block)
#         num_scene_images = images_per_block - num_face_images

#         selected_faces = random.sample(face_images, num_face_images)
#         selected_scenes = random.sample(scene_images, num_scene_images)

#         # Combine, shuffle, and rename the images
#         combined = selected_faces + selected_scenes
#         random.shuffle(combined)

#         # Create a dataframe to record the sequence
#         df = pd.DataFrame(columns=["Instructions"])

#         for idx, image_path in enumerate(combined):
#             output_path = os.path.join(block_folder, f'{idx}.jpg')
#             convert_to_bw_and_add_fixation(image_path, output_path)

#             # Determine whether it's a face or scene image for the Excel file
#             if "face" in image_path.lower():
#                 df.loc[idx] = "F"
#             else:
#                 df.loc[idx] = "s"

#         # Save the dataframe as an Excel file in the block folder
#         df.to_excel(os.path.join(block_folder, f'Block{block}_sequence.xlsx'), index=False)

# # Paths
# scene_path = r"C:\Users\tnlab\OneDrive\Documents\GitHub\AlphaFold\Neurofeedback-Based-BCI\Images\scene"
# face_path = r"C:\Users\tnlab\OneDrive\Documents\GitHub\AlphaFold\Neurofeedback-Based-BCI\Images\face"
# target_folder = r"C:\Users\tnlab\OneDrive\Documents\GitHub\AlphaFold\Neurofeedback-Based-BCI\Blocks"

# # Proportions of Face images in each block
# face_proportions = [0.5, 0.6, 0.7, 0.55, 0.75, 0.45, 0.65, 0.4]

# create_and_populate_blocks(scene_path, face_path, target_folder, face_proportions)
from PIL import Image, ImageDraw
import os
import random
import shutil
import pandas as pd

def convert_to_bw_and_add_fixation(image_path, output_path):
    """Convert an image to black and white, add a bold white cross fixation, and save it."""
    image = Image.open(image_path).convert("L")
    draw = ImageDraw.Draw(image)
    center_x, center_y = image.width // 2, image.height // 2
    cross_size = 20
    cross_width = 4

    draw.line((center_x, center_y - cross_size, center_x, center_y + cross_size), fill=255, width=cross_width)
    draw.line((center_x - cross_size, center_y, center_x + cross_size, center_y), fill=255, width=cross_width)

    image.save(output_path)

def create_and_populate_blocks(scene_path, face_path, target_folder, face_proportions, images_per_block=40):
    # Ensure the target folder is clean
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)

    # Get all image paths
    scene_images = [os.path.join(scene_path, f) for f in os.listdir(scene_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    face_images = [os.path.join(face_path, f) for f in os.listdir(face_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for block, face_prop in zip(range(1, len(face_proportions) + 1), face_proportions):
        block_folder = os.path.join(target_folder, f'Block{block}')
        os.makedirs(block_folder)

        # Select face and scene images based on proportions
        num_face_images = round(face_prop * images_per_block)
        num_scene_images = images_per_block - num_face_images

        selected_faces = random.sample(face_images, num_face_images)
        selected_scenes = random.sample(scene_images, num_scene_images)

        # Combine, shuffle, and rename the images
        combined = selected_faces + selected_scenes
        random.shuffle(combined)

        # Create a dataframe to record the sequence
        df = pd.DataFrame(columns=["Instructions"])

        for idx, image_path in enumerate(combined):
            output_path = os.path.join(block_folder, f'{idx}.jpg')
            convert_to_bw_and_add_fixation(image_path, output_path)

            # Determine whether it's a face or scene image for the CSV file
            if "face" in image_path.lower():
                df.loc[idx] = "F"
            else:
                df.loc[idx] = "s"

        # Save the dataframe as a CSV file in the block folder
        df.to_csv(os.path.join(block_folder, f'Block{block}_key.csv'), index=False)

# Paths
scene_path = r"C:\Users\tnlab\OneDrive\Documents\GitHub\AlphaFold\Neurofeedback-Based-BCI\Images\scene"
face_path = r"C:\Users\tnlab\OneDrive\Documents\GitHub\AlphaFold\Neurofeedback-Based-BCI\Images\face"
target_folder = r"C:\Users\tnlab\OneDrive\Documents\GitHub\AlphaFold\Neurofeedback-Based-BCI\Blocks"

# Proportions of Face images in each block
face_proportions = [0.1, 0.11, 0.12, 0.15, 0.85, 0.82, 0.9, 0.8]

create_and_populate_blocks(scene_path, face_path, target_folder, face_proportions)
