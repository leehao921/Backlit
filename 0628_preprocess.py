import os
from PIL import Image

# Directory containing the images
directory = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/classicalone/code/Users/461902/Backlit_data/Original/train/output"

# Output directory for compressed images
output_directory = "compressed_images"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Check if the file is an image (you can modify the list of valid extensions as per your requirements)
    if filename.lower().endswith(('.jpg', '.jpeg',)):
        # Open the original image
        original_image = Image.open(os.path.join(directory, filename))

        # Resize from 5760x 3480 the image to the desired dimensions
        compressed_image = original_image.resize((384, 384))
        filename = "_low_"+filename
        # Save the compressed image in the output directory
        compressed_image.save(os.path.join(output_directory, filename))
