import os
from PIL import Image

# Directory containing the images
directory =os.path.join( os.getcwd(), "full_resInput")

# Output directory for compressed images
output_directory = "low_resInput"
input_path = os.path.join(output_directory)
# output_path = os.path.join(output_directory, "output")
# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    # if not os.path.exists(output_path):
        # os.makedirs(output_path)
# Iterate over all files in the directory
# subdirectory
# for subdirectory in os.listdir(directory):
    # Iterate over all files in the directory
    # if not subdirectory.startswith('.'):
        # dirPath = os.path.join(directory, subdirectory)
        # print(dirPath)
for filename in os.listdir(directory):
    print("prcessing : " + filename)
    # Check if the file is an image 
    if filename.lower().endswith(('.jpg', '.jpeg',)):
        # Open the original image
        original_image = Image.open(os.path.join(directory, filename))
        # Resize from 5760x 3480 the image to the desired dimensions
        compressed_image = original_image.resize((384, 384))
        filename = "low_"+filename
        # Save the compressed image in the output directory
        # if(subdirectory=="input"):
        compressed_image.save(os.path.join(input_path, filename))
        print(os.path.join(input_path, filename))
                # elif (subdirectory=="output"):
                    # compressed_image.save(os.path.join(output_path, filename))
print("done Preprocessing and saving image to Dir Compressed image")
