import os
from PIL import Image

def rotate_images_in_directory(directory):

    for filename in os.listdir(directory):
        if filename.endswith(".JPG"):
            image_path = os.path.join(directory, filename)
            img = Image.open(image_path)

            # Check the image resolution
            width, height = img.size
            if (width == 5472 and height == 3648):
                # No need to rotate, the image is already in the desired resolution
                rotated_img = img
            else:
                # Rotate the image 90 degrees clockwise to swap the dimensions
                rotated_img = img.transpose(Image.Transpose.ROTATE_90)
                print("Rotated image: ",output_path)
                # Save the rotated image to the output_directory
                output_path = os.path.join(directory, filename)
                rotated_img.save(output_path)

if __name__ == "__main__":
    directory =os.path.join( os.getcwd(), "full_resInput")
    print("Running")
    rotate_images_in_directory(directory)
