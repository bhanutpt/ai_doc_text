#HareKrsna

# Program to take input character images and standardize them to a given standard size images.
# It will be helpful to load them and process them in AI/ML model training.
#
# Usage : get_std_size_images.py <source_folder_name> <destination_folder_name>
#
# Example :
# python get_std_size_images.py divya_numerals_chars divya_numerals_chars_dataset

import os
import cv2
import numpy as np
import argparse
 
def pad_to_square(image, size, color=255):
    # Create a new square canvas of the specified size
    canvas = np.full((size, size), color, dtype=np.uint8)

    # Get the dimensions of the input image
    height, width = image.shape

    # Calculate the scaling factor to fit the image into the canvas
    scale = min(size / width, size / height)

    # Resize the image while preserving the aspect ratio
    resized_image = cv2.resize(image, None, fx=scale, fy=scale)

    # Calculate the position to paste the resized image
    x_offset = (size - resized_image.shape[1]) // 2
    y_offset = (size - resized_image.shape[0]) // 2

    # Paste the resized image onto the canvas
    canvas[y_offset:y_offset + resized_image.shape[0], x_offset:x_offset + resized_image.shape[1]] = resized_image

    return canvas

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate dataset of padded character images")
    parser.add_argument("source_folder", help="Path to the folder containing input images")
    parser.add_argument("dest_folder", help="Path to the folder where the output images will be saved")
    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    os.makedirs(args.dest_folder, exist_ok=True)

    # Iterate through each image file in the source folder
    for image_file in os.listdir(args.source_folder):
        # Check if the file is an image
        if image_file.endswith('.png') or image_file.endswith('.jpg') or image_file.endswith('.jpeg'):
            # Open the image
            image_path = os.path.join(args.source_folder, image_file)
            character_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Pad the character image to a square canvas of size 32x32
            padded_image = pad_to_square(character_image, size=32)
            
            # Save the image
            output_path = os.path.join(args.dest_folder, image_file)
            cv2.imwrite(output_path, padded_image)

    print("Dataset generated.")