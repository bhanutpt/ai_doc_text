
# Program to preprocess the images, prepare the images for character extraction.
# make sure to keep the input documents in a folder.
# Specify input folder and output floder as command line arguments.
#
# Usage :-
# python preprocess_images <source_folder> <destnation_folder>
#
# Example :-
# python preprocess_images.py src_images out_images 

# Code inspired by assistance from ChatGPT
# OpenAI GPT-3.5 model trained by OpenAI
# https://openai.com

import cv2
import os
import argparse

def apply_adaptive_threshold(image_path, block_size, constant):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)

    return binary_image

def preprocess_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    image_files = os.listdir(input_folder)

    #target_size = (32, 32)  # for resizing

    # Iterate through each image file
    for image_file in image_files:
        # Check if the file is an image
        if image_file.endswith('.png') or image_file.endswith('.jpg') or image_file.endswith('.jpeg'):
            # Construct the input and output paths
            input_image_path = os.path.join(input_folder, image_file)
            output_image_path = os.path.join(output_folder, image_file)

            # Apply adaptive thresholding
            processed_image = apply_adaptive_threshold(input_image_path, 81, 16)

            # Save the processed image
            cv2.imwrite(output_image_path, processed_image)

            print(f"Processed image saved as: {output_image_path}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Prepare images for character extraction")
    parser.add_argument("source_folder", help="Path to the folder containing input images")
    parser.add_argument("dest_folder", help="Path to the folder where the output images will be saved")
    args = parser.parse_args()
   
    # Preprocess images
    preprocess_images(args.source_folder, args.dest_folder)