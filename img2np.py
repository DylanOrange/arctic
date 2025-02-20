import numpy as np
import os

data_path = '/ssd/dylu/data/arctic/arctic_data/data/cropped_images'

input_dir = '/ssd/dylu/data/arctic/arctic_data/data/cropped_images/'
output_dir = '/ssd/dylu/data/arctic/arctic_data/data/arctic_npys/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over the directory structure
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.jpg'):
            # Construct the input and output paths
            img_path = os.path.join(root, file)
            rel_path = os.path.relpath(img_path, input_dir)  # Relative path from input directory
            output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.npy')

            # Load the image and convert to numpy array
            img = Image.open(img_path)
            img_array = np.array(img)

            # Save the numpy array
            os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directories if necessary
            np.save(output_path, img_array)

            print(f"Image {img_path} saved as {output_path}")

print("Done!")