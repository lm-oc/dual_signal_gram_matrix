import cv2
import numpy as np
import random

# Load the uploaded image
image_path = './pg3.png'
image = cv2.imread(image_path)

# Get image dimensions
h, w, _ = image.shape

# Define the size of each grid cell
cell_h, cell_w = h // 3, w // 3

# Initialize a mask with all ones (no erasure)
mask = np.ones((h, w), dtype=np.uint8)

# Create a black block for erasure
black_block = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

# Randomly choose two different blocks to erase
blocks_to_erase = random.sample(range(9), 2)

# Erase the selected blocks by placing the black block
for block in blocks_to_erase:
    row = block // 3
    col = block % 3
    image[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w] = black_block

# Save and output the modified image
output_path = './pg33.png'
cv2.imwrite(output_path, image)
#output_path