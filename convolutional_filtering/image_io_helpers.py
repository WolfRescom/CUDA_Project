%%writefile image_io_helpers.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to load image, convert to grayscale, save as .bin
def save_image_as_bin(input_image_path, bin_output_path, width=1024, height=1024):
    # Load the image (grayscale)
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {input_image_path}")

    # Resize if needed
    img = cv2.resize(img, (width, height))
    print(f"Saving image of size: {img.shape}")

    # Save as flat .bin file
    img.tofile(bin_output_path)
    print(f"Saved binary image to: {bin_output_path}")

# Function to load a .bin and save/display as image
def load_bin_and_display(bin_path, width=1024, height=1024, save_output_path=None):
    # Load flat binary file
    img = np.fromfile(bin_path, dtype=np.uint8)
    img = img.reshape((height, width))

    # Display the image
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

    # Save output if requested
    if save_output_path:
        cv2.imwrite(save_output_path, img)
        print(f"Saved output image to: {save_output_path}")

if __name__ == '__main__':
    