import cv2
import numpy as np

# Load image
img = cv2.imread("resize.png")

# Flatten pixels
flat_pixels = img.flatten()

# Convert each byte to bits (0/1 string)
bitstring = ''.join([f'{byte:08b}' for byte in flat_pixels])

# Save to file (text file with 0/1)
with open("resize_bits.txt", "w") as f:
    f.write(bitstring)

print("✅ Bitstring file created:", "resize_bits.txt")
print("Length in bits:", len(bitstring))