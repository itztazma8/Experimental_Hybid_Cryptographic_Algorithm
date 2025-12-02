import time

start_time = time.time()  # record start

import numpy as np
from Matrix.MatrixGen import matrix  # Importing the generated matrix from MatrixGen.py

def text_to_numbers(text):
    """Convert text to numbers (A=0, B=1, ..., Z=25)."""
    return [ord(char) - ord('A') for char in text.upper() if char.isalpha()]


def numbers_to_text(numbers):
    """Convert numbers back to text (0=A, 1=B, ..., 25=Z)."""
    return ''.join(chr(num + ord('A')) for num in numbers)


def mod_26(matrix):
    """Apply modulo 26 to each element of the matrix."""
    return np.mod(matrix, 26)


def encrypt(plaintext, key_matrix):
    """Encrypt the plaintext using the Hill cipher."""
    # Convert plaintext to numbers
    plaintext_numbers = text_to_numbers(plaintext)

    # Ensure plaintext length is even
    original_length = len(plaintext_numbers)
    if original_length % 2 != 0:
        plaintext_numbers.append(0)  # Padding with 'A' (0)

    # Display the plaintext as a matrix
    #print("Plaintext as Matrix:")
    for i in range(0, len(plaintext_numbers), 2):
        plain_matrix = np.array([[plaintext_numbers[i]], [plaintext_numbers[i + 1]]])
        #print(plain_matrix)

    # Create 2x1 matrices and encrypt
    ciphertext_numbers = []

    for i in range(0, len(plaintext_numbers), 2):
        # Create a 2x1 matrix from the plaintext
        plain_matrix = np.array([[plaintext_numbers[i]], [plaintext_numbers[i + 1]]])

        # Multiply by the key matrix and apply mod 26
        encrypted_matrix = mod_26(np.dot(key_matrix, plain_matrix))

        # Append the result to ciphertext
        ciphertext_numbers.extend(encrypted_matrix.flatten().tolist())

    # Convert encrypted numbers back to text
    ciphertext = numbers_to_text(ciphertext_numbers)

    return ciphertext, ciphertext_numbers, original_length


def decrypt(ciphertext_numbers, key_matrix, original_length):
    """Decrypt the ciphertext using the inverse of the key matrix."""

    # Calculate the inverse of the key matrix modulo 26
    det = int(np.round(np.linalg.det(key_matrix)))  # Determinant of K
    inv_det = pow(det, -1, 26)  # Modular multiplicative inverse of determinant

    # Calculate adjugate matrix for inverse calculation
    adjugate_matrix = np.array([[key_matrix[1][1], -key_matrix[0][1]],
                                [-key_matrix[1][0], key_matrix[0][0]]])

    inverse_key_matrix = mod_26(inv_det * adjugate_matrix)

    # Display the inverse key matrix being used for decryption
    #print("Using Inverse Key Matrix (K^-1):\n", inverse_key_matrix)

    # Create a list to hold decrypted numbers
    decrypted_numbers = []

    for i in range(0, len(ciphertext_numbers), 2):
        # Create a 2x1 matrix from the ciphertext
        cipher_matrix = np.array([[ciphertext_numbers[i]], [ciphertext_numbers[i + 1]]])

        # Multiply by the inverse key matrix and apply mod 26
        decrypted_matrix = mod_26(np.dot(inverse_key_matrix, cipher_matrix))

        # Append the result to decrypted numbers
        decrypted_numbers.extend(decrypted_matrix.flatten().tolist())

    # Convert decrypted numbers back to text
    decrypted_text = numbers_to_text(decrypted_numbers)

    # Remove padding if necessary (if last character is 'A')
    if len(decrypted_text) > original_length:
        decrypted_text = decrypted_text[:-1]  # Remove padding 'A'

    return decrypted_text


# Display the key matrix being used
#print("Using Key Matrix (K):\n", matrix)

# Ask user for input plaintext message
plaintext = "DNGRSHEWROTE"

# Encrypt the message using the imported key matrix
encrypted_message, ciphertext_numbers, original_length = encrypt(plaintext, matrix)

print(f"Plaintext: {plaintext}")
print(f"Encrypted: {encrypted_message}")

# Decrypting the message using the inverse of K
#decrypted_message = decrypt(ciphertext_numbers, matrix, original_length)

#print(f"Decrypted: {decrypted_message}")




import numpy as np
def parameter_generation(message):
    bin_bits = []
    final = []
    result1=[]

    # Convert each character to 8-bit binary
    for i in message:
        ascii_val = format(ord(i), '08b')
        bin_bits.append(ascii_val)
    
    # XOR every 3 binary groups
    for i in range(0, len(bin_bits), 3):
        if i + 2 >= len(bin_bits):  # safety check
            break
        b1 = np.array(list(bin_bits[i]), dtype=int)
        b2 = np.array(list(bin_bits[i+1]), dtype=int)
        b3 = np.array(list(bin_bits[i+2]), dtype=int)

        result = np.bitwise_xor(np.bitwise_xor(b1, b2), b3)
        binary_result = ''.join(result.astype(str))
        final.append(binary_result)

    # Convert to decimals
    decimals = [int(b, 2) for b in final]

    # Scale to parameters
    if len(decimals) < 4:
        raise ValueError("Need at least 12 characters to generate 4 parameters")

    x0 = decimals[0] / 255
    r  = 3.57 + (decimals[1] / 255) * (4 - 3.57)
    p  = int(1 + (decimals[2] / 255) * 49)
    q  = int(1 + (decimals[3] / 255) * 49)

    result1.append(x0)
    result1.append(r)
    result1.append(p)
    result1.append(q)

    return result1

parametric_values=parameter_generation(encrypted_message)





import Logistic.logisticKey as key   # Importing the key generating function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img


# Accepting an image
path = str(input('Lena.png\n'))
image = img.imread(path)


# ✅ Handle different image channel cases safely
if image.ndim == 2:
    # grayscale image → expand to 3 channels
    image = np.stack((image,)*3, axis=-1)
elif image.shape[-1] == 4:
    # RGBA image → remove alpha channel
    image = image[..., :3]
elif image.shape[-1] == 3:
    # RGB → fine, do nothing
    pass
else:
    raise ValueError(f"Unexpected image shape: {image.shape}")

# Convert to uint8 if not already (some libraries load floats 0–1)
if image.dtype != np.uint8:
    image = (image * 255).astype(np.uint8)


# Displaying the image
#plt.imshow(image)
plt.show()

# Generating dimensions of the image
height = image.shape[0]
width = image.shape[1]
#print(height, width)

# Generating keys
# Calling logistic_key and providing r value such that the keys are pseudo-random
# and generating a key for every pixel of the image
generatedKey = key.logistic_key(parametric_values[0], parametric_values[1], height*width) 
#print(generatedKey)

# Encryption using XOR
z = 0

# Initializing the encrypted image
encryptedImage = np.zeros(shape=[height, width, 3], dtype=np.uint8)

# Substituting all the pixels in original image with nested for
for i in range(height):
    for j in range(width):
        # USing the XOR operation between image pixels and keys
        encryptedImage[i, j] = image[i, j].astype(int) ^ generatedKey[z]
        z += 1

# Displaying the encrypted image
plt.imshow(encryptedImage)
plt.show()

from PIL import Image

# Convert NumPy array to Image
img_encrypted = Image.fromarray(encryptedImage)

# Save as PNG or JPG
img_encrypted.save("encrypted_image.png")


import numpy as np
import math, time, sys
from PIL import Image
from Arnold.arnold import Arnold

image_name = "encrypted_image.png"
image_path = "" + image_name

# Arnold Transform Parameters
a = parametric_values[2]
b = parametric_values[3]
rounds = 33 #For now constant

# Open the images
lena = np.array(Image.open(image_path).convert("L"))

print(" ~~~~~~  * PARAMETERS *  ~~~~~~ ")
arnold = Arnold(a, b, rounds)
print("\ta:\t", a)
print("\tb:\t", b)
print("\trounds:\t", rounds)

print("\n ~~~~~~  *  RESULTS   *  ~~~~~~ ")
    
start_time = time.time()
scrambled = arnold.applyTransformTo(lena)
exec_time = time.time() - start_time
print("Transform  execution time: %.6f " % exec_time, "sec")
im = Image.fromarray(scrambled).convert("L")
im.save("scrambled.png", format="PNG")


#SHA-256

"""This Python module is an implementation of the SHA-256 algorithm.
From https://github.com/keanemind/Python-SHA-256"""

K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

def generate_hash(message: bytearray) -> bytearray:
    """Return a SHA-256 hash from the message passed.
    The argument should be a bytes, bytearray, or
    string object."""

    if isinstance(message, str):
        message = bytearray(message, 'ascii')
    elif isinstance(message, bytes):
        message = bytearray(message)
    elif not isinstance(message, bytearray):
        raise TypeError

    # Padding
    length = len(message) * 8 # len(message) is number of BYTES!!!
    message.append(0x80)
    while (len(message) * 8 + 64) % 512 != 0:
        message.append(0x00)

    message += length.to_bytes(8, 'big') # pad to 8 bytes or 64 bits

    assert (len(message) * 8) % 512 == 0, "Padding did not complete properly!"

    # Parsing
    blocks = [] # contains 512-bit chunks of message
    for i in range(0, len(message), 64): # 64 bytes is 512 bits
        blocks.append(message[i:i+64])

    # Setting Initial Hash Value
    h0 = 0x6a09e667
    h1 = 0xbb67ae85
    h2 = 0x3c6ef372
    h3 = 0xa54ff53a
    h5 = 0x9b05688c
    h4 = 0x510e527f
    h6 = 0x1f83d9ab
    h7 = 0x5be0cd19

    # SHA-256 Hash Computation
    for message_block in blocks:
        # Prepare message schedule
        message_schedule = []
        for t in range(0, 64):
            if t <= 15:
                # adds the t'th 32 bit word of the block,
                # starting from leftmost word
                # 4 bytes at a time
                message_schedule.append(bytes(message_block[t*4:(t*4)+4]))
            else:
                term1 = _sigma1(int.from_bytes(message_schedule[t-2], 'big'))
                term2 = int.from_bytes(message_schedule[t-7], 'big')
                term3 = _sigma0(int.from_bytes(message_schedule[t-15], 'big'))
                term4 = int.from_bytes(message_schedule[t-16], 'big')

                # append a 4-byte byte object
                schedule = ((term1 + term2 + term3 + term4) % 2**32).to_bytes(4, 'big')
                message_schedule.append(schedule)

        assert len(message_schedule) == 64

        # Initialize working variables
        a = h0
        b = h1
        c = h2
        d = h3
        e = h4
        f = h5
        g = h6
        h = h7

        # Iterate for t=0 to 63
        for t in range(64):
            t1 = ((h + _capsigma1(e) + _ch(e, f, g) + K[t] +
                   int.from_bytes(message_schedule[t], 'big')) % 2**32)

            t2 = (_capsigma0(a) + _maj(a, b, c)) % 2**32

            h = g
            g = f
            f = e
            e = (d + t1) % 2**32
            d = c
            c = b
            b = a
            a = (t1 + t2) % 2**32

        # Compute intermediate hash value
        h0 = (h0 + a) % 2**32
        h1 = (h1 + b) % 2**32
        h2 = (h2 + c) % 2**32
        h3 = (h3 + d) % 2**32
        h4 = (h4 + e) % 2**32
        h5 = (h5 + f) % 2**32
        h6 = (h6 + g) % 2**32
        h7 = (h7 + h) % 2**32

    return ((h0).to_bytes(4, 'big') + (h1).to_bytes(4, 'big') +
            (h2).to_bytes(4, 'big') + (h3).to_bytes(4, 'big') +
            (h4).to_bytes(4, 'big') + (h5).to_bytes(4, 'big') +
            (h6).to_bytes(4, 'big') + (h7).to_bytes(4, 'big'))

def _sigma0(num: int):
    """As defined in the specification."""
    num = (_rotate_right(num, 7) ^
           _rotate_right(num, 18) ^
           (num >> 3))
    return num

def _sigma1(num: int):
    """As defined in the specification."""
    num = (_rotate_right(num, 17) ^
           _rotate_right(num, 19) ^
           (num >> 10))
    return num

def _capsigma0(num: int):
    """As defined in the specification."""
    num = (_rotate_right(num, 2) ^
           _rotate_right(num, 13) ^
           _rotate_right(num, 22))
    return num

def _capsigma1(num: int):
    """As defined in the specification."""
    num = (_rotate_right(num, 6) ^
           _rotate_right(num, 11) ^
           _rotate_right(num, 25))
    return num

def _ch(x: int, y: int, z: int):
    """As defined in the specification."""
    return (x & y) ^ (~x & z)

def _maj(x: int, y: int, z: int):
    """As defined in the specification."""
    return (x & y) ^ (x & z) ^ (y & z)

def _rotate_right(num: int, shift: int, size: int = 32):
    """Rotate an integer right."""
    return (num >> shift) | (num << size - shift)

key_for_aes=generate_hash("DNGRSHEWROTE")
print(key_for_aes)





# AES 

import sys
import cv2
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

# This program encrypts a jpg With AES-256. The encrypted image contains more data than the original image (e.g. because of 
# padding, IV etc.). Therefore the encrypted image has one row more. Supported are CBC and ECB mode.

# Set mode
mode = AES.MODE_CBC
#mode = AES.MODE_ECB
if mode != AES.MODE_CBC and mode != AES.MODE_ECB:
    print('Only CBC and ECB mode supported...')
    sys.exit()

# Set sizes
keySize = 32
ivSize = AES.block_size if mode == AES.MODE_CBC else 0

#
# Start Encryption ----------------------------------------------------------------------------------------------
#

# Load original image
imageOrig = cv2.imread(r"E:\Crypto\scrambled.png")
rowOrig, columnOrig, depthOrig = imageOrig.shape

# Check for minimum width
minWidth = (AES.block_size + AES.block_size) // depthOrig + 1
if columnOrig < minWidth:
    print('The minimum width of the image must be {} pixels, so that IV and padding can be stored in a single additional row!'.format(minWidth))
    sys.exit()

# Display original image
cv2.imshow("Original image", imageOrig)
cv2.waitKey()

# Convert original image data to bytes
imageOrigBytes = imageOrig.tobytes()

# Encrypt
key = key_for_aes
iv = get_random_bytes(ivSize)
cipher = AES.new(key, AES.MODE_CBC, iv) if mode == AES.MODE_CBC else AES.new(key, AES.MODE_ECB)
imageOrigBytesPadded = pad(imageOrigBytes, AES.block_size)
ciphertext = cipher.encrypt(imageOrigBytesPadded)

# Convert ciphertext bytes to encrypted image data
#    The additional row contains columnOrig * DepthOrig bytes. Of this, ivSize + paddedSize bytes are used 
#    and void = columnOrig * DepthOrig - ivSize - paddedSize bytes unused
paddedSize = len(imageOrigBytesPadded) - len(imageOrigBytes)
void = columnOrig * depthOrig - ivSize - paddedSize
ivCiphertextVoid = iv + ciphertext + bytes(void)
imageEncrypted = np.frombuffer(ivCiphertextVoid, dtype = imageOrig.dtype).reshape(rowOrig + 1, columnOrig, depthOrig)

# Display encrypted image

cv2.imwrite("final_image.png", imageEncrypted)
cv2.imshow("Encrypted image", imageEncrypted)
cv2.waitKey()

resized_image = cv2.resize(imageEncrypted, (512, 512), interpolation=cv2.INTER_AREA)
cv2.imwrite("resize.png", resized_image)


end_time = time.time()  # record end

total_time = end_time - start_time
print(f"Total time: {total_time:.2f} seconds")

# Save the encrypted image (optional)
#    If the encrypted image is to be stored, a format must be chosen that does not change the data. Otherwise, 
#    decryption is not possible after loading the encrypted image. bmp does not change the data, but jpg does. 
#    When saving with imwrite, the format is controlled by the extension (.jpg, .bmp). The following works:
#    cv2.imwrite("topsecretEnc.bmp", imageEncrypted)
#    imageEncrypted = cv2.imread("topsecretEnc.bmp")

#
# Start Decryption ----------------------------------------------------------------------------------------------
#

# Convert encrypted image data to ciphertext bytes
#rowEncrypted, columnOrig, depthOrig = imageEncrypted.shape 
#rowOrig = rowEncrypted - 1
#encryptedBytes = imageEncrypted.tobytes()
#iv = encryptedBytes[:ivSize]
#imageOrigBytesSize = rowOrig * columnOrig * depthOrig
#paddedSize = (imageOrigBytesSize // AES.block_size + 1) * AES.block_size - imageOrigBytesSize
#encrypted = encryptedBytes[ivSize : ivSize + imageOrigBytesSize + paddedSize]

# Decrypt
#cipher = AES.new(key, AES.MODE_CBC, iv) if mode == AES.MODE_CBC else AES.new(key, AES.MODE_ECB)
#decryptedImageBytesPadded = cipher.decrypt(encrypted)
#decryptedImageBytes = unpad(decryptedImageBytesPadded, AES.block_size)

# Convert bytes to decrypted image data
#decryptedImage = np.frombuffer(decryptedImageBytes, imageEncrypted.dtype).reshape(rowOrig, columnOrig, depthOrig)

# Display decrypted image
#cv2.imshow("Decrypted Image", decryptedImage)
#cv2.waitKey()

# Close all windows
#cv2.destroyAllWindows()



#ARNOLD DECRYPT


#start_time = time.time()
#reconstructed = arnold.applyInverseTransformTo(scrambled)
#exec_time = time.time() - start_time
#print("Inverse T. execution time: %.6f " % exec_time, "sec")
#im = Image.fromarray(reconstructed).convert("L")
#im.save("reconstructed.tif", format="TIFF")

#counter = 0
#for i in range(scrambled.shape[0]):
    #for j in range(scrambled.shape[0]):
        #if(lena[i, j] != reconstructed[i, j]):
            #print(lena[i, j], " != ", reconstructed[i, j])
            #counter += 1
#print("\nDIFFERENT PIXELS\n\toriginal  VS reconstructed:\t\t", counter)


#DECRYPTION LOGISTIC MAP


# Decryption using XOR
#z = 0

# Initializing the decrypted image
#decryptedImage = np.zeros(shape=[height, width, 3], dtype=np.uint8)

# Substituting all the pixels in encrypted image with nested for
#for i in range(height):
    #for j in range(width):
        # USing the XOR operation between encrypted image pixels and keys
        #decryptedImage[i, j] = encryptedImage[i, j].astype(int) ^ generatedKey[z]
        #z += 1

# Displaying the decrypted image
#plt.imshow(decryptedImage)
#plt.show()

