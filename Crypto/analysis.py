import cv2
import numpy as np
from scipy.stats import pearsonr

# ---------- Load Images ----------
plain_img = cv2.imread("Lena.png")
encrypted_img = cv2.imread("final_image.png")

# Convert to RGB for consistency (cv2 loads as BGR)
plain_img = cv2.cvtColor(plain_img, cv2.COLOR_BGR2RGB)
encrypted_img = cv2.cvtColor(encrypted_img, cv2.COLOR_BGR2RGB)

# Resize encrypted image to match plain if needed
if plain_img.shape != encrypted_img.shape:
    encrypted_img = cv2.resize(encrypted_img, (plain_img.shape[1], plain_img.shape[0]))

# ---------- Function to calculate correlation in directions ----------
def channel_correlations(img):
    """Returns horizontal, vertical, diagonal correlation coefficients for one channel."""
    h, w = img.shape
    # Horizontal pairs
    x_h = img[:, :-1].flatten()
    y_h = img[:, 1:].flatten()
    
    # Vertical pairs
    x_v = img[:-1, :].flatten()
    y_v = img[1:, :].flatten()
    
    # Diagonal pairs
    x_d = img[:-1, :-1].flatten()
    y_d = img[1:, 1:].flatten()
    
    # Compute Pearson correlation
    def corr(x, y):
        return pearsonr(x, y)[0]
    
    return corr(x_h, y_h), corr(x_v, y_v), corr(x_d, y_d)

# ---------- Compute for Each Channel ----------
channels = ['Red', 'Green', 'Blue']
results = {'Plain': {}, 'Encrypted': {}}

for i, ch in enumerate(channels):
    plain_ch = plain_img[:, :, i]
    enc_ch = encrypted_img[:, :, i]

    results['Plain'][ch] = channel_correlations(plain_ch)
    results['Encrypted'][ch] = channel_correlations(enc_ch)

# ---------- Display Results ----------
print("Table: Correlation Coefficients of Individual Channels (LENA)")
print("--------------------------------------------------------------")
print(f"{'Channel':<10} {'Image':<10} {'Horizontal':>12} {'Vertical':>12} {'Diagonal':>12}")
print("--------------------------------------------------------------")

for ch in channels:
    ph, pv, pd = results['Plain'][ch]
    eh, ev, ed = results['Encrypted'][ch]
    print(f"{ch:<10} {'Plain':<10} {ph:>12.4f} {pv:>12.4f} {pd:>12.4f}")
    print(f"{'':<10} {'Encrypted':<10} {eh:>12.4f} {ev:>12.4f} {ed:>12.4f}")
    print("--------------------------------------------------------------")
