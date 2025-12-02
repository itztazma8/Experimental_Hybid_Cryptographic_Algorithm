#!/usr/bin/env python3
"""
Compute NPCR and UACI between Lena.png and final_image.png
"""

from PIL import Image
import numpy as np
import os

# ================================
# --- PARAMETERS ---
# ================================
plain_path = "Lena.png"
encrypted_path = "final_image.png"

# ================================
# --- LOAD IMAGES ---
# ================================

def load_image(path, target_size=None):
    """Load image and return as uint8 numpy array (RGB)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.NEAREST)
    return np.array(img, dtype=np.uint8)

plain_img = load_image(plain_path)
encrypted_img = load_image(encrypted_path, target_size=(plain_img.shape[1], plain_img.shape[0]))

if plain_img.shape != encrypted_img.shape:
    raise ValueError(f"Image size mismatch: {plain_img.shape} vs {encrypted_img.shape}")

# ================================
# --- NPCR & UACI CALC ---
# ================================

def npcr_uaci_per_channel(img1, img2):
    """
    Compute NPCR and UACI for a single channel.
    """
    H, W = img1.shape
    total = H * W

    # NPCR
    diff_pixels = np.count_nonzero(img1 != img2)
    npcr = (diff_pixels / total) * 100.0

    # UACI
    abs_diff = np.abs(img1.astype(np.int32) - img2.astype(np.int32))
    uaci = (np.sum(abs_diff) / (255.0 * total)) * 100.0

    return npcr, uaci


def compute_npcr_uaci(img1, img2):
    """
    Compute NPCR & UACI per channel and average.
    """
    per_channel = []
    for c in range(3):  # R,G,B
        npcr, uaci = npcr_uaci_per_channel(img1[..., c], img2[..., c])
        per_channel.append((npcr, uaci))

    npcr_avg = sum(x[0] for x in per_channel) / 3
    uaci_avg = sum(x[1] for x in per_channel) / 3
    return per_channel, (npcr_avg, uaci_avg)


# ================================
# --- EXECUTION ---
# ================================

channels = ["Red", "Green", "Blue"]
(per_channel, (npcr_avg, uaci_avg)) = compute_npcr_uaci(plain_img, encrypted_img)

print("\n=== NPCR & UACI Results ===")
print("Channel    NPCR (%)     UACI (%)")
print("----------------------------------")
for name, (npcr, uaci) in zip(channels, per_channel):
    print(f"{name:<8}  {npcr:10.4f}   {uaci:10.4f}")
print("----------------------------------")
print(f"Average   {npcr_avg:10.4f}   {uaci_avg:10.4f}")
print()

# Optional: Save to a text report
with open("npcr_uaci_report.txt", "w") as f:
    f.write("NPCR & UACI Results\n")
    f.write("Channel    NPCR (%)     UACI (%)\n")
    f.write("----------------------------------\n")
    for name, (npcr, uaci) in zip(channels, per_channel):
        f.write(f"{name:<8}  {npcr:10.4f}   {uaci:10.4f}\n")
    f.write("----------------------------------\n")
    f.write(f"Average   {npcr_avg:10.4f}   {uaci_avg:10.4f}\n")
    f.write("\n")
print("✅ Results also saved to npcr_uaci_report.txt")
