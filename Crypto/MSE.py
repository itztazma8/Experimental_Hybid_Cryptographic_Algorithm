import cv2
import numpy as np

def calculate_metrics(original_path, encrypted_path, size=(512, 512)):
    """
    Calculate MSE, PSNR, NPCR, and UACI between two images.
    """
    # === Load images ===
    original = cv2.imread(original_path)
    encrypted = cv2.imread(encrypted_path)

    if original is None:
        raise ValueError(f"Original image not found at {original_path}")
    if encrypted is None:
        raise ValueError(f"Encrypted image not found at {encrypted_path}")

    # === Resize to same dimensions ===
    original = cv2.resize(original, size)
    encrypted = cv2.resize(encrypted, size)

    # === Convert to float for MSE/PSNR calculation ===
    orig_float = original.astype(np.float32)
    enc_float = encrypted.astype(np.float32)

    # === MSE ===
    mse = np.mean((orig_float - enc_float) ** 2)

    # === PSNR ===
    psnr = float('inf') if mse == 0 else 10 * np.log10((255 ** 2) / mse)

    # === NPCR (Number of Pixels Change Rate) ===
    diff_pixels = np.not_equal(original, encrypted)
    npcr = (np.count_nonzero(diff_pixels) / diff_pixels.size) * 100

    # === UACI (Unified Average Changing Intensity) ===
    uaci = (np.sum(np.abs(orig_float - enc_float)) / (orig_float.size * 255)) * 100

    return mse, psnr, npcr, uaci

if __name__ == "__main__":
    original_path = "original.jpg"
    encrypted_path = "dec.jpg"

    mse, psnr, npcr, uaci = calculate_metrics(original_path, encrypted_path)

    print(f"✅ Mean Squared Error (MSE): {mse:.4f}")
    print(f"🔸 Peak Signal-to-Noise Ratio (PSNR): {psnr:.4f} dB")
    print(f"🔹 NPCR: {npcr:.4f}%")
    print(f"🔹 UACI: {uaci:.4f}%")
