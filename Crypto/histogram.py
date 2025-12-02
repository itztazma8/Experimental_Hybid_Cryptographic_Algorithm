import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def calculate_histogram(image):
    """Calculate histogram for each RGB channel"""
    # Split into channels (OpenCV uses BGR)
    b, g, r = cv2.split(image)
    
    # Calculate histograms
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()
    
    return hist_r, hist_g, hist_b

def plot_overlapping_histogram(hist_r, hist_g, hist_b, title, ax):
    """Plot overlapping histogram exactly like the reference image"""
    x = np.arange(256)
    
    # Plot overlapping filled areas with higher alpha for better blending
    # The order and alpha create the mountain effect
    ax.fill_between(x, 0, hist_b, color='#4444FF', alpha=0.7, linewidth=0, label='Blue Channel')
    ax.fill_between(x, 0, hist_g, color='#44FF44', alpha=0.7, linewidth=0, label='Green Channel')
    ax.fill_between(x, 0, hist_r, color='#FF4444', alpha=0.7, linewidth=0, label='Red Channel')
    
    # Styling
    ax.set_xlabel('Intensity', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlim([0, 255])
    ax.set_ylim(bottom=0)
    
    # Set x-axis ticks
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    
    # Add border/box
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    # Grid off
    ax.grid(False)
    
    # Legend with correct order
    legend_elements = [
        Patch(facecolor='#FF4444', alpha=0.7, label='Red Channel'),
        Patch(facecolor='#44FF44', alpha=0.7, label='Green Channel'),
        Patch(facecolor='#4444FF', alpha=0.7, label='Blue Channel')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
             frameon=True, fancybox=False, shadow=False, framealpha=0.95)
    
    # Set background
    ax.set_facecolor('white')

def main():
    # Load images
    try:
        img1 = cv2.imread('Lena.png')
        img2 = cv2.imread('final_image.png')
        
        if img1 is None:
            print("Error: Could not load Lena.png")
            print("Make sure Lena.png is in the same directory as this script")
            return
        if img2 is None:
            print("Error: Could not load final_image.png")
            print("Make sure final_image.png is in the same directory as this script")
            return
        
        print(f"✓ Loaded Lena.png - Size: {img1.shape}")
        print(f"✓ Loaded final_image.png - Size: {img2.shape}")
        
        # Calculate histograms
        hist1_r, hist1_g, hist1_b = calculate_histogram(img1)
        hist2_r, hist2_g, hist2_b = calculate_histogram(img2)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot histograms
        plot_overlapping_histogram(hist1_r, hist1_g, hist1_b, 
                                   'Histograms of RGB Channels', ax1)
        plot_overlapping_histogram(hist2_r, hist2_g, hist2_b, 
                                   'Histograms of RGB Channels', ax2)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig('rgb_histograms.png', dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print("✓ Histograms saved as 'rgb_histograms.png'")
        print("✓ Displaying histograms...")
        
        # Display
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()