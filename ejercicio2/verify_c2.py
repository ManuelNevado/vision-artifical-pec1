import cv2 as cv
import numpy as np

def verify_c2():
    w, h = 100, 100
    
    # Create meshgrid for destination coordinates (x', y')
    map_x_float, map_y_float = np.meshgrid(np.arange(w), np.arange(h))

    # Normalize to [0, 1]
    x_prime = map_x_float / (w - 1)
    y_prime = map_y_float / (h - 1)

    # Inverse mapping:
    # y = y' - (0.4*(x-0.5)**2 - 0.1)
    x_src_norm = x_prime
    y_src_norm = y_prime - (0.4 * (x_prime - 0.5)**2 - 0.1)

    # Check center point (0.5, 0.5)
    # Indices for center
    cx, cy = w // 2, h // 2
    
    src_y_at_center = y_src_norm[cy, cx]
    src_x_at_center = x_src_norm[cy, cx]
    
    print(f"Destination Center ({x_prime[cy, cx]:.2f}, {y_prime[cy, cx]:.2f}) maps to Source ({src_x_at_center:.2f}, {src_y_at_center:.2f})")
    
    expected_y = 0.5 - (0.4 * (0.5 - 0.5)**2 - 0.1)
    print(f"Expected Source Y: {expected_y:.2f}")
    
    if np.isclose(src_y_at_center, expected_y, atol=0.01):
        print("SUCCESS: Center mapping is correct.")
    else:
        print("FAILURE: Center mapping is incorrect.")

if __name__ == "__main__":
    verify_c2()
