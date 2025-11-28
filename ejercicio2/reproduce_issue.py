import cv2 as cv
import numpy as np
import os

def reproduce():
    # Path to image
    img_path = "../data/zigzag.jpg"
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        return

    img1 = cv.imread(img_path)
    h, w = img1.shape[:2]
    print(f"Original size: w={w}, h={h}")

    # Homography from parte_c1
    H = np.array([
        [0.5, 0, 0],
        [0, 1, 0],
        [-1.0/(2.0*w), 0, 1]
    ], dtype=np.float32)

    # Points to check: 0, w/2, w
    points_x = [0, w/2, w]
    print("\nChecking horizontal mapping constraints:")
    for x in points_x:
        # Point (x, 0)
        pt = np.array([x, 0, 1], dtype=np.float32)
        new_pt = H @ pt
        new_pt /= new_pt[2]
        print(f"x={x} -> x'={new_pt[0]:.2f}")
        
    # Check if w/2 maps to w/3
    mid_mapped = (H @ np.array([w/2, 0, 1])) 
    mid_mapped /= mid_mapped[2]
    
    expected_third = w / 3.0
    if np.isclose(mid_mapped[0], expected_third, atol=1.0):
        print(f"\nSUCCESS: Left half (w/2={w/2}) maps to approx w/3 ({mid_mapped[0]:.2f} vs {expected_third:.2f})")
    else:
        print(f"\nFAILURE: Left half (w/2={w/2}) DOES NOT map to w/3. Got {mid_mapped[0]:.2f}, expected {expected_third:.2f}")

    # Check vertical bounds
    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)
    corners_homog = np.hstack([corners, np.ones((4, 1))])
    transformed_corners = []
    
    for pt in corners_homog:
        new_pt = H @ pt
        new_pt /= new_pt[2]
        transformed_corners.append(new_pt[:2])
        
    transformed_corners = np.array(transformed_corners)
    max_y = transformed_corners[:, 1].max()
    
    print(f"\nVertical bounds check:")
    print(f"Max Y: {max_y:.2f} (Original H: {h})")
    if max_y > h:
        print("ISSUE CONFIRMED: Image is cut off vertically.")

if __name__ == "__main__":
    reproduce()
