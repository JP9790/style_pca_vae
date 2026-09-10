import cv2
import numpy as np
import os
from pathlib import Path
import sys

def zhang_suen_thinning(image):
    """
    Applies Zhang-Suen thinning algorithm to a binary image
    """
    # Ensure binary image
    binary = (image > 0).astype(np.uint8)
    
    # Zhang-Suen thinning algorithm implementation
    def neighbours(x, y, image):
        """Return 8-neighbours of point p1 of picture, in order"""
        img = image
        x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
        return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]
    
    def transitions(neighbours):
        """Count the number of 0,1 patterns (transitions) in the ordered sequence"""
        n = neighbours + neighbours[0:1]
        return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))
    
    changing1 = changing2 = 1
    rows, cols = binary.shape
    while changing1 or changing2:
        changing1 = []
        changing2 = []
        
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                if binary[x][y] == 0:
                    continue
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, binary)
                if (2 <= sum(n) <= 6 and
                    transitions(n) == 1 and
                    P2 * P4 * P6 == 0 and
                    P4 * P6 * P8 == 0):
                    changing1.append((x,y))
        
        for x, y in changing1:
            binary[x][y] = 0
        
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                if binary[x][y] == 0:
                    continue
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, binary)
                if (2 <= sum(n) <= 6 and
                    transitions(n) == 1 and
                    P2 * P4 * P8 == 0 and
                    P2 * P6 * P8 == 0):
                    changing2.append((x,y))
        
        for x, y in changing2:
            binary[x][y] = 0
    
    return binary * 255

def find_connected_components(image):
    """
    Find connected components in binary image
    """
    binary = (image > 0).astype(np.uint8)
    num_labels4, labels4 = cv2.connectedComponents(binary, connectivity=4)
    num_labels8, labels8 = cv2.connectedComponents(binary, connectivity=8)
    return num_labels4 - 1, num_labels8 - 1  # Subtract 1 to exclude background

def write_image_txt(filename, image):
    """
    Write image as tab-separated text file
    """
    with open(filename, 'w') as f:
        for row in image:
            f.write('\t'.join(map(str, row)) + '\n')

def adaptive_threshold_image(image):
    """
    Apply adaptive thresholding similar to the C++ version
    """
    # Find original connected components
    nrcomponents_org4, nrcomponents_org8 = find_connected_components(image)
    orgnrpoints = np.count_nonzero(image)
    
    thr = 0
    stepsize = 25
    done = False
    thresholded = image.copy()
    
    while not done:
        _, thresholded = cv2.threshold(image, thr, 255, cv2.THRESH_BINARY)
        
        nrcomponents4, nrcomponents8 = find_connected_components(thresholded)
        nrpoints = np.count_nonzero(thresholded)
        
        if nrpoints < 0.5 * orgnrpoints:
            done = True
        elif (nrcomponents4 != nrcomponents_org4) or (nrcomponents8 != nrcomponents_org8) or (thr >= 250):
            done = True
        else:
            thr += stepsize
    
    thr = max(0, thr - stepsize)
    _, thresholded = cv2.threshold(image, thr, 255, cv2.THRESH_BINARY)
    
    return thresholded

def process_image(input_path, output_dir, image_name):
    """
    Process a single image: threshold and thin
    """
    print(f"Processing: {input_path}")
    
    # Read image in grayscale
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not read image: {input_path}")
        return
    
    # Invert image if background is white (assumes black text on white background)
    # Check if mean pixel value is high (white background)
    if np.mean(image) > 127:
        image = 255 - image  # Invert to white text on black background
    
    base_name = os.path.join(output_dir, image_name)
    
    # Save original
    cv2.imwrite(f"{base_name}-input.png", image)
    write_image_txt(f"{base_name}-input.txt", image)
    
    # Apply adaptive thresholding
    thresholded = adaptive_threshold_image(image)
    cv2.imwrite(f"{base_name}-thresholded.png", thresholded)
    write_image_txt(f"{base_name}-thresholded.txt", thresholded)
    
    # Apply Zhang-Suen thinning
    thinned = zhang_suen_thinning(thresholded)
    cv2.imwrite(f"{base_name}-thinned.png", thinned)
    write_image_txt(f"{base_name}-thinned.txt", thinned)

def main():
    if len(sys.argv) < 3:
        print("Usage: python process_images.py <input-directory> <output-directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all PNG files recursively, preserving directory structure
    img_count = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.png'):
                input_path = os.path.join(root, file)
                image_name = os.path.splitext(file)[0]
                
                # Get relative path from input_dir to preserve structure
                rel_path = os.path.relpath(root, input_dir)
                if rel_path == '.':
                    output_subdir = output_dir
                else:
                    output_subdir = os.path.join(output_dir, rel_path)
                
                # Create output subdirectory if needed
                os.makedirs(output_subdir, exist_ok=True)
                
                process_image(input_path, output_subdir, image_name)
                img_count += 1
    
    print(f"Processed {img_count} images.")

if __name__ == "__main__":
    main()
