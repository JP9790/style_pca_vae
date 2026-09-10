import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

def get_point_list(image):
    """
    Get list of all non-zero points in the image
    Returns: array of (row, col) coordinates
    """
    points = np.argwhere(image > 0)
    return points

def find_endpoints(image):
    """
    Find endpoints in the thinned image (points with only 1 neighbor)
    """
    nrrows, nrcols = image.shape
    endpoints = []
    
    for row in range(nrrows):
        for col in range(nrcols):
            if image[row, col]:
                # Count neighbors
                row_from = max(0, row - 1)
                row_to = min(nrrows - 1, row + 1)
                col_from = max(0, col - 1)
                col_to = min(nrcols - 1, col + 1)
                
                neighbors = image[row_from:row_to+1, col_from:col_to+1]
                nr_neighbors = np.sum(neighbors) - 1  # Subtract self
                
                if nr_neighbors <= 1:
                    endpoints.append([row, col])
    
    return np.array(endpoints) if endpoints else np.array([])

def remove_small_components(image):
    """
    Remove connected components with only 1 or 2 points
    """
    endpoints = find_endpoints(image)
    
    if len(endpoints) == 0:
        return image
    
    nrrows, nrcols = image.shape
    result = image.copy()
    
    for endpoint in endpoints:
        row, col = endpoint
        component = np.zeros((nrrows, nrcols))
        
        row_from = max(0, row - 1)
        row_to = min(nrrows - 1, row + 1)
        col_from = max(0, col - 1)
        col_to = min(nrcols - 1, col + 1)
        
        neighbors = image[row_from:row_to+1, col_from:col_to+1]
        component[row_from:row_to+1, col_from:col_to+1] = neighbors
        
        if np.sum(component > 0) <= 2:
            result = result * (component == 0)
    
    return result

def solve_tsp_greedy(points, start_idx):
    """
    Solve TSP using nearest neighbor heuristic with cost function encouraging minimal distances
    """
    n = len(points)
    if n == 0:
        return []
    
    # Calculate distance matrix with cost = sum(abs_dist^3)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = np.abs(points[i] - points[j])
            cost = np.sum(dist ** 3)  # Encourage minimal distances
            dist_matrix[i, j] = cost
    
    # Greedy nearest neighbor starting from start_idx
    unvisited = set(range(n))
    tour = [start_idx]
    unvisited.remove(start_idx)
    
    current = start_idx
    while unvisited:
        nearest = min(unvisited, key=lambda x: dist_matrix[current, x])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return tour

def extract_strokes_tsp(image):
    """
    Extract strokes from thinned image using TSP
    Returns points in (col, row) format to match R version
    """
    pointlist = get_point_list(image)
    
    if len(pointlist) == 0:
        return np.array([[-2, -2]])
    
    startingpointlist = find_endpoints(image)
    
    # If no endpoints, use all points
    if len(startingpointlist) == 0:
        startingpointlist = pointlist
    
    # Use top-left point as starting point
    dist = np.sum(startingpointlist, axis=1)
    start_idx_in_endpoints = np.argmin(dist)
    starting_point = startingpointlist[start_idx_in_endpoints]
    
    # Find index of starting point in full pointlist
    start_idx = np.where((pointlist == starting_point).all(axis=1))[0][0]
    
    # Solve TSP
    tour = solve_tsp_greedy(pointlist, start_idx)
    
    # Get points in tour order, swap to (col, row) format
    points_tsp = pointlist[tour][:, ::-1]  # Swap row,col to col,row
    
    # Separate strokes with marker lines (-1, -1)
    result = []
    for i in range(len(points_tsp)):
        result.append(points_tsp[i])
        if i < len(points_tsp) - 1:
            # Check if next point is more than 1 pixel away
            if np.max(np.abs(points_tsp[i+1] - points_tsp[i])) > 1:
                result.append([-1, -1])  # End of stroke marker
    
    result.append([-2, -2])  # End of sequence marker
    
    return np.array(result)

def construct_input_data(points, label=0, add_class_outputs=False):
    """
    Construct input data from points
    """
    nrpoints = len(points)
    
    if add_class_outputs:
        result = np.zeros((nrpoints, 14))  # 10 class + 2 dx/dy + 2 markers
        result[:, label] = 1  # One-hot encoding
        nrcols = 14
    else:
        result = np.zeros((nrpoints, 4))  # 2 dx/dy + 2 markers
        nrcols = 4
    
    xprev = 0
    yprev = 0
    
    for p in range(nrpoints):
        point = points[p]
        x, y = point[0], point[1]
        
        if x >= 0 and y >= 0:
            dx = x - xprev
            dy = y - yprev
            xprev = x
            yprev = y
            result[p, nrcols-4:nrcols-2] = [dx, dy]
        else:
            result[p, nrcols-2] = 1  # End of stroke
    
    result[nrpoints-1, nrcols-1] = 1  # End of sequence
    
    return result

def process_thinned_images(image_folder, result_folder):
    """
    Process all thinned images and extract strokes
    """
    # Find all thinned.txt files
    thinned_files = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith('-thinned.txt'):
                thinned_files.append(os.path.join(root, file))
    
    print(f"Found {len(thinned_files)} thinned images to process")
    
    if len(thinned_files) == 0:
        print("No thinned images found!")
        return
    
    # Create result folder if it doesn't exist
    os.makedirs(result_folder, exist_ok=True)
    
    for thinned_file in thinned_files:
        # Extract base name
        base_name = os.path.basename(thinned_file).replace('-thinned.txt', '')
        
        # Get relative path from image_folder to preserve structure
        rel_dir = os.path.relpath(os.path.dirname(thinned_file), image_folder)
        if rel_dir == '.':
            output_subdir = result_folder
        else:
            output_subdir = os.path.join(result_folder, rel_dir)
        
        # Create output subdirectory if needed
        os.makedirs(output_subdir, exist_ok=True)
        
        print(f"Processing image: {base_name}")
        
        # Read thinned image
        thinned = np.loadtxt(thinned_file)
        img = thinned > 0
        
        # Remove small components
        img_denoised = remove_small_components(img)
        
        # Extract strokes
        points = extract_strokes_tsp(img_denoised)
        
        # Save points
        points_file = os.path.join(output_subdir, f"{base_name}-points.txt")
        np.savetxt(points_file, points, delimiter=',', header='x,y', comments='', fmt='%d')
        
        # Create input data (without class outputs)
        inputdata = construct_input_data(points, label=0, add_class_outputs=False)
        inputdata_file = os.path.join(output_subdir, f"{base_name}-inputdata.txt")
        np.savetxt(inputdata_file, inputdata, fmt='%.6f')
        
        # Create visualization
        try:
            plt.figure(figsize=(6, 6))
            
            # Filter out marker points for plotting
            valid_points = points[(points[:, 0] >= 0) & (points[:, 1] >= 0)]
            
            if len(valid_points) > 0:
                # Plot strokes
                current_stroke = []
                for i, point in enumerate(points):
                    if point[0] >= 0 and point[1] >= 0:
                        current_stroke.append(point)
                    else:
                        if current_stroke:
                            stroke_array = np.array(current_stroke)
                            plt.plot(stroke_array[:, 0], stroke_array[:, 1], 'b-', linewidth=2)
                            current_stroke = []
                
                # Plot final stroke if exists
                if current_stroke:
                    stroke_array = np.array(current_stroke)
                    plt.plot(stroke_array[:, 0], stroke_array[:, 1], 'b-', linewidth=2)
                
                # Mark all points
                plt.plot(valid_points[:, 0], valid_points[:, 1], 'r.', markersize=3)
                
                # Mark start point
                plt.plot(valid_points[0, 0], valid_points[0, 1], 'go', markersize=10, 
                        markerfacecolor='green', markeredgecolor='darkgreen', markeredgewidth=2)
            
            plt.title(f'Strokes: {base_name}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.gca().set_aspect('equal')
            plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates (origin top-left)
            plt.grid(True, alpha=0.3)
            
            plot_file = os.path.join(output_subdir, f"{base_name}-strokes.png")
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Could not create plot for {base_name}: {e}")
    
    print(f"Processed {len(thinned_files)} images successfully!")

def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_strokes.py <thinned-image-folder> <output-folder>")
        sys.exit(1)
    
    image_folder = sys.argv[1]
    result_folder = sys.argv[2]
    
    print(f"Thinned image folder: {image_folder}")
    print(f"Output folder: {result_folder}")
    
    process_thinned_images(image_folder, result_folder)

if __name__ == "__main__":
    main()
