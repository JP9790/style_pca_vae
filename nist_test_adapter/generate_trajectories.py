import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def generate_trajectory(points, time_step=0.01, lift_duration=0.05):
    """
    Generate trajectory with time from stroke points
    
    Args:
        points: Array of (x, y) coordinates with markers (-1,-1) for stroke breaks, (-2,-2) for end
        time_step: Time increment between adjacent points (seconds)
        lift_duration: Time duration for pen lifts (seconds)
    
    Returns:
        DataFrame with columns: time, x, y
    """
    trajectory = []
    current_time = 0.0
    
    for i, point in enumerate(points):
        x, y = point[0], point[1]
        
        if x == -1 and y == -1:
            # End of stroke marker - pen lift
            current_time += lift_duration
            # Don't add this point to trajectory (it's just a marker)
            continue
        elif x == -2 and y == -2:
            # End of sequence marker
            break
        else:
            # Regular point - add to trajectory
            trajectory.append([current_time, x, y])
            current_time += time_step
    
    return pd.DataFrame(trajectory, columns=['time', 'x', 'y'])

def plot_trajectory(df, output_path, title):
    """
    Plot the x-y trajectory to verify it looks correct
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: X-Y trajectory with time colormap
    scatter = ax1.scatter(df['x'], df['y'], c=df['time'], cmap='viridis', s=10)
    ax1.plot(df['x'], df['y'], 'b-', alpha=0.3, linewidth=1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f'{title} - Trajectory (colored by time)')
    ax1.set_aspect('equal')
    ax1.invert_yaxis()  # Match image coordinates
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Time (s)')
    
    # Mark start and end
    ax1.plot(df['x'].iloc[0], df['y'].iloc[0], 'go', markersize=12, 
            markerfacecolor='green', markeredgecolor='darkgreen', markeredgewidth=2,
            label='Start')
    ax1.plot(df['x'].iloc[-1], df['y'].iloc[-1], 'ro', markersize=12,
            markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2,
            label='End')
    ax1.legend()
    
    # Plot 2: Time series (X and Y over time)
    ax2.plot(df['time'], df['x'], 'b-', label='X position', linewidth=2)
    ax2.plot(df['time'], df['y'], 'r-', label='Y position', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position')
    ax2.set_title(f'{title} - Position over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def process_all_strokes(input_folder, output_folder):
    """
    Process all stroke files and generate trajectories
    """
    # Find all points files
    points_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('-points.txt'):
                points_files.append(os.path.join(root, file))
    
    print(f"Found {len(points_files)} stroke files to process")
    
    if len(points_files) == 0:
        print("No points files found!")
        return
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each file
    for points_file in points_files:
        base_name = os.path.basename(points_file).replace('-points.txt', '')
        
        # Get relative path from input_folder to preserve structure
        rel_dir = os.path.relpath(os.path.dirname(points_file), input_folder)
        if rel_dir == '.':
            output_subdir = output_folder
        else:
            output_subdir = os.path.join(output_folder, rel_dir)
        
        # Create output subdirectory if needed
        os.makedirs(output_subdir, exist_ok=True)
        
        print(f"Processing: {base_name}")
        
        # Read points (skip header if exists)
        try:
            points = np.loadtxt(points_file, delimiter=',', skiprows=1)
        except:
            points = np.loadtxt(points_file, delimiter=',')
        
        # Generate trajectory
        trajectory_df = generate_trajectory(points, time_step=0.01, lift_duration=0.05)
        
        # Save as CSV
        csv_path = os.path.join(output_subdir, f"{base_name}-trajectory.csv")
        trajectory_df.to_csv(csv_path, index=False)
        
        # Plot trajectory
        plot_path = os.path.join(output_subdir, f"{base_name}-trajectory.png")
        plot_trajectory(trajectory_df, plot_path, base_name)
        
        # Print statistics
        duration = trajectory_df['time'].iloc[-1]
        num_points = len(trajectory_df)
        print(f"  Duration: {duration:.2f}s, Points: {num_points}")
    
    print(f"\nProcessed {len(points_files)} trajectories successfully!")
    print(f"Output saved to: {output_folder}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_trajectories.py <input-folder> <output-folder>")
        print("  input-folder: folder containing *-points.txt files")
        print("  output-folder: folder to save trajectory CSVs and plots")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print()
    
    process_all_strokes(input_folder, output_folder)

if __name__ == "__main__":
    main()
