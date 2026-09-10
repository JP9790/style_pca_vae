#!/usr/bin/env python3
"""
Process all user_X_digit_library folders (X from 1 to 24):
1. Extract trajectories using nist_test_adapter
2. Learn ProMP for each task using ProMP_basic.py
3. Save results to userX_digit_generic_library
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(f"Exit code: {result.returncode}")
        if result.stderr:
            print(f"Error output:\n{result.stderr}")
        return False
    
    if result.stdout:
        print(result.stdout)
    return True


def process_user_digit_library(user_id: int, base_dir: Path):
    """Process a single user's digit library."""
    print(f"\n{'#'*80}")
    print(f"Processing User {user_id} Digit Library")
    print(f"{'#'*80}")
    
    # Define paths
    input_library = base_dir / f"user_{user_id}_digit_library"
    output_trajectories = base_dir / f"user_{user_id}_digit_library_output"
    output_promp = base_dir / f"user{user_id}_digit_generic_library"
    
    # Check if input library exists
    if not input_library.exists():
        print(f"WARNING: {input_library} does not exist, skipping user {user_id}")
        return False
    
    # Step 1: Process images (adaptive thresholding + thinning)
    print(f"\nStep 1: Processing images for user {user_id}...")
    thinned_dir = base_dir / f"thinned_user_{user_id}_digit"
    if not run_command(
        ["python", "nist_test_adapter/process_images.py", str(input_library), str(thinned_dir)],
        f"Image processing for user {user_id}"
    ):
        return False
    
    # Step 2: Extract strokes
    print(f"\nStep 2: Extracting strokes for user {user_id}...")
    if not run_command(
        ["python", "nist_test_adapter/extract_strokes.py", str(thinned_dir), str(output_trajectories)],
        f"Stroke extraction for user {user_id}"
    ):
        return False
    
    # Step 3: Generate trajectories
    print(f"\nStep 3: Generating trajectories for user {user_id}...")
    if not run_command(
        ["python", "nist_test_adapter/generate_trajectories.py", str(output_trajectories), str(output_trajectories)],
        f"Trajectory generation for user {user_id}"
    ):
        return False
    
    # Step 4: Learn ProMP for each task
    print(f"\nStep 4: Learning ProMP for user {user_id}...")
    if not run_command(
        ["python", "ProMP_basic.py", "--input_dir", str(output_trajectories), "--output_dir", str(output_promp)],
        f"ProMP learning for user {user_id}"
    ):
        return False
    
    print(f"\n✓ Successfully processed user {user_id} digit library!")
    print(f"  Trajectories: {output_trajectories}")
    print(f"  ProMP library: {output_promp}")
    return True


def main():
    """Process all user digit libraries from 1 to 24."""
    base_dir = Path(".").resolve()
    
    print("="*80)
    print("Processing All User Digit Libraries (Users 1-24)")
    print("="*80)
    
    success_count = 0
    failed_users = []
    
    for user_id in range(1, 25):  # Users 1 to 24
        if process_user_digit_library(user_id, base_dir):
            success_count += 1
        else:
            failed_users.append(user_id)
    
    # Summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Total users processed: {success_count}/24")
    if failed_users:
        print(f"Failed users: {failed_users}")
    else:
        print("All users processed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
