#!/usr/bin/env python3
"""
Script to clean up training run folders that don't contain checkpoint .pth files.

This script will:
1. Scan the 'Training Runs' directory
2. Find folders that don't contain any .pth files
3. Delete those folders (with confirmation)
4. Provide a summary of what was deleted
"""

import os
import shutil
from pathlib import Path
import argparse

def find_training_runs_without_checkpoints(training_runs_dir):
    """
    Find training run folders that don't contain any .pth checkpoint files.
    
    Args:
        training_runs_dir (str): Path to the 'Training Runs' directory
        
    Returns:
        list: List of folder paths that don't contain .pth files
    """
    training_runs_path = Path(training_runs_dir)
    
    if not training_runs_path.exists():
        print(f"Error: Directory '{training_runs_dir}' does not exist!")
        return []
    
    folders_without_checkpoints = []
    total_folders = 0
    
    print(f"Scanning directory: {training_runs_path}")
    print("=" * 60)
    
    # Iterate through all subdirectories in Training Runs
    for folder_path in training_runs_path.iterdir():
        if folder_path.is_dir():
            total_folders += 1
            folder_name = folder_path.name
            
            # Check if this folder contains any .pth files
            pth_files = list(folder_path.rglob("*.pth"))
            
            if not pth_files:
                folders_without_checkpoints.append(folder_path)
                print(f"❌ No checkpoints found: {folder_name}")
            else:
                print(f"✅ Checkpoints found ({len(pth_files)}): {folder_name}")
    
    print("=" * 60)
    print(f"Total folders scanned: {total_folders}")
    print(f"Folders without checkpoints: {len(folders_without_checkpoints)}")
    print(f"Folders with checkpoints: {total_folders - len(folders_without_checkpoints)}")
    
    return folders_without_checkpoints

def delete_folders(folders_to_delete, dry_run=True):
    """
    Delete folders (with confirmation if not dry run).
    
    Args:
        folders_to_delete (list): List of folder paths to delete
        dry_run (bool): If True, only show what would be deleted without actually deleting
    """
    if not folders_to_delete:
        print("No folders to delete!")
        return
    
    print(f"\n{'DRY RUN - ' if dry_run else ''}Folders to be deleted:")
    print("=" * 60)
    
    total_size = 0
    for folder_path in folders_to_delete:
        folder_size = get_folder_size(folder_path)
        total_size += folder_size
        print(f"📁 {folder_path.name} ({format_size(folder_size)})")
    
    print("=" * 60)
    print(f"Total size to be freed: {format_size(total_size)}")
    
    if dry_run:
        print("\n🔍 This was a dry run. No folders were actually deleted.")
        print("To actually delete these folders, run with --execute flag")
        return
    
    # Ask for confirmation
    response = input(f"\n⚠️  Are you sure you want to delete {len(folders_to_delete)} folders? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("❌ Deletion cancelled.")
        return
    
    # Actually delete the folders
    deleted_count = 0
    deleted_size = 0
    
    for folder_path in folders_to_delete:
        try:
            folder_size = get_folder_size(folder_path)
            shutil.rmtree(folder_path)
            deleted_count += 1
            deleted_size += folder_size
            print(f"🗑️  Deleted: {folder_path.name}")
        except Exception as e:
            print(f"❌ Error deleting {folder_path.name}: {e}")
    
    print("=" * 60)
    print(f"✅ Successfully deleted {deleted_count} folders")
    print(f"💾 Freed up {format_size(deleted_size)} of disk space")

def get_folder_size(folder_path):
    """Calculate the total size of a folder in bytes."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
    except Exception:
        pass
    return total_size

def format_size(size_bytes):
    """Format bytes into human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def main():
    parser = argparse.ArgumentParser(description="Clean up training run folders without checkpoints")
    parser.add_argument(
        "--training-runs-dir", 
        default="Training Runs",
        help="Path to the Training Runs directory (default: 'Training Runs')"
    )
    parser.add_argument(
        "--execute", 
        action="store_true",
        help="Actually delete the folders (default is dry run)"
    )
    parser.add_argument(
        "--no-confirmation", 
        action="store_true",
        help="Skip confirmation prompt (use with caution!)"
    )
    
    args = parser.parse_args()
    
    print("🧹 Training Runs Cleanup Script")
    print("=" * 60)
    
    # Find folders without checkpoints
    folders_to_delete = find_training_runs_without_checkpoints(args.training_runs_dir)
    
    if not folders_to_delete:
        print("\n✅ No folders to delete! All training runs have checkpoints.")
        return
    
    # Delete folders (or show what would be deleted)
    delete_folders(folders_to_delete, dry_run=not args.execute)

if __name__ == "__main__":
    main() 