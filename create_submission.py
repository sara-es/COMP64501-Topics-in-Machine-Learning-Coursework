"""
create_submission.py - Package your coursework for submission

This script creates a zip file containing only the required files for submission.
The zip will be named with your student ID.

Usage:
    uv run create_submission.py
"""

import os
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

try:
    from submission.STUDENT_ID import STUDENT_ID
except ImportError:
    print("ERROR: Could not import STUDENT_ID from submission/STUDENT_ID.py")
    print("Please make sure this file exists and you have set STUDENT_ID.")
    sys.exit(1)

STUDENT_ID = str(STUDENT_ID).strip() # in case it was entered as a number

# Verify student ID is set
if not (STUDENT_ID.isdigit() and len(STUDENT_ID) == 8):
    print("ERROR: STUDENT_ID not set, or not an 8-digit number!")
    print("Please edit submission/STUDENT_ID.py to set your student ID.")
    sys.exit(1)

# Files to include in submission
REQUIRED_FILES = [
    "submission/fashion_model.py",
    "submission/fashion_training.py",
    "submission/STUDENT_ID.py",
    "submission/__init__.py",
    "submission/model_weights.pth",
    "model_calls.py", # not required in final submission, but necessary if you wish to test locally
    "utils.py", # not required in final submission, but necessary if you wish to test locally
    "Dockerfile",
    "pyproject.toml",
    "uv.lock",
]

OPTIONAL_FILES = [
    ".dockerignore",
    "submission/engine.py",
    # add any other files you wish to include
]

def create_submission():
    """Create submission zip file"""
    
    print("Creating submission zip file.")
    print(f"Student ID: {STUDENT_ID}")
    print()
    
    # Create output filename and temp directory with student ID
    output_file = f"{STUDENT_ID}.zip"
    temp_dir = STUDENT_ID
    
    # Clean up if temp dir exists
    temp_path = Path(temp_dir)
    if temp_path.exists():
        shutil.rmtree(temp_path)
    temp_path.mkdir(parents=True)
    
    # Check and copy required files
    print("Checking required files...")
    missing_files = []
    copied_files = []
    
    for file_path in REQUIRED_FILES:
        src = Path(file_path)
        if not src.exists():
            missing_files.append(file_path)
            print(f"Missing: {file_path}")
        else:
            dst = temp_path / file_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied_files.append(file_path)
            print(f"Added: {file_path}")
    
    if missing_files:
        print()
        print("Please ensure all required files exist before creating submission.")
        shutil.rmtree(temp_path)
        sys.exit(1)
    
    # Copy optional files
    for file_path in OPTIONAL_FILES:
        src = Path(file_path)
        if src.exists():
            dst = temp_path / file_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied_files.append(file_path)
            print(f"Added (optional): {file_path}")

    # Copy any other python files in submission/ directory
    # If you wish to include additional files that do not end in .py, add them to OPTIONAL_FILES above
    submission_dir = Path("submission")
    for item in submission_dir.rglob('*'):
        if item.is_file() and item.as_posix() not in REQUIRED_FILES + OPTIONAL_FILES and item.suffix == '.py':
            relative_path = item.relative_to('.')
            dst = temp_path / relative_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)
            copied_files.append(str(relative_path))
            print(f"Added (extra): {relative_path}")
    
    # Create zip file
    print() 
    print(f"Creating {output_file}...")
    with ZipFile(output_file, 'w', ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_path):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(temp_path.parent)
                zipf.write(file_path, arcname)
    
    # Clean up temp directory
    shutil.rmtree(temp_path)
    
    # Get final size
    zip_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    
    print()
    print("Submission package created successfully.")
    print(f"File: {output_file}")
    print(f"Size: {zip_size_mb:.2f} MB")
    if zip_size_mb > 2: # most submissions should be under 1MB
        print("WARNING: this zip is larger than expected. Check that it only contains the files required for submission (no Jupyter notebooks or other unnecessary files).")
    print(f"Files included: {len(copied_files)}")
    print()
    print("Next steps:")
    print("1. Test your submission:")
    print(f"   unzip {output_file} -d .")
    print(f"   cd {STUDENT_ID}")
    print("   docker build -t test .")
    print("   docker run test")
    print("2. If tests pass, submit the .zip file to the course portal on Canvas.")
    print()

if __name__ == "__main__":
    create_submission()