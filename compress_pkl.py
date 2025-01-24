import gzip
import shutil
import os

# Get a list of all .pkl files in the current directory
pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]

for file in pkl_files:
    with open(file, 'rb') as f_in:
        with gzip.open(f"{file}.gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Compressed {file} to {file}.gz")

print("All .pkl files compressed successfully!")
