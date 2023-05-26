import os
import sys

# Get the current directory of the __init__.py file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the relative path to the sys.path list
relative_path = "IndividualTreeExtraction/voxel_region_grow/"
absolute_path = os.path.join(current_dir, relative_path)
sys.path.append(absolute_path)