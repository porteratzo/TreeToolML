pip install cython
git clone https://github.com/HiphonL/IndividualTreeExtraction.git
cd IndividualTreeExtraction
cd voxel_region_grow
python VoxelRegionGrow_Setup.py build_ext --inplace
cd ..
cp -r voxel_traversal ../treetoolml/IndividualTreeExtraction_utils/voxel_traversal
cp -r voxel_region_grow ../treetoolml/IndividualTreeExtraction_utils/voxel_region_grow
cp -r accessible_region ../treetoolml/IndividualTreeExtraction_utils/accessible_region