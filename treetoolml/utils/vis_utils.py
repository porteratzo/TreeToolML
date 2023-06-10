from treetoolml.utils.py_util import (
    bb_intersection_over_union,
    combine_IOU,
    data_preprocess,
    get_center_scale,
    makesphere,
    shuffle_data,
)
from treetoolml.Libraries.open3dvis import open3dpaint

def vis_trees_centers(cloud, centers):
    spheres = [makesphere(p,0.05) for p in centers]
    open3dpaint([cloud] + spheres, pointsize=2, axis=1)