from treetoolml.utils.py_util import (
    bb_intersection_over_union,
    combine_IOU,
    data_preprocess,
    get_center_scale,
    makesphere,
    shuffle_data,
)
from matplotlib import cm
from porteratzolibs.visualization_o3d.create_geometries import make_arrow, make_cylinder
if True:
    from porteratzolibs.visualization_o3d.open3dvis import open3dpaint
    from porteratzolibs.visualization_o3d.open3d_pointsetClass import o3d_pointSetClass
else:
    from porteratzolibs.visualization_o3d.open3dvis import open3dpaint
    from porteratzolibs.visualization_o3d.open3d_pointsetClass import o3d_pointSetClass
import numpy as np
import open3d

def _make_arrow(x, vector_scale):
    if isinstance(x, np.ndarray):
        if len(x) == 6:
            return make_arrow(x[:3],x[3:], scale=vector_scale)
        else:
            return make_arrow(x, scale=vector_scale)
    else:
        return make_arrow(x[0],x[1], scale=vector_scale)

def tree_vis_tool(points=None, centers=None, cylinders=None, gt_cylinders=None, vectors=None, vector_scale=1, axis=1, sphere_rad=0.05, for_thesis=False, pointsize=6, point_colors=None):
    spheres = check_if_list(centers, lambda x:makesphere(x, sphere_rad), name='centers')
    lines = check_if_list(vectors, lambda x:_make_arrow(x, vector_scale), name='vectors')
    gt_cyls = check_if_list(gt_cylinders, lambda x:make_cylinder([*x[:3],0,0,1, x[-1]/2], length=30, dense=30), name='gt_trunks')

    objects = []
    if isinstance(points, dict):
        for n,(key,val) in enumerate(points.items()):
            if point_colors is not None:
                if len(point_colors) == len(points):
                    objects.append(o3d_pointSetClass(val, point_colors[n], name=key))
                else:
                    objects.append(o3d_pointSetClass(val, point_colors, name=key))
            else:
                if isinstance(val, list):
                    if '_$' in key:
                        for _n,i in enumerate(val):
                            objects.append(o3d_pointSetClass(val, name=key.replace('_$',str(_n))))
                    else:
                        objects.append(o3d_pointSetClass(val, name=key))
                elif isinstance(val, o3d_pointSetClass):
                    val.name = key
                    objects.append(val)
                else:
                    objects.append(o3d_pointSetClass(val, cm.jet(n / len(points))[:3], name=key))
    else:
        if point_colors is not None:
            points = o3d_pointSetClass(points, point_colors)
        if isinstance(points, list):
            objects += points
        else:
            objects.append(points)
    objects.append(spheres)
    #objects.append(lines)
    objects.extend(lines)
    objects.append(gt_cyls)
    return open3dpaint(objects, pointsize=pointsize, axis=axis, for_thesis=for_thesis)

def check_if_list(vectors, func, name):
    output = []
    if vectors is not None:
        if isinstance(vectors, list):
            output = [func(i ) for i in vectors]
        elif isinstance(vectors, np.ndarray) and vectors.ndim == 2:
            output = [func(i ) for i in vectors]
        else:
            output = [func(vectors)]
    if len(output) > 0:
        if isinstance(output, list):
            if type(output[0] ) == open3d.geometry.TriangleMesh:
                return output
        output = o3d_pointSetClass(output, name=name)
    return output


def vis_trees_centers(cloud, centers):
    spheres = [makesphere(p,0.05) for p in centers]
    open3dpaint([cloud] + spheres, pointsize=2, axis=1)