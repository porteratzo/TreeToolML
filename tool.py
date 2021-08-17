import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import open3d
from Libraries.Corners import eulerAnglesToRotationMatrix
from collections import defaultdict


def convertcloud(points):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    # open3d.write_point_cloud(Path+'sync.ply', pcd)
    # pcd_load = open3d.read_point_cloud(Path+'sync.ply')
    return pcd


def open3dpaint(nppoints, color_map="jet", pointsize=0.1):
    assert (
        (type(nppoints) == np.ndarray)
        or (type(nppoints) is list)
        or (type(nppoints) is tuple)
    ), "Not valid point_cloud"

    if (type(nppoints) is not list) & (type(nppoints) is not tuple):
        nppoints = [nppoints]

    try:
        vis = open3d.visualization.Visualizer()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        opt.point_size = pointsize

        view = vis.get_view_control()
        T = np.eye(4)
        T[:, 3] = [-100, 0, 600, 1]
        T[:3, :3] = eulerAnglesToRotationMatrix(np.deg2rad([-180, 0, 0]))

        if len(nppoints) > 1:
            for n, i in enumerate(nppoints):
                workpoints = i

                points = convertcloud(workpoints)
                colNORM = n / len(nppoints) / 2 + n % 2 * 0.5
                if type(color_map) == np.ndarray:
                    points.colors = open3d.utility.Vector3dVector(color_map)
                elif color_map == "jet":
                    color = cm.jet(colNORM)[:3]
                    points.colors = open3d.utility.Vector3dVector(
                        np.ones_like(workpoints) * color
                    )
                else:
                    color = cm.Set1(colNORM)[:3]
                    points.colors = open3d.utility.Vector3dVector(
                        np.ones_like(workpoints) * color
                    )
                # points.colors = open3d.utility.Vector3dVector(color)
                vis.add_geometry(points)
        else:
            workpoints = nppoints[0]

            points = convertcloud(workpoints)
            if type(color_map) == np.ndarray:
                points.colors = open3d.utility.Vector3dVector(color_map)
            vis.add_geometry(points)
        # view.rotate(0,45)
        # view.translate(1000,1000)
        cam = view.convert_to_pinhole_camera_parameters()
        cam.extrinsic = T
        view.convert_from_pinhole_camera_parameters(cam)
        vis.run()
        vis.destroy_window()

    except Exception as e:
        print(type(e))
        print(e.args)
        print(e)
        vis.destroy_window()


def open3dpaint(nppoints, color_map="jet", pointsize=0.1):
    assert (
        (type(nppoints) == np.ndarray)
        or (type(nppoints) is list)
        or (type(nppoints) is tuple)
    ), "Not valid point_cloud"

    if (type(nppoints) is not list) & (type(nppoints) is not tuple):
        nppoints = [nppoints]

    try:
        vis = open3d.visualization.Visualizer()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        opt.point_size = pointsize

        view = vis.get_view_control()
        T = np.eye(4)
        T[:, 3] = [-100, 0, 600, 1]
        T[:3, :3] = eulerAnglesToRotationMatrix(np.deg2rad([-180, 0, 0]))

        if len(nppoints) > 1:
            for n, i in enumerate(nppoints):
                workpoints = i

                points = convertcloud(workpoints)
                colNORM = n / len(nppoints) / 2 + n % 2 * 0.5
                if type(color_map) == np.ndarray:
                    points.colors = open3d.utility.Vector3dVector(color_map)
                elif color_map == "jet":
                    color = cm.jet(colNORM)[:3]
                    points.colors = open3d.utility.Vector3dVector(
                        np.ones_like(workpoints) * color
                    )
                else:
                    color = cm.Set1(colNORM)[:3]
                    points.colors = open3d.utility.Vector3dVector(
                        np.ones_like(workpoints) * color
                    )
                # points.colors = open3d.utility.Vector3dVector(color)
                vis.add_geometry(points)
        else:
            workpoints = nppoints[0]

            points = convertcloud(workpoints)
            if type(color_map) == np.ndarray:
                points.colors = open3d.utility.Vector3dVector(color_map)
            vis.add_geometry(points)
        # view.rotate(0,45)
        # view.translate(1000,1000)
        cam = view.convert_to_pinhole_camera_parameters()
        cam.extrinsic = T
        view.convert_from_pinhole_camera_parameters(cam)
        vis.run()
        vis.destroy_window()

    except Exception as e:
        print(type(e))
        print(e.args)
        print(e)
        vis.destroy_window()


class open3dpaint_non_block:
    def __init__(
        self, color_map="jet", pointsize=0.1, file_name=None, axis=False
    ) -> None:
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window()
        self.opt = self.vis.get_render_option()
        self.opt.background_color = np.asarray([0.1, 0.1, 0.1])
        self.opt.point_size = pointsize
        self.opt.line_width = 100

        self.view = self.vis.get_view_control()
        self.view.set_constant_z_far(3000)
        self.T = np.eye(4)
        # self.T[:,3] = [-100, 0, 600, 1]
        # self.T[:3,:3] = eulerAnglesToRotationMatrix(np.deg2rad([165,0,0]))

        self.T[:, 3] = [0, 0, 0, 1]
        self.T[:3, :3] = eulerAnglesToRotationMatrix(np.deg2rad([0, 0, 0]))

        self.all_bb = []
        self.frame = 0
        self.file_name = file_name

        self.pointsets = {}

        self.vis.add_geometry(
            open3d.geometry.TriangleMesh.create_coordinate_frame(100)
        )

    def update_points(self, nppoints, pointset=0, color_map="jet"):
        create = False

        assert (
            (type(nppoints) == np.ndarray)
            or (type(nppoints) is list)
            or (type(nppoints) is tuple)
        ), "Not valid point_cloud"

        if (type(nppoints) is not list) & (type(nppoints) is not tuple):
            nppoints = [nppoints]

        if len(nppoints) > 1:
            group_points = []
            group_colors = []
            for n, i in enumerate(nppoints):
                workpoints = i
                group_points.append(workpoints)
                colNORM = n / len(nppoints) / 2 + n % 2 * 0.5

                if color_map == "jet":
                    color = cm.jet(colNORM)[:3]
                    group_colors.append(np.ones_like(workpoints) * color)
                else:
                    group_colors.append(np.ones_like(workpoints) * color_map)

            if not self.pointsets.get(pointset):
                self.pointsets[pointset] = open3d.geometry.PointCloud()
                create = True
            self.pointsets[pointset].points = open3d.utility.Vector3dVector(
                np.concatenate(group_points)
            )
            self.pointsets[pointset].colors = open3d.utility.Vector3dVector(
                np.concatenate(group_colors)
            )
            if create:
                self.vis.add_geometry(self.pointsets[pointset])
            else:
                self.vis.update_geometry(self.pointsets[pointset])
        elif len(nppoints) == 0:
            pass
        else:
            workpoints = nppoints[0]
            if not self.pointsets.get(pointset):
                self.pointsets[pointset] = open3d.geometry.PointCloud()
                create = True
            self.pointsets[pointset].points = open3d.utility.Vector3dVector(
                workpoints
            )
            if type(color_map) == np.ndarray:
                if color_map.shape[0] == workpoints.shape[0]:
                    self.pointsets[
                        pointset
                    ].colors = open3d.utility.Vector3dVector(color_map)
                else:
                    self.pointsets[
                        pointset
                    ].colors = open3d.utility.Vector3dVector(
                        np.ones_like(workpoints) * np.array(color_map)
                    )
            if create:
                self.vis.add_geometry(self.pointsets[pointset])
            else:
                self.vis.update_geometry(self.pointsets[pointset])

    def update_points_of_interest(
        self, nppoints, color=np.array([1.0, 0.0, 0.0])
    ):
        assert (
            (type(nppoints) == np.ndarray)
            or (type(nppoints) is list)
            or (type(nppoints) is tuple)
        ), "Not valid point_cloud"

        if (type(nppoints) is not list) & (type(nppoints) is not tuple):
            nppoints = [nppoints]

        if len(nppoints) > 1:
            for n, i in enumerate(nppoints):
                workpoints = i
                if len(self.all_bb) < n + 1:
                    if len(workpoints) > 3:
                        bb = open3d.geometry.OrientedBoundingBox.create_from_points(
                            open3d.utility.Vector3dVector(workpoints)
                        )
                        bb.color = color
                        self.vis.add_geometry(bb)
                        self.all_bb.append(bb)
                else:
                    if len(workpoints) > 3:
                        bb = open3d.geometry.OrientedBoundingBox.create_from_points(
                            open3d.utility.Vector3dVector(workpoints)
                        )
                        self.all_bb[n].center = bb.center
                        self.all_bb[n].extent = bb.extent
                        self.all_bb[n].R = bb.R
                        self.vis.update_geometry(self.all_bb[n])
        elif len(nppoints) == 0:
            pass
        else:
            workpoints = nppoints[0]

            if len(self.all_bb) == 0:
                if len(workpoints) > 3:
                    bb = (
                        open3d.geometry.OrientedBoundingBox.create_from_points(
                            open3d.utility.Vector3dVector(workpoints)
                        )
                    )
                    bb.color = color
                    self.vis.add_geometry(bb)
                    self.all_bb.append(bb)
            else:
                if len(workpoints) > 3:
                    bb = (
                        open3d.geometry.OrientedBoundingBox.create_from_points(
                            open3d.utility.Vector3dVector(workpoints)
                        )
                    )
                    self.all_bb[0].center = bb.center
                    self.all_bb[0].extent = bb.extent
                    self.all_bb[0].R = bb.R
                    self.vis.update_geometry(self.all_bb[0])

        while len(nppoints) < len(self.all_bb):
            self.vis.remove_geometry(self.all_bb[-1])
            self.all_bb.pop(-1)

    def rotate(self, x, y, z):
        T = np.eye(4)
        T[:3, :3] = eulerAnglesToRotationMatrix(np.deg2rad([x, y, z]))
        self.T = T @ self.T

    def translate(self, x, y, z):
        T = np.eye(4)
        T[:, 3] = np.array([x, y, z, 1])
        self.T = T @ self.T

    def set_perspective(self):
        self.cam = self.view.convert_to_pinhole_camera_parameters()
        self.cam.extrinsic = self.T
        self.view.convert_from_pinhole_camera_parameters(self.cam)

    def draw(self, saveim=False):
        self.vis.poll_events()
        self.vis.update_renderer()

        if saveim:
            if self.file_name is not None:
                self.vis.capture_screen_image(
                    self.file_name + str(self.frame).zfill(5) + ".jpg"
                )
                self.frame += 1

    def stop(self):
        self.vis.destroy_window()


if False:
    for i in tqdm(glob.glob("data/Paris/training_10_classes/*.ply")):
        with open(i, "rb") as f:
            plydata = PlyData.read(f)

        p =[i if i.name not in ['label','class'] else PlyProperty(i.name, 'float') for i in plydata['vertex'].properties ]
        plydata['vertex'].properties = p

        with open("data/Paris/" + i.split('/')[-1], 'wb') as f:
            plydata.write(f)