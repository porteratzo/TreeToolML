import numpy as np
from matplotlib import cm
from porteratzolibs.visualization_o3d import open3d_utils
import os
import time
from porteratzolibs.visualization_o3d.open3d_pointsetClass import check_point_format, o3d_pointSetClass, pointSetClass, multi_bounding_box
from porteratzolibs.Misc.transforms import eulerAnglesToRotationMatrix

try:
    import open3d
    V3V = open3d.utility.Vector3dVector
    use_headless = False
except ImportError:
    use_headless = True
import pickle


def sidexsidepaint(*point_clusters, color_map="jet", pointsize=0.1, axis=False, for_thesis=False):
    new_points = []
    trans_list = {}
    for arg in point_clusters:
        trans = 0
        for n, points in enumerate(arg):
            if len(points) == 0:
                continue
            _points = points + np.array([[trans, 0, 0]])
            if trans_list.get(n, None) is None:
                trans_list[n] = np.max(_points, 0)[0]
            trans = trans_list[n]
            new_points.append(_points)
    open3dpaint(new_points, pointsize=pointsize, axis=axis, for_thesis=for_thesis)


def open3dpaint(nppoints, color_map="jet", pointsize=0.1, axis=False, for_thesis=False, voxel_size=None):
    nppoints = check_point_format(nppoints)
    if len(nppoints) == 0:
        return
    try:
        if not use_headless:
            vis = open3d.visualization.Visualizer()
            vis.create_window()
            opt = vis.get_render_option()
            if for_thesis:
                opt.background_color = np.asarray([1.0, 1.0, 1.0])
            else:
                opt.background_color = np.asarray([0.1, 0.1, 0.1])
                
            opt.point_size = pointsize
            current_pointset = o3d_pointSetClass()
        else:
            current_pointset = pointSetClass()

        if len(nppoints) > 1:
            if len(nppoints) < 3:
                color_list = [cm.jet(i)[:3] for i in np.linspace(0, 1, len(nppoints))]
            else:
                color_list = [cm.nipy_spectral(i)[:3] for i in np.linspace(0, 1, len(nppoints))]
            choice_list = np.random.choice(len(nppoints), len(nppoints), replace=False)
            for n, i in enumerate(nppoints):
                if type(i) != o3d_pointSetClass:
                    workpoints = i
                    color = color_list[choice_list[n]]
                    if voxel_size is not None:
                        idx = open3d_utils.downsample(workpoints, leaf_size=voxel_size, return_idx=True)
                        workpoints = workpoints[idx]
                    current_pointset = o3d_pointSetClass()
                    current_pointset.update(workpoints, color)
                    current_pointset.draw(vis)
                elif type(i) == o3d_pointSetClass:
                    i.in_vis = False
                    i.update()
                    i.draw(vis)

        else:
            if type(nppoints[0]) != o3d_pointSetClass:
                workpoints = nppoints[0]
                if voxel_size is not None:
                    idx = open3d_utils.downsample(workpoints, leaf_size=voxel_size, return_idx=True)
                    workpoints = workpoints[idx]
                current_pointset.update(workpoints)
                current_pointset.draw(vis)
            elif type(nppoints[0]) == o3d_pointSetClass:
                nppoints[0].in_vis = False
                nppoints[0].update()
                nppoints[0].draw(vis)
        if axis:
            vis.add_geometry(open3d.geometry.TriangleMesh.create_coordinate_frame(axis))
        vis.run()
        vis.clear_geometries()
        vis.destroy_window()

    except Exception as e:
        print(type(e))
        print(e.args)
        print(e)
        vis.destroy_window()


class open3dpaint_non_block:
    def __init__(
        self,
        pointsize=0.1,
        file_name=None,
        axis=False,
        disable=False,
        headless=False,
    ) -> None:
        self.disable = disable
        if use_headless:
            self.headless = True
        else:
            self.headless = headless

        if not self.disable:
            if not self.headless:
                self.vis = open3d.visualization.Visualizer()
                self.vis.create_window(width=640, height=480)
                self.opt = self.vis.get_render_option()
                self.opt.background_color = np.asarray([0.1, 0.1, 0.1])
                self.opt.point_size = pointsize

                self.view = self.vis.get_view_control()
                self.view.set_constant_z_far(10000)
                if axis:
                    self.vis.add_geometry(
                        open3d.geometry.TriangleMesh.create_coordinate_frame(100)
                    )
            self.T = np.eye(4)
            self.all_bb = []
            self.frame = 0
            self.file_name = file_name
            self.pointsets = {}
            self.state_save = []

    def update_points(self, nppoints, pointset=0, color_map="jet", persistant=False):
        if not self.disable:

            if len(nppoints) < 1:
                return
            nppoints = check_point_format(nppoints)

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
                    if not self.headless:
                        self.pointsets[pointset] = o3d_pointSetClass(
                            persistant=persistant
                        )
                    else:
                        self.pointsets[pointset] = pointSetClass(persistant=persistant)
                self.pointsets[pointset].update(
                    np.concatenate(group_points), np.concatenate(group_colors)
                )

            elif len(nppoints) == 0:
                pass
            else:
                workpoints = nppoints[0]
                if not self.pointsets.get(pointset):
                    if not self.headless:
                        self.pointsets[pointset] = o3d_pointSetClass(
                            persistant=persistant
                        )
                    else:
                        self.pointsets[pointset] = pointSetClass(persistant=persistant)

                self.pointsets[pointset].update(workpoints, color_map)

    def rotate_pointset(self, pointset=0, rotation=(0.0, 0.0, 0.0), center="self"):
        R = self.pointsets[pointset].get_rotation_matrix_from_xyz(np.deg2rad(rotation))
        if type(center) is str:
            self.pointsets[pointset].rotate(R)
        else:
            self.pointsets[pointset].rotate(R, center=center)

    def update_points_of_interest_multiline(
        self, nppoints, color=np.array([1.0, 0.0, 0.0])
    ):
        if not self.disable:
            nppoints = check_point_format(nppoints)
            if not self.headless:
                if len(nppoints) > 1:
                    for n, i in enumerate(nppoints):
                        workpoints = i
                        if len(self.all_bb) < n + 1:
                            if len(workpoints) > 3:
                                bb = multi_bounding_box(workpoints, color)
                                bb.draw(self.vis)
                                # self.vis.add_geometry(bb)
                                self.all_bb.append(bb)
                        else:
                            if len(workpoints) > 3:
                                self.all_bb[n].set_points(workpoints)
                                self.all_bb[n].update(self.vis)
                                # self.vis.update_geometry(self.all_bb[n])
                elif len(nppoints) == 0:
                    pass
                else:
                    workpoints = nppoints[0]

                    if len(self.all_bb) == 0:
                        if len(workpoints) > 3:
                            bb = multi_bounding_box(workpoints, color)
                            bb.draw(self.vis)
                            # self.vis.add_geometry(bb)
                            self.all_bb.append(bb)
                    else:
                        if len(workpoints) > 3:
                            self.all_bb[0].set_points(workpoints)
                            self.all_bb[0].update(self.vis)

                while len(nppoints) < len(self.all_bb):
                    # self.vis.remove_geometry(self.all_bb[-1])
                    self.all_bb[-1].remove(self.vis)
                    self.all_bb.pop(-1)

    def rotate(self, x, y, z):
        if not self.disable:
            T = np.eye(4)
            T[:3, :3] = eulerAnglesToRotationMatrix(np.deg2rad([x, y, z]))
            self.T = T @ self.T

    def translate(self, x, y, z):
        if not self.disable:
            T = np.eye(4)
            T[:, 3] = np.array([x, y, z, 1])
            self.T = T @ self.T

    def set_perspective(self):
        if not self.disable:
            if not self.headless:
                self.cam = self.view.convert_to_pinhole_camera_parameters()
                self.cam.extrinsic = self.T
                self.view.convert_from_pinhole_camera_parameters(self.cam)
                time.sleep(0.02)

    def draw(self, saveim=False, save=None, inc=False):
        if not self.disable:
            for keys in self.pointsets.keys():
                if not self.headless:
                    self.pointsets[keys].draw(self.vis)

            if not self.headless:
                self.vis.poll_events()
                self.vis.update_renderer()

                if saveim:
                    if self.file_name is not None:
                        if save is None:
                            save_path = (
                                self.file_name + str(self.frame).zfill(5) + ".jpg"
                            )
                        else:
                            save_path = save
                            if inc:
                                n = 0
                                while os.path.isfile(save_path):
                                    save_path = (
                                        save.split(".jpg")[0] + "_" + str(n) + ".jpg"
                                    )
                                    n += 1
                        self.vis.capture_screen_image(save_path)
                        self.frame += 1
                return not self.vis.poll_events()
            else:
                if saveim:
                    self.state_save.append(
                        {
                            key: self.pointsets[key].state_dict()
                            for key in self.pointsets.keys()
                        }
                    )
                return False

    def save_state(self):
        with open(self.file_name + ".pkl", "wb") as f:
            pickle.dump(self.state_save, f, pickle.HIGHEST_PROTOCOL)

    def load_state(self, load_path):
        with open(load_path, "rb") as f:
            self.state_save = pickle.load(f)

    def run_from_state(self, save):
        state = self.state_save
        exit_vis = False
        for step in state:
            for pointset_key in step.keys():
                self.update_points(
                    step[pointset_key]["points"],
                    pointset=pointset_key,
                    color_map=step[pointset_key]["colors"],
                )
            self.set_perspective()
            if self.draw(save):
                exit_vis = True
                break
        return exit_vis

    def stop(self):
        if not self.disable:
            if not self.headless:
                self.vis.destroy_window()
