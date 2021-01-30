import ipdb
st = ipdb.set_trace
# st()
import habitat_sim
import habitat
print("HJ")
from habitat.config.default import get_config
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb
from habitat_sim.utils.common import quat_from_two_vectors, quat_from_angle_axis, angle_between_quats
import cv2
import math
import random
#%matplotlib inline
import matplotlib.pyplot as plt
plt.figure(figsize=(12 ,8))
import time
import numpy as np
import quaternion
import ipdb

import os 
import sys
import glob, os.path
import pickle
import json
from habitat_sim.utils import common as utils
# from habitat_sim.utils import viz_utils as vut
from scipy.spatial.transform import Rotation
# from scipy.spatial.transform import Rotation as Rs
import subprocess
from associate import associate
from mpl_toolkits.mplot3d import Axes3D

EPSILON = 1e-8


class AutomatedMultiview():
    def __init__(self):   
        self.visualize = False
        self.verbose = False
        self.save_imgs = True
        self.do_orbslam = False
        self.do_depth_noise = True
        self.min_orbslam_views = 150
        self.orbslam_path = '/home/nel/ORB_SLAM2/Examples/RGB-D/replica3'
        self.ORBSLAM_Cam_Traj = '/home/nel/gsarch/habitat/habitat-lab/HabitatScripts/HabitatScripts_Third/CameraTrajectory.txt'
        self.ORBSLAM_Cam_Keypoints = '/home/nel/gsarch/habitat/habitat-lab/HabitatScripts/HabitatScripts_Third/KeyFrameTrajectory.txt'
        # st()
        self.mapnames = os.listdir('/home/nel/gsarch/Replica-Dataset/out/')
        # self.mapnames = os.listdir('/hdd/replica/Replica-Dataset/out/')
        self.num_episodes = len(self.mapnames)
        
        # self.num_episodes = 1 # temporary
        self.ignore_classes = ['book','base-cabinet','beam','blanket','blinds','cloth','clothing','coaster','comforter','curtain','ceiling','countertop','floor','handrail','mat','paper-towel','picture','pillar','pipe','scarf','shower-stall','switch','tissue-paper','towel','vent','wall','wall-plug','window','rug','logo','set-of-clothing']
        
        self.include_classes = ['chair', 'bed', 'toilet', 'sofa', 'indoor-plant', 'refrigerator', 'tv-screen', 'table']
        # self.include_classes = ['chair', 'bed', 'toilet', 'sofa', 'indoor-plant', 'refrigerator', 'tv-screen', 'table']
        # self.small_classes = ['indoor-plant', 'toilet']
        # self.include_classes = ['beanbag', 'cushion', 'nightstand', 'shelf']
        self.small_classes = []
        self.rot_interval = 5.0
        self.radius_max = 1.75
        self.radius_min = 1.25
        self.num_flat_views = 3
        self.num_any_views = 7
        self.num_views = 25

        self.origin_quaternion = np.quaternion(1, 0, 0, 0)
        self.origin_rot_vector = quaternion.as_rotation_vector(self.origin_quaternion) 

        # self.env = habitat.Env(config=config, dataset=None)
        # st()
        # self.test_navigable_points()

        self.W = 256
        self.H = 256

        self.fov = 90
        hfov = float(self.fov) * np.pi / 180.
        self.pix_T_camX = np.array([
            [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        self.pix_T_camX[0,2] = self.W/2.
        self.pix_T_camX[1,2] = self.H/2.


        self.run_episodes()

    def run_episodes(self):
        self.ep_idx = 0
        if os.path.exists('/home/nel/gsarch/habitat/habitat-lab/HabitatScripts/HabitatScripts_Third/maps'):
            dir_name = "/home/nel/gsarch/habitat/habitat-lab/HabitatScripts/HabitatScripts_Third/maps/"
            test = os.listdir(dir_name)
            for item in test:
                if item.endswith(".png"):
                    os.remove(os.path.join(dir_name, item))
        for episode in range(self.num_episodes):
            print("STARTING EPISODE ", episode)

            print("DELETING PREVIOUS FILES")
            # remove png files from previous episode
            filelist = glob.glob(os.path.join(self.orbslam_path,"rgb", "*.png"))
            for f in filelist:
                os.remove(f)
            filelist = glob.glob(os.path.join(self.orbslam_path,"depth", "*.tiff"))
            for f in filelist:
                os.remove(f)
            
            # reset trajectory files
            deleted = []
            if os.path.exists(self.ORBSLAM_Cam_Traj):
                deleted.append(['cam_traj'])
                os.remove(self.ORBSLAM_Cam_Traj)
            if os.path.exists(self.ORBSLAM_Cam_Keypoints):
                deleted.append(['keypoint_traj'])
                os.remove(self.ORBSLAM_Cam_Keypoints)
            associations_file = "/home/nel/ORB_SLAM2/Examples/RGB-D/associations/replica3.txt"
            if os.path.exists(associations_file):
                deleted.append(['associations2'])
                os.remove(associations_file)
            if os.path.exists(self.orbslam_path + "/" + 'rgb3.txt'):
                deleted.append(['rgb3.txt'])
                os.remove(self.orbslam_path + "/" + 'rgb3.txt')
            if os.path.exists(self.orbslam_path + "/" + 'depth3.txt'):
                deleted.append(['depth3.txt'])
                os.remove(self.orbslam_path + "/" + 'depth3.txt')

            print('Deleted: ', deleted)

            print("DONE. ", "DELETED ", len(filelist), " files.")

            # mapname = np.random.choice(self.mapnames)
            mapname = self.mapnames[episode] # KEEP THIS
            # mapname = 'apartment_0'
            #mapname = 'frl_apartment_4'
            self.test_scene = "/home/nel/gsarch/Replica-Dataset/out/{}/habitat/mesh_semantic.ply".format(mapname)
            self.object_json = "/home/nel/gsarch/Replica-Dataset/out/{}/habitat/info_semantic.json".format(mapname)
            # self.test_scene = "/hdd/replica/Replica-Dataset/out/{}/habitat/mesh_semantic.ply".format(mapname)
            # self.object_json = "/hdd/replica/Replica-Dataset/out/{}/habitat/info_semantic.json".format(mapname)
            self.sim_settings = {
                "width": 256,  # Spatial resolution of the observations
                "height": 256,
                "scene": self.test_scene,  # Scene path
                "default_agent": 0,
                "sensor_height": 1.5,  # Height of sensors in meters
                "color_sensor": True,  # RGB sensor
                "semantic_sensor": True,  # Semantic sensor
                "depth_sensor": True,  # Depth sensor
                "seed": 1,
            }

            # self.basepath = f"/home/nel/gsarch/replica_novel_categories/{mapname}_{episode}"
            # self.basepath = '/home/nel/gsarch/replica_orbslam_noisy_pose'
            # self.basepath = '/home/nel/gsarch/replica_orbslam_noisy_depth_pose'
            self.basepath = '/home/nel/gsarch/replica_orbslam_noisy_depth'
            # if os.path.exists(self.basepath):
            #     os.remove(self.basepath + "/*")
            self.basepath = self.basepath + f"/{mapname}_{episode}"
            print("BASEPATH: ", self.basepath)

            # self.basepath = f"/hdd/ayushj/habitat_data/{mapname}_{episode}"
            if not os.path.exists(self.basepath):
                os.mkdir(self.basepath)

            self.cfg, self.sim_cfg = self.make_cfg(self.sim_settings)
            self.sim = habitat_sim.Simulator(self.cfg)
            random.seed(self.sim_settings["seed"])
            self.sim.seed(self.sim_settings["seed"])
            self.set_agent(self.sim_settings)
            self.nav_pts = self.get_navigable_points()

            config = get_config()
            config.defrost()
            config.TASK.SENSORS = []
            config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
            config.freeze()

            self.run()
            
            self.sim.close()
            time.sleep(3)

            self.ep_idx += 1


    def set_agent(self, sim_settings):
        agent = self.sim.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([1.5, 1.072447, 0.0])
        #agent_state.position = np.array([1.0, 3.0, 1.0])
        agent.set_state(agent_state)
        agent_state = agent.get_state()
        if self.verbose:
            print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
        self.agent = agent

    def make_cfg(self, settings):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene.id = settings["scene"]

        # Note: all sensors must have the same resolution
        sensors = {
            "color_sensor": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
            "depth_sensor": {
                "sensor_type": habitat_sim.SensorType.DEPTH,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
            "semantic_sensor": {
                "sensor_type": habitat_sim.SensorType.SEMANTIC,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
        }

        sensor_specs = []
        for sensor_uuid, sensor_params in sensors.items():
            if settings[sensor_uuid]:
                sensor_spec = habitat_sim.SensorSpec()
                sensor_spec.uuid = sensor_uuid
                sensor_spec.sensor_type = sensor_params["sensor_type"]
                sensor_spec.resolution = sensor_params["resolution"]
                sensor_spec.position = sensor_params["position"]

                if sensor_uuid == 'depth_sensor' and self.do_depth_noise:
                    # add depth noise
                    sensor_spec.noise_model = "RedwoodDepthNoiseModel"
                    sensor_spec.noise_model_kwargs = dict(noise_multiplier=1)

                sensor_specs.append(sensor_spec)

        # Here you can specify the amount of displacement in a forward action and the turn angle
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "do_nothing": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.)
            ),
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=self.rot_interval)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=self.rot_interval)
            ),
            "look_up":habitat_sim.ActionSpec(
                "look_up", habitat_sim.ActuationSpec(amount=self.rot_interval)
            ),
            "look_down":habitat_sim.ActionSpec(
                "look_down", habitat_sim.ActuationSpec(amount=self.rot_interval)
            ),
            "look_down_init":habitat_sim.ActionSpec(
                "look_down", habitat_sim.ActuationSpec(amount=100.0)
            )
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg]), sim_cfg 
    

    def display_sample(self, rgb_obs, semantic_obs, depth_obs, mainobj=None, visualize=False):
        rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        # st()
        
        
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")

        display_img = cv2.cvtColor(np.asarray(rgb_img), cv2.COLOR_RGB2BGR)
        #print(display_img.shape)

        # mask_image = False
        # if mask_image and mainobj is not None:
        #     main_id = int(mainobj.id[1:])
        #     print("MAINID ", main_id)
        #     # semantic = observations["semantic_sensor"]
        #     display_img[semantic_obs == main_id] = [1, 0, 1]
            # st()

        #display_img = cv2
        plt.imshow(display_img)
        plt.show()
        # cv2.imshow('img',display_img)
        if visualize:
            arr = [rgb_img, semantic_img, depth_img]
            titles = ['rgb', 'semantic', 'depth']
            plt.figure(figsize=(12 ,8))
            for i, data in enumerate(arr):
                ax = plt.subplot(1, 3, i+1)
                ax.axis('off')
                ax.set_title(titles[i])
                plt.imshow(data)
                # plt.pause()
            plt.show()
            # plt.pause(0.5)
            # cv2.imshow()
            # plt.close()

    def save_datapoint(self, agent, observations, data_path, viewnum, mainobj_id, flat_view):
        if self.verbose:
            print("Print Sensor States.", self.agent.state.sensor_states)
        rgb = observations["color_sensor"]
        semantic = observations["semantic_sensor"]
        
        # Extract objects from instance segmentation
        object_list = []
        obj_ids = np.unique(semantic)
        if self.verbose:
            print("Unique semantic ids: ", obj_ids)

        # st()
        for obj_id in obj_ids:
            if obj_id < 1 or obj_id > len(self.sim.semantic_scene.objects):
                continue
            if self.sim.semantic_scene.objects[obj_id] == None:
                continue
            if self.sim.semantic_scene.objects[obj_id].category == None:
                continue
            try:
                class_name = self.sim.semantic_scene.objects[obj_id].category.name()
                
                if self.verbose:
                    print("Class name is : ", class_name)
            except Exception as e:
                print(e)
                print("done")
            #if class_name not in self.ignore_classes:
            if class_name in self.include_classes:
                obj_instance = self.sim.semantic_scene.objects[obj_id]
                # st()
                mask = np.zeros_like(semantic)
                mask[semantic == obj_id] = 1
                y, x = np.where(mask)
                pred_box = np.array([min(y), min(x), max(y), max(x)]) # ymin, xmin, ymax, xmax
                # print("Object name {}, Object category id {}, Object instance id {}".format(class_name, obj_instance['id'], obj_instance['class_id']))
                # st()
                obj_data = {'instance_id': obj_id, 'category_id': obj_instance.category.index(), 'category_name': obj_instance.category.name(),
                                 'bbox_center': obj_instance.obb.center, 'bbox_size': obj_instance.obb.sizes,
                                  'world_T_local': obj_instance.obb.local_to_world, 'mask_2d': mask, 'box_2d': pred_box}
                # object_list.append(obj_instance)
                object_list.append(obj_data)

        # st()
        depth = observations["depth_sensor"]
        # self.display_sample(rgb, semantic, depth, visualize=True)
        agent_pos = observations["positions"] #agent.state.position
        agent_rot = observations["rotations"]
        # Assuming all sensors have same extrinsics
        color_sensor_pos = observations["positions"] #agent.state.sensor_states['color_sensor'].position
        color_sensor_rot = observations["rotations"] #agent.state.sensor_states['color_sensor'].rotation
        print("POS ", agent_pos)
        print("ROT ", color_sensor_rot)

        if self.do_orbslam:
            agent_pos_orbslam = observations["positions_orb"]
            agent_rot_orbslam = observations["rotations_orb"]

        save_data = {'flat_view': flat_view, 'mainobj_id': mainobj_id,
            'objects_info': object_list,'rgb_camX':rgb, 'depth_camX': depth,
            'semantic_camX': semantic, 'agent_pos':agent_pos, 'agent_rot': agent_rot,
            'sensor_pos': color_sensor_pos, 'sensor_rot': color_sensor_rot}

        if self.do_orbslam:
            save_data['orbslam_rot'] = agent_rot_orbslam
            save_data['orbslam_pos'] = agent_pos_orbslam
        
        with open(os.path.join(data_path, str(viewnum) + ".p"), 'wb') as f:
            pickle.dump(save_data, f)
        f.close()

    def is_valid_datapoint(self, observations, mainobj):
        main_id = int(mainobj.id[1:])
        semantic = observations["semantic_sensor"]
        # st()
        num_occ_pixels = np.where(semantic == main_id)[0].shape[0]
        #print(semantic.shape)
        #print("Number of pixels: ", num_occ_pixels)
        small_objects = []
        if mainobj.category.name() in self.small_classes:
            if  num_occ_pixels < 0.9*256*256:
                return True
        else:
            if num_occ_pixels < 0.9*256*256: 
                return True
        return False
    
    def rotation_matrix_from_vectors(vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    def quaternion_from_two_vectors(self, v0: np.array, v1: np.array) -> np.quaternion:
        r"""Computes the quaternion representation of v1 using v0 as the origin."""

        # if v0[0] == 0.0 and v0[1] == 0.0 and v0[2] == 0.0:
        #     pass
        # else:
        #     v0 = v0 / np.linalg.norm(v0)
        v0 = v0 / np.linalg.norm(v0)
        v1 = v1 / np.linalg.norm(v1)
        c = v0.dot(v1)
        # Epsilon prevents issues at poles.
        if c < (-1 + EPSILON):
            c = max(c, -1)
            m = np.stack([v0, v1], 0)
            _, _, vh = np.linalg.svd(m, full_matrices=True)
            axis = vh.T[:, 2]
            w2 = (1 + c) * 0.5
            w = np.sqrt(w2)
            axis = axis * np.sqrt(1 - w2)
            return np.quaternion(w, *axis)
        axis = np.cross(v0, v1)
        s = np.sqrt((1 + c) * 2)
        return np.quaternion(s * 0.5, *(axis / s))

    def safe_inverse_single(self,a):
        r, t = self.split_rt_single(a)
        t = np.reshape(t, (3,1))
        r_transpose = r.T
        inv = np.concatenate([r_transpose, -np.matmul(r_transpose, t)], 1)
        bottom_row = a[3:4, :] # this is [0, 0, 0, 1]
        # bottom_row = torch.tensor([0.,0.,0.,1.]).view(1,4) 
        inv = np.concatenate([inv, bottom_row], 0)
        return inv
    
    def split_rt_single(self,rt):
        r = rt[:3, :3]
        t = np.reshape(rt[:3, 3], 3)
        return r, t

    def run(self):

        scene = self.sim.semantic_scene
        objects = scene.objects
        plt_id = 0
        # np.random.shuffle(objects)
        for obj in objects:

            if obj == None or obj.category == None or obj.category.name() not in self.include_classes:
                continue

            print("DELETING PREVIOUS FILES")
            # remove png files from previous episode
            filelist = glob.glob(os.path.join(self.orbslam_path,"rgb", "*.png"))
            for f in filelist:
                os.remove(f)
            filelist = glob.glob(os.path.join(self.orbslam_path,"depth", "*.tiff"))
            for f in filelist:
                os.remove(f)
            
            # reset trajectory files
            deleted = []
            if os.path.exists(self.ORBSLAM_Cam_Traj):
                deleted.append(['cam_traj'])
                os.remove(self.ORBSLAM_Cam_Traj)
            if os.path.exists(self.ORBSLAM_Cam_Keypoints):
                deleted.append(['keypoint_traj'])
                os.remove(self.ORBSLAM_Cam_Keypoints)
            associations_file = "/home/nel/ORB_SLAM2/Examples/RGB-D/associations/replica3.txt"
            if os.path.exists(associations_file):
                deleted.append(['associations2'])
                os.remove(associations_file)
            if os.path.exists(self.orbslam_path + "/" + 'rgb3.txt'):
                deleted.append(['rgb3.txt'])
                os.remove(self.orbslam_path + "/" + 'rgb3.txt')
            if os.path.exists(self.orbslam_path + "/" + 'depth3.txt'):
                deleted.append(['depth3.txt'])
                os.remove(self.orbslam_path + "/" + 'depth3.txt')

            # st()
            if self.verbose:
                print(f"Object name is: {obj.category.name()}")
            # Calculate distance to object center
            obj_center = obj.obb.to_aabb().center
            #print(obj_center)
            obj_center = np.expand_dims(obj_center, axis=0)
            #print(obj_center)
            distances = np.sqrt(np.sum((self.nav_pts - obj_center)**2, axis=1))

            # Get points with r_min < dist < r_max
            valid_pts = self.nav_pts[np.where((distances > self.radius_min)*(distances<self.radius_max))]
            # if not valid_pts:
                # continue

            # plot valid points that we happen to select
            # self.plot_navigable_points(valid_pts)

            # Bin points based on angles [vertical_angle (10 deg/bin), horizontal_angle (10 deg/bin)]
            valid_pts_shift = valid_pts - obj_center

            dz = valid_pts_shift[:,2]
            dx = valid_pts_shift[:,0]
            dy = valid_pts_shift[:,1]

            # Get yaw for binning 
            valid_yaw = np.degrees(np.arctan2(dz,dx))

            # # pitch calculation 
            # dxdz_norm = np.sqrt((dx * dx) + (dz * dz))
            # valid_pitch = np.degrees(np.arctan2(dy,dxdz_norm))

            # binning yaw around object
            # nbins = 18

            nbins = 200
            bins = np.linspace(-180, 180, nbins+1)
            bin_yaw = np.digitize(valid_yaw, bins)

            num_valid_bins = np.unique(bin_yaw).size

            spawns_per_bin = 1 #int(self.num_views / num_valid_bins) + 2
            print(f'spawns_per_bin: {spawns_per_bin}')

            if False:
                import matplotlib.cm as cm
                colors = iter(cm.rainbow(np.linspace(0, 1, nbins)))
                plt.figure(2)
                plt.clf()
                print(np.unique(bin_yaw))
                for bi in range(nbins):
                    cur_bi = np.where(bin_yaw==(bi+1))
                    points = valid_pts[cur_bi]
                    x_sample = points[:,0]
                    z_sample = points[:,2]
                    plt.plot(z_sample, x_sample, 'o', color = next(colors))
                plt.plot(obj_center[:,2], obj_center[:,0], 'x', color = 'black')
                plt.show()


            pos_s_all = []
            bin_start = 0
            count = 0
            max_count = 0
            first = True
            for b in range(nbins):
            
                # get all angle indices in the current bin range
                # st()
                inds_bin_cur = np.where(bin_yaw==(b+1)) # bins start 1 so need +1
                # if b>0:
                #     inds_bin_prev = np.where(bin_yaw==(b))
                
                if inds_bin_cur[0].size == 0:
                    print("Continuing")
                    count += 1
                    if count > max_count and not b == 0:
                        max_count = count
                        bin_start = b 
                    # break
                    continue

                
                    

                for s in range(spawns_per_bin):
                    # st()
                    if first:
                        np.random.seed(1)
                        s_ind = np.random.choice(inds_bin_cur[0])
                        pos_s = valid_pts[s_ind]
                        pos_s_prev = pos_s
                        y_height_fixed = pos_s[1] + 1.5
                        first = False
                    else:
                        # get closest valid position for smooth trajectory
                        pos_s_cur_all = valid_pts[inds_bin_cur[0]]
                        distances_to_prev = np.sqrt(np.sum((pos_s_cur_all - pos_s_prev)**2, axis=1))
                        argmin_s_ind_cur = np.argmin(distances_to_prev)
                        distances_to_prev_cur = distances_to_prev[argmin_s_ind_cur]
                        if distances_to_prev_cur > 20:
                            print("TOO FAR: CONTINUING")
                            count += 1
                            continue
                        s_ind = inds_bin_cur[0][argmin_s_ind_cur]
                        pos_s = valid_pts[s_ind]
                        pos_s_prev = pos_s

                    count = 0
                    
                    pos_s_all.append(pos_s)
            
            pos_s_all = np.array(pos_s_all)
            # print("STOP")
            
            if False:
                import matplotlib.cm as cm
                colors = iter(cm.rainbow(np.linspace(0, 1, nbins)))
                plt.figure(2)
                plt.clf()
                print(np.unique(bin_yaw))
                for bi in range(nbins):
                    cur_bi = np.where(bin_yaw==(bi+1))
                    points = valid_pts[cur_bi]
                    x_sample = points[:,0]
                    z_sample = points[:,2]
                    plt.plot(z_sample, x_sample, 'o', color = next(colors))
                    plt.plot(pos_s_all[:,2], pos_s_all[:,0], 'x', color = 'red')
                plt.plot(pos_s_all[bin_start,2], pos_s_all[bin_start,0], 'x', color = 'black')
                plt.plot(obj_center[:,2], obj_center[:,0], 'x', color = 'black')
                plt.show()

            bins_all = np.arange(bin_start, nbins)
            bins_all = np.hstack((bins_all, np.arange(0, bin_start)))
            # bins_all = np.
            count = 0
            pos_s_all_two = []
            first = True
            for b in list(bins_all):
            
                # get all angle indices in the current bin range
                # st()
                inds_bin_cur = np.where(bin_yaw==(b+1)) # bins start 1 so need +1
                # if b>0:
                #     inds_bin_prev = np.where(bin_yaw==(b))
                if inds_bin_cur[0].size == 0:
                    print("Continuing")
                    count += 1
                    if count == 5:
                        print("BREAK")
                        break
                    # break
                    continue

                
                    

                for s in range(spawns_per_bin):
                    # st()
                    if first:
                        np.random.seed(1)
                        s_ind = np.random.choice(inds_bin_cur[0])
                        pos_s = valid_pts[s_ind]
                        pos_s_prev = pos_s
                        y_height_fixed = pos_s[1] + 1.5
                        first = False
                    else:
                        # get closest valid position for smooth trajectory
                        pos_s_cur_all = valid_pts[inds_bin_cur[0]]
                        distances_to_prev = np.sqrt(np.sum((pos_s_cur_all - pos_s_prev)**2, axis=1))
                        argmin_s_ind_cur = np.argmin(distances_to_prev)
                        if distances_to_prev_cur > 20:
                            print("TOO FAR: CONTINUING")
                            count += 1
                            continue
                        s_ind = inds_bin_cur[0][argmin_s_ind_cur]
                        pos_s = valid_pts[s_ind]
                        pos_s_prev = pos_s
                    
                    count = 0
                    
                    pos_s_all_two.append(pos_s)
            
            pos_s_all_two = np.array(pos_s_all_two)

            if pos_s_all_two.shape[0]==0:
                print("NO VALID VIEWS DETECTED IN TRAJECTORY")
                continue

            if True:
                import matplotlib.cm as cm
                colors = iter(cm.rainbow(np.linspace(0, 1, nbins)))
                plt.figure(2)
                plt.clf()
                # print(np.unique(bin_yaw))
                for bi in range(nbins):
                    cur_bi = np.where(bin_yaw==(bi+1))
                    points = valid_pts[cur_bi]
                    x_sample = points[:,0]
                    z_sample = points[:,2]
                    plt.plot(z_sample, x_sample, 'o', color = next(colors))
                    plt.plot(pos_s_all_two[:,2], pos_s_all_two[:,0], 'x', color = 'red')
                plt.plot(pos_s_all_two[0,2], pos_s_all_two[0,0], 'x', color = 'black')
                plt.plot(pos_s_all_two[-1,2], pos_s_all_two[-1,0], 'x', color = 'blue')
                plt.plot(obj_center[:,2], obj_center[:,0], 'x', color = 'black')
                # plt.show()
                plt_name = 'samples/' + str(plt_id) + '_' + 'samples.png'
                plt.savefig(plt_name)
            
            print("Generated ", pos_s_all_two.shape[0], "views on trajectory")
            if pos_s_all_two.shape[0] < self.min_orbslam_views:
                print("NOT ENOUGH VIEWS FOR ORBSLAM")
                continue
            
            action = "do_nothing"
            episodes = []
            valid_pts_selected = []
            cnt = 0
            texts_images = []
            texts_depths = []
            rots_cam0 = []
            camXs_T_camX0_4x4 = []
            camX0_T_camXs_4x4 = []
            origin_T_camXs = []
            origin_T_camXs_t = []
            # rots2 = []
            rots_cam0.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            # rots2.append([0, 0, 0, 0, 0, 0, 0, 1])
            rots_origin = []
            for b in range(pos_s_all_two.shape[0]):
                
                # get all angle indices in the current bin range
                # st()
                # inds_bin_cur = np.where(bin_yaw==(b+1)) # bins start 1 so need +1
                # if b>0:
                #     inds_bin_prev = np.where(bin_yaw==(b))
                
                # if inds_bin_cur[0].size == 0:
                #     print("Continuing")
                #     break
                #     # continue

                # for s in range(spawns_per_bin):
                #     # st()
                #     if b==0:
                #         np.random.seed(1)
                #         s_ind = np.random.choice(inds_bin_cur[0])
                #         pos_s = valid_pts[s_ind]
                #         pos_s_prev = pos_s
                #         y_height_fixed = pos_s[1] + 1.5
                #     else:
                #         # get closest valid position for smooth trajectory
                #         pos_s_cur_all = valid_pts[inds_bin_cur[0]]
                #         distances_to_prev = np.sqrt(np.sum((pos_s_cur_all - pos_s_prev)**2, axis=1))
                #         argmin_s_ind_cur = np.argmin(distances_to_prev)
                #         s_ind = inds_bin_cur[0][argmin_s_ind_cur]
                #         pos_s = valid_pts[s_ind]
                #         pos_s_prev = pos_s
                #         # pos_s[1] = y_height_fixed
                    
                    pos_s = pos_s_all_two[b]
                    
                    valid_pts_selected.append(pos_s)
                    agent_state = habitat_sim.AgentState()
                    pos_s = pos_s + np.array([0, 1.5, 0])
                    pos_s[1] = y_height_fixed # keep y height constant as much as possible
                    agent_state.position = pos_s

                    # initiate agent
                    self.agent.set_state(agent_state)
                    self.sim.step(action)


                    # YAW calculation - rotate to object
                    agent_to_obj = np.squeeze(obj_center) - agent_state.position
                    agent_local_forward = np.array([0, 0, -1.0]) # y, z, x
                    flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
                    flat_dist_to_obj = np.linalg.norm(flat_to_obj)
                    flat_to_obj /= flat_dist_to_obj

                    det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
                    turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))
                    quat_yaw = quat_from_angle_axis(turn_angle, np.array([0, 1.0, 0]))

                    # Set agent yaw rotation to look at object
                    agent_state.rotation = quat_yaw
                    
                    # change sensor state to default 
                    # need to move the sensors too
                    if self.verbose:
                        print(self.agent.state.sensor_states)
                    for sensor in self.agent.state.sensor_states:
                        # st()
                        self.agent.state.sensor_states[sensor].rotation = agent_state.rotation
                        self.agent.state.sensor_states[sensor].position = agent_state.position # + np.array([0, 1.5, 0]) # ADDED IN UP TOP
                        # print("PRINT", self.agent.state.sensor_states[sensor].rotation)

                    # Calculate Pitch from head to object
                    turn_pitch = np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj))
                    num_turns = np.abs(np.floor(turn_pitch/self.rot_interval)).astype(int) # compute number of times to move head up or down by rot_interval
                    if self.verbose:
                        print("MOVING HEAD ", num_turns, " TIMES")
                    movement = "look_up" if turn_pitch>0 else "look_down"

                    # initiate agent
                    self.agent.set_state(agent_state)
                    self.sim.step(action)


                    # Rotate "head" of agent up or down based on calculated pitch angle to object to center it in view
                    if num_turns == 0:
                        pass
                    else: 
                        for turn in range(num_turns):
                            # st()
                            self.sim.step(movement)
                            if self.verbose:
                                for sensor in self.agent.state.sensor_states:
                                    print(self.agent.state.sensor_states[sensor].rotation)
                    
                    # get observations after centiering
                    observations = self.sim.step(action)
                    
                    # Assuming all sensors have same rotation and position
                    observations["rotations"] = self.agent.state.sensor_states['color_sensor'].rotation #agent_state.rotation
                    observations["positions"] = self.agent.state.sensor_states['color_sensor'].position

                    if self.do_orbslam: 
                        im = observations["color_sensor"]
                        im = Image.fromarray(im, mode="RGBA")
                        im = im.convert("RGB")
                        str_id = str(cnt)
                        im_name = str_id + '.png'
                        im.save(self.orbslam_path + "/" + "rgb/" + im_name)
                        # img = cv2.imread(self.orbslam_path + "/" + "rgb/" + im_name, cv2.IMREAD_UNCHANGED) 
                        image_line_text = str_id + " " + "rgb/" + im_name
                        texts_images.append(image_line_text)
                        
                        
                        # img2 = cv2.imread('/home/nel/ORB_SLAM2/Examples/RGB-D/rgbd_dataset_freiburg1_desk/rgb/1305031463.759674.png', cv2.IMREAD_UNCHANGED) 
                        # test = Image.open('/home/nel/ORB_SLAM2/Examples/RGB-D/rgbd_dataset_freiburg1_desk/depth/1305031466.748734.png')
                        # img = cv2.imread('/home/nel/ORB_SLAM2/Examples/RGB-D/rgbd_dataset_freiburg1_desk/depth/1305031466.748734.png', cv2.IMREAD_UNCHANGED) 

                        depths = observations["depth_sensor"] #*1000000
                        # depths = depths.astype(np.uint32)
                        depth_name = str_id + '.tiff'
                        cv2.imwrite(self.orbslam_path + "/" + "depth/" + depth_name, depths)
                        # img = cv2.imread(self.orbslam_path + "/" + "depth/" + depth_name, cv2.IMREAD_UNCHANGED) 
                        depths_line_text = str_id + " " + "depth/" + depth_name
                        texts_depths.append(depths_line_text)




                    
                    if self.is_valid_datapoint(observations, obj):
                        if self.verbose:
                            print("episode is valid......")
                        episodes.append(observations)
                        if self.visualize:
                            rgb = observations["color_sensor"]
                            semantic = observations["semantic_sensor"]
                            depth = observations["depth_sensor"]
                            self.display_sample(rgb, semantic, depth, mainobj=obj, visualize=False)
                    # if self.visualize:
                    #         rgb = observations["color_sensor"]
                    #         semantic = observations["semantic_sensor"]
                    #         depth = observations["depth_sensor"]
                    #         self.display_sample(rgb, semantic, depth, mainobj=obj, visualize=False)

                    #print("agent_state: position", self.agent.state.position, "rotation", self.agent.state.rotation)

                    
                    
                    if cnt > 0:

                        
                        origin_T_camX = quaternion.as_rotation_matrix(episodes[cnt]["rotations"]) 
                        camX0_T_camX = np.matmul(camX0_T_origin, origin_T_camX)
                        # camX0_T_camX = np.matmul(camX0_T_origin, origin_T_camX)

                        origin_T_camXs.append(origin_T_camX)
                        origin_T_camXs_t.append(episodes[cnt]["positions"])

                        origin_T_camX_4x4 = np.eye(4)
                        origin_T_camX_4x4[0:3, 0:3] = origin_T_camX
                        origin_T_camX_4x4[:3,3] = episodes[cnt]["positions"]
                        camX0_T_camX_4x4 = np.matmul(camX0_T_origin_4x4, origin_T_camX_4x4)
                        camX_T_camX0_4x4 = self.safe_inverse_single(camX0_T_camX_4x4)

                        camXs_T_camX0_4x4.append(camX_T_camX0_4x4)
                        camX0_T_camXs_4x4.append(camX0_T_camX_4x4)

                        # # trans_camX0_T_camX = episodes[cnt]["positions"] - episodes[0]["positions"]

                        # r_camX_T_camX0, t_camX_T_camX0, = self.split_rt_single(camX_T_camX0_4x4)
                        # r_camX_T_camX0_quat = quaternion.as_float_array(quaternion.from_rotation_matrix(r_camX_T_camX0))

                        # # print(t_camX_T_camX0)
                        # # print(trans_camX0_T_camX)
                        
                        # # full_trans1 = np.hstack((cnt, trans, quat_diff1))
                        # full_trans_camX_T_camX0 = np.hstack((cnt, t_camX_T_camX0, r_camX_T_camX0_quat))

                        # # rots1.append(list(full_trans1))
                        # rots_cam0.append(list(full_trans_camX_T_camX0))
                        # # rots_origin.append(list(full_trans_origin)) 

                        camX0_T_camX_4x4 = self.safe_inverse_single(camX_T_camX0_4x4)
                        origin_T_camX_4x4 = np.matmul(origin_T_camX0_4x4, camX0_T_camX_4x4)
                        r_origin_T_camX, t_origin_T_camX, = self.split_rt_single(origin_T_camX_4x4)

                        if self.verbose:
                            print(r_origin_T_camX)
                            print(origin_T_camX)

                    else:
                        origin_T_camX0_quat = episodes[0]["rotations"]
                        origin_T_camX0 = quaternion.as_rotation_matrix(origin_T_camX0_quat)
                        camX0_T_origin = np.linalg.inv(origin_T_camX0)
                        # camX0_T_origin = self.safe_inverse_single(origin_T_camX0)

                        origin_T_camXs.append(origin_T_camX0)
                        origin_T_camXs_t.append(episodes[0]["positions"])

                        origin_T_camX0_4x4 = np.eye(4)
                        origin_T_camX0_4x4[0:3, 0:3] = origin_T_camX0
                        origin_T_camX0_4x4[:3,3] = episodes[0]["positions"]
                        camX0_T_origin_4x4 = self.safe_inverse_single(origin_T_camX0_4x4)

                        camXs_T_camX0_4x4.append(np.eye(4))

                        camX0_T_camXs_4x4.append(np.eye(4))

                        origin_T_camX0_t = episodes[0]["positions"]
                        
                        # r0 = quaternion.as_rotation_vector(episodes[0]["rotations"])
                        # quat_diff_origin = quaternion.as_float_array(episodes[0]["rotations"]) #quaternion.as_float_array(self.quaternion_from_two_vectors(self.origin_rot_vector,r0))
                        # full_trans_origin = np.hstack((cnt, episodes[0]["positions"], quat_diff_origin))
                        # rots_origin.append(list(full_trans_origin))

                    cnt +=1
            
            print("Generated ", len(episodes), " views.")
            if len(episodes) < self.min_orbslam_views:
                print("NOT ENOUGH VIEWS FOR ORBSLAM")
                continue
                


            if self.do_orbslam:
                with open(self.orbslam_path + "/" + 'rgb3.txt', 'w') as f:
                    for item in texts_images:
                        f.write("%s\n" % item)
                
                with open(self.orbslam_path + "/" + 'depth3.txt', 'w') as f:
                    for item in texts_depths:
                        f.write("%s\n" % item)

                # with open(self.orbslam_path + "/" + 'CamTraj1.txt', 'w') as f:
                #     for item in rots1:
                #         f.write("%s\n" % item)
                
                # with open(self.orbslam_path + "/" + 'CamTraj.txt', 'w') as f:
                #     for item in rots_cam0:
                #         f.write("%s\n" % item)
            
            ############ ORBSLAM ################
            if self.do_orbslam:
                camXs_T_camX0_4x4 = np.array(camXs_T_camX0_4x4)
                # origin_T_camXs = np.array(origin_T_camXs)
                # origin_T_camXs_t = np.array(origin_T_camXs_t)
                camX0_T_camXs_4x4 = np.array(camX0_T_camXs_4x4)

                orbslam_dir = "/home/nel/ORB_SLAM2"
                executable = os.path.join(orbslam_dir, "Examples/RGB-D/rgbd_tum")
                vocabulary_file = os.path.join(orbslam_dir, "Vocabulary/ORBvoc.txt")
                settings_file = os.path.join(orbslam_dir, "Examples/RGB-D/REPLICA.yaml")
                data_dir = os.path.join(orbslam_dir, "Examples/RGB-D/replica3")
                associations_file = os.path.join(orbslam_dir, "Examples/RGB-D/associations/replica3.txt")
                associations_executable = os.path.join(orbslam_dir, "Examples/RGB-D/associations/associate.py")
                rgb_file = os.path.join(orbslam_dir, "Examples/RGB-D/replica3/rgb3.txt")
                depth_file = os.path.join(orbslam_dir, "Examples/RGB-D/replica3/depth3.txt")

                # executable = os.path.join(orbslam_dir, "Examples/Monocular/mono_tum")
                # vocabulary_file = os.path.join(orbslam_dir, "Vocabulary/ORBvoc.txt")
                # settings_file = os.path.join(orbslam_dir, "Examples/Monocular/REPLICA.yaml")
                # data_dir = os.path.join(orbslam_dir, "Examples/RGB-D/replica")

                # associate rgb and depth files
                print("ASSOCIATING FILES")
                associate(rgb_file, depth_file, associations_file)

                # run ORBSLAM2
                print("RUNNING ORBSLAM2...")
                try:
                    # RGB-D
                    output = subprocess.run([executable, vocabulary_file, settings_file, data_dir, associations_file], capture_output=True, text=True, check=True, timeout=900)
                    # Monocular
                    # output = subprocess.run([executable, vocabulary_file, settings_file, data_dir], capture_output=True, text=True, check=True, timeout=900)

                # except subprocess.CalledProcessError as e:
                #     print(e)
                #     print("PROCESS TIMED OUT")
                #     continue
                except Exception as e:
                    print(e)
                    print("PROCESS TIMED OUT")
                    continue

                # load camera rotation and translation estimated by orbslam - these are wrt first frame camX0
                camXs_T_camX0_orb_output = np.loadtxt(self.ORBSLAM_Cam_Traj)
                print(output)

                inds_keep = camXs_T_camX0_orb_output[:,0].astype(np.uint8)
                if 0 not in inds_keep:
                    print("NO FRAME 1: SKIPPING")
                    continue
                
                if len(episodes) != camXs_T_camX0_orb_output.shape[0]:
                    # keep only inds kept by orbslam
                    camXs_T_camX0_4x4 = camXs_T_camX0_4x4[inds_keep]
                    # origin_T_camXs = origin_T_camXs[inds_keep]
                    # origin_T_camXs_t = origin_T_camXs_t[inds_keep]
                    camX0_T_camXs_4x4 = camX0_T_camXs_4x4[inds_keep]
                    episodes = [episodes[i] for i in list(inds_keep)]
                
                # assert len(episodes) == camXs_T_camX0_orb_output.shape[0], f"{camXs_T_camX0_orb_output.shape[0]}, {len(episodes)}"

                # np.save("camXs_T_camX0_orb_output.npy", camXs_T_camX0_orb_output)
                # np.save("origin_T_camX0_4x4.npy", origin_T_camX0_4x4)
                # np.save("camXs_T_camX0_4x4.npy", camXs_T_camX0_4x4)
                # pos = []
                # for i in range(camXs_T_camX0_orb_output.shape[0]):
                #     pos.append(episodes[i]["positions"])
                # pos = np.array(pos)
                # np.save("pos.npy", pos)


                # get ORBSLAM estimated rot
                camXs_T_camX0_quant = []
                camXs_T_camX0 = []
                for i in range(camXs_T_camX0_orb_output.shape[0]):
                    cur = camXs_T_camX0_orb_output[i]
                    camX_T_camX0_quant = np.quaternion(cur[7], cur[4], cur[5], cur[6])
                    camX_T_camX0 = quaternion.as_rotation_matrix(camX_T_camX0_quant) 
                    camXs_T_camX0_quant.append(camX_T_camX0_quant)
                    camXs_T_camX0.append(camX_T_camX0)
                camXs_T_camX0 = np.array(camXs_T_camX0)
                camXs_T_camX0_quant = np.array(camXs_T_camX0_quant)

                # get ORBSLAM estimated pos
                t = []
                for i in range(camXs_T_camX0_orb_output.shape[0]):
                    cur = camXs_T_camX0_orb_output[i]
                    t_cur = np.array([-cur[1], -cur[2], -cur[3]])    
                    t.append(t_cur)
                t = np.array(t)
                
                # Get 4x4 matrix and convert to origin coords
                camXs_T_camX0_4x4_orb = []
                for i in range(camXs_T_camX0_orb_output.shape[0]):
                    # assert len(episodes) == camXs_T_camX0_orb_output.shape[0], f"{camXs_T_camX0_orb_output.shape[0]}, {len(episodes)}"
                    # get estimated 4x4
                    camX_T_camX0_4x4_orb = np.eye(4)
                    camX_T_camX0_4x4_orb[0:3, 0:3] = camXs_T_camX0[i]
                    camX_T_camX0_4x4_orb[:3,3] = t[i]

                    # invert
                    camX0_T_camX_4x4_orb = self.safe_inverse_single(camX_T_camX0_4x4_orb)

                    # convert to origin coordinates
                    origin_T_camX_4x4_orb = np.matmul(origin_T_camX0_4x4, camX0_T_camX_4x4_orb)
                    r_origin_T_camX_orb, t_origin_T_camX_orb = self.split_rt_single(origin_T_camX_4x4_orb)
                    r_origin_T_camX_orb_quat = quaternion.from_rotation_matrix(r_origin_T_camX_orb)

                    # testing for plotting
                    camX0_T_camX_4x4 = camX0_T_camXs_4x4[i]
                    origin_T_camX_4x4 = np.matmul(origin_T_camX0_4x4, camX0_T_camX_4x4)
                    r_origin_T_camX, t_origin_T_camX = self.split_rt_single(origin_T_camX_4x4)
                    r_origin_T_camX_quat = quaternion.from_rotation_matrix(r_origin_T_camX)
                    episodes[i]["positions_gt"] = t_origin_T_camX
                    episodes[i]["rotations_gt"] = r_origin_T_camX_quat
                    
                    # # for some reason that I have to figure out - we need this adjustment for y
                    # if i==0:
                    #     y_t0 = t_origin_T_camX_orb[1]
                    # else:
                    #     dist_to_t0 = y_t0 - t_origin_T_camX_orb[1]
                    #     t_origin_T_camX_orb[1] = t_origin_T_camX_orb[1] + 2*dist_to_t0
                    
                    t_origin_T_camX_orb[1] = t_origin_T_camX[1]

                    #save
                    episodes[i]["positions_orb"] = t_origin_T_camX_orb
                    episodes[i]["rotations_orb"] = r_origin_T_camX_orb_quat
                    camXs_T_camX0_4x4_orb.append(camX_T_camX0_4x4_orb)
                camXs_T_camX0_4x4_orb = np.array(camXs_T_camX0_4x4_orb)
                



                ## PLOTTTING #########
                print("PLOTTING")
                plot_camX0_traj = True
                plot_origin_traj = True

                if plot_camX0_traj:
                    x_orbslam_camX0 = []
                    y_orbslam_camX0 = []
                    z_orbslam_camX0 = []
                    for i in range(camXs_T_camX0_orb_output.shape[0]):
                        camX_T_camX0_4x4_orb_r, camX_T_camX0_4x4_orb_t = self.split_rt_single(camXs_T_camX0_4x4_orb[i])
                        x_orbslam_camX0.append(camX_T_camX0_4x4_orb_t[0])
                        y_orbslam_camX0.append(camX_T_camX0_4x4_orb_t[1])
                        z_orbslam_camX0.append(camX_T_camX0_4x4_orb_t[2])
                    x_orbslam_camX0 = np.array(x_orbslam_camX0)
                    y_orbslam_camX0 = np.array(y_orbslam_camX0)
                    z_orbslam_camX0 = np.array(z_orbslam_camX0)

                    x_gt_camX0 = []
                    y_gt_camX0 = []
                    z_gt_camX0 = []
                    for i in range(camXs_T_camX0_orb_output.shape[0]):
                        camX_T_camX0_4x4_r, camX_T_camX0_4x4_t = self.split_rt_single(camXs_T_camX0_4x4[i])
                        x_gt_camX0.append(camX_T_camX0_4x4_t[0])
                        y_gt_camX0.append(camX_T_camX0_4x4_t[1])
                        z_gt_camX0.append(camX_T_camX0_4x4_t[2])
                    x_gt_camX0 = np.array(x_gt_camX0)
                    y_gt_camX0 = np.array(y_gt_camX0)
                    z_gt_camX0 = np.array(z_gt_camX0)

                    plt.figure()
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                    ax1.plot(y_orbslam_camX0, x_orbslam_camX0, label='ORB-SLAM-yx', color='green', linestyle='dashed')
                    ax1.plot(y_gt_camX0, x_gt_camX0, label='GT', color='blue', linestyle='solid')
                    ax1.legend()
                    ax2.plot(y_orbslam_camX0, z_orbslam_camX0, label='ORB-SLAM-yz', color='green', linestyle='dashed')
                    ax2.plot(y_gt_camX0, z_gt_camX0, label='GT', color='blue', linestyle='solid')
                    ax2.legend()
                    ax3.plot(x_orbslam_camX0, z_orbslam_camX0, label='ORB-SLAM-xz', color='green', linestyle='dashed')
                    ax3.plot(x_gt_camX0, z_gt_camX0, label='GT', color='blue', linestyle='solid')
                    ax3.legend()
                    plt_name = 'maps/' + str(self.ep_idx) + '_' + str(plt_id) + '_' + 'camX0_' + 'map.png'
                    plt.savefig(plt_name)

                    plt.figure()
                    ax = plt.axes(projection='3d')
                    ax.plot3D(x_orbslam_camX0, y_orbslam_camX0, z_orbslam_camX0, 'green')
                    ax.plot3D(x_gt_camX0, y_gt_camX0, z_gt_camX0, 'blue')
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')
                    # plt.plot(-np.array(x_orbslam), np.array(y_orbslam), label='ORB-SLAM', color='green', linestyle='dashed')
                    # plt.plot(x_gt, y_gt, label='GT', color='blue', linestyle='solid')
                    # plt.legend()
                    plt_name = 'maps/' + str(self.ep_idx) + '_' + str(plt_id) + '_' + 'camX0_3D' + 'map.png'
                    plt.savefig(plt_name)

                if plot_origin_traj:
                
                    x_orbslam = []
                    y_orbslam = []
                    z_orbslam = []
                    for i in range(camXs_T_camX0_orb_output.shape[0]):
                        x_orbslam.append(episodes[i]["positions_orb"][0])
                        y_orbslam.append(episodes[i]["positions_orb"][1])
                        z_orbslam.append(episodes[i]["positions_orb"][2])
                    x_orbslam = np.array(x_orbslam)
                    y_orbslam = np.array(y_orbslam)
                    z_orbslam = np.array(z_orbslam)
                    
                    x_gt = []
                    y_gt = []
                    z_gt = []
                    for i in range(camXs_T_camX0_orb_output.shape[0]):
                        x_gt.append(episodes[i]["positions"][0])
                        y_gt.append(episodes[i]["positions"][1])
                        z_gt.append(episodes[i]["positions"][2])
                    x_gt = np.array(x_gt)
                    y_gt = np.array(y_gt)
                    z_gt = np.array(z_gt)

                    x_gt_mat = []
                    y_gt_mat = []
                    z_gt_mat = []
                    for i in range(camXs_T_camX0_orb_output.shape[0]):
                        x_gt_mat.append(episodes[i]["positions_gt"][0])
                        y_gt_mat.append(episodes[i]["positions_gt"][1])
                        z_gt_mat.append(episodes[i]["positions_gt"][2])
                    x_gt_mat = np.array(x_gt_mat)
                    y_gt_mat = np.array(y_gt_mat)
                    z_gt_mat = np.array(z_gt_mat)

                        

                    plt.figure()
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                    ax1.plot(y_orbslam, x_orbslam, label='ORB-SLAM-yx', color='green', linestyle='dashed')
                    ax1.plot(y_gt, x_gt, label='GT', color='blue', linestyle='solid')
                    ax1.plot(y_gt_mat, x_gt_mat, label='GT_mat', color='red', linestyle='solid')
                    ax1.legend()
                    ax2.plot(y_orbslam, z_orbslam, label='ORB-SLAM-yz', color='green', linestyle='dashed')
                    ax2.plot(y_gt, z_gt, label='GT', color='blue', linestyle='solid')
                    ax2.plot(y_gt_mat, z_gt_mat, label='GT_mat', color='red', linestyle='solid')
                    ax2.legend()
                    ax3.plot(x_orbslam, z_orbslam, label='ORB-SLAM-xz', color='green', linestyle='dashed')
                    ax3.plot(x_gt, z_gt, label='GT', color='blue', linestyle='solid')
                    ax3.plot(x_gt_mat, z_gt_mat, label='GT_mat', color='red', linestyle='solid')
                    ax3.legend()
                    plt_name = 'maps/' + str(self.ep_idx) + '_' + str(plt_id) + '_' + 'origin' + 'map.png'
                    plt.savefig(plt_name)
                    
                    plt.figure()
                    ax = plt.axes(projection='3d')
                    ax.plot3D(x_orbslam, y_orbslam, z_orbslam, 'green')
                    ax.plot3D(x_gt, y_gt, z_gt, 'blue')
                    ax.plot3D(x_gt_mat, y_gt_mat, z_gt_mat, 'red')
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')
                    # plt.plot(-np.array(x_orbslam), np.array(y_orbslam), label='ORB-SLAM', color='green', linestyle='dashed')
                    # plt.plot(x_gt, y_gt, label='GT', color='blue', linestyle='solid')
                    # plt.legend()
                    plt_name = 'maps/' + str(self.ep_idx) + '_' + str(plt_id) + '_' + 'origin_3D' + 'map.png'
                    plt.savefig(plt_name)
                
                plt_id += 1

            if not self.do_orbslam:
                if len(episodes) >= self.num_views:
                    print(f'num episodes: {len(episodes)}')
                    data_folder = obj.category.name() + '_' + obj.id
                    data_path = os.path.join(self.basepath, data_folder)
                    print("Saving to ", data_path)
                    os.mkdir(data_path)
                    np.random.seed(1)
                    flat_obs = np.random.choice(episodes, self.num_views, replace=False)
                    viewnum = 0
                    for obs in flat_obs:
                            self.save_datapoint(self.agent, obs, data_path, viewnum, obj.id, True)
                            viewnum += 1


                else:
                    print(f"Not enough episodes: f{len(episodes)}")
            
            # the len of episodes is sometimes greater than camXs_T_camX0_orb_output.shape[0]
            # so we need to sample from episode number less than camXs_T_camX0_orb_output.shape[0]
            else:
                num_valid_episodes =  camXs_T_camX0_orb_output.shape[0]
                if num_valid_episodes >= self.num_views:
                    print(f'num episodes: {num_valid_episodes}')
                    data_folder = obj.category.name() + '_' + obj.id
                    data_path = os.path.join(self.basepath, data_folder)
                    print("Saving to ", data_path)
                    os.mkdir(data_path)
                    np.random.seed(1)

                    episodes = episodes[:num_valid_episodes]
                    interval = np.floor(camXs_T_camX0_orb_output.shape[0]/self.num_views)
                    episodes = episodes[::interval]
                    flat_obs = np.random.choice(episodes, self.num_views, replace=False)
                    viewnum = 0
                    for obs in flat_obs:
                        self.save_datapoint(self.agent, obs, data_path, viewnum, obj.id, True)
                        viewnum += 1


                else:
                    print(f"Not enough episodes: f{len(episodes)}")


            if self.visualize:
                valid_pts_selected = np.vstack(valid_pts_selected)
                self.plot_navigable_points(valid_pts_selected)



    def get_navigable_points(self):
        navigable_points = np.array([0,0,0])
        for i in range(20000):
            navigable_points = np.vstack((navigable_points,self.sim.pathfinder.get_random_navigable_point()))
        return navigable_points
    
    def plot_navigable_points(self, points):
        # print(points)
        x_sample = points[:,0]
        z_sample = points[:,2]
        plt.plot(z_sample, x_sample, 'o', color = 'red')
        
        plt.show()


if __name__ == '__main__':
    AutomatedMultiview()