import ipdb
st = ipdb.set_trace
# st()
import habitat_sim
import habitat
from habitat.config.default import get_config
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb
from habitat_sim.utils.common import quat_from_two_vectors, quat_from_angle_axis
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

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from greedy_geodesic_follower import GreedyGeodesicFollower
# from env.habitat.utils.noisy_actions import CustomActionSpaceConfiguration
# from habitat_sim.simulator import make_greedy_follower
import habitat_sim.simulator
from habitat_sim.nav import (  # type: ignore
    GreedyFollowerCodes,
    GreedyGeodesicFollowerImpl,
    PathFinder,
)
import sys
sys.path.append("..")

from scipy.ndimage.morphology import binary_fill_holes

from habitat.utils.visualizations import maps
import torch

import os 
import sys
import pickle
import json
from habitat_sim.utils import common as utils
# from habitat_sim.utils import viz_utils as vut
from scipy.spatial.transform import Rotation
EPSILON = 1e-8

from argparse import Namespace


class AutomatedMultiview():
    def __init__(self):

        self.visualize = False
        self.verbose = False

        self.save_this = False
        self.gen_maps = False
        # st()
        self.mapnames = os.listdir('/home/nel/gsarch/habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/')
        # self.mapnames = os.listdir('/hdd/replica/Replica-Dataset/out/')
        # self.mapnames = ['room_1']
        self.num_episodes = len(self.mapnames)
        # self.num_episodes = 1 # temporary
        # self.ignore_classes = ['book','base-cabinet','beam','blanket','blinds','cloth','clothing','coaster','comforter','curtain','ceiling','countertop','floor','handrail','mat','paper-towel','picture','pillar','pipe','scarf','shower-stall','switch','tissue-paper','towel','vent','wall','wall-plug','window','rug','logo','set-of-clothing']
        self.ignore_classes = ['','objects','misc','base-cabinet','beam','blanket','blinds','coaster','curtain','ceiling','countertop','floor','handrail','pillar','pipe','scarf','vent','wall','window','rug','logo','void']
        self.include_rooms = ['bathroom']

        # self.include_classes = ['chair', 'bed', 'toilet', 'sofa', 'indoor-plant', 'refrigerator', 'tv-screen', 'table']
        self.small_classes = ['indoor-plant', 'toilet']
        self.rot_interval = 5.0
        self.radius_max = 3
        self.radius_min = 0.5
        self.num_flat_views = 3
        self.num_any_views = 7
        self.num_views = 10

        self.num_objects_per_episode = 3 #30 #30
        # Initialize maskRCNN
        cfg_det = get_cfg()
        cfg_det.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg_det.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2 #0.5  # set threshold for this model
        cfg_det.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg_det.MODEL.DEVICE='cpu'
        self.cfg_det = cfg_det
        self.maskrcnn = DefaultPredictor(cfg_det)
        
        # Filter only the five categories we care about
        '''
        class mapping between replica and maskRCNN
        class-name      replica ID      maskRCNN ID
        chair           20              56
        bed             7               59
        dining table    80              60
        toilet          84              61
        couch           76              57
        potted plant    44              58
        # bottle          14              39
        # clock           22              74
        refrigerator    67              72
        tv(tv-screen)   87              62
        # vase            91              75
        '''
        self.maskrcnn_to_catname = {56: "chair", 59: "bed", 61: "toilet", 57: "couch", 58: "indoor-plant", 
                            72: "refrigerator", 62: "tv", 60: "dining-table"}
        self.replica_to_maskrcnn = {20: 56, 7: 59, 84: 61, 76: 57, 44: 58, 67: 72, 87: 62, 80: 60}

        # self.env = habitat.Env(config=config, dataset=None)
        # st()
        # self.test_navigable_points()
        self.run_episodes()

    def run_episodes(self):
        self.object_labels = []
        for episode in range(self.num_episodes):
            print("STARTING EPISODE ", episode)
            # mapname = np.random.choice(self.mapnames)
            mapname = self.mapnames[episode] # KEEP THIS
            #mapname = 'apartment_0'
            #mapname = 'frl_apartment_4'
            self.test_scene = "/home/nel/gsarch/habitat/data/scene_datasets/mp3d/v1/tasks/mp3d/{}/{}.glb".format(mapname, mapname)
            # self.object_json = "/home/nel/gsarch/Replica-Dataset/out/{}/habitat/info_semantic.json".format(mapname)
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

            

            self.fov = 90
            self.camera_matrix = self.get_camera_matrix(self.sim_settings["width"], self.sim_settings["height"], self.fov)
            self.K = self.get_habitat_pix_T_camX(self.fov)

            self.basepath = f"/home/nel/gsarch/data/matterport_bathrooms_test/{mapname}_{episode}"
            # self.basepath = f"/hdd/ayushj/habitat_data/{mapname}_{episode}"
            if not os.path.exists(self.basepath):
                os.mkdir(self.basepath)

            if len(os.listdir(self.basepath)) != 0:
                print("ALREADY GEN")
                continue

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

            self.path_finder = self.sim.make_greedy_follower() #GreedyGeodesicFollower(agent=self.agent, pathfinder=PathFinder)

            self.run2()
            
            self.sim.close()
            time.sleep(1)
        unique_objs = set(self.object_labels)
        st()


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

                sensor_specs.append(sensor_spec)

        # Here you can specify the amount of displacement in a forward action and the turn angle
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.1)
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
            ),
            "do_nothing": habitat_sim.agent.ActionSpec(
                "look_down", habitat_sim.agent.ActuationSpec(amount=0.)
            ),
        }

        # self.sensor_noise_fwd = \
        #         pickle.load(open("noise_models/sensor_noise_fwd.pkl", 'rb'))
        # self.sensor_noise_right = \
        #         pickle.load(open("noise_models/sensor_noise_right.pkl", 'rb'))
        # self.sensor_noise_left = \
        #         pickle.load(open("noise_models/sensor_noise_left.pkl", 'rb'))

        # habitat.SimulatorActions.extend_action_space("NOISY_FORWARD")
        # habitat.SimulatorActions.extend_action_space("NOISY_RIGHT")
        # habitat.SimulatorActions.extend_action_space("NOISY_LEFT")
        # agent_cfg.action_space = {
        #     "do_nothing": habitat_sim.agent.ActionSpec(
        #         "move_forward", habitat_sim.agent.ActuationSpec(amount=0.)
        #     ),
        #     "move_forward": habitat_sim.agent.ActionSpec(
        #         "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        #     ),
        #     "turn_left": habitat_sim.agent.ActionSpec(
        #         "turn_left", habitat_sim.agent.ActuationSpec(amount=self.rot_interval)
        #     ),
        #     "turn_right": habitat_sim.agent.ActionSpec(
        #         "turn_right", habitat_sim.agent.ActuationSpec(amount=self.rot_interval)
        #     ),
        #     "look_up":habitat_sim.ActionSpec(
        #         "look_up", habitat_sim.ActuationSpec(amount=self.rot_interval)
        #     ),
        #     "look_down":habitat_sim.ActionSpec(
        #         "look_down", habitat_sim.ActuationSpec(amount=self.rot_interval)
        #     ),
        #     "look_down_init":habitat_sim.ActionSpec(
        #         "look_down", habitat_sim.ActuationSpec(amount=100.0)
        #     )
        # }
        # agent_cfg.action_space = {
        #     "do_nothing": habitat_sim.agent.ActionSpec(
        #         "move_forward", habitat_sim.agent.ActuationSpec(amount=0.)
        #     ),
        #     "move_forward": habitat_sim.agent.ActionSpec(
        #         "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        #     ),
        #     "turn_left": habitat_sim.agent.ActionSpec(
        #         "turn_left", habitat_sim.agent.ActuationSpec(amount=self.rot_interval)
        #     ),
        #     "turn_right": habitat_sim.agent.ActionSpec(
        #         "turn_right", habitat_sim.agent.ActuationSpec(amount=self.rot_interval)
        #     ),
        #     "look_up":habitat_sim.ActionSpec(
        #         "look_up", habitat_sim.ActuationSpec(amount=self.rot_interval)
        #     ),
        #     "look_down":habitat_sim.ActionSpec(
        #         "look_down", habitat_sim.ActuationSpec(amount=self.rot_interval)
        #     ),
        #     "look_down_init":habitat_sim.ActionSpec(
        #         "look_down", habitat_sim.ActuationSpec(amount=100.0)
        #     )
        # }

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

        im = rgb_img[...,:3]
        im = im[:, :, ::-1]
        outputs = self.maskrcnn(im)

        pred_masks = outputs['instances'].pred_masks
        pred_boxes = outputs['instances'].pred_boxes.tensor
        pred_classes = outputs['instances'].pred_classes
        pred_scores = outputs['instances'].scores
        
        # converts instance segmentation to individual masks and bbox
        # visualisations
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs['instances'].to("cpu"))
        seg_im = out.get_image()

        # cv2.imshow('img',display_img)
        if visualize:
            arr = [rgb_img, semantic_img, depth_img, seg_im]
            titles = ['rgb', 'semantic', 'depth', 'seg_im']
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
            if class_name not in self.ignore_classes:
            # if class_name in self.include_classes:
                obj_instance = self.sim.semantic_scene.objects[obj_id]
                # st()
                mask = np.zeros_like(semantic)
                mask[semantic == obj_id] = 1
                y, x = np.where(mask)
                pred_box = np.array([min(y), min(x), max(y), max(x)]) # ymin, xmin, ymax, xmax
                # print("Object name {}, Object category id {}, Object instance id {}".format(class_name, obj_instance['id'], obj_instance['class_id']))
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

        save_data = {'flat_view': flat_view, 'mainobj_id': mainobj_id, 'objects_info': object_list,'rgb_camX':rgb, 'depth_camX': depth, 'semantic_camX': semantic, 'agent_pos':agent_pos, 'agent_rot': agent_rot, 'sensor_pos': color_sensor_pos, 'sensor_rot': color_sensor_rot}
        
        with open(os.path.join(data_path, str(viewnum) + ".p"), 'wb') as f:
            pickle.dump(save_data, f)
        f.close()



    def is_valid_datapoint(self, observations):
        semantic = observations["semantic_sensor"]
        depth = observations["depth_sensor"]
        valid_views = []
        num_bad = 0
        # mindepth = np.min(depth)
        # depth = depths[view]
        not_ok = depth < 0.3
        yes_ok = depth > 0.3
        A = np.sum(yes_ok)
        B = np.sum(not_ok)
        if A < B*4: # view does not have space in front of it 
            # im = observations["color_sensor"]
            # plt.imshow(im)
            # plt_name = f'images/not_ok.png'
            # plt.savefig(plt_name)
            # st()
            return False

        else: # ok view
            return True

    def quaternion_from_two_vectors(self, v0: np.array, v1: np.array) -> np.quaternion:
        r"""Computes the quaternion representation of v1 using v0 as the origin."""
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
    

    def get_habitat_pix_T_camX(self, fov):
        hfov = float(self.fov) * np.pi / 180.
        pix_T_camX = np.array([
            [(self.sim_settings["width"]/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (self.sim_settings["height"]/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        return pix_T_camX

    def get_camera_matrix(self, width, height, fov):
        """Returns a camera matrix from image size and fov."""
        xc = (width - 1.) / 2.
        zc = (height - 1.) / 2.
        f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))
        camera_matrix = {'xc': xc, 'zc': zc, 'f': f}
        camera_matrix = Namespace(**camera_matrix)
        return camera_matrix

    def get_point_cloud_from_z(self, Y, camera_matrix, scale=1):
        """Projects the depth image Y into a 3D point cloud.
        Inputs:
            Y is ...xHxW
            camera_matrix
        Outputs:
            X is positive going right
            Y is positive into the image
            Z is positive up in the image
            XYZ is ...xHxWx3
        """
        x, z = np.meshgrid(np.arange(Y.shape[-1]),
                        np.arange(Y.shape[-2] - 1, -1, -1))
        for i in range(Y.ndim - 2):
            x = np.expand_dims(x, axis=0)
            z = np.expand_dims(z, axis=0)
        X = (x[::scale, ::scale] - camera_matrix.xc) * Y[::scale, ::scale] / camera_matrix.f
        Z = (z[::scale, ::scale] - camera_matrix.zc) * Y[::scale, ::scale] / camera_matrix.f
        XYZ = np.concatenate((X[..., np.newaxis], Y[::scale, ::scale][..., np.newaxis],
                            Z[..., np.newaxis]), axis=X.ndim)
        return XYZ

    def convert_points_to_topdown(self, pathfinder, points, meters_per_pixel):
        points_topdown = []
        bounds = pathfinder.get_bounds()
        for point in points:
            # convert 3D x,z to topdown x,y
            px = (point[0] - bounds[0][0]) / meters_per_pixel
            py = (point[2] - bounds[0][2]) / meters_per_pixel
            points_topdown.append(np.array([px, py]))
        return points_topdown


    # display a topdown map with matplotlib
    def display_map(self, topdown_map, tag, key_points=None, obs_points=None, obj_center=None, obj_points=None, objs_point_boxes=None):
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(1, 1, 1)
        ax.axis("off")
        plt.imshow(topdown_map)
        # plot points on map
        if obs_points is not None:
            for point in obs_points:
                plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8, color='blue')
        if objs_point_boxes is not None:
            for points in objs_point_boxes:
                points_np = np.array(points)
                plt.scatter(points_np[:,0], points_np[:,1], marker="o", linewidths=4, alpha=0.8, color='lightgrey')
        obj_center = np.array(obj_center)
        plt.plot(obj_center[:,0], obj_center[:,1], marker='o', markersize=10, alpha=0.8, color='red')
        key_points = np.array(key_points)
        plt.plot(key_points[:,0], key_points[:,1])
        plt_name = f'images/map{tag}.png'
        plt.savefig(plt_name)
        # plt.show(block=False)

    def update_b_ind(self, current_bin_ind, current_direction, num_bins):
        current_bin_ind += current_direction
        if current_bin_ind == 0:
            current_bin_ind = num_bins
        elif current_bin_ind > num_bins:
            current_bin_ind = 1
        return current_bin_ind
    
    def run2(self):
        scene = self.sim.semantic_scene
        # objects = scene.objects
        id = 0
        for region in scene.regions:
            if region.category.name() not in self.include_rooms:
                continue
            print("Room name is: ", region.category.name())
            self.object_labels.append(region.category.name())
            break
            # objects = region.objects

            # for obj in objects:
            #     if obj == None or obj.category == None or obj.category.name() in self.ignore_classes:
            #         continue
            #     self.object_labels.append(obj.category.name())
        
        # for level in scene.levels:
        #     print(
        #         f"Level id:{level.id}, center:{level.aabb.center},"
        #         f" dims:{level.aabb.sizes}"
        #     )
        #     for region in level.regions:
        #         print(
        #             f"Region id:{region.id}, category:{region.category.name()},"
        #             f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
        #         )
        # unique_objs = set(object_labels)



    def run(self):

        scene = self.sim.semantic_scene
        # objects = scene.objects
        id = 0
        
        for region in scene.regions:
            if region.category.name() not in self.include_rooms:
                continue
            print("Room name is: ", region.category.name())

            action = "move_forward"
            agent_state = self.agent.get_state()
            self.agent.set_state(agent_state)
            # val = self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount
            # self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount = 0.
            self.sim.step(action)
            # self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount = val
            
            region_center = region.aabb.center
            region_size = region.aabb.sizes
            xmin, xmax = region_center[0]-region_size[0]/2., region_center[0]+region_size[0]/2.
            ymin, ymax = region_center[1]-region_size[1]/2., region_center[1]+region_size[1]/2.
            zmin, zmax = region_center[2]-region_size[2]/2., region_center[2]+region_size[2]/2.
            if True:
                objects = region.objects
                object_centers = []
                object_categories = []
                for obj in objects:
                    if obj == None or obj.category == None or obj.category.name() in self.ignore_classes:
                        continue
                    # st()
                    if self.verbose:
                        print(f"Object name is: {obj.category.name()}")
                    # Calculate distance to object center
                    obj_center = obj.obb.to_aabb().center
                    object_centers.append(obj_center)
                    object_categories.append(obj.category.name())
                object_centers = np.array(object_centers)
            else:
                object_centers = self.get_midpoint_obj_conf()

            # print(object_centers)
            print("NUMBER OF OBJECTS: ", object_centers.shape)
            
            # for obj in objects:
            #     if obj == None or obj.category == None or obj.category.name() not in self.include_classes:
            #         continue
            #     obj_center = obj.obb.to_aabb().center
            #     #print(obj_center)
            #     obj_center = np.expand_dims(obj_center, axis=0)
            #     print("CENTER: ", obj_center)
            num_obj_save = 0
            for obj_idx in range(object_centers.shape[0]):
            # while count < self.num_objects_per_episode:
                # object_centers = self.get_midpoint_obj_conf()

                #Calculate distance to object center
                obj_center = object_centers[obj_idx]
                
                print(obj_center)
                obj_center = np.expand_dims(obj_center, axis=0)

                print("OBJECT CATEGORY IS: ", object_categories[obj_idx])
                print(self.nav_pts.shape)
                #print(obj_center)
                distances = np.sqrt(np.sum((self.nav_pts - obj_center)**2, axis=1))

                # Get points with r_min < dist < r_max
                valid_pts = self.nav_pts[np.where((distances > self.radius_min)*(distances<self.radius_max))]
                # stacked_check = np.vstack((valid_pts[:,0]>xmin, valid_pts[:,0]<xmax, valid_pts[:,1]>ymin, valid_pts[:,1]<ymax, valid_pts[:,2]>zmin, valid_pts[:,2]<zmax))
                # stacked_check = np.vstack((valid_pts[:,0]>xmin, valid_pts[:,0]<xmax, valid_pts[:,1]>ymin, valid_pts[:,1]<ymax))#, valid_pts[:,2]<zmin, valid_pts[:,2]<zmax))
                # valid_pts = valid_pts[np.all(stacked_check, axis=0)]
                if valid_pts.shape[0] == 0:
                    print("No valid points.. continuing")
                    continue
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

                # pitch calculation 
                dxdz_norm = np.sqrt((dx * dx) + (dz * dz))
                valid_pitch = np.degrees(np.arctan2(dy,dxdz_norm))

                # binning yaw around object
                nbins = 22
                bins = np.linspace(-180, 180, nbins+1)
                bin_yaw = np.digitize(valid_yaw, bins)

                num_valid_bins = np.unique(bin_yaw).size

                spawns_per_bin = int(self.num_views / num_valid_bins) + 1
                print(f'spawns_per_bin: {spawns_per_bin}')

                if False:
                    print("PLOTTING")
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
                    plt_name = 'images/samples.png'
                    plt.savefig(plt_name)
                
                action = "move_forward"
                # action = "move_forward"
                episodes = []
                valid_pts_selected = []
                cnt = 0
                positions_traveled_all = []
                obs_locations = []
                current_direction = 1 # move to "right" bins first
                current_bin_ind = int(np.random.randint(nbins)) #1
                # for b in range(nbins):
                    
                #     # # get all angle indices in the current bin range
                #     # # st()
                #     # inds_bin_cur = np.where(bin_yaw==(b+1)) # bins start 1 so need +1
                #     # inds_bin_cur = list(inds_bin_cur[0])
                #     # if len(inds_bin_cur) == 0:
                #     #     continue
                if False:
                    capture_output = False
                    first_capture = True
                    cap = 0
                else:
                    capture_output = True
                    first_capture = True

                aleady_changed = False

                #     for s in range(spawns_per_bin):
                if True:
                    num_iters = self.num_views + self.num_views 

                    for idx_s in range(num_iters):
                
                        # st()
                        # if len(inds_bin_cur) == 0:
                        #     continue
                        
                        # Get next step
                        current_bin_ind_prev = current_bin_ind
                        current_bin_ind = self.update_b_ind(current_bin_ind, current_direction, nbins)
                        inds_bin_cur = np.where(bin_yaw==current_bin_ind) # bins start 1 so need +1
                        inds_bin_cur = list(inds_bin_cur[0])
                        if cnt==0:
                            num_check = nbins
                        else:
                            num_check = 3
                        if len(inds_bin_cur) == 0:
                            for i in range(num_check):
                                current_bin_ind = self.update_b_ind(current_bin_ind, current_direction, nbins)
                                inds_bin_cur = np.where(bin_yaw==current_bin_ind) # bins start 1 so need +1
                                inds_bin_cur = list(inds_bin_cur[0])
                                if len(inds_bin_cur) > 0:
                                    break
                        if len(inds_bin_cur) == 0 and aleady_changed:
                            current_bin_ind = int(np.random.randint(nbins))
                            aleady_changed = False
                            # print(2)
                            if False: # for plotting
                                if first_capture:
                                    capture_output = True
                                else:
                                    capture_output = False
                            continue
                            # switch searching direction
                        elif len(inds_bin_cur) == 0:
                            aleady_changed = True
                            current_bin_ind = current_bin_ind_prev
                            current_direction = -current_direction
                            # print(1)
                            if False:
                                if first_capture:
                                    capture_output = True
                                else:
                                    capture_output = False
                            # print(capture_output)
                            continue
                        aleady_changed = False
                        # print(capture_output)

                        if first_capture and capture_output:
                            first_capture = False

                        
                        rand_ind = np.random.randint(0, len(inds_bin_cur))
                        s_ind = inds_bin_cur.pop(rand_ind)

                        pos_s = valid_pts[s_ind]
                        # valid_pts_selected.append(pos_s)
                        # pos_s = pos_s + np.array([0, 1.5, 0])
                        # # get valid point closest
                        # pos_s = valid_pts[np.argmin(np.linalg.norm(valid_pts[:,[True, False, True]] - pos_s[[True, False, True]], axis=1))]
                        # if np.all(agent_state.position==np.zeros(3)):
                        #     agent_state.position = pos_s
                        #     self.agent.set_state(agent_state)
                        if cnt==0:
                            agent_state = habitat_sim.AgentState()
                            agent_state.position = pos_s
                            self.agent.set_state(agent_state)
                        # val = self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount
                        # self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount = 0.
                        # self.sim.step(action)
                        # self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount = val

                        valid_pts_selected.append(pos_s)
                        # pos_s = pos_s + np.array([0, 1.5, 0])
                        # pos_s[1] = y_height_fixed # keep y height constant as much as possible
                        # agent_state.position = pos_s

                        # # initiate agent
                        # self.agent.set_state(agent_state)
                        # self.sim.step(action)
                        
                        ### Execute actions from path_finder #########
                        # agent_state = self.agent.get_state()
                        # print("BEFORE", agent_state.position)
                        # print("ROT", agent_state.rotation)
                        try:
                            actions_path = self.path_finder.find_path(pos_s)
                        except:
                            print("ACTION PATH FAILED")
                            continue
                        for action_path in actions_path:
                            if action_path is None:
                                continue
                            self.sim.step(action_path)
                            if action_path=="move_forward":
                                agent_state_cur = self.agent.get_state()
                                if capture_output:
                                    positions_traveled_all.append(agent_state_cur.position)
                        if actions_path[0] == None:
                            agent_state_cur = self.agent.get_state()
                            if capture_output:
                                positions_traveled_all.append(agent_state_cur.position)
                        if capture_output:
                            obs_locations.append(agent_state_cur.position)
                        
                        # obs_locations.append(agent_state_cur.position)
                        agent_state = self.agent.get_state()
                        # print("AFTER", agent_state.position)
                        # print("ROT", agent_state.rotation)
                        # print(self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount)

                        # agent_state.position = pos_s
                        self.agent.set_state(agent_state)
                        val = self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount
                        self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount = 0.
                        self.sim.step(action)
                        self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount = val

                        # agent_state = self.agent.get_state()
                        # print("AFTER", agent_state.position)
                        # print("ROT", agent_state.rotation)

                        ##############################################


                        # agent_state = habitat_sim.AgentState()
                        # agent_state.position = pos_s + np.array([0, 1.5, 0])

                        # agent_state = habitat_sim.AgentState()
                        # self.agent.set_state(agent_state)


                        # YAW calculation - rotate to object
                        # agent_state = self.agent.get_state()
                        agent_to_obj = np.squeeze(obj_center) - (agent_state.position + np.array([0, 1.5, 0]))
                        agent_local_forward = np.array([0, 0, -1.0]) # y, z, x
                        flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
                        flat_dist_to_obj = np.linalg.norm(flat_to_obj)
                        flat_to_obj /= flat_dist_to_obj

                        det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
                        turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))
                        quat_yaw = quat_from_angle_axis(turn_angle, np.array([0, 1.0, 0]))

                        # Set agent yaw rotation to look at object
                        # agent_state = self.agent.get_state()
                        agent_state.rotation = quat_yaw
                        # self.agent.set_state(agent_state)
                        # print("ROT", agent_state.rotation)
                        
                        # change sensor state to default 
                        # need to move the sensors too
                        # print(self.agent.state.sensor_states)
                        for sensor in self.agent.state.sensor_states:
                            # st()
                            self.agent.state.sensor_states[sensor].rotation = agent_state.rotation
                            self.agent.state.sensor_states[sensor].position = agent_state.position + np.array([0, 1.5, 0]) # ADDED IN UP TOP
                            # print("PRINT", self.agent.state.sensor_states[sensor].rotation)

                        # Calculate Pitch from head to object
                        turn_pitch = np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj))
                        num_turns = np.abs(np.floor(turn_pitch/self.rot_interval)).astype(int) # compute number of times to move head up or down by rot_interval
                        # print("MOVING HEAD ", num_turns, " TIMES")
                        movement = "look_up" if turn_pitch>0 else "look_down"

                        # # initiate agent
                        # # self.agent.set_state(agent_state)
                        # # agent_state = habitat_sim.AgentState()
                        self.agent.set_state(agent_state)
                        val = self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount
                        self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount = 0.
                        self.sim.step(action)
                        self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount = val

                        # # initiate agent
                        # self.agent.set_state(agent_state)
                        # self.sim.step(action)

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
                        
                        # # get observations after centiering
                        val = self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount
                        self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount = 0.
                        observations = self.sim.step(action)
                        self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount = val
                        # print("ROT", agent_state.rotation)

                        # # get observations after centiering
                        # observations = self.sim.step(action)
                        
                        # Assuming all sensors have same rotation and position
                        observations["rotations"] = self.agent.state.sensor_states['color_sensor'].rotation #agent_state.rotation
                        observations["positions"] = self.agent.state.sensor_states['color_sensor'].position

                        if self.is_valid_datapoint(observations):
                            if self.verbose:
                                print("episode is valid......")
                            episodes.append(observations)

                        if self.visualize: 
                            im = observations["color_sensor"]
                            im = Image.fromarray(im, mode="RGBA")
                            im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
                            plt.imshow(im)
                            # plt.show()
                            plt.savefig(f'images/test{s}_{b}.png')

                        # agent_state = self.agent.get_state()
                        # print("AFTER", agent_state.position)

                        cnt +=1

                if self.gen_maps and len(positions_traveled_all)>0:
                    # convert 3d points to 2d topdown coordinates
                    map_count += 1
                    height = 1
                    meters_per_pixel = 0.01
                    positions_traveled_all = np.array(positions_traveled_all)
                    obs_locations = np.array(obs_locations)
                    h_y = np.mean(valid_pts, axis=0)[1]
                    top_down_map = maps.get_topdown_map(
                        self.sim.pathfinder, h_y, meters_per_pixel=meters_per_pixel, draw_border=False
                        )
                    # top_down_map = maps.get_topdown_map(
                    #     self.sim.pathfinder, h_y, map_resolution=1024, draw_border=False
                    #     )
                        
                    top_down_map_filled = binary_fill_holes(top_down_map).astype(int)
                    tdm_diff = top_down_map_filled - top_down_map
                    top_down_map[tdm_diff==1] = 2
                    vis_points = positions_traveled_all #np.unique(positions_traveled_all, axis=0)
                    xy_vis_points = self.convert_points_to_topdown(
                        self.sim.pathfinder, vis_points, meters_per_pixel
                    )
                    xy_obs_points = self.convert_points_to_topdown(
                        self.sim.pathfinder, obs_locations, meters_per_pixel
                    )
                    obj_center_vis_points = self.convert_points_to_topdown(
                        self.sim.pathfinder, obj_center, meters_per_pixel
                    )
                    # objs_point_boxes = []
                    # for obj in objects:
                    #     if obj == None or obj.category == None or obj.category.name() in self.ignore_classes: #not in self.include_classes:
                    #         continue
                    #     bbox_size = obj.obb.sizes
                    #     bbox_center = obj.obb.center
                    #     xmin, xmax = bbox_center[0]-bbox_size[0]/2., bbox_center[0]+bbox_size[0]/2.
                    #     ymin, ymax = bbox_center[2]-bbox_size[2]/2., bbox_center[2]+bbox_size[2]/2.
                    #     x = np.arange(xmin, xmax, 0.1)
                    #     y = np.arange(ymin, ymax, 0.1)
                    #     xx, yy = np.meshgrid(x, y)
                    #     xx = xx.flatten()
                    #     yy = yy.flatten()
                    #     obj_box_locs = np.vstack((xx, np.ones(xx.shape[0])*h_y, yy)).T
                    #     xy_points_obj = self.convert_points_to_topdown(
                    #         self.sim.pathfinder, obj_box_locs, meters_per_pixel
                    #     )
                    #     objs_point_boxes.append(xy_points_obj)


                    recolor_map = np.array(
                        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
                    )
                    top_down_map = recolor_map[top_down_map]
                    tag = os.path.split(self.basepath)[-1] + '_' + str(map_count)
                    self.display_map(top_down_map, tag=tag, key_points=xy_vis_points, obs_points=xy_obs_points, obj_center=obj_center_vis_points, objs_point_boxes=None)
                            
                if self.save_this:
                    if len(episodes) >= self.num_views:
                        print(f'num episodes: {len(episodes)}')
                        data_folder = f'{num_obj_save}' #obj.category.name() + '_' + obj.id
                        data_path = os.path.join(self.basepath, data_folder)
                        print("Saving to ", data_path)
                        os.mkdir(data_path)
                        # flat_obs = np.random.choice(episodes, self.num_views, replace=False)
                        flat_obs = episodes[:self.num_views]
                        viewnum = 0
                        for obs in flat_obs:
                            self.save_datapoint(self.agent, obs, data_path, viewnum, None, True)
                            viewnum += 1
                        num_obj_save += 1
                    else:
                        print(f"Not enough episodes: f{len(episodes)}")
                
                # if num_obj_save > 15:
                #     break

                if self.visualize:
                    valid_pts_selected = np.vstack(valid_pts_selected)
                    self.plot_navigable_points(valid_pts_selected)

    def get_midpoint_obj_conf(self):
        
        scene = self.sim.semantic_scene
        objects = scene.objects
        print(objects)
        #objects = random.sample(list(objects), self.num_objects_per_episode)
        xyz_obj_mids = []
        print(self.num_objects_per_episode)
        print(len(objects))
        count = 0
        obj_ind = 0
        while count < self.num_objects_per_episode:
            print("GETTING OBJECT #", count)
            if obj_ind >= len(objects):
                obj_ind = 0
                # break
            # obj_ind = np.random.randint(low = 0, high = len(objects))
            obj = objects[obj_ind]
            obj_ind += 1
            if obj == None or obj.category == None or obj.category.name() not in self.include_classes:
                continue
            # st()
            # if self.verbose:
            print(f"Object name is: {obj.category.name()}")
            # Calculate distance to object center
            obj_center = obj.obb.to_aabb().center
            #print(obj_center)
            obj_center = np.expand_dims(obj_center, axis=0)
            #print(obj_center)
            distances = np.sqrt(np.sum((self.nav_pts - obj_center)**2, axis=1))

            # Get points with r_min < dist < r_max
            valid_pts = self.nav_pts[np.where((distances > self.radius_min)*(distances<self.radius_max))]


            # Bin points based on angles [vertical_angle (10 deg/bin), horizontal_angle (10 deg/bin)]
            valid_pts_shift = valid_pts - obj_center

            dz = valid_pts_shift[:,2]
            dx = valid_pts_shift[:,0]
            dy = valid_pts_shift[:,1]

            # Get yaw for binning 
            valid_yaw = np.degrees(np.arctan2(dz,dx))

            # pitch calculation 
            dxdz_norm = np.sqrt((dx * dx) + (dz * dz))
            valid_pitch = np.degrees(np.arctan2(dy,dxdz_norm))

            # binning yaw around object
            nbins = 7
            bins = np.linspace(-180, 180, nbins+1)
            bin_yaw = np.digitize(valid_yaw, bins)

            num_valid_bins = np.unique(bin_yaw).size
            
            action = "do_nothing"
            episodes = []
            valid_pts_selected = []
            bin_inds = list(range(nbins))
            bin_inds = random.sample(bin_inds, len(bin_inds))
            
            for b in bin_inds: #b_inds_notempty:

                inds_bin_cur = np.where(bin_yaw==(b+1)) # bins start 1 so need +1
                if inds_bin_cur[0].size == 0:
                    continue
                
                # get all angle indices in the current bin range
                # st()
                inds_bin_cur = np.where(bin_yaw==(b+1)) # bins start 1 so need +1

                # st()
                s_ind = np.random.choice(inds_bin_cur[0])
                #s_ind = inds_bin_cur[0][0]
                pos_s = valid_pts[s_ind]
                valid_pts_selected.append(pos_s)
                agent_state = habitat_sim.AgentState()
                agent_state.position = pos_s + np.array([0, 1.5, 0])


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
                    # print("PRINT", self.agent.state.sensor_states[sensor].rotation)


                # NOTE: for finding an object, i dont think we'd want to center it
                # Calculate Pitch from head to object
                turn_pitch = np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj))
                num_turns = np.abs(np.floor(turn_pitch/self.rot_interval)).astype(int) # compute number of times to move head up or down by rot_interval
                print("MOVING HEAD ", num_turns, " TIMES")
                movement = "look_up" if turn_pitch>0 else "look_down"

                # initiate agent
                val = self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount
                self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount = 0.
                self.agent.set_state(agent_state)
                self.sim.step(action)
                self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount = val

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
                val = self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount
                self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount = 0.
                observations = self.sim.step(action)
                self.sim.agents[0].agent_config.action_space["move_forward"].actuation.amount = val

                print(self.agent.state.sensor_states)
                for sensor in self.agent.state.sensor_states:
                    # st()
                    self.agent.state.sensor_states[sensor].rotation = agent_state.rotation
                    self.agent.state.sensor_states[sensor].position = agent_state.position # + np.array([0, 1.5, 0]) # ADDED IN UP TOP

                ####### %%%%%%%%%%%%%%%%%%%%%%% ######### MASK RCNN

                im = observations["color_sensor"]
                im = Image.fromarray(im, mode="RGBA")
                im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

                # plt.imshow(im)
                # plt.show()

                outputs = self.maskrcnn(im)

                v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg_det.DATASETS.TRAIN[0]), scale=1.2)
                out = v.draw_instance_predictions(outputs['instances'].to("cpu"))
                seg_im = out.get_image()

                if False:
                    plt.figure(1)
                    plt.clf()
                    plt.imshow(seg_im)
                    plt.savefig('images/test.png')
                
                pred_masks = outputs['instances'].pred_masks
                pred_boxes = outputs['instances'].pred_boxes.tensor
                pred_classes = outputs['instances'].pred_classes
                pred_scores = outputs['instances'].scores

                maskrcnn_to_catname = {56: "chair", 59: "bed", 61: "toilet", 57: "couch", 58: "indoor-plant", 72: "refrigerator", 62: "tv", 60: "dining-table"}

                obj_ids = []
                obj_catids = []
                obj_scores = []
                obj_masks = []
                obj_all_catids = []
                obj_all_scores = []
                obj_all_boxes = []
                for segs in range(len(pred_masks)):
                    if pred_classes[segs].item() in maskrcnn_to_catname:
                        if pred_scores[segs] >= 0.70:
                            obj_ids.append(segs)
                            obj_catids.append(pred_classes[segs].item())
                            obj_scores.append(pred_scores[segs].item())
                            obj_masks.append(pred_masks[segs])

                            obj_all_catids.append(pred_classes[segs].item())
                            obj_all_scores.append(pred_scores[segs].item())
                            y, x = torch.where(pred_masks[segs])
                            pred_box = torch.Tensor([min(y), min(x), max(y), max(x)]) # ymin, xmin, ymax, xmax
                            obj_all_boxes.append(pred_box)

                print("MASKS ", len(pred_masks))
                print("VALID ", len(obj_scores))
                print(obj_scores)
                print(pred_scores.shape)

                translation_ = self.agent.state.sensor_states['depth_sensor'].position
                quaternion_ = self.agent.state.sensor_states['depth_sensor'].rotation
                rotation_ = quaternion.as_rotation_matrix(quaternion_)
                T_world_cam = np.eye(4)
                T_world_cam[0:3,0:3] =  rotation_
                T_world_cam[0:3,3] = translation_

                if not obj_masks:
                    continue
                else: 

                    # randomly choose a high confidence object
                    # instead of this I think we should iterate over ALL the high confident objects and fixate on them
                    obj_mask_focus = random.choice(obj_masks)

                    # if True:
                    #     plt.figure(1)
                    #     plt.clf()
                    #     plt.imshow(obj_mask_focus)
                    #     plt.savefig('images/test.png')
                    # st()

                    depth = observations["depth_sensor"]

                    xs, ys = np.meshgrid(np.linspace(-1*256/2.,1*256/2.,256), np.linspace(1*256/2.,-1*256/2., 256))
                    depth = depth.reshape(1,256,256)
                    xs = xs.reshape(1,256,256)
                    ys = ys.reshape(1,256,256)

                    xys = np.vstack((xs * depth , ys * depth, -depth, np.ones(depth.shape)))
                    xys = xys.reshape(4, -1)
                    xy_c0 = np.matmul(np.linalg.inv(self.K), xys)
                    xyz = xy_c0.T[:,:3].reshape(256,256,3)
                    xyz_obj_masked = xyz[obj_mask_focus]

                    xyz_obj_masked = np.matmul(rotation_, xyz_obj_masked.T) + translation_.reshape(3,1)
                    xyz_obj_mid = np.mean(xyz_obj_masked, axis=1)

                    print("MIDPOINT=", xyz_obj_mid)

                    xyz_obj_mids.append(xyz_obj_mid)

                    count += 1

                    break # got an object

        xyz_obj_mids = np.array(xyz_obj_mids)

        return xyz_obj_mids

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