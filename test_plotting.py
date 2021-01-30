
import numpy as np
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os 
import sys
import glob, os.path
from PIL import Image

import ipdb
st = ipdb.set_trace

plt.figure(0)
a = np.load("/home/nel/gsarch/replica_orbslam_noisy_depth_pose/room_2_0/chair__3/0.p", allow_pickle=True)
im = a['rgb_camX']
im = Image.fromarray(im, mode="RGBA")
plt.imshow(im)
# plt.show()

plt.figure(1)
a = np.load("/home/nel/gsarch/replica_orbslam_noisy_depth_pose/room_2_0/chair__3/10.p", allow_pickle=True)
im = a['rgb_camX']
im = Image.fromarray(im, mode="RGBA")
plt.imshow(im)
# plt.show()

plt.figure(2)
a = np.load("/home/nel/gsarch/replica_orbslam_noisy_depth_pose/room_2_0/chair__3/20.p", allow_pickle=True)
im = a['rgb_camX']
im = Image.fromarray(im, mode="RGBA")
plt.imshow(im)
plt.show()

# def safe_inverse_single(a):
#     r, t = split_rt_single(a)
#     t = np.reshape(t, (3,1))
#     r_transpose = r.T
#     inv = np.concatenate([r_transpose, -np.matmul(r_transpose, t)], 1)
#     bottom_row = a[3:4, :] # this is [0, 0, 0, 1]
#     # bottom_row = torch.tensor([0.,0.,0.,1.]).view(1,4) 
#     inv = np.concatenate([inv, bottom_row], 0)
#     return inv

# def split_rt_single(rt):
#     r = rt[:3, :3]
#     t = np.reshape(rt[:3, 3], 3)
#     return r, t

# # if os.path.exists('/home/nel/gsarch/habitat/habitat-lab/HabitatScripts/maps'):
# #     dir_name = "/home/nel/gsarch/habitat/habitat-lab/HabitatScripts/maps/"
# #     test = os.listdir(dir_name)
# #     for item in test:
# #         if item.endswith(".png"):
# #             os.remove(os.path.join(dir_name, item))

# camXs_T_camX0_orb_output = np.load("camXs_T_camX0_orb_output.npy")
# origin_T_camX0_4x4 = np.load("origin_T_camX0_4x4.npy")
# camXs_T_camX0_4x4 = np.load("camXs_T_camX0_4x4.npy")
# pos = np.load("pos.npy")


# episodes = {}
# for i in range(camXs_T_camX0_orb_output.shape[0]):
#     episodes[i] = {}

# ep_idx = 1
# plt_id = 1
# # get ORBSLAM estimated rot
# camXs_T_camX0_quant = []
# camXs_T_camX0 = []
# for i in range(camXs_T_camX0_orb_output.shape[0]):
#     cur = camXs_T_camX0_orb_output[i]
#     camX_T_camX0_quant = np.quaternion(cur[7], cur[4], cur[5], cur[6])
#     camX_T_camX0 = quaternion.as_rotation_matrix(camX_T_camX0_quant) # Need to negative rotation because axes are flipped 
#     camXs_T_camX0_quant.append(camX_T_camX0_quant)
#     camXs_T_camX0.append(camX_T_camX0)
# camXs_T_camX0 = np.array(camXs_T_camX0)
# camXs_T_camX0_quant = np.array(camXs_T_camX0_quant)

# # get ORBSLAM estimated pos
# t = []
# for i in range(camXs_T_camX0_orb_output.shape[0]):
#     cur = camXs_T_camX0_orb_output[i]
#     # t_cur = np.array([-cur[1], cur[3], cur[2]]) 
#     t_cur = np.array([-cur[1], -cur[2], -cur[3]])        # i think x shouldn't be inverted
#     t.append(t_cur)
# t = np.array(t)

# # adjust y
# # st()

# # Get 4x4 matrix and convert to origin coords
# camXs_T_camX0_4x4_orb = []
# for i in range(camXs_T_camX0_orb_output.shape[0]):
#     # assert len(episodes) == camXs_T_camX0_orb_output.shape[0], f"{camXs_T_camX0_orb_output.shape[0]}, {len(episodes)}"
#     # get estimated 4x4
#     camX_T_camX0_4x4_orb = np.eye(4)
#     camX_T_camX0_4x4_orb[0:3, 0:3] = camXs_T_camX0[i]
#     camX_T_camX0_4x4_orb[:3,3] = t[i]

#     # invert
#     camX0_T_camX_4x4_orb = safe_inverse_single(camX_T_camX0_4x4_orb)

#     # convert to origin coordinates
#     origin_T_camX_4x4 = np.matmul(origin_T_camX0_4x4, camX0_T_camX_4x4_orb)
#     r_origin_T_camX_orb, t_origin_T_camX_orb = split_rt_single(origin_T_camX_4x4)
#     r_origin_T_camX_orb_quat = quaternion.from_rotation_matrix(r_origin_T_camX_orb)

#     #save
#     episodes[i]["positions_orb"] = t_origin_T_camX_orb
#     episodes[i]["rotations_orb"] = r_origin_T_camX_orb_quat
#     camXs_T_camX0_4x4_orb.append(camX_T_camX0_4x4_orb)
# camXs_T_camX0_4x4_orb = np.array(camXs_T_camX0_4x4_orb)

# ## PLOTTTING #########
# print("PLOTTING")
# plot_camX0_traj = True
# plot_origin_traj = True

# if plot_camX0_traj:
#     x_orbslam_camX0 = []
#     y_orbslam_camX0 = []
#     z_orbslam_camX0 = []
#     for i in range(camXs_T_camX0_orb_output.shape[0]):
#         camX_T_camX0_4x4_orb_r, camX_T_camX0_4x4_orb_t = split_rt_single(camXs_T_camX0_4x4_orb[i])
#         x_orbslam_camX0.append(camX_T_camX0_4x4_orb_t[0])
#         y_orbslam_camX0.append(camX_T_camX0_4x4_orb_t[1])
#         z_orbslam_camX0.append(camX_T_camX0_4x4_orb_t[2])
#     x_orbslam_camX0 = np.array(x_orbslam_camX0)
#     y_orbslam_camX0 = np.array(y_orbslam_camX0)
#     z_orbslam_camX0 = np.array(z_orbslam_camX0)

#     np.vstack(((x_orbslam_camX0, y_orbslam_camX0, z_orbslam_camX0))).T

#     x_gt_camX0 = []
#     y_gt_camX0 = []
#     z_gt_camX0 = []
#     for i in range(camXs_T_camX0_orb_output.shape[0]):
#         camX_T_camX0_4x4_r, camX_T_camX0_4x4_t = split_rt_single(camXs_T_camX0_4x4[i])
#         x_gt_camX0.append(camX_T_camX0_4x4_t[0])
#         y_gt_camX0.append(camX_T_camX0_4x4_t[1])
#         z_gt_camX0.append(camX_T_camX0_4x4_t[2])
#     x_gt_camX0 = np.array(x_gt_camX0)
#     y_gt_camX0 = np.array(y_gt_camX0)
#     z_gt_camX0 = np.array(z_gt_camX0)

#     np.vstack(((x_gt_camX0, y_gt_camX0, z_gt_camX0))).T

#     # st()



#     plt.figure()
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#     ax1.plot(y_orbslam_camX0, x_orbslam_camX0, label='ORB-SLAM-yx', color='green', linestyle='dashed')
#     ax1.plot(y_gt_camX0, x_gt_camX0, label='GT', color='blue', linestyle='solid')
#     ax1.legend()
#     ax2.plot(y_orbslam_camX0, z_orbslam_camX0, label='ORB-SLAM-yz', color='green', linestyle='dashed')
#     ax2.plot(y_gt_camX0, z_gt_camX0, label='GT', color='blue', linestyle='solid')
#     ax2.legend()
#     ax3.plot(x_orbslam_camX0, z_orbslam_camX0, label='ORB-SLAM-xz', color='green', linestyle='dashed')
#     ax3.plot(x_gt_camX0, z_gt_camX0, label='GT', color='blue', linestyle='solid')
#     ax3.legend()
#     plt_name = 'maps/' + str(ep_idx) + '_' + str(plt_id) + '_' + 'camX0_' + 'map.png'
#     plt.savefig(plt_name)

#     plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.plot3D(x_orbslam_camX0, y_orbslam_camX0, z_orbslam_camX0, 'green')
#     ax.plot3D(x_gt_camX0, y_gt_camX0, z_gt_camX0, 'blue')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     plt_name = 'maps/' + str(ep_idx) + '_' + str(plt_id) + '_' + 'camX0_3D_' + 'map.png'
#     plt.savefig(plt_name)


# if plot_origin_traj:

#     x_orbslam = []
#     y_orbslam = []
#     z_orbslam = []
#     for i in range(camXs_T_camX0_orb_output.shape[0]):
#         x_orbslam.append(episodes[i]["positions_orb"][0])
#         y_orbslam.append(episodes[i]["positions_orb"][1])
#         z_orbslam.append(episodes[i]["positions_orb"][2])
#     x_orbslam = np.array(x_orbslam)
#     y_orbslam = np.array(y_orbslam)
#     z_orbslam = np.array(z_orbslam)
    
#     x_gt = []
#     y_gt = []
#     z_gt = []
#     for i in range(camXs_T_camX0_orb_output.shape[0]):
#         x_gt.append(pos[i][0])
#         y_gt.append(pos[i][1])
#         z_gt.append(pos[i][2])
#     x_gt = np.array(x_gt)
#     y_gt = np.array(y_gt)
#     z_gt = np.array(z_gt)
    
#     plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.plot3D(x_orbslam, y_orbslam, z_orbslam, 'green')
#     ax.plot3D(x_gt, y_gt, z_gt, 'blue')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     # plt.plot(y_orbslam, x_orbslam, label='ORB-SLAM', color='green', linestyle='dashed')
#     # plt.plot(y_gt, z_gt, label='GT', color='blue', linestyle='solid')
#     # plt.plot(-np.array(x_orbslam), np.array(y_orbslam), label='ORB-SLAM', color='green', linestyle='dashed')
#     # plt.plot(x_gt, y_gt, label='GT', color='blue', linestyle='solid')
#     # plt.legend()
#     plt_name = 'maps/' + str(ep_idx) + '_' + str(plt_id) + '_' + 'origin_' + 'map.png'
#     plt.savefig(plt_name)


