# imports
import os
import numpy as np
import json


# function to interpolate missing pose estimates
# https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
def interpolate_pose_estimates(a):
    found = a != 0
    not_found = a == 0
    xp = found.nonzero()[0]
    fp = a[found]
    x  = not_found.nonzero()[0]

    a[not_found] = np.interp(x, xp, fp)
    return a


# function to load pose estimates
def load_pose_estimates(load_dir, vid_name, fps):
    json_files = sorted(os.listdir(os.path.join(load_dir, vid_name)))
    json_paths = [(int(json_file.split('_')[1]), os.path.join(load_dir, vid_name, json_file)) for json_file in json_files if '.json' in json_file]
    n_frames = len(json_paths)
    
    pose_x = np.empty((n_frames, 18))
    pose_y = np.empty((n_frames, 18))
    pose_c = np.empty((n_frames, 18))
    time_stamps = np.empty((n_frames))
    for frm, json_path in json_paths: 
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
        x = json_dict['people'][0]['pose_keypoints_2d'][0::3]
        y = json_dict['people'][0]['pose_keypoints_2d'][1::3]
        c = json_dict['people'][0]['pose_keypoints_2d'][2::3]
        pose_x[frm] = x
        pose_y[frm] = y
        pose_c[frm] = c
        time_stamps[frm] = frm/fps
        
    # interpolate missing pose estimates
    for limb in range(18):
        pose_x[:,limb] = interpolate_pose_estimates(pose_x[:,limb])
        pose_y[:,limb] = interpolate_pose_estimates(pose_y[:,limb])
        
    print(f'Interpolation correct? {(pose_x != 0).all() and (pose_y != 0).all()}')
        
    return pose_x, pose_y, pose_c, time_stamps
