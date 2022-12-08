# imports
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# function to detect impact points
def detect_impact_points(pose_x, pose_y, pose_c, time_stamps, limb_indices, gm_thresh=[3, 3, 1, 1], ggm_thresh=[-0.4, -0.4, -0.1, -0.1], fps=60, fig_dir=None):
    assert len(pose_x) == len(pose_y) and len(pose_x) == len(pose_c) and len(pose_x) == len(time_stamps)
    
    # detect impact points for each limb
    n_frames = len(time_stamps)
    n_limbs = len(limb_indices)
    impact_points = np.zeros((n_frames, n_limbs), dtype=bool)
    for i, limb in enumerate(limb_indices):
        x = pose_x[:,limb]
        y = pose_y[:,limb]
        c = pose_c[:,limb] # just a confidence value, probably not necessary
        
        # parameters
        sigma_0 = 5
        sigma_1 = 3
        peak_distance = 10
        
        # smooth coordinates
        x_filt = gaussian_filter1d(x, sigma_0)
        y_filt = gaussian_filter1d(y, sigma_0)
        
        # gradients
        x_grad = np.gradient(x_filt)
        y_grad = np.gradient(y_filt)
        grad_mag = np.sqrt(x_grad**2 + y_grad**2)
        grad_mag_filt = gaussian_filter1d(grad_mag, sigma_1)
        grad_grad_mag = np.gradient(grad_mag_filt)
        
        # impact detection: troughs in ggm that exceed gm threshold
        impacts_idx, _ = find_peaks(-grad_grad_mag, height=-ggm_thresh[i], distance=peak_distance)
        impacts_idx = impacts_idx[grad_mag_filt[impacts_idx] >= gm_thresh[i]]
        
        limb_impacts = np.zeros((n_frames), dtype=bool)
        limb_impacts[impacts_idx] = True
        impact_points[:,i] = limb_impacts
        
        # plot
        if fig_dir:
            start_second = 0
            num_seconds = 5
            start_i = int(fps * start_second)
            end_i = int(start_i + fps * num_seconds)
            
            t = np.arange(start_i, end_i) / fps
            impacts_plot = impacts_idx[np.logical_and(impacts_idx >= start_i, impacts_idx <= end_i)]
            fig = plt.figure(figsize=(15, 4), facecolor='w')
            
            plt.subplot(1, 3, 1)
            plt.plot(t, x[start_i:end_i], label='X')
            plt.plot(t, y[start_i:end_i], label='Y')
            plt.plot(t, x_filt[start_i:end_i], label='Smooth X')
            plt.plot(t, y_filt[start_i:end_i], label='Smooth Y')
            plt.plot(impacts_plot / fps, x_filt[impacts_plot], 'x', label='Impacts')
            plt.plot(impacts_plot / fps, y_filt[impacts_plot], 'x', label='Impacts')
            plt.title(f'Joint {limb} Pose Estimates')
            plt.xlabel('Time [s]')
            plt.ylabel('Coordinate')
            plt.xlim(start_second, start_second + num_seconds)
            plt.legend()

            plt.subplot(1, 3, 2)
            plt.plot(t, grad_mag[start_i:end_i], label='Grad magnitude')
            plt.plot(t, grad_mag_filt[start_i:end_i], label='Smooth grad magnidue')          
            plt.plot(impacts_plot / fps, grad_mag_filt[impacts_plot], 'x', label='Impacts')
            plt.title(f'Joint {limb} Gradient Magnitude of Pose')
            plt.xlabel('Time [s]')
            plt.ylabel('Magnitude')
            plt.xlim(start_second, start_second + num_seconds)
            plt.legend()

            plt.subplot(1, 3, 3)
            plt.plot(t, grad_grad_mag[start_i:end_i], label='Grad of grad magnitude')            
            plt.plot(impacts_plot / fps, grad_grad_mag[impacts_plot], 'x', label='Impacts')
            plt.title(f'Joint {limb} Gradient of Gradient Magnitude')
            plt.xlabel('Time [s]')
            plt.ylabel('Value')
            plt.xlim(start_second, start_second + num_seconds)
            plt.legend()
            plt.tight_layout()
            # plt.show(block=False)
            fig.savefig(os.path.join(fig_dir, f'joint_{limb}_impacts.png'))
    
    assert len(impact_points) == len(time_stamps)
    return impact_points, time_stamps


# function to quantize impact points
def quantize_impact_points(impact_points, time_stamps, quant_params):
    assert len(impact_points) == len(time_stamps)
    n_frames, n_limbs = impact_points.shape
    
    # quantize time stamps
    dt = 60 / (quant_params['bpm'] * quant_params['subdiv'])
    quantized_time_stamps = np.round((time_stamps - quant_params['beat_ref']) / dt) * dt + quant_params['beat_ref']
    impact_points[quantized_time_stamps < 0] = np.zeros((n_limbs), dtype=bool)
    quantized_time_stamps[quantized_time_stamps < 0] = 0
    
    # format quantized impacts
    quantized_impacts = []
    for l in range(n_limbs):
        quantized_impacts.append(np.unique(quantized_time_stamps[impact_points[:,l]]))
    
    return quantized_impacts
