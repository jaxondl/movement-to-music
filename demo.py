# imports
import os
from functions.load_poses import load_pose_estimates
from functions.impact_detection import detect_impact_points, quantize_impact_points
from functions.audio_construction import strip_audio, construct_audio, combine_audio_video

# parameters for demo
vid_name = 'j2'
limb_indices = [4, 7, 10, 13] # hands and feet indices in COCO format
input_dir = os.path.join('demo_files', 'input')
output_dir = os.path.join('demo_files', 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# print start
print(50 * '-')
print(f'Demo for video {vid_name}')

# load pose estimates
print('\nLoading pose estimates...')
pose_fps = 60
pose_load_dir = os.path.join('data', 'output_jsons')
pose_x, pose_y, pose_c, time_stamps = load_pose_estimates(pose_load_dir, vid_name, pose_fps)

# detect and save impact points and timestamps
print('\nDetecting impact points...')
impact_points, time_stamps = detect_impact_points(pose_x, pose_y, pose_c, time_stamps, limb_indices, fps=pose_fps, fig_dir=output_dir)

# quantize impact points
print('\nQuantizing impact points...')
quant_params = {
    'beat_ref': 2.4,
    'bpm': 159,
    'subdiv': 2
}
quantized_impacts = quantize_impact_points(impact_points, time_stamps, quant_params)

# strip audio
print('\nStripping original audio...')
video_original_path = os.path.join(input_dir, f'{vid_name}.mov')
audio_original_path = os.path.join(input_dir, f'{vid_name}.wav')
strip_audio(video_original_path, audio_original_path)

# get instrument paths
instrument_dir = os.path.join('audio', 'instruments')
instrument_names = ['hihat.wav', 'cowbell.wav', 'kick.wav', 'snare.wav']
instrument_paths = [os.path.join(instrument_dir, name) for name in instrument_names]

# construct audio
print('\nConstructing output audio...')
audio_original_path = os.path.join(input_dir, f'{vid_name}.wav')
audio_output_path = os.path.join(output_dir, f'{vid_name}.wav')
construct_audio(quantized_impacts, instrument_paths, audio_output_path, audio_load_path=audio_original_path)

# combine audio and video
print('\nConstructing output video...')
video_original_path = os.path.join(input_dir, f'{vid_name}.avi')
video_output_path = os.path.join(output_dir, f'{vid_name}.mp4')
combine_audio_video(video_original_path, audio_output_path, video_output_path)

# print end
print(f'\nSuccess!\nOutput files in {output_dir}\n')
