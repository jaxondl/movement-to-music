# imports
from pydub import AudioSegment
import moviepy.editor as mp


# stripping sound from video and re-saving
def strip_audio(video_load_path, audio_save_path, video_save_path=None):
    video = mp.VideoFileClip(video_load_path)
    video.audio.write_audiofile(audio_save_path)
    
    if video_save_path != None:
        video_no_audio = video.without_audio()
        video_no_audio.write_videofile(video_save_path)


# function to construct audio
def construct_audio(quantized_impacts, instrument_paths, audio_save_path, audio_load_path=None, video_length:float=None):
    
    # quantized impacts and instrument paths should both have length n_limbs
    assert len(quantized_impacts) == len(instrument_paths)
        
    # if audio_load_path provided, construct audio on top of original audio
    if audio_load_path:
        base_audio = AudioSegment.from_wav(audio_load_path)
        
    # otherwise, construct blank audio file
    elif video_length:
        base_audio = AudioSegment.silent(duration=(1000*video_length))
    else:
        raise Exception('audio_load_path or video_length must be provided')
    
    # quantized_impacts is a list of numpy arrays
    # each array corresponds to a limb and contains timestamps in seconds of the impact points
    # each limb has a corresponding instrument sound in instrument_paths
    # instrument sounds should occur in the audio file at every time stamp in the respective array
    for impacts, instrument_path in zip(quantized_impacts, instrument_paths):
        print(f'Instrument: {instrument_path}')
        impact_sound = AudioSegment.from_wav(instrument_path)
        
        # for each time stamp in array, make sound in file & combine sound file with base_audio
        for time in impacts:
            beat = AudioSegment.silent(duration=(1000*time)) + impact_sound
            quite_beat = beat - 9
            base_audio = base_audio.overlay(quite_beat)
    
    final_sound = base_audio
    
    # save audio to audio_save_path
    final_sound.export(audio_save_path, format=audio_save_path[-3:])


# combining video and audio
def combine_audio_video(video_load_path, audio_load_path, combined_save_path, fps=30):
    video = mp.VideoFileClip(video_load_path)
    audio = mp.AudioFileClip(audio_load_path)
    
    video.audio = audio
    video.write_videofile(combined_save_path, fps=fps, audio_codec='aac')
