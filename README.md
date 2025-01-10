# Audio-Video Recorder with Background Segmentation

This project is a Python application that records audio and video with real-time background segmentation using OpenCV, MediaPipe, and other libraries. The program allows users to apply custom backgrounds and save synchronized audio and video recordings.

## Features

- **Real-Time Background Segmentation**: Uses MediaPipe's Selfie Segmentation model to replace the background with a custom image.
- **Audio and Video Recording**: Records high-quality audio and video with synchronized timestamps.
- **Smoothness Adjustment**: Allows fine-tuning of segmentation smoothness through a trackbar.
- **Compressed Audio**: Saves audio in MP3 format with 64kbps bitrate for reduced file size.
- **Audio-Video Synchronization**: Ensures that the audio duration matches the video duration.


# Install them using:

```pip install -r requirements.txt```

# How to Use


Steps to Run:
Clone or download this repository.

Ensure you have a camera connected to your system.

Place a background image named background.png in the script's directory.

Set the output_folder in the script to your desired location for saving recordings.

Run the script: ```background_removal .py```

# Key Controls:

r: Start/Stop recording.

q: Quit the application.


# Adjustable Smoothness: 
Use the trackbar in the "Segmented View" window to adjust the segmentation smoothness.

# Outputs

Video File: Saved in MP4 format with synchronized timestamps.
Audio File: Saved in MP3 format at 64kbps bitrate.
Synchronized Recording: Ensures both audio and video durations match.


#How It Works

# Real-Time Segmentation:
MediaPipe's Selfie Segmentation model processes each video frame to replace the background with a custom image.  


# Audio-Video Recording:
Audio is recorded using PyAudio and synchronized with video frames using timestamps.
Video frames are captured, processed, and saved using OpenCV.


# File Saving:
Audio and video are saved separately and then synchronized.
The audio is compressed into MP3 format to reduce file size.


# File Structure

# AudioVideoRecorder Class:
Handles all recording tasks (audio/video).
Synchronizes and saves the recordings.


# Background Segmentation:
Utilizes MediaPipe's Selfie Segmentation for real-time mask creation.
Replaces the background with an image (background.png).


# Smoothness Adjustment:
Applies Gaussian blur to improve segmentation quality based on user-defined parameters.


# Prerequisites
Python Version: 3.7+

Hardware:
Camera for video input.

Microphone for audio input.

# Files:
A background.png image file in the same directory as the script.


# Known Issues
Ensure background.png exists and matches the camera's resolution.
Adjust segmentation smoothness if the background blending is not optimal.


# Example Output
Video: video_YYYYMMDD_HHMMSS.mp4
Audio: audio_YYYYMMDD_HHMMSS_64kbps.mp3


# Acknowledgments
This project leverages:

MediaPipe for real-time segmentation.

OpenCV for video processing.

PyAudio for high-quality audio recording.

MoviePy for audio-video synchronization.
