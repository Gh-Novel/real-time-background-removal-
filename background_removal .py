import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import wave
import threading
import os
import time
from datetime import datetime
from pydub import AudioSegment
import moviepy as mpeditor


class AudioVideoRecorder:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.is_recording = False
        self.is_running = True
        self.audio_frames = []
        self.video_frames = []
        self.fps = 24  # Fixed video FPS
        
        # Audio settings
        self.audio = pyaudio.PyAudio()
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.audio_format = pyaudio.paInt16
        
        # Start audio stream
        self.stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        # Recording start time
        self.start_time = None
        
    def start_recording(self, frame):
        self.is_recording = True
        self.audio_frames = []
        self.video_frames = []
        self.start_time = time.time()
        self.add_timestamp(frame)  # Add initial frame with timestamp
        
        # Start audio recording thread
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.start()

    def _record_audio(self):
        while self.is_recording:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                timestamp = time.time() - self.start_time  # Add timestamp to each audio chunk
                self.audio_frames.append((timestamp, data))  # Store audio chunk with timestamp
            except Exception as e:
                print(f"Audio recording error: {e}")
                break

    def add_timestamp(self, frame):
        timestamp = time.time() - self.start_time
        self.video_frames.append((timestamp, frame))  # Store frame with timestamp

    def stop_recording(self):
        if not self.is_recording:
            return None
            
        # Calculate actual duration
        duration = time.time() - self.start_time
        self.is_recording = False
        
        # Wait for audio thread to complete
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join()
        
        # Save files separately
        return self._save_recording(duration)

    def _save_recording(self, duration):
        if not self.video_frames:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save video
        video_path = os.path.join(self.output_folder, f"video_{timestamp}.mp4")
        height, width = self.video_frames[0][1].shape[:2]
        out = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (width, height)
        )
        
        # Write video frames
        for frame in self.video_frames:
            out.write(frame[1])  # Write frame without timestamp
        out.release()
        
        # Save audio
        audio_path = os.path.join(self.output_folder, f"audio_{timestamp}.wav")
        with wave.open(audio_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
            wf.setframerate(self.rate)
            for audio_chunk in self.audio_frames:
                wf.writeframes(audio_chunk[1])  # Write audio chunk without timestamp

        # Compress the audio to 64kbps MP3
        compressed_audio_path = self._compress_audio_to_64kbps(audio_path)
        
        # Check video duration and adjust audio to match
        adjusted_audio_path = self._adjust_audio_to_video(video_path, compressed_audio_path)
        
        return video_path, adjusted_audio_path  # Return the paths of the saved video and adjusted audio files

    def _compress_audio_to_64kbps(self, audio_path):
        # Convert WAV to MP3 with 64kbps bitrate
        audio = AudioSegment.from_wav(audio_path)
        compressed_audio_path = audio_path.replace(".wav", "_64kbps.mp3")
        audio.export(compressed_audio_path, format="mp3", bitrate="64k")
        
        return compressed_audio_path

    def _adjust_audio_to_video(self, video_path, audio_path):
        # Load the video file
        video_clip = mpeditor.VideoFileClip(video_path)
        video_duration = video_clip.duration  # Duration in seconds
        
        # Load the audio file
        audio = AudioSegment.from_mp3(audio_path)
        audio_duration = len(audio) / 1000.0  # Duration in seconds

        # Stretch or shrink audio to match the video duration
        if audio_duration > video_duration:
            # Shrink the audio
            audio = audio[:int(video_duration * 1000)]  # Trim audio to match video length
        else:
            # Stretch the audio
            audio = audio.speedup(playback_speed=audio_duration / video_duration)
        
        # Save adjusted audio
        adjusted_audio_path = audio_path.replace(".mp3", "_adjusted.mp3")
        audio.export(adjusted_audio_path, format="mp3")
        
        return adjusted_audio_path

    def cleanup(self):
        self.is_recording = False
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

def main():
    # Initialize MediaPipe
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup recorder
    output_folder = r"add a folder"
    os.makedirs(output_folder, exist_ok=True)
    recorder = AudioVideoRecorder(output_folder)
    
    # Load background
    background_image = cv2.imread("background.png")
    if background_image is None:
        print("Error: Background image not found")
        return
    background_image = cv2.resize(background_image, (width, height))
    
    # Setup window and trackbar
    cv2.namedWindow("Segmented View")
    smoothness = 0
    
    def on_trackbar(val):
        nonlocal smoothness
        smoothness = val
    
    cv2.createTrackbar("Smoothness", "Segmented View", 0, 100, on_trackbar)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        # Process frame
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(rgb_image)
        
        # Create mask with improved smoothness
        mask = results.segmentation_mask > 0.1
        mask = mask.astype(np.uint8) * 255
        
        # Apply smoothness
        smooth_val = smoothness / 100.0
        kernel_size = max(1, int(smooth_val * 30)) * 2 + 1
        if kernel_size > 1:
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 
                                  smooth_val * 10)
        
        mask_3d = np.stack((mask,) * 3, axis=-1)
        
        # Create output image
        alpha = mask_3d / 255.0
        output_image = (image * alpha + background_image * (1 - alpha)).astype(np.uint8)
        
        # Add recording indicator
        if recorder.is_recording:
            cv2.putText(output_image, "R", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            recorder.add_timestamp(output_image)  # Add timestamp with frame
        
        # Display frame
        cv2.imshow("Segmented View", output_image)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            if not recorder.is_recording:
                recorder.start_recording(output_image)
                print("Recording started...")
            else:
                video_file, audio_file = recorder.stop_recording()
                print(f"Recording saved: Video - {video_file}, Audio - {audio_file}")
    
    # Cleanup
    cap.release()
    recorder.cleanup()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
