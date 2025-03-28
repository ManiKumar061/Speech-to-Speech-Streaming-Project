import subprocess
import os

def sync_audio_to_video(video_path, audio_path, output_path):
    """
    Syncs the audio to the video and creates a new video file.
    
    Parameters:
    video_path (str): Path to the original video file
    audio_path (str): Path to the new audio file
    output_path (str): Path for the output video file
    
    Returns:
    str: Path to the output video file if successful, None otherwise
    """
    try:
        # Get video duration
        video_duration = float(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True
        ).stdout.strip())

        # Get audio duration
        audio_duration = float(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, check=True
        ).stdout.strip())

        # Calculate speed factor
        speed_factor = video_duration / audio_duration
        adjusted_audio_path = os.path.splitext(audio_path)[0] + "_adjusted.mp3"

        # Adjust the audio speed
        print("⚠ Adjusting audio speed to match video duration...")
        subprocess.run(
            ["ffmpeg", "-i", audio_path, "-filter:a", f"atempo={speed_factor}", "-vn", adjusted_audio_path, "-y"],
            check=True, capture_output=True
        )

        # Ensure adjusted audio matches video duration exactly
        final_audio_path = os.path.splitext(audio_path)[0] + "_final.aac"
        subprocess.run(
            ["ffmpeg", "-i", adjusted_audio_path, "-af", f"apad,atrim=end={video_duration}", "-ar", "48000", "-ac", "2", "-c:a", "aac", final_audio_path, "-y"],
            check=True, capture_output=True
        )

        # Merge the adjusted audio with the original video
        print("🔄 Replacing original audio with new synced audio...")
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-i", final_audio_path, "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", output_path, "-y"],
            check=True, capture_output=True
        )

        print(f"✅ Syncing complete! Saved as: {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return None

if __name__ == "__main__":
    video_path = input("📂 Enter the path of the original video file: ").strip()
    audio_path = input("🎵 Enter the path of the new TTS-generated audio file: ").strip()
    output_path = input("💾 Enter the output video file name (e.g., final_synced_video.mp4): ").strip()

    sync_audio_to_video(video_path, audio_path, output_path)