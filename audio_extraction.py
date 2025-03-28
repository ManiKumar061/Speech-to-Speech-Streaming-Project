import ffmpeg
import os

def extract_audio(video_path, output_audio_path):
    """
    Extracts audio from a video file and saves it as a WAV file.
    """
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .run(overwrite_output=True, quiet=True)
        )
        print(f"✅ Audio extracted successfully: {output_audio_path}")
        return output_audio_path
    except Exception as e:
        print(f"❌ Error extracting audio: {e}")
        return None

if __name__ == "__main__":
    video_file = input("Enter the path of the video file: ").strip()

    if not os.path.exists(video_file):
        print("❌ Error: File not found! Please check the path.")
    else:
        audio_output = os.path.splitext(video_file)[0] + "_audio.wav"  # Save with same name
        extract_audio(video_file, audio_output)