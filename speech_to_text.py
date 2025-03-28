import whisper
import os
import sys

def transcribe_audio(audio_path, language=None):
    """
    Transcribes the given audio file using Whisper.
    
    Parameters:
    audio_path (str): Path to the audio file
    language (str, optional): Language code to use for transcription
    
    Returns:
    str: Transcribed text
    """
    try:
        print("⏳ Loading Whisper model... (This may take a few seconds)")
        model = whisper.load_model("small", download_root="venv/")  # Ensure it downloads in venv
        print("✅ Model loaded. Starting transcription...")

        # Use the specified language if provided, otherwise let whisper detect
        result = model.transcribe(audio_path, language="en")

        print("\n🎤 Transcription:\n")
        print(result["text"])  # Print to console
        return result["text"]

    except Exception as e:
        print(f"❌ Error in transcription: {e}")
        return None

if __name__ == "__main__":
    audio_file = input("Enter the path of the audio file: ").strip()

    if not os.path.exists(audio_file):
        print("❌ Error: File not found! Please check the path.")
        sys.exit(1)  # Exit the script if the file is missing

    transcript = transcribe_audio(audio_file)

    if transcript:
        text_output = os.path.splitext(audio_file)[0] + "_transcription.txt"

        with open(text_output, "w", encoding="utf-8") as f:
            f.write(transcript)

        print(f"\n✅ Transcription saved to: {text_output}")