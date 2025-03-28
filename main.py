import torchaudio
import whisperx
import torch
import os
import logging
from langchain.tools import Tool
import deepfilternet as dfn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Check for CUDA support
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

### STEP 1: Background Noise Removal & Enhancement ###
def remove_noise(audio_path: str) -> str:
    """
    Removes background noise and enhances speech using DeepFilterNet.
    """
    try:
        logging.info(f"Starting noise removal for {audio_path}...")
        model = dfn.DeepFilterNet2.from_pretrained("deepfilternet2")  # Explicit model

        enhanced_audio_path = "enhanced_output.wav"

        # Load input audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Process audio
        enhanced_waveform = model.enhance(waveform, sample_rate)

        # Save enhanced audio
        torchaudio.save(enhanced_audio_path, enhanced_waveform, sample_rate)
        logging.info(f"Noise removed. Enhanced audio saved at {enhanced_audio_path}")

        return enhanced_audio_path
    except Exception as e:
        logging.error(f"Error in noise removal: {e}")
        return audio_path  # Return original if enhancement fails

noise_removal_tool = Tool(
    name="Noise Removal & Enhancement",
    func=remove_noise,
    description="Removes background noise and enhances speech in an audio file."
)

### STEP 2: Syncing Translated Voice with Video ###
def sync_audio_with_video(translated_audio: str, original_video: str) -> str:
    """
    Syncs translated voice with original video using WhisperX.
    """
    try:
        logging.info(f"Loading WhisperX model on {DEVICE} for audio alignment...")
        model = whisperx.load_model("large-v2", device=DEVICE)

        # Load alignment model
        alignment_model, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)

        # Transcription with word-level timestamps
        logging.info(f"Transcribing {translated_audio} for alignment...")
        audio = whisperx.load_audio(translated_audio, device=DEVICE)
        result = model.transcribe(audio, batch_size=16)

        # Align transcription with the original video timestamps
        logging.info("Performing forced alignment...")
        aligned_result = whisperx.align(result["segments"], alignment_model, metadata, translated_audio, DEVICE)

        # Convert NumPy array to PyTorch tensor before saving
        tensor_audio = torch.tensor(audio).unsqueeze(0)  # Ensure correct tensor shape

        # Save aligned audio
        aligned_audio = "aligned_audio.wav"
        torchaudio.save(aligned_audio, tensor_audio, 16000)  # Ensure correct sample rate

        logging.info(f"Audio synced successfully. Saved at {aligned_audio}")
        return aligned_audio
    except Exception as e:
        logging.error(f"Error in syncing audio: {e}")
        return translated_audio  # Return original if syncing fails

sync_tool = Tool(
    name="Sync Translated Voice with Video",
    func=sync_audio_with_video,
    description="Aligns the translated voice with the original video timing."
)

### PIPELINE EXECUTION ###
def process_pipeline(input_audio: str, input_video: str) -> str:
    """
    Executes the full pipeline: noise removal -> syncing.
    """
    logging.info("Starting pipeline execution...")

    cleaned_audio = noise_removal_tool.func(input_audio)  # Correct function call
    synced_audio = sync_tool.func(cleaned_audio, input_video)

    logging.info(f"Final processed audio saved at: {synced_audio}")
    return synced_audio

# Example Usage
if __name__ == "__main__":
    final_audio = process_pipeline("input_audio.wav", "original_video.mp4")
    print(f"Final processed audio saved at: {final_audio}")
