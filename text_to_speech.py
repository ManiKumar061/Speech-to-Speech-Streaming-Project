import edge_tts
import asyncio
import os

# Language and voice options
LANGUAGE_VOICES = {
    "Spanish": {"male": "es-ES-AlvaroNeural", "female": "es-ES-ElviraNeural"},
    "French": {"male": "fr-FR-HenriNeural", "female": "fr-FR-DeniseNeural"},
    "German": {"male": "de-DE-ConradNeural", "female": "de-DE-KatjaNeural"},
    "Hindi": {"male": "hi-IN-MadhurNeural", "female": "hi-IN-SwaraNeural"},
    "Tamil": {"male": "ta-IN-ValluvarNeural", "female": "ta-IN-PallaviNeural"},
    "Arabic": {"male": "ar-SA-FareedNeural", "female": "ar-SA-ZariyahNeural"},
    "Bengali": {"male": "bn-IN-BashkarNeural", "female": "bn-IN-TanishaaNeural"},
    "Chinese": {"male": "zh-CN-YunxiNeural", "female": "zh-CN-XiaoxiaoNeural"},
    "Portuguese": {"male": "pt-PT-FernandoNeural", "female": "pt-PT-FernandaNeural"},
    "Russian": {"male": "ru-RU-DmitryNeural", "female": "ru-RU-SvetlanaNeural"},
    "English": {"male": "en-US-GuyNeural", "female": "en-US-JennyNeural"}
}

async def generate_speech(text, output_path, language="English", gender="female"):
    """
    Converts text to speech using Edge TTS.
    
    Parameters:
    text (str): Text to convert to speech
    output_path (str): Path to save the audio file
    language (str): Target language (e.g., "Spanish", "French")
    gender (str): "male" or "female"
    
    Returns:
    str: Path to the generated audio file
    """
    try:
        # Get the appropriate voice
        if language not in LANGUAGE_VOICES:
            print(f"❌ Language '{language}' not found. Defaulting to English.")
            language = "English"
        
        voice = LANGUAGE_VOICES[language][gender.lower()]
        
        # Convert text to speech
        print(f"⏳ Generating {gender} {language} speech...")
        tts = edge_tts.Communicate(text, voice, rate="-10%")  # Decrease speed by 10%
        await tts.save(output_path)
        
        print(f"✅ Audio saved as: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error generating speech: {e}")
        return None

async def interactive_tts(file_path):
    """Interactive function for command-line usage"""
    # Show language options
    print("\n🌍 Choose the language:")
    for i, lang in enumerate(LANGUAGE_VOICES.keys(), 1):
        print(f"{i}. {lang}")

    # Get user input
    lang_choice = input("\nEnter the number corresponding to the language: ").strip()
    try:
        language = list(LANGUAGE_VOICES.keys())[int(lang_choice) - 1]
    except (ValueError, IndexError):
        print("❌ Invalid choice! Defaulting to English.")
        language = "English"

    # Read text file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print("❌ Error: File not found! Please check the path.")
        return

    # Voice selection
    print("\n🎙️ Choose a voice option:")
    print("1. Male Voice")
    print("2. Female Voice")
    voice_choice = input("\nEnter your choice (1/2): ").strip()

    # Assign the correct voice
    gender = "male" if voice_choice == "1" else "female"

    # Convert text to speech
    audio_file = file_path.replace(".txt", f"_{language.lower()}.mp3")
    await generate_speech(text, audio_file, language, gender)

# Run the async function
if __name__ == "__main__":
    file_path = input("\n📂 Enter the full path of the translated text file: ").strip()
    asyncio.run(interactive_tts(file_path))