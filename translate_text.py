from googletrans import Translator
import os

# Language options
LANGUAGE_OPTIONS = {
    "Spanish": "es", 
    "French": "fr", 
    "German": "de",
    "Hindi": "hi", 
    "Tamil": "ta", 
    "Arabic": "ar",
    "Bengali": "bn", 
    "Chinese": "zh-cn",
    "Portuguese": "pt", 
    "Russian": "ru",
    "English": "en"
}

def translate_text(text, target_language):
    """
    Translates the given text to the specified target language.
    
    Parameters:
    text (str): Text to translate
    target_language (str): Target language name (e.g., "Spanish", "French")
    
    Returns:
    str: Translated text
    """
    translator = Translator()
    
    # Get the language code
    target_language_code = LANGUAGE_OPTIONS.get(target_language)
    
    if not target_language_code:
        print(f"❌ Invalid language: {target_language}. Defaulting to English.")
        target_language_code = "en"
    
    print(f"⏳ Translating to {target_language}...")
    
    try:
        translated_text = translator.translate(text, dest=target_language_code).text
        print("✅ Translation complete!")
        return translated_text
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return None

def translate_from_file(text_file):
    """Translates the text file into a user-selected language."""
    # Read original text
    with open(text_file, "r", encoding="utf-8") as f:
        original_text = f.read()

    # Display language options
    print("\n🌍 Select a language for translation:")
    for i, lang in enumerate(LANGUAGE_OPTIONS.keys(), 1):
        print(f"{i}. {lang}")

    choice = input("\nEnter the number of your choice: ").strip()
    
    try:
        target_language = list(LANGUAGE_OPTIONS.keys())[int(choice) - 1]
    except (ValueError, IndexError):
        print("❌ Invalid choice. Exiting.")
        return None

    translated_text = translate_text(original_text, target_language)
    
    if translated_text:
        # Save the translated text
        translated_file = text_file.replace("_transcription.txt", f"_{target_language.lower()}_translation.txt")
        with open(translated_file, "w", encoding="utf-8") as f:
            f.write(translated_text)

        print(f"✅ Translation saved to: {translated_file}")
        return translated_file
    
    return None

if __name__ == "__main__":
    file_path = input("Enter the path of the transcribed text file: ").strip()

    if not os.path.exists(file_path):
        print("❌ Error: File not found! Please check the path.")
    else:
        translated_file = translate_from_file(file_path)