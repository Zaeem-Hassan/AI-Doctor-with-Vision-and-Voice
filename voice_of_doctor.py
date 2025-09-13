from gtts import gTTS
from pydub import AudioSegment
import subprocess
import platform
import os
import pyttsx3

def text_to_speech_with_gtts(input_text, output_filepath):
    """
    Convert text to speech using gTTS and play it based on OS.
    Returns the path of the generated audio file.
    """
    language = "en"

    # Save as MP3
    audioobj = gTTS(text=input_text, lang=language, slow=False)
    audioobj.save(output_filepath)

    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(["afplay", output_filepath])
        elif os_name == "Windows":  # Windows needs WAV
            wav_path = output_filepath.replace(".mp3", ".wav")
            sound = AudioSegment.from_mp3(output_filepath)
            sound.export(wav_path, format="wav")
            subprocess.run(
                ["powershell", "-c", f'(New-Object Media.SoundPlayer "{wav_path}").PlaySync();']
            )
        elif os_name == "Linux":
            subprocess.run(["mpg123", output_filepath])  # mpg123 handles MP3 directly
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"An error occurred while playing the audio: {e}")

    return output_filepath  # ✅ important for Gradio

def text_to_speech_with_pyttsx3(input_text, output_filepath="final.mp3"):
    engine = pyttsx3.init()
    engine.save_to_file(input_text, output_filepath)
    engine.runAndWait()
    return output_filepath