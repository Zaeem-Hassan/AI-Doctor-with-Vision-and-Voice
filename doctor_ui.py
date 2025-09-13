import os
import gradio as gr
from brain_of_doctor import encode_image, analyze_image_with_query
from voice_of_patient import transcribe_with_groq
from voice_of_doctor import text_to_speech_with_gtts
from voice_of_doctor import text_to_speech_with_pyttsx3

system_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose.
What's in this image?. Do you find anything wrong with it medically? 
If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Donot say 'In the image I see' but say 'With what I see, I think you have ....'
Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""


def process_inputs(audio_filepath, image_filepath):
    # 1. Speech → Text
    speech_to_text_output = transcribe_with_groq(
        GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
        audio_filepath=audio_filepath,
        stt_model="whisper-large-v3",
    )

    # 2. Text + Image → Doctor response
    if image_filepath:
        full_query = f"{system_prompt} {speech_to_text_output}"
        doctor_response = analyze_image_with_query(
            query=full_query,
            encoded_image=encode_image(image_filepath),
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )
    else:
        doctor_response = "No image provided for me to analyze."

    # 3. Doctor response → Voice (offline TTS)
    voice_of_doctor = text_to_speech_with_pyttsx3(
        input_text=doctor_response, output_filepath="final.mp3"
    )

    return speech_to_text_output, doctor_response, voice_of_doctor

# Gradio UI
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Patient's Voice"),
        gr.Image(type="filepath", label="Patient's Image"),
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Voice"),
    ],
    title="AI Doctor with Vision and Voice",
)

if __name__ == "__main__":
    iface.launch(debug=True)
