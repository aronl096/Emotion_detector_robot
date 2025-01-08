import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from audio_processor import record_audio
from emotion_detector import load_model, predict_emotion
from config import MODEL_NAME, SAMPLE_RATE, DURATION
import os  # To run shell commands
import serial
import time

# Initialize Arduino connection
# arduino = serial.Serial('COM5', 9600)  # Replace 'COM5' with your Arduino's port
# time.sleep(2)  # Allow the connection to initialize

# TinyLlama handler
from chat_handler import TinyLlamaHandler

# Servo movement functions
# def move_servo(command):
#     """Send a command to move the servo"""
#     arduino.write(f"{command}\n".encode())  # Send the command
#     print(f"Sent: {command}")
#     time.sleep(0.5)  # Allow time for the servo to move

# def animate_mouth():
#     """Simulate mouth movement during speech"""
#     for _ in range(3):
#         move_servo("mouth_open")
#         time.sleep(0.3)
#         move_servo("mouth_close")
#         time.sleep(0.3)

# def move_head_based_on_response(response):
#     """Move the head based on TinyLlama's response"""
#     if "yes" in response.lower():
#         move_servo("head_yes")  # Move head to simulate a nod
#     elif "no" in response.lower():
#         move_servo("head_no")  # Move head to simulate a shake

# Load Wav2Vec2 models
print("Loading models...")
emotion_processor, emotion_model = load_model(MODEL_NAME)  # Emotion model and processor

# Load Wav2Vec2 model for speech-to-text
speech_model_name = "facebook/wav2vec2-large-960h"
speech_processor = Wav2Vec2Processor.from_pretrained(speech_model_name)
speech_model = Wav2Vec2ForCTC.from_pretrained(speech_model_name)
speech_model.eval()

def speak_with_festival(response):
    """Use Festival to read text aloud with sanitized input."""
    print("Speaking the response...")
    max_length = 800  # Festival handles ~300 characters well
    
    # Sanitize the response to remove problematic characters
    safe_response = (
        response.replace('"', '')   # Remove double quotes
               .replace('(', '')   # Remove opening parentheses
               .replace(')', '')   # Remove closing parentheses
               .replace("'", '')   # Remove single quotes
    )
    
    # Split the sanitized response into manageable chunks
    chunks = [safe_response[i:i + max_length] for i in range(0, len(safe_response), max_length)]
    
    for chunk in chunks:
        # Prepare the command for Festival
        command = f'echo "(voice_rab_diphone) (SayText \\"{chunk}\\")" | festival'
        os.system(command)


def main():
    try:
        # Start TinyLlama connection
        llama = TinyLlamaHandler()

        # Record audio
        print("Starting the audio recording...")
        speak_with_festival("Please speak into the microphone")
        audio_data = record_audio(duration=DURATION, sample_rate=SAMPLE_RATE)
        print("Recording complete.")

        # Predict emotion
        print("Detecting emotion...")
        emotion = predict_emotion(audio_data, emotion_processor, emotion_model)
        print(f"Detected emotion: {emotion}")
        speak_with_festival(f"We detect that your emotion is: {emotion}")

        # Speech-to-text conversion
        print("Transcribing speech...")
        inputs = speech_processor(audio_data, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = speech_model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = speech_processor.batch_decode(predicted_ids)[0]
        print(f"Transcription: {transcription}")

        # Combine transcription and emotion for Ollama
        query = f"{transcription}. Please answer for a person that feels {emotion} in one or two sentences."
        print(f"User Query for Ollama: {query}")

        # Send to TinyLlama
        try:
            print("Sending to TinyLlama...")
            response = llama.send_query(query)
            # Ensure the response is not too long
            response = response[:600]
            print(f"TinyLlama Response: {response}")

            # Read response aloud
            print("Speaking the response...")
            speak_with_festival(response)

        except Exception as e:
            # Handle communication errors
            error_message = "I'm unable to process that right now."
            print(f"Error communicating with TinyLlama: {e}")
            speak_with_festival(error_message)

        # Move head based on response
        # move_head_based_on_response(response)

    finally:
        # Close TinyLlama connection
        llama.close()
        print("TinyLlama connection closed.")

if __name__ == "__main__":
    main()
