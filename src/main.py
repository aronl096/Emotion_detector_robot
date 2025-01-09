import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from audio_processor import record_audio
from emotion_detector import load_model, predict_emotion
from config import MODEL_NAME, SAMPLE_RATE, DURATION
import os  # To run shell commands
import serial
from time import sleep
import time

# Initialize Arduino connection
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=2)  # Ensure this matches your port
sleep(2)  # Allow Arduino to reset after connection
print("Arduino connected successfully!")

# TinyLlama handler
from chat_handler import TinyLlamaHandler


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

def calculate_speaking_time(text, speed=0.9):
    """
    Estimate the time it takes for Festival to speak the text.
    - Speed affects the duration (1.0 is normal, higher is slower).
    """
    avg_char_per_second = 10  # Average speaking rate for Festival (~300 characters/min)
    estimated_duration = len(text) / avg_char_per_second * speed
    return max(1, int(estimated_duration))  # Ensure at least 1 second

def speak_with_festival(response, arduino=None, speed=0.9):
    """Use Festival to read text aloud and synchronize with Arduino."""
    print("Speaking the response...")

    # Ensure speed is within a reasonable range
    if speed < 1.0:
        speed = 1.0  # Minimum speed
    elif speed > 3.0:
        speed = 3.0  # Maximum speed

    max_length = 800  # Festival handles ~300 characters well

    # Sanitize the response to remove problematic characters
    safe_response = (
        response.replace('"', '')   # Remove double quotes
               .replace('(', '')   # Remove opening parentheses
               .replace(')', '')   # Remove closing parentheses
               .replace("'", '')   # Remove single quotes
    )

    # Calculate total speaking duration
    total_duration = calculate_speaking_time(safe_response, speed)
    print(f"Estimated speaking time: {total_duration} seconds")

    # Send "speak(duration)" command to Arduino
    if arduino:
        try:
            
            send_command_to_arduino(f"speak({total_duration})", arduino)
            # print(f"Sent to Arduino: {command}")
            print(f"Sent 'speak({total_duration})' command to Arduino.")
        except Exception as e:
            print(f"Error sending 'speak' command to Arduino: {e}")

    # Split the sanitized response into manageable chunks
    chunks = [safe_response[i:i + max_length] for i in range(0, len(safe_response), max_length)]

    # Speak each chunk with Festival
    for chunk in chunks:
        # Prepare the command for Festival with slower speech
        command = f'echo "(Parameter.set \'Duration_Stretch {speed}) (voice_rab_diphone) (SayText \\"{chunk}\\")" | festival'
        os.system(command)

    # Inform Arduino that speaking has finished
    if arduino:
        try:
            arduino.close()
            print("Sent 'stop speaking' command to Arduino.")
        except Exception as e:
            print(f"Error sending 'stop speaking' command to Arduino: {e}")


def send_command_to_arduino(command, arduino):
    """Send a command to the Arduino via serial communication."""
    try:
        # Send the command
        arduino.write(f"{command}\n".encode())  # Add newline for Arduino's `readStringUntil`
        print(f"Sent to Arduino: {command}")
        
        # Wait for a response
        sleep(1)  # Give Arduino time to process
        if arduino.in_waiting > 0:
            response = arduino.readline().decode().strip()
            print(f"Arduino response: {response}")
        else:
            print("No response from Arduino.")
    except Exception as e:
        print(f"Error communicating with Arduino: {e}")




def main():
    try:
        speak_with_festival("Helloo , I'm the ROBOT of Aaron and Sagi , I'm here to help you to detect your emotions.")
        # Start TinyLlama connection
        sleep(2)
        speak_with_festival("Test me")
        test_robot = record_audio(duration=4, sample_rate=SAMPLE_RATE)
        inputs = speech_processor(test_robot, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = speech_model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        text_robot = speech_processor.batch_decode(predicted_ids)[0]
        send_command_to_arduino(f"{text_robot}", arduino)
    

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
            # response = response[:10000]
            print(f"TinyLlama Response: {response}")

            # Read response aloud
            print("Speaking the response...")
            speak_with_festival(response)
            sleep(5)

        except Exception as e:
            # Handle communication errors
            error_message = "I'm unable to process that right now."
            print(f"Error communicating with TinyLlama: {e}")
            speak_with_festival(error_message)


    finally:
        # Close TinyLlama connection
        llama.close()
        print("TinyLlama connection closed.")

if __name__ == "__main__":
    main()
