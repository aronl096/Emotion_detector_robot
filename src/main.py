import os
import serial
from time import sleep
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from audio_processor import record_audio
from emotion_detector import load_model, predict_emotion
from config import MODEL_NAME, SAMPLE_RATE, DURATION
from chat_handler import TinyLlamaHandler


# Initialize Arduino connection
try:
    arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=2)
    sleep(2)  # Allow Arduino to reset after connection
    print("Arduino connected successfully!")
except Exception as e:
    print(f"Error initializing Arduino: {e}")
    arduino = None


def calculate_speaking_time(text, speed=1.1):
    """Estimate the speaking time for Festival based on the text length."""
    avg_char_per_second = 12  # Adjust this as needed
    return max(1, int(len(text) / avg_char_per_second * speed))


def send_command_to_arduino(command):
    """Send a command to the Arduino and handle the response."""
    if not arduino:
        print("Arduino not connected. Skipping command.")
        return

    try:
        arduino.write(f"{command}\n".encode())
        print(f"Sent to Arduino: {command}")

        # Wait for a response
        sleep(2)
        if arduino.in_waiting > 0:
            response = arduino.readline().decode().strip()
            print(f"Arduino response: {response}")
        else:
            print("No response from Arduino.")
    except Exception as e:
        print(f"Error communicating with Arduino: {e}")


def speak_with_festival(response, speed=0.9):
    """Use Festival to read text aloud and synchronize with Arduino."""
    print("Speaking the response...")

    # Ensure speed is within a reasonable range
    speed = max(0.9, min(speed, 3.0))

    # Sanitize the response
    safe_response = response.replace('"', '').replace("'", "").replace("(", "").replace(")", "")

    # Split into manageable chunks
    max_length = 800
    chunks = [safe_response[i:i + max_length] for i in range(0, len(safe_response), max_length)]

    # Calculate speaking time
    total_duration = calculate_speaking_time(safe_response, speed)
    print(f"Estimated speaking time: {total_duration} seconds")

    # Notify Arduino
    send_command_to_arduino(f"speak({total_duration})")

    # Use Festival to speak each chunk
    for chunk in chunks:
        command = f'echo "(Parameter.set \'Duration_Stretch {speed}) (voice_rab_diphone) (SayText \\"{chunk}\\")" | festival'
        os.system(command)

    # Notify Arduino that speaking has finished
    send_command_to_arduino("stop")


def main():
    try:


        
        # Initialize TinyLlama
        llama = TinyLlamaHandler()
        send_command_to_arduino("go sleep")
        send_command_to_arduino("wake up")

        # Initial greeting
        speak_with_festival("Hello, I'm the robot of Aaron and Sagghi. I'm here to help you detect your emotions.")
        
        sleep(4)
        speak_with_festival("Test me")
        # Record initial test input
        test_robot = record_audio(duration=3, sample_rate=SAMPLE_RATE)
        inputs = speech_processor(test_robot, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = speech_model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        text_robot = speech_processor.batch_decode(predicted_ids)[0]
        print(f"Transcription: {text_robot}")
        send_command_to_arduino(text_robot)

        # Main interaction loop
        print("Starting interaction...")
        speak_with_festival("Please speak into the microphone.")
        audio_data = record_audio(duration=DURATION, sample_rate=SAMPLE_RATE)

        # Predict emotion
        emotion = predict_emotion(audio_data, emotion_processor, emotion_model)
        print(f"Detected emotion: {emotion}")
        speak_with_festival(f"We detect that your emotion is: {emotion}")

        # Transcribe speech
        inputs = speech_processor(audio_data, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = speech_model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = speech_processor.batch_decode(predicted_ids)[0]
        print(f"Transcription: {transcription}")

        # Combine transcription and emotion for TinyLlama
        query = f"{transcription}. Please answer for a person that feels {emotion} in one or two sentences."
        response = llama.send_query(query)
        print(f"TinyLlama Response: {response}")

        # Speak the response
        speak_with_festival(response)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        if arduino:
            arduino.close()
            print("Arduino connection closed.")

        print("Program finished.")
        send_command_to_arduino("go sleep")


if __name__ == "__main__":
    # Load models
    print("Loading models...")
    emotion_processor, emotion_model = load_model(MODEL_NAME)

    # Load speech-to-text models
    speech_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    speech_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    speech_model.eval()

    # Run the main program
    main()
