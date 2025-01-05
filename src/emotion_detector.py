import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

def load_model(model_name):
    """
    Loads the Wav2Vec2 model and processor for emotion detection.
    """
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
    model.eval()  # Set model to evaluation mode
    return processor, model

LABEL_MAP = {
    "LABEL_0": "Neutral",
    "LABEL_1": "Happy",
    "LABEL_2": "Sad",
    "LABEL_3": "Angry",
    "LABEL_4": "Disgust",
    "LABEL_5": "Fear",
    "LABEL_6": "Surprised"
}


def predict_emotion(audio_data, processor, model):
    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    label = f"LABEL_{predicted_id}"
    return LABEL_MAP.get(label, "Unknown Emotion")

