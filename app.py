import gradio as gr
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = "goemotions/goemotions"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
LABELS = model.config.id2label

# -------------------------
# Emotion mapping
# -------------------------
GOEMO_TO_FLOWERS = {
    "admiration": "Admiration",
    "amusement": "Amusement",
    "approval": "Approval",
    "caring": "Caring",
    "curiosity": "Curiosity",
    "desire": "Desire",
    "excitement": "Excitement",
    "gratitude": "Gratitude",
    "joy": "Joy",
    "love": "Love",
    "optimism": "Optimism",
    "pride": "Pride",
    "relief": "Relief",
    "surprise": "Surprise",
    "realization": "Realization",
    "confusion": "Confusion",
    "anger": "Anger",
    "annoyance": "Annoyance",
    "disapproval": "Disapproval",
    "disgust": "Disgust",
    "embarrassment": "Embarrassment",
    "fear": "Fear",
    "grief": "Grief",
    "nervousness": "Nervousness",
    "remorse": "Remorse",
    "sadness": "Sadness",
    "disappointment": "Disappointment"
}

# Flower type mapping
def get_flower_type(emotion: str) -> str:
    flower_map = {
        "Admiration": "Orchid",
        "Amusement": "Yellow Tulip",
        "Approval": "Hydrangea",
        "Caring": "Pink Carnation",
        "Curiosity": "Blue Iris",
        "Desire": "Red Tulip",
        "Excitement": "Gerbera Daisy",
        "Gratitude": "Pink Rose",
        "Joy": "Sunflower",
        "Love": "Red Rose",
        "Optimism": "Yellow Rose",
        "Pride": "Gladiolus",
        "Relief": "Lavender",
        "Surprise": "Purple Tulip",
        "Realization": "White Rose",
        "Confusion": "Foxglove",
        "Anger": "Red Chrysanthemum",
        "Annoyance": "Orange Lily",
        "Disapproval": "Petunia",
        "Disgust": "Marigold",
        "Embarrassment": "Peony",
        "Fear": "White Poppy",
        "Grief": "White Lily",
        "Nervousness": "Daisy",
        "Remorse": "Purple Hyacinth",
        "Sadness": "Blue Hyacinth",
        "Disappointment": "Willow"
    }
    return flower_map.get(emotion, "Unknown")

# -------------------------
# Prediction function for Gradio
# -------------------------
def predict_flower(text: str, threshold: float = 0.3, top_k: int = 3):
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.sigmoid(logits)[0].cpu().numpy()

    # Filter by threshold or pick top_k
    indices = np.where(probs > threshold)[0]
    if len(indices) == 0:
        indices = np.argsort(probs)[-top_k:][::-1]

    # Aggregate flower emotions
    flower_scores = {}
    for idx in indices:
        label = LABELS[idx]
        score = float(probs[idx])
        flower = GOEMO_TO_FLOWERS.get(label, "Neutral/Unknown")
        flower_scores[flower] = max(score, flower_scores.get(flower, 0))

    results = [
        {
            "flower_emotion": flower,
            "score": score,
            "flower_type": get_flower_type(flower)
        }
        for flower, score in flower_scores.items()
    ]

    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:top_k]

    dominant = results[0] if results else {
        "flower_emotion": "Neutral/Unknown",
        "score": 0.0,
        "flower_type": "Unknown"
    }

    return dominant["flower_emotion"], dominant["score"]

# -------------------------
# Gradio interface
# -------------------------
iface = gr.Interface(
    fn=predict_flower,
    inputs=[
        gr.Textbox(lines=2, placeholder="Type your text here...", label="Text"),
        gr.Slider(minimum=0, maximum=1, step=0.05, value=0.3, label="Threshold"),
        gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Top K")
    ],
    outputs=[
        gr.Textbox(label="Dominant Flower Emotion"),
        gr.Number(label="Score")
    ],
    title="Princesa Emotion API 🌸",
    description="Enter a sentence to detect the dominant flower emotion."
)

iface.launch()