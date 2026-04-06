from transformers import AutoConfig
import os

model_name = "garystafford/wav2vec2-deepfake-voice-detector"
try:
    config = AutoConfig.from_pretrained(model_name)
    print(f"Labels for {model_name}:")
    print(config.id2label)
except Exception as e:
    print(f"Error: {e}")
