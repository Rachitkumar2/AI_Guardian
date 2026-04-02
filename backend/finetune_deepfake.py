import os
import glob
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from torch.optim import AdamW
from tqdm import tqdm

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"
SAMPLE_RATE = 16000
MAX_DURATION_SEC = 5 
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 2e-5

# We assume you have two folders containing your 20-50 local audio samples:
# data/real/*.wav
# data/fake/*.wav
DATASET_DIR = "data"

class DeepfakeAudioDataset(Dataset):
    def __init__(self, data_dir, feature_extractor):
        self.feature_extractor = feature_extractor
        self.samples = []
        
        real_dir = os.path.join(data_dir, "real")
        fake_dir = os.path.join(data_dir, "fake")
        
        # 0 for Real, 1 for Fake
        for f in glob.glob(os.path.join(real_dir, "*.wav")) + glob.glob(os.path.join(real_dir, "*.mp3")) + glob.glob(os.path.join(real_dir, "*.flac")) + glob.glob(os.path.join(real_dir, "*.ogg")):
            self.samples.append({"path": f, "label": 0})
            
        for f in glob.glob(os.path.join(fake_dir, "*.wav")) + glob.glob(os.path.join(fake_dir, "*.mp3")) + glob.glob(os.path.join(fake_dir, "*.flac")) + glob.glob(os.path.join(fake_dir, "*.ogg")):
            self.samples.append({"path": f, "label": 1})
            
        print(f"Discovered {len(self.samples)} audio files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        audio, _ = librosa.load(item["path"], sr=SAMPLE_RATE, duration=MAX_DURATION_SEC)
        
        # Ensure exact uniform length for batching
        target_length = int(SAMPLE_RATE * MAX_DURATION_SEC)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
        else:
            audio = audio[:target_length]
            
        inputs = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        # remove batch dimension since DataLoader will batch it
        return {"input_values": inputs.input_values.squeeze(0), "labels": torch.tensor(item["label"], dtype=torch.long)}


def main():
    print("Loading pre-trained model and extractor...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    if not os.path.exists(DATASET_DIR):
        print(f"Error: Could not find '{DATASET_DIR}/' folder.")
        print("Please create 'dataset/real/' and 'dataset/fake/' and put your audio files there.")
        return

    dataset = DeepfakeAudioDataset(DATASET_DIR, feature_extractor)
    if len(dataset) < 20:
        print("Warning: It is recommended to have at least 20-50 samples total for stable fine-tuning!")
        
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting Training Loop...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            optimizer.zero_grad()
            
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_values=input_values, labels=labels)
            loss = outputs.loss
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch + 1}/{EPOCHS} -> Average Loss: {total_loss / len(loader):.4f}")

    # Save the tuned model locally
    output_dir = "./checkpoints/finetuned-wav2vec2"
    model.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
    print(f"\nFine-tuning complete! Model saved to {output_dir}")
    print("To use this model, update HF_MODEL_NAME in your backend to point to this local directory.")

if __name__ == "__main__":
    main()
