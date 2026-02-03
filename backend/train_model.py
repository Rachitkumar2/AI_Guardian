import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import librosa
import os
import random
from sklearn.preprocessing import StandardScaler
import pickle

# Ensure data folders exist
os.makedirs("data/real", exist_ok=True)
os.makedirs("data/fake", exist_ok=True)

# Enhanced Feature Extraction
def extract_features(file_path, sr=16000, n_mfcc=40, max_len=100):
    """Extract comprehensive audio features for deepfake detection."""
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=5)  # Limit to 5 seconds
        
        # Pad or truncate audio to consistent length
        target_length = sr * 5
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        # 1. MFCCs (40 coefficients) - captures spectral envelope
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_delta = np.mean(librosa.feature.delta(mfcc), axis=1)
        
        # 2. Spectral features - detect artifacts in fake audio
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr), axis=1)
        
        # 3. Zero crossing rate - voice naturalness
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # 4. RMS Energy - amplitude patterns
        rms = np.mean(librosa.feature.rms(y=audio))
        
        # 5. Chroma features - harmonic content
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)
        
        # 6. Mel spectrogram statistics
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
        mel_mean = np.mean(mel_spec, axis=1)
        
        # Combine all features
        features = np.concatenate([
            mfcc_mean,           # 40 features
            mfcc_std,            # 40 features
            mfcc_delta,          # 40 features
            [spectral_centroid], # 1 feature
            [spectral_bandwidth],# 1 feature
            [spectral_rolloff],  # 1 feature
            spectral_contrast,   # 7 features
            [zcr],               # 1 feature
            [rms],               # 1 feature
            chroma,              # 12 features
            mel_mean             # 40 features
        ])
        
        return features  # Total: ~184 features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Custom Dataset Class with Enhanced Features
class AudioDataset(Dataset):
    def __init__(self, real_dir, fake_dir, sr=16000):
        self.real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                          if f.endswith((".wav", ".mp3", ".flac"))]
        self.fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                          if f.endswith((".wav", ".mp3", ".flac"))]
        self.all_files = [(f, 0) for f in self.real_files] + [(f, 1) for f in self.fake_files]
        
        if not self.all_files:
            raise ValueError("No audio files found! Please add audio files to 'data/real' and 'data/fake'.")
        
        print(f"📊 Found {len(self.real_files)} real and {len(self.fake_files)} fake audio files")
        
        random.shuffle(self.all_files)
        self.sr = sr
        
        # Pre-extract and cache features
        print("🔄 Extracting features from audio files...")
        self.features = []
        self.labels = []
        
        for file_path, label in self.all_files:
            feat = extract_features(file_path, sr=self.sr)
            if feat is not None:
                self.features.append(feat)
                self.labels.append(label)
        
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        
        # Normalize features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
        print(f"✅ Extracted {len(self.features)} feature vectors of size {self.features.shape[1]}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (torch.tensor(self.features[idx], dtype=torch.float32), 
                torch.tensor(self.labels[idx], dtype=torch.long))

# Enhanced Neural Network Model
class AudioClassifier(nn.Module):
    def __init__(self, input_size=184):
        super(AudioClassifier, self).__init__()
        
        self.network = nn.Sequential(
            # First layer
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Third layer
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Fourth layer
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Output layer
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)

# Training Function with Validation
def train_model(real_dir, fake_dir, epochs=50, batch_size=16, lr=0.001):
    try:
        dataset = AudioDataset(real_dir, fake_dir)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Split into train/validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📈 Training on {train_size} samples, validating on {val_size} samples")
    
    # Get input size from features
    input_size = dataset.features.shape[1]
    model = AudioClassifier(input_size=input_size)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience_counter = 0
    early_stop_patience = 10
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
        
        train_acc = train_correct / train_size
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
        
        val_acc = val_correct / val_size
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "audio_model.pth")
            # Save the scaler for inference
            with open("scaler.pkl", "wb") as f:
                pickle.dump(dataset.scaler, f)
            patience_counter = 0
            print(f"  💾 Saved best model with Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print("📁 Model saved as 'audio_model.pth'")
    print("📁 Scaler saved as 'scaler.pkl'")

# Run Training
if __name__ == "__main__":
    train_model("data/real", "data/fake")
