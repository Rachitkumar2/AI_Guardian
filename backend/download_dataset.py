"""
Dataset downloader for AI Guardian deepfake audio detection.
Downloads sample audio files for training.
"""

import os
import urllib.request
import zipfile
import shutil

# Create data directories
os.makedirs("data/real", exist_ok=True)
os.makedirs("data/fake", exist_ok=True)

print("=" * 60)
print("AI Guardian - Training Data Setup")
print("=" * 60)

# Option 1: Download from ASVspoof (mini subset for testing)
# Note: Full dataset requires registration at https://www.asvspoof.org/

SAMPLE_DATASETS = {
    "ljspeech_sample": {
        "url": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        "description": "LJ Speech Dataset (real human speech samples)",
        "size": "2.6 GB - LARGE"
    }
}

print("""
📁 MANUAL DATASET DOWNLOAD OPTIONS:

For the best results, download these datasets manually:

1. ASVspoof 2019 (Recommended for deepfake detection)
   URL: https://www.asvspoof.org/
   - Contains real and synthetic speech
   - Register for free access
   - Download LA (Logical Access) subset

2. FakeAVCeleb Dataset
   URL: https://github.com/DASH-Lab/FakeAVCeleb
   - Contains real and fake celebrity audio/video

3. In-the-Wild Dataset  
   URL: https://github.com/piotrmirowski/fakeyou-detection
   - Real-world deepfake samples

4. WaveFake Dataset
   URL: https://github.com/RUB-SysSec/WaveFake
   - Synthetic audio from various TTS systems

After downloading, place files in:
   - Real audio → data/real/
   - Fake audio → data/fake/

""")

# Quick synthetic data generation for testing the pipeline
print("=" * 60)
print("🔧 Generating synthetic test data to verify pipeline...")
print("=" * 60)

try:
    import numpy as np
    from scipy.io import wavfile
    
    def generate_test_audio(filename, duration=3, sample_rate=16000, is_fake=False):
        """Generate simple test audio files."""
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        if is_fake:
            # Fake: more synthetic sounding (pure tones with harmonics)
            freq = np.random.choice([200, 250, 300, 350])
            audio = 0.3 * np.sin(2 * np.pi * freq * t)
            audio += 0.15 * np.sin(2 * np.pi * freq * 2 * t)
            audio += 0.1 * np.sin(2 * np.pi * freq * 3 * t)
            # Add slight robotic modulation
            audio *= (1 + 0.3 * np.sin(2 * np.pi * 5 * t))
        else:
            # Real: more natural sounding (noise + varying frequencies)
            freq = np.random.choice([150, 180, 220, 260])
            audio = 0.3 * np.sin(2 * np.pi * freq * t * (1 + 0.1 * np.sin(2 * np.pi * 0.5 * t)))
            # Add natural noise
            audio += 0.05 * np.random.randn(len(t))
            # Add breath-like pauses
            envelope = np.ones_like(t)
            for i in range(3):
                pause_start = int(np.random.uniform(0.2, 0.8) * len(t))
                pause_len = int(0.1 * sample_rate)
                envelope[pause_start:pause_start+pause_len] *= 0.1
            audio *= envelope
        
        audio = np.clip(audio, -1, 1)
        audio = (audio * 32767).astype(np.int16)
        wavfile.write(filename, sample_rate, audio)
    
    # Generate test files
    print("\n📝 Creating 20 test audio files (10 real, 10 fake)...")
    
    for i in range(10):
        generate_test_audio(f"data/real/real_sample_{i+1:03d}.wav", 
                           duration=np.random.uniform(2, 4), is_fake=False)
        generate_test_audio(f"data/fake/fake_sample_{i+1:03d}.wav", 
                           duration=np.random.uniform(2, 4), is_fake=True)
    
    print("✅ Generated 10 real and 10 fake test audio files!")
    print("\n⚠️  NOTE: These are SYNTHETIC test files for pipeline verification only.")
    print("   For real training, download actual speech datasets listed above.")
    
except ImportError:
    print("❌ scipy not installed. Installing...")
    os.system("pip install scipy")
    print("Please run this script again.")

print("\n" + "=" * 60)
print("Next steps:")
print("1. Run: python train_model.py")
print("2. For production, replace test data with real datasets")
print("=" * 60)
