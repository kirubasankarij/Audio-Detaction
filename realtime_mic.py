import sounddevice as sd
import numpy as np
import torch
from model import AudioCNN
from features import extract_mfcc_from_array
import pandas as pd

SAMPLE_RATE = 22050
CHUNK_DURATION = 1.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

device = "cuda" if torch.cuda.is_available() else "cpu"

ESC_META = "data/ESC50/meta/esc50.csv"
df = pd.read_csv(ESC_META)
id_to_label = df.groupby("target")["category"].first().to_dict()

model = AudioCNN(num_classes=50).to(device)
model.load_state_dict(torch.load("audio_cnn_esc50.pth", map_location=device))
model.eval()

def classify_chunk(audio_chunk):
    mfcc = extract_mfcc_from_array(audio_chunk, sr=SAMPLE_RATE)
    x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    class_id = pred.item()
    label = id_to_label.get(class_id, "unknown")
    return class_id, label, conf.item()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_mono = indata[:, 0]
    class_id, label, conf = classify_chunk(audio_mono)
    if conf > 0.70:
        print(f"Detected: {label} (class {class_id}) with confidence {conf:.2f}")

def main():
    print("Listening from microphone... Press Ctrl+C to stop.")
    with sd.InputStream(channels=1,
                        samplerate=SAMPLE_RATE,
                        callback=audio_callback,
                        blocksize=CHUNK_SAMPLES):
        while True:
            sd.sleep(1000)

if __name__ == "__main__":
    main()
