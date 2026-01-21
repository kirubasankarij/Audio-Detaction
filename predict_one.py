import torch
import numpy as np
from model import AudioCNN
from features import extract_mfcc_file
import pandas as pd

ESC_META = "data/ESC50/meta/esc50.csv"
df = pd.read_csv(ESC_META)
id_to_label = df.groupby("target")["category"].first().to_dict()

def predict_file(path, device="cpu"):
    model = AudioCNN(num_classes=50).to(device)
    model.load_state_dict(torch.load("audio_cnn_esc50.pth", map_location=device))
    model.eval()

    mfcc = extract_mfcc_file(path)
    x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    class_id = pred.item()
    label = id_to_label.get(class_id, "unknown")
    print(f"File: {path}")
    print(f"Predicted class: {class_id} ({label}) with confidence {conf.item():.2f}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_path = "data/ESC50/audio/" + df.iloc[0]["filename"]
    predict_file(test_path, device=device)
