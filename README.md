# Audio Event Detection using ESC-50

## Problem Statement
In real-world environments (shops, malls, construction sites), critical sound events
(e.g., glass breaking, dog barking, alarms) often go unnoticed because humans cannot
monitor audio continuously. This project uses the ESC-50 environmental sound dataset
and a CNN-based classifier to automatically detect and classify sound events from audio
clips, enabling faster and safer responses.

## Dataset
- ESC-50: 2,000 clips, 50 environmental sound classes (5 seconds each).[web:98]
- Stored under `data/ESC50/audio` and `data/ESC50/meta/esc50.csv`.

## Algorithms
- MFCC feature extraction using Librosa.
- Convolutional Neural Network (CNN) in PyTorch for 50-class classification.[web:93]

## Project Structure
- `esc50_loader.py` – loads file paths and labels from ESC-50.
- `features.py` – extracts MFCC features and saves `features_X.npy`, `features_y.npy`.
- `model.py` – defines `ESC50Dataset` and `AudioCNN` model.
- `train.py` – trains CNN and saves `audio_cnn_esc50.pth`.
- `predict_one.py` – predicts class for a single audio file.

## How to Run
1. Create venv and install requirements:
   ```bash
   py -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
