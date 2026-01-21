import numpy as np
import librosa
from esc50_loader import load_esc50_metadata

# Extract MFCC from a file path (used for dataset feature extraction)
def extract_mfcc_file(file_path, n_mfcc=40, max_len=216):
    # Load audio file
    y, sr = librosa.load(file_path, sr=22050)
    # Compute MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Pad or cut to fixed length (time dimension)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

# Build feature dataset from ESC-50 (offline clips)
def build_feature_dataset():
    meta = load_esc50_metadata()
    X, y = [], []
    for idx, row in meta.iterrows():
        mfcc = extract_mfcc_file(row["file_path"])
        X.append(mfcc)
        y.append(row["target"])
        if (idx + 1) % 100 == 0:
            print("Processed", idx + 1, "files")
    X = np.array(X)
    y = np.array(y)
    np.save("features_X.npy", X)
    np.save("features_y.npy", y)
    print("Saved features:", X.shape, y.shape)

# NEW: Extract MFCC from a raw audio array (for real-time mic chunks)
def extract_mfcc_from_array(audio, sr=22050, n_mfcc=40, max_len=216):
    """
    audio: 1D numpy array of samples
    sr: sample rate (Hz)
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

if __name__ == "__main__":
    # Run this file directly to build features from ESC-50 dataset
    build_feature_dataset()
