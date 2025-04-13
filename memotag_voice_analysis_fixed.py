
import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    features = {
        "zcr": np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)[0],
        "rmse": np.mean(librosa.feature.rms(y).T, axis=0)[0],
        "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)[0],
        "spectral_bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)[0],
        "rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)[0],
        "mfcc_mean": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
    }
    return features

def main_pipeline(audio_dir="audio_clips", n_clusters=2):
    data = []
    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav"):
            path = os.path.join(audio_dir, filename)
            features = extract_features(path)
            features["filename"] = filename
            data.append(features)

    df = pd.DataFrame(data)

    # Drop transcript and filename columns safely
    for col in ["transcript", "filename"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df["cluster"] = kmeans.fit_predict(scaled_features)

    # Save output
    df.to_csv("audio_features_with_clusters.csv", index=False)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=df["cluster"])
    plt.title("Audio Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig("clustering_plot.png")
    plt.close()

    return df, scaler, kmeans

if __name__ == "__main__":
    df, scaler, kmeans = main_pipeline()
    print(df.head())
