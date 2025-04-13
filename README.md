# MemoTag-Cognitive-Detection
# MemoTag Cognitive Decline Detection (Proof of Concept)

This project implements a basic AI/ML pipeline to detect early cognitive stress or decline based on voice recordings.

## 🧠 Objective

Analyze voice recordings and extract speech features to cluster voice patterns that may indicate early signs of cognitive impairment.

## 📁 Structure

- `memotag_voice_analysis_fixed.py`: Main pipeline to process audio, extract features, and apply unsupervised learning.
- `audio_clips/`: Sample `.wav` voice recordings.
- `MemoTag_Report_Aastha_Verma.pdf`: Short report on methodology and insights.
- `README.md`: Project overview.

## 🔧 Requirements

Install required packages:

```bash
pip install numpy pandas matplotlib librosa scikit-learn fpdf
