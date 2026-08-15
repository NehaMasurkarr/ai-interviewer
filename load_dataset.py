import os

import kagglehub
import pandas as pd


# --------------------------------------------------
# Dataset configuration
# --------------------------------------------------

DATASET_NAME = "yaswanthkumary/ai-recruitment-pipeline-dataset"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

dataset_path = kagglehub.dataset_download(DATASET_NAME)

csv_files = [
    file_name
    for file_name in os.listdir(dataset_path)
    if file_name.endswith(".csv")
]

if not csv_files:
    raise FileNotFoundError("No CSV file found.")

csv_path = os.path.join(dataset_path, csv_files[0])

df = pd.read_csv(csv_path)


# --------------------------------------------------
# Dataset overview
# --------------------------------------------------

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------
# Inspect interview transcript
# --------------------------------------------------

print("\n" + "=" * 80)
print("INTERVIEW TRANSCRIPT")
print("=" * 80)

print(df.iloc[0]["Transcript"])