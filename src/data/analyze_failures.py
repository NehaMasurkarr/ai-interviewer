import os

import kagglehub
import pandas as pd

from src.data.transcript_parser import (
    parse_transcript,
    create_qa_sequences,
)


# --------------------------------------------------
# Configuration
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
# Find interviews our parser cannot process
# --------------------------------------------------

failures = []

for _, row in df.iterrows():

    turns = parse_transcript(
        transcript=row["Transcript"],
        candidate_name=row["Name"],
    )

    sequences = create_qa_sequences(turns)

    if not sequences:
        failures.append(
            {
                "interview_id": row["ID"],
                "candidate_name": row["Name"],
                "role": row["Role"],
                "transcript": row["Transcript"],
            }
        )


# --------------------------------------------------
# Summary
# --------------------------------------------------

failure_df = pd.DataFrame(failures)

failure_rate = (
    len(failure_df) / len(df) * 100
    if len(df) > 0
    else 0
)

print(f"Total interviews: {len(df):,}")
print(f"Failed interviews: {len(failure_df):,}")
print(f"Failure rate: {failure_rate:.2f}%")


# --------------------------------------------------
# Inspect examples
# --------------------------------------------------

sample_size = min(5, len(failure_df))

print(f"\nShowing {sample_size} failed transcripts:")


for i in range(sample_size):

    row = failure_df.iloc[i]

    print("\n" + "=" * 100)

    print(f"INTERVIEW ID: {row['interview_id']}")
    print(f"CANDIDATE: {row['candidate_name']}")
    print(f"ROLE: {row['role']}")

    print("\nTRANSCRIPT:")
    print(row["transcript"])

    print("=" * 100)