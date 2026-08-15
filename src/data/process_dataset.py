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

OUTPUT_PATH = "data/processed/interview_qa_sequences.csv"


# --------------------------------------------------
# Load raw Kaggle dataset
# --------------------------------------------------

print("Loading dataset...")

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

print(f"Loaded {len(df):,} interviews.")


# --------------------------------------------------
# Parse every interview
# --------------------------------------------------

processed_rows = []

failed_interviews = 0

for _, row in df.iterrows():

    interview_id = row["ID"]
    candidate_name = row["Name"]
    role = row["Role"]
    transcript = row["Transcript"]
    job_description = row["Job_Description"]
    decision = row["decision"]

    turns = parse_transcript(
        transcript=transcript,
        candidate_name=candidate_name,
    )

    sequences = create_qa_sequences(turns)

    if not sequences:
        failed_interviews += 1
        continue

    for turn_number, sequence in enumerate(sequences, start=1):

        processed_rows.append(
            {
                "interview_id": interview_id,
                "role": role,
                "turn_number": turn_number,
                "question": sequence["question"],
                "answer": sequence["answer"],
                "next_question": sequence["next_question"],
                "job_description": job_description,
                "decision": decision,
            }
        )


# --------------------------------------------------
# Create processed DataFrame
# --------------------------------------------------

processed_df = pd.DataFrame(processed_rows)


# --------------------------------------------------
# Save processed dataset
# --------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True,
)

processed_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# --------------------------------------------------
# Validation summary
# --------------------------------------------------

print("\nProcessing complete.")

print(f"Original interviews: {len(df):,}")
print(f"Successfully parsed: {len(df) - failed_interviews:,}")
print(f"Failed interviews: {failed_interviews:,}")
print(f"Q/A sequences created: {len(processed_df):,}")

print("\nProcessed columns:")
print(processed_df.columns.tolist())

print("\nSample sequence:")

if not processed_df.empty:

    sample = processed_df.iloc[0]

    print(f"\nRole: {sample['role']}")
    print(f"Turn: {sample['turn_number']}")

    print("\nQUESTION:")
    print(sample["question"])

    print("\nANSWER:")
    print(sample["answer"])

    print("\nNEXT QUESTION:")
    print(sample["next_question"])

print(f"\nSaved to: {OUTPUT_PATH}")