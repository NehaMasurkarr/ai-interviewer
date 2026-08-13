import json
import pandas as pd

file_path = "data/scripts/interview_001.json"

with open(file_path, "r") as file:
    data = json.load(file)

conversation = data["conversation"]

qa_pairs = []

for i in range(0, len(conversation) - 2, 2):
    question = conversation[i]["text"]
    answer = conversation[i + 1]["text"]
    next_question = conversation[i + 2]["text"]

    qa_pair = {
        "interview_id": data["interview_id"],
        "role": data["role"],
        "difficulty": data["difficulty"],
        "question": question,
        "answer": answer,
        "next_question": next_question
    }

    qa_pairs.append(qa_pair)

df = pd.DataFrame(qa_pairs)

print(df)

output_path = "data/processed/interview_qa_pairs.csv"

df.to_csv(output_path, index=False)

print("Dataset saved successfully!")