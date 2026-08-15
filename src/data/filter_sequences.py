import re

import pandas as pd


INPUT_PATH = "data/processed/interview_qa_sequences.csv"
OUTPUT_PATH = "data/processed/interview_qa_sequences_clean.csv"


GREETING_PATTERNS = [
    r"^good morning",
    r"^good afternoon",
    r"^good evening",
    r"^hello",
    r"^hi ",
    r"^welcome",
]

CLOSING_PATTERNS = [
    r"thank you .* for coming",
    r"thank you .* for your time",
    r"we'll be in touch",
    r"we will be in touch",
    r"have a great day",
    r"do you have any questions for me",
    r"do you have any questions",
]

CANDIDATE_QUESTION_PATTERNS = [
    r"can you tell me more about the team",
    r"what is the company culture",
    r"what's the company culture",
    r"what are the next steps",
    r"what does a typical day",
    r"what would a typical day",
]


def normalize_text(text: str) -> str:
    """
    Normalize text for rule-based filtering.
    """

    if not isinstance(text, str):
        return ""

    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)

    return text


def matches_any_pattern(
    text: str,
    patterns: list[str],
) -> bool:
    """
    Return True if the text matches any configured pattern.
    """

    normalized = normalize_text(text)

    return any(
        re.search(pattern, normalized)
        for pattern in patterns
    )


def is_useful_interview_question(
    question: str,
    answer: str,
) -> bool:
    """
    Decide whether a Q/A pair is useful for the RAG corpus.
    """

    if not isinstance(question, str):
        return False

    if not isinstance(answer, str):
        return False

    question = question.strip()
    answer = answer.strip()

    if not question or not answer:
        return False

    # --------------------------------------------------
    # Remove greetings
    # --------------------------------------------------

    if matches_any_pattern(
        question,
        GREETING_PATTERNS,
    ):
        return False

    # --------------------------------------------------
    # Remove closings / logistics
    # --------------------------------------------------

    if matches_any_pattern(
        question,
        CLOSING_PATTERNS,
    ):
        return False

    # --------------------------------------------------
    # Remove questions that are really candidate
    # logistics / company questions
    # --------------------------------------------------

    if matches_any_pattern(
        question,
        CANDIDATE_QUESTION_PATTERNS,
    ):
        return False

    # --------------------------------------------------
    # Very short exchanges usually aren't useful
    # --------------------------------------------------

    if len(question.split()) < 4:
        return False

    if len(answer.split()) < 3:
        return False

    return True


def main():

    print("Loading processed interview sequences...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Original sequences: {len(df):,}")

    # --------------------------------------------------
    # Filter
    # --------------------------------------------------

    keep_mask = df.apply(
        lambda row: is_useful_interview_question(
            row["question"],
            row["answer"],
        ),
        axis=1,
    )

    clean_df = df[keep_mask].copy()

    # --------------------------------------------------
    # Remove exact duplicate Q/A pairs
    # --------------------------------------------------

    before_dedup = len(clean_df)

    clean_df = clean_df.drop_duplicates(
        subset=[
            "question",
            "answer",
        ]
    )

    duplicates_removed = (
        before_dedup - len(clean_df)
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    clean_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    removed = (
        len(df) - before_dedup
    )

    print("\nQuality filtering complete.")

    print(
        f"Removed by quality rules: {removed:,}"
    )

    print(
        f"Exact duplicates removed: {duplicates_removed:,}"
    )

    print(
        f"Final clean sequences: {len(clean_df):,}"
    )

    print(
        f"Retention rate: "
        f"{len(clean_df) / len(df) * 100:.2f}%"
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )

    # --------------------------------------------------
    # Show one clean example
    # --------------------------------------------------

    if not clean_df.empty:

        sample = clean_df.iloc[0]

        print("\nSample clean sequence:")

        print("\nQUESTION:")
        print(sample["question"])

        print("\nANSWER:")
        print(sample["answer"])

        print("\nNEXT QUESTION:")
        print(sample["next_question"])


if __name__ == "__main__":
    main()