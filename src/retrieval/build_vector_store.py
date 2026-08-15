import os
import shutil

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = "data/processed/interview_qa_sequences_clean.csv"

CHROMA_PATH = "data/vector_store"

COLLECTION_NAME = "interview_sequences"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 256


# ============================================================
# Load data
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Load the cleaned Q -> A -> Next-Q dataset.
    """

    print("Loading cleaned interview dataset...")

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "interview_id",
        "role",
        "turn_number",
        "question",
        "answer",
        "next_question",
        "job_description",
        "decision",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print(
        f"Loaded {len(df):,} clean interview sequences."
    )

    return df


# ============================================================
# Build retrieval documents
# ============================================================

def build_document(row: pd.Series) -> str:
    """
    Convert one structured interview sequence into the text
    that will be embedded.

    We include the role, interviewer question, and candidate
    answer because those are the strongest signals for finding
    similar historical interview situations.

    The next question is intentionally NOT embedded as part of
    the search text. It is stored as metadata and retrieved as
    the historical target/follow-up.
    """

    role = str(row["role"]).strip()
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()

    document = (
        f"Role: {role}\n"
        f"Interviewer Question: {question}\n"
        f"Candidate Answer: {answer}"
    )

    return document


# ============================================================
# Metadata cleaning
# ============================================================

def clean_metadata_value(value):
    """
    Chroma metadata values must be simple scalar values.

    Convert NaN/None to empty strings and normalize numeric
    values where needed.
    """

    if pd.isna(value):
        return ""

    return value


# ============================================================
# Create vector store
# ============================================================

def build_vector_store(
    df: pd.DataFrame,
) -> None:
    """
    Embed interview sequences and persist them in Chroma.
    """

    print(
        f"\nLoading embedding model: "
        f"{EMBEDDING_MODEL_NAME}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("Embedding model loaded.")


    # --------------------------------------------------------
    # Rebuild index from scratch
    # --------------------------------------------------------

    if os.path.exists(CHROMA_PATH):

        print(
            "\nExisting vector store found. "
            "Removing old index..."
        )

        shutil.rmtree(
            CHROMA_PATH
        )


    # --------------------------------------------------------
    # Create persistent Chroma client
    # --------------------------------------------------------

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": (
                "Historical interview "
                "Q/A/next-question sequences"
            )
        },
    )


    # --------------------------------------------------------
    # Prepare documents
    # --------------------------------------------------------

    documents = [
        build_document(row)
        for _, row in df.iterrows()
    ]

    ids = [
        f"sequence_{index}"
        for index in range(len(df))
    ]


    # --------------------------------------------------------
    # Process in batches
    # --------------------------------------------------------

    total_rows = len(df)

    print(
        f"\nCreating embeddings for "
        f"{total_rows:,} sequences..."
    )

    for start in range(
        0,
        total_rows,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total_rows,
        )

        batch_df = df.iloc[
            start:end
        ]

        batch_documents = documents[
            start:end
        ]

        batch_ids = ids[
            start:end
        ]


        # ----------------------------------------------------
        # Create embeddings
        # ----------------------------------------------------

        embeddings = model.encode(
            batch_documents,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )


        # ----------------------------------------------------
        # Build metadata
        # ----------------------------------------------------

        metadatas = []

        for _, row in batch_df.iterrows():

            metadata = {
                "interview_id": clean_metadata_value(
                    row["interview_id"]
                ),

                "role": clean_metadata_value(
                    row["role"]
                ),

                "turn_number": int(
                    row["turn_number"]
                ),

                "question": clean_metadata_value(
                    row["question"]
                ),

                "answer": clean_metadata_value(
                    row["answer"]
                ),

                "next_question": clean_metadata_value(
                    row["next_question"]
                ),

                "job_description": clean_metadata_value(
                    row["job_description"]
                ),

                "decision": clean_metadata_value(
                    row["decision"]
                ),
            }

            metadatas.append(
                metadata
            )


        # ----------------------------------------------------
        # Store batch
        # ----------------------------------------------------

        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )


        print(
            f"Indexed {end:,} / {total_rows:,}"
        )


    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    count = collection.count()

    print(
        "\nVector store build complete."
    )

    print(
        f"Stored vectors: {count:,}"
    )

    print(
        f"Vector store location: "
        f"{CHROMA_PATH}"
    )


# ============================================================
# Main
# ============================================================

def main():

    df = load_data()

    build_vector_store(
        df
    )


if __name__ == "__main__":
    main()
    