import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

CHROMA_PATH = "data/vector_store"

COLLECTION_NAME = "interview_sequences"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# Interview Retriever
# ============================================================

class InterviewRetriever:
    """
    Retrieve historical interview situations that are similar
    to the current interview question and candidate answer.
    """

    def __init__(self):

        print(
            f"Loading embedding model: "
            f"{EMBEDDING_MODEL_NAME}"
        )

        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        print(
            f"Retriever ready. "
            f"{self.collection.count():,} vectors available."
        )


    # ========================================================
    # Build query
    # ========================================================

    @staticmethod
    def build_query(
        role: str,
        question: str,
        answer: str,
    ) -> str:

        return (
            f"Role: {role.strip()}\n"
            f"Interviewer Question: {question.strip()}\n"
            f"Candidate Answer: {answer.strip()}"
        )


    # ========================================================
    # Retrieve similar interview situations
    # ========================================================

    def retrieve(
        self,
        role: str,
        question: str,
        answer: str,
        top_k: int = 5,
    ) -> list[dict]:

        query_text = self.build_query(
            role=role,
            question=question,
            answer=answer,
        )

        query_embedding = self.model.encode(
            query_text,
            normalize_embeddings=True,
        )

        results = self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        retrieved = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):

            retrieved.append(
                {
                    "document": document,
                    "role": metadata.get(
                        "role",
                        "",
                    ),
                    "question": metadata.get(
                        "question",
                        "",
                    ),
                    "answer": metadata.get(
                        "answer",
                        "",
                    ),
                    "next_question": metadata.get(
                        "next_question",
                        "",
                    ),
                    "distance": float(distance),
                }
            )

        return retrieved


# ============================================================
# Manual test
# ============================================================

def main():

    retriever = InterviewRetriever()

    role = "Data Scientist"

    question = (
        "Tell me about your experience "
        "with machine learning."
    )

    answer = (
        "I built classification models using "
        "Python and scikit-learn. I also worked "
        "with TensorFlow on a deep learning project."
    )

    results = retriever.retrieve(
        role=role,
        question=question,
        answer=answer,
        top_k=5,
    )

    print("\n" + "=" * 80)

    print("CURRENT INTERVIEW")

    print("=" * 80)

    print(f"\nROLE:\n{role}")

    print(
        f"\nQUESTION:\n{question}"
    )

    print(
        f"\nANSWER:\n{answer}"
    )

    print(
        "\n\nTOP RETRIEVED "
        "INTERVIEW SEQUENCES"
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        print("\n" + "-" * 80)

        print(
            f"RESULT {index}"
        )

        print(
            f"\nDistance: "
            f"{result['distance']:.4f}"
        )

        print(
            f"\nRole:\n"
            f"{result['role']}"
        )

        print(
            f"\nHistorical Question:\n"
            f"{result['question']}"
        )

        print(
            f"\nHistorical Answer:\n"
            f"{result['answer']}"
        )

        print(
            f"\nHistorical Next Question:\n"
            f"{result['next_question']}"
        )


if __name__ == "__main__":
    main()