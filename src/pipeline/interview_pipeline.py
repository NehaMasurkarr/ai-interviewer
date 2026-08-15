from src.generation.gemini_generator import generate_next_question
from src.generation.prompt_builder import build_interviewer_prompt
from src.retrieval.retriever import InterviewRetriever


class InterviewPipeline:
    """
    End-to-end RAG interview pipeline.

    Flow:

        role + job description
                +
        current interview question
                +
        candidate answer
                ↓
        retrieve similar historical interviews
                ↓
        build RAG prompt
                ↓
        Gemini
                ↓
        next interview question
    """

    def __init__(self):
        self.retriever = InterviewRetriever()

    def generate_next_question(
        self,
        role: str,
        job_description: str,
        current_question: str,
        candidate_answer: str,
        top_k: int = 5,
    ) -> dict:
        """
        Generate the next interview question using RAG.
        """

        # --------------------------------------------------
        # 1. Retrieve similar historical interview examples
        # --------------------------------------------------

        retrieved_examples = self.retriever.retrieve(
            role=role,
            question=current_question,
            answer=candidate_answer,
            top_k=top_k,
        )

        # --------------------------------------------------
        # 2. Build the RAG prompt
        # --------------------------------------------------

        prompt = build_interviewer_prompt(
            role=role,
            job_description=job_description,
            current_question=current_question,
            candidate_answer=candidate_answer,
            retrieved_examples=retrieved_examples,
        )

        # --------------------------------------------------
        # 3. Send prompt to Gemini
        # --------------------------------------------------

        next_question = generate_next_question(
            prompt
        )

        # --------------------------------------------------
        # 4. Return structured result
        # --------------------------------------------------

        return {
            "next_question": next_question,
            "retrieved_examples": retrieved_examples,
        }


def main():
    """
    End-to-end manual test.
    """

    pipeline = InterviewPipeline()

    role = "Data Scientist"

    job_description = """
We are looking for a Data Scientist with experience in
machine learning, Python, statistical modeling, deep learning,
and communicating technical insights to stakeholders.
"""

    current_question = (
        "Tell me about your experience with machine learning."
    )

    candidate_answer = (
        "I built classification models using Python and scikit-learn. "
        "I also worked with TensorFlow on a deep learning project "
        "for image classification."
    )

    print("\n" + "=" * 80)
    print("RUNNING END-TO-END RAG INTERVIEW PIPELINE")
    print("=" * 80)

    result = pipeline.generate_next_question(
        role=role,
        job_description=job_description,
        current_question=current_question,
        candidate_answer=candidate_answer,
        top_k=5,
    )

    print("\nCURRENT QUESTION:")
    print(current_question)

    print("\nCANDIDATE ANSWER:")
    print(candidate_answer)

    print("\nGENERATED NEXT QUESTION:")
    print(result["next_question"])

    print("\n" + "=" * 80)
    print("RETRIEVED HISTORICAL EXAMPLES")
    print("=" * 80)

    for index, example in enumerate(
        result["retrieved_examples"],
        start=1,
    ):
        print("\n" + "-" * 80)
        print(f"EXAMPLE {index}")

        print("\nHistorical Question:")
        print(example["question"])

        print("\nHistorical Answer:")
        print(example["answer"])

        print("\nHistorical Next Question:")
        print(example["next_question"])

        print(
            f"\nDistance: "
            f"{example['distance']:.4f}"
        )


if __name__ == "__main__":
    main()