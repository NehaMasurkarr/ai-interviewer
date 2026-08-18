from src.generation.gemini_generator import generate_next_question
from src.generation.prompt_builder import build_interviewer_prompt
from src.memory.interview_memory import InterviewMemory
from src.retrieval.retriever import InterviewRetriever


class InterviewPipeline:
    """
    End-to-end RAG interview pipeline with session memory.
    """

    def __init__(self):
        self.retriever = InterviewRetriever()
        self.memory = InterviewMemory()

    def generate_next_question(
        self,
        role: str,
        job_description: str,
        current_question: str,
        candidate_answer: str,
        top_k: int = 5,
    ) -> dict:
        """
        Generate the next question while preserving
        the interview history.
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
        # 2. Get interview history BEFORE adding current turn
        # --------------------------------------------------

        interview_history = self.memory.format_history(
            max_turns=5
        )

        # --------------------------------------------------
        # 3. Build RAG prompt
        # --------------------------------------------------

        prompt = build_interviewer_prompt(
            role=role,
            job_description=job_description,
            current_question=current_question,
            candidate_answer=candidate_answer,
            retrieved_examples=retrieved_examples,
            interview_history=interview_history,
        )

        # --------------------------------------------------
        # 4. Generate next question
        # --------------------------------------------------

        next_question = generate_next_question(
            prompt
        )

        # --------------------------------------------------
        # 5. Store completed current turn in memory
        # --------------------------------------------------

        self.memory.add_turn(
            question=current_question,
            answer=candidate_answer,
        )

        # --------------------------------------------------
        # 6. Return structured output
        # --------------------------------------------------

        return {
            "next_question": next_question,
            "retrieved_examples": retrieved_examples,
            "history": self.memory.get_history(),
        }

    def reset_interview(self) -> None:
        """
        Start a completely new interview session.
        """

        self.memory.clear()


def main():
    """
    Test a multi-turn interview.
    """

    pipeline = InterviewPipeline()

    role = "Data Scientist"

    job_description = """
We are looking for a Data Scientist with experience in
machine learning, Python, statistical modeling, deep learning,
and communicating technical insights to stakeholders.
"""

    # ========================================================
    # TURN 1
    # ========================================================

    question_1 = (
        "Tell me about your experience with machine learning."
    )

    answer_1 = (
        "I built classification models using Python and "
        "scikit-learn. I also worked with TensorFlow on "
        "an image classification project."
    )

    print("\n" + "=" * 80)
    print("TURN 1")
    print("=" * 80)

    print("\nQUESTION:")
    print(question_1)

    print("\nANSWER:")
    print(answer_1)

    result_1 = pipeline.generate_next_question(
        role=role,
        job_description=job_description,
        current_question=question_1,
        candidate_answer=answer_1,
    )

    question_2 = result_1["next_question"]

    print("\nGENERATED NEXT QUESTION:")
    print(question_2)

    # ========================================================
    # TURN 2
    # ========================================================

    answer_2 = (
        "The project classified medical images into multiple "
        "categories. I used a convolutional neural network "
        "in TensorFlow and evaluated it using precision, "
        "recall, and F1-score."
    )

    print("\n" + "=" * 80)
    print("TURN 2")
    print("=" * 80)

    print("\nQUESTION:")
    print(question_2)

    print("\nANSWER:")
    print(answer_2)

    result_2 = pipeline.generate_next_question(
        role=role,
        job_description=job_description,
        current_question=question_2,
        candidate_answer=answer_2,
    )

    print("\nGENERATED NEXT QUESTION:")
    print(result_2["next_question"])

    # ========================================================
    # Show stored memory
    # ========================================================

    print("\n" + "=" * 80)
    print("INTERVIEW MEMORY")
    print("=" * 80)

    print(
        pipeline.memory.format_history(
            max_turns=10
        )
    )


if __name__ == "__main__":
    main()