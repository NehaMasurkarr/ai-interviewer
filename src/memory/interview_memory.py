from typing import Dict, List


class InterviewMemory:
    """
    Stores the question-answer history for one interview session.

    This allows the interviewer to understand what has already
    been discussed and avoid asking repetitive questions.
    """

    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def add_turn(self, question: str, answer: str) -> None:
        """
        Add one completed interview question-answer turn.
        """

        self.history.append(
            {
                "question": question.strip(),
                "answer": answer.strip(),
            }
        )

    def get_history(self) -> List[Dict[str, str]]:
        """
        Return the complete interview history.
        """

        return self.history.copy()

    def get_recent_history(self, max_turns: int = 5) -> List[Dict[str, str]]:
        """
        Return only the most recent interview turns.
        """

        if max_turns <= 0:
            return []

        return self.history[-max_turns:]

    def format_history(self, max_turns: int = 5) -> str:
        """
        Convert recent interview history into text that can
        later be inserted into the LLM prompt.
        """

        recent_history = self.get_recent_history(max_turns)

        if not recent_history:
            return "No previous interview questions have been asked."

        formatted_turns = []

        for turn_number, turn in enumerate(recent_history, start=1):
            formatted_turns.append(
                f"""
Turn {turn_number}

Question:
{turn["question"]}

Candidate Answer:
{turn["answer"]}
""".strip()
            )

        return "\n\n".join(formatted_turns)

    def clear(self) -> None:
        """
        Clear the interview session.
        """

        self.history.clear()

    def __len__(self) -> int:
        return len(self.history)


def main():
    """
    Simple test for the interview memory.
    """

    memory = InterviewMemory()

    memory.add_turn(
        question="Tell me about your experience with machine learning.",
        answer=(
            "I built classification models using Python and "
            "scikit-learn and worked with TensorFlow."
        ),
    )

    memory.add_turn(
        question="Tell me about your TensorFlow project.",
        answer=(
            "I built an image classification model using a "
            "convolutional neural network."
        ),
    )

    print("=" * 80)
    print("INTERVIEW MEMORY TEST")
    print("=" * 80)

    print(f"\nStored turns: {len(memory)}")

    print("\nFORMATTED INTERVIEW HISTORY:\n")
    print(memory.format_history())


if __name__ == "__main__":
    main()
