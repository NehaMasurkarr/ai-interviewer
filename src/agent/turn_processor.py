from dataclasses import dataclass
from typing import Dict

from src.agent.interviewer_agent import (
    InterviewerDecision,
)
from src.memory.interview_memory import (
    InterviewMemory,
)
from src.state.competency_tracker import (
    CompetencyState,
    CompetencyTracker,
)


@dataclass
class TurnResult:
    """
    Result of processing one completed interview turn.
    """

    current_question: str
    candidate_answer: str
    next_question: str

    competency_changes: Dict[
        str,
        CompetencyState,
    ]


class TurnProcessor:
    """
    Coordinates interview state after the interviewer
    agent has evaluated a candidate response.

    Responsibilities:

    1. Store the completed question and answer in memory.
    2. Apply competency updates safely.
    3. Return the next interview question.

    This class does NOT call an LLM.
    """

    def __init__(
        self,
        memory: InterviewMemory,
        competency_tracker: CompetencyTracker,
    ):
        self.memory = memory

        self.competency_tracker = (
            competency_tracker
        )


    def process_turn(
        self,
        current_question: str,
        candidate_answer: str,
        decision: InterviewerDecision,
    ) -> TurnResult:
        """
        Process one completed candidate response.
        """

        current_question = (
            current_question.strip()
        )

        candidate_answer = (
            candidate_answer.strip()
        )

        next_question = (
            decision.next_question.strip()
        )

        if not current_question:
            raise ValueError(
                "Current question cannot be empty."
            )

        if not candidate_answer:
            raise ValueError(
                "Candidate answer cannot be empty."
            )

        if not next_question:
            raise ValueError(
                "Next question cannot be empty."
            )

        # Save the completed interview turn.
        self.memory.add_turn(
            question=current_question,
            answer=candidate_answer,
        )

        # Safely apply competency updates.
        # The tracker prevents state regression.
        changed_states = (
            self.competency_tracker.apply_updates(
                decision.competency_updates
            )
        )

        return TurnResult(
            current_question=current_question,
            candidate_answer=candidate_answer,
            next_question=next_question,
            competency_changes=changed_states,
        )


def main():
    """
    Local integration test.

    No Gemini calls are made.
    """

    print("=" * 80)
    print("TURN PROCESSOR TEST")
    print("=" * 80)

    memory = InterviewMemory()

    tracker = CompetencyTracker(
        competencies=[
            "Python",
            "SQL",
            "Statistical Modeling",
            "Machine Learning",
            "Communication",
        ]
    )

    processor = TurnProcessor(
        memory=memory,
        competency_tracker=tracker,
    )

    # --------------------------------------------------------
    # TURN 1
    # --------------------------------------------------------

    question_1 = (
        "Tell me about a machine learning "
        "project you worked on."
    )

    answer_1 = (
        "I built a customer churn model using "
        "Python and scikit-learn. I compared "
        "logistic regression and random forest "
        "and evaluated the models using "
        "precision, recall, F1-score, and ROC-AUC."
    )

    decision_1 = InterviewerDecision(
        next_question=(
            "How did you validate the model and "
            "decide which approach should be used?"
        ),
        competency_updates={
            "Python":
                CompetencyState.MENTIONED,

            "Machine Learning":
                CompetencyState.EXPLORED,

            "Statistical Modeling":
                CompetencyState.MENTIONED,
        },
    )

    result_1 = processor.process_turn(
        current_question=question_1,
        candidate_answer=answer_1,
        decision=decision_1,
    )

    print("\nTURN 1 COMPLETE")

    print("\nNEXT QUESTION")
    print(
        result_1.next_question
    )

    print("\nSTATE CHANGES")

    for competency, state in (
        result_1.competency_changes.items()
    ):
        print(
            f"- {competency}: "
            f"{state.value}"
        )

    # --------------------------------------------------------
    # TURN 2
    # --------------------------------------------------------

    question_2 = (
        result_1.next_question
    )

    answer_2 = (
        "I used cross-validation and compared "
        "the models using F1-score and ROC-AUC. "
        "The random forest performed better, "
        "but I also checked feature importance "
        "and whether the improvement justified "
        "the additional complexity."
    )

    decision_2 = InterviewerDecision(
        next_question=(
            "How would you use SQL to prepare "
            "customer-level features for this model?"
        ),
        competency_updates={
            "Machine Learning":
                CompetencyState.ASSESSED,

            "Statistical Modeling":
                CompetencyState.EXPLORED,

            # Deliberately lower than the existing
            # Python state. This should be ignored.
            "Python":
                CompetencyState.NOT_COVERED,
        },
    )

    result_2 = processor.process_turn(
        current_question=question_2,
        candidate_answer=answer_2,
        decision=decision_2,
    )

    print("\n" + "=" * 80)
    print("TURN 2 COMPLETE")
    print("=" * 80)

    print("\nNEXT QUESTION")
    print(
        result_2.next_question
    )

    print("\nSTATE CHANGES")

    for competency, state in (
        result_2.competency_changes.items()
    ):
        print(
            f"- {competency}: "
            f"{state.value}"
        )

    # --------------------------------------------------------
    # FINAL STATE
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("INTERVIEW STATE")
    print("=" * 80)

    print("\nCOMPETENCIES")

    print(
        tracker.format_status()
    )

    print("\nINTERVIEW MEMORY")

    print(
        memory.format_history()
    )

    print("\nGemini was NOT called.")


if __name__ == "__main__":
    main()