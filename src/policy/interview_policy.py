from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from src.policy.question_source import QuestionSource


class InterviewPhase(Enum):
    """
    High-level phases of the interview.
    """

    INTRODUCTION = "INTRODUCTION"
    TECHNICAL = "TECHNICAL"
    BEHAVIORAL = "BEHAVIORAL"
    CLOSING = "CLOSING"
    COMPLETE = "COMPLETE"


@dataclass
class InterviewPolicyConfig:
    """
    Configurable rules controlling interview structure.

    These rules are deterministic.

    The LLM can choose WHAT to ask, but it does not
    have unlimited control over HOW LONG it can remain
    on one topic or skip required interview sections.
    """

    max_followups_per_target: int = 2

    behavioral_questions_required: int = 3

    require_jd_technical: bool = True

    require_jd_scenario: bool = True

    require_resume_validation_when_evidence: bool = True


class InterviewPolicy:
    """
    Controls the structural behavior of an interview.

    Responsibilities:

    - track interview phase
    - track follow-up counts
    - enforce follow-up limits
    - track behavioral questions
    - determine when behavioral interviewing is complete

    This class does NOT generate questions.
    It does NOT call an LLM.
    """

    def __init__(
        self,
        config: Optional[InterviewPolicyConfig] = None,
    ):
        self.config = (
            config
            if config is not None
            else InterviewPolicyConfig()
        )

        self.phase = (
            InterviewPhase.INTRODUCTION
        )

        self.followup_counts: Dict[
            str,
            int,
        ] = {}

        self.behavioral_questions_completed = 0

        self.question_source_counts: Dict[QuestionSource, int] = {
            source: 0 for source in QuestionSource
        }


    # ========================================================
    # Phase management
    # ========================================================

    def get_phase(
        self,
    ) -> InterviewPhase:
        """
        Return the current interview phase.
        """

        return self.phase


    def start_technical_phase(
        self,
    ) -> None:
        """
        Move from introduction into technical interviewing.
        """

        if self.phase == InterviewPhase.INTRODUCTION:
            self.phase = InterviewPhase.TECHNICAL


    def start_behavioral_phase(
        self,
    ) -> None:
        """
        Move into the behavioral section.
        """

        if self.phase in {
            InterviewPhase.INTRODUCTION,
            InterviewPhase.TECHNICAL,
        }:
            self.phase = InterviewPhase.BEHAVIORAL


    def start_closing_phase(
        self,
    ) -> None:
        """
        Move into the closing section.
        """

        if self.phase != InterviewPhase.COMPLETE:
            self.phase = InterviewPhase.CLOSING


    def complete_interview(
        self,
    ) -> None:
        """
        Mark the interview as complete.
        """

        self.phase = InterviewPhase.COMPLETE


    # ========================================================
    # Follow-up management
    # ========================================================

    def get_followup_count(
        self,
        competency: str,
    ) -> int:
        """
        Return how many follow-up questions have already
        been used for a competency.
        """

        return self.followup_counts.get(
            competency,
            0,
        )


    def can_ask_followup(
        self,
        competency: str,
    ) -> bool:
        """
        Determine whether another follow-up is allowed
        for a competency.
        """

        count = self.get_followup_count(
            competency
        )

        return (
            count
            < self.config.max_followups_per_target
        )


    def record_followup(
        self,
        competency: str,
    ) -> bool:
        """
        Record that a follow-up question was asked.

        Returns True if the follow-up was allowed
        and recorded.

        Returns False if the competency has already
        reached its follow-up limit.
        """

        if not self.can_ask_followup(
            competency
        ):
            return False

        current_count = (
            self.get_followup_count(
                competency
            )
        )

        self.followup_counts[
            competency
        ] = current_count + 1

        return True


    def remaining_followups(
        self,
        competency: str,
    ) -> int:
        """
        Return how many follow-ups remain for a target.
        """

        used = self.get_followup_count(
            competency
        )

        remaining = (
            self.config.max_followups_per_target
            - used
        )

        return max(
            0,
            remaining,
        )


    # ========================================================
    # Behavioral section
    # ========================================================

    def record_behavioral_question(
        self,
    ) -> bool:
        """
        Record one completed behavioral question.

        Returns False if the required behavioral count
        has already been reached.
        """

        if self.behavioral_complete():
            return False

        self.behavioral_questions_completed += 1

        return True


    def behavioral_complete(
        self,
    ) -> bool:
        """
        Return True once the required number of behavioral
        questions has been completed.
        """

        return (
            self.behavioral_questions_completed
            >= self.config.behavioral_questions_required
        )


    def behavioral_questions_remaining(
        self,
    ) -> int:
        """
        Return number of required behavioral questions
        still remaining.
        """

        remaining = (
            self.config.behavioral_questions_required
            - self.behavioral_questions_completed
        )

        return max(
            0,
            remaining,
        )

    # ========================================================
    # Question-source coverage
    # ========================================================

    def record_question_source(self, source: QuestionSource) -> None:
        """Record one accepted interviewer question."""

        self.question_source_counts[source] += 1

    def get_question_source_count(self, source: QuestionSource) -> int:
        return self.question_source_counts.get(source, 0)

    def required_question_sources_remaining(
        self,
        *,
        has_resume_evidence: bool,
        has_technical_targets: bool,
    ) -> List[QuestionSource]:
        """Return missing technical coverage in deterministic order."""

        required: List[QuestionSource] = []
        if has_technical_targets and self.config.require_jd_technical:
            required.append(QuestionSource.JD_TECHNICAL)
        if has_technical_targets and self.config.require_jd_scenario:
            required.append(QuestionSource.JD_SCENARIO)
        if (
            has_resume_evidence
            and self.config.require_resume_validation_when_evidence
        ):
            required.append(QuestionSource.RESUME_VALIDATION)

        return [
            source for source in required
            if self.get_question_source_count(source) == 0
        ]


    # ========================================================
    # Formatting
    # ========================================================

    def format_policy_state(
        self,
    ) -> str:
        """
        Format current policy state for debugging and,
        later, for the interviewer-agent prompt.
        """

        lines = [
            f"Interview Phase: {self.phase.value}",
            (
                "Maximum Follow-ups Per Target: "
                f"{self.config.max_followups_per_target}"
            ),
            (
                "Behavioral Questions Required: "
                f"{self.config.behavioral_questions_required}"
            ),
            (
                "Behavioral Questions Completed: "
                f"{self.behavioral_questions_completed}"
            ),
            (
                "Behavioral Questions Remaining: "
                f"{self.behavioral_questions_remaining()}"
            ),
            (
                "Question Sources Used: "
                + ", ".join(
                    f"{source.value}={self.get_question_source_count(source)}"
                    for source in QuestionSource
                )
            ),
        ]

        if self.followup_counts:

            lines.append("")
            lines.append("Follow-ups Used:")

            for competency, count in (
                self.followup_counts.items()
            ):

                remaining = (
                    self.remaining_followups(
                        competency
                    )
                )

                lines.append(
                    f"- {competency}: "
                    f"{count} used, "
                    f"{remaining} remaining"
                )

        else:

            lines.append("")
            lines.append(
                "Follow-ups Used: None"
            )

        return "\n".join(lines)


def main():
    """
    Test the deterministic interview policy.

    No Gemini calls.
    """

    print("=" * 80)
    print("INTERVIEW POLICY TEST")
    print("=" * 80)

    policy = InterviewPolicy(
        config=InterviewPolicyConfig(
            max_followups_per_target=2,
            behavioral_questions_required=3,
        )
    )

    print("\nINITIAL STATE")
    print("-" * 80)

    print(
        policy.format_policy_state()
    )

    # --------------------------------------------------------
    # Start technical interview
    # --------------------------------------------------------

    policy.start_technical_phase()

    print("\n" + "=" * 80)
    print("TECHNICAL PHASE")
    print("=" * 80)

    print(
        f"Phase: {policy.get_phase().value}"
    )

    # --------------------------------------------------------
    # Test follow-up budget
    # --------------------------------------------------------

    competency = (
        "Time Series Forecasting"
    )

    print(
        f"\nTesting follow-ups for: "
        f"{competency}"
    )

    print(
        "Follow-up 1 allowed:",
        policy.record_followup(
            competency
        ),
    )

    print(
        "Follow-up 2 allowed:",
        policy.record_followup(
            competency
        ),
    )

    print(
        "Follow-up 3 allowed:",
        policy.record_followup(
            competency
        ),
    )

    print(
        "\nFollow-ups used:",
        policy.get_followup_count(
            competency
        ),
    )

    print(
        "Follow-ups remaining:",
        policy.remaining_followups(
            competency
        ),
    )

    # --------------------------------------------------------
    # Behavioral section
    # --------------------------------------------------------

    policy.start_behavioral_phase()

    print("\n" + "=" * 80)
    print("BEHAVIORAL PHASE")
    print("=" * 80)

    print(
        f"Phase: {policy.get_phase().value}"
    )

    for number in range(1, 5):

        recorded = (
            policy.record_behavioral_question()
        )

        print(
            f"Behavioral question {number}: "
            f"{recorded}"
        )

    print(
        "\nBehavioral complete:",
        policy.behavioral_complete(),
    )

    print(
        "Behavioral remaining:",
        policy.behavioral_questions_remaining(),
    )

    # --------------------------------------------------------
    # Closing
    # --------------------------------------------------------

    policy.start_closing_phase()

    print("\n" + "=" * 80)
    print("CLOSING")
    print("=" * 80)

    print(
        f"Phase: {policy.get_phase().value}"
    )

    policy.complete_interview()

    print(
        f"Final phase: "
        f"{policy.get_phase().value}"
    )

    # --------------------------------------------------------
    # Final state
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL POLICY STATE")
    print("=" * 80)

    print(
        policy.format_policy_state()
    )

    print("\nGemini was NOT called.")


if __name__ == "__main__":
    main()
