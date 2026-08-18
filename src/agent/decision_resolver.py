from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from src.agent.interview_engine import (
    InterviewEngine,
    InterviewPolicyViolation,
)
from src.agent.interviewer_agent import (
    InterviewerDecision,
    QuestionType,
)
from src.state.competency_tracker import (
    CompetencyState,
)
from src.policy.interview_policy import InterviewPhase


@dataclass
class ResolutionResult:
    """
    Result of resolving an interviewer-agent decision.
    """

    decision: InterviewerDecision
    attempts: int
    correction_messages: List[str]


class DecisionResolutionError(Exception):
    """
    Raised when a valid interviewer decision cannot be
    obtained within the allowed number of attempts.
    """

    pass


class DecisionResolver:
    """
    Validates interviewer-agent decisions and retries
    when a proposed question violates interview policy.

    The resolver itself does not know how the decision
    is generated.

    A decision_generator function is supplied to it.
    Later that function will call Gemini.
    """

    def __init__(
        self,
        max_attempts: int = 3,
    ):
        if max_attempts < 1:
            raise ValueError(
                "max_attempts must be at least 1."
            )

        self.max_attempts = max_attempts


    def resolve(
        self,
        engine: InterviewEngine,
        initial_decision: InterviewerDecision,
        decision_generator: Callable[
            [str],
            InterviewerDecision,
        ],
        validation_phase: Optional[
            Callable[[InterviewerDecision], InterviewPhase]
        ] = None,
        validation_states: Optional[
            Callable[
                [InterviewerDecision],
                Dict[str, CompetencyState],
            ]
        ] = None,
    ) -> ResolutionResult:
        """
        Attempt to obtain a policy-valid next decision.

        Flow:

        initial decision
            ->
        validate
            ->
        valid: return

        invalid:
            ->
        create correction message
            ->
        decision_generator(correction)
            ->
        validate again
        """

        decision = initial_decision

        correction_messages: List[str] = []

        for attempt in range(
            1,
            self.max_attempts + 1,
        ):

            try:

                phase = (
                    validation_phase(decision)
                    if validation_phase is not None
                    else None
                )

                engine.validate_decision(
                    decision,
                    phase=phase,
                    competency_states=(
                        validation_states(decision)
                        if validation_states is not None
                        else None
                    ),
                )

                return ResolutionResult(
                    decision=decision,
                    attempts=attempt,
                    correction_messages=(
                        correction_messages
                    ),
                )

            except InterviewPolicyViolation as error:

                correction = (
                    self.build_correction_message(
                        engine=engine,
                        decision=decision,
                        violation=str(error),
                    )
                )

                correction_messages.append(
                    correction
                )

                if attempt == self.max_attempts:
                    break

                decision = (
                    decision_generator(
                        correction
                    )
                )

        raise DecisionResolutionError(
            "Unable to obtain a valid interviewer "
            f"decision after {self.max_attempts} attempts."
        )


    def build_correction_message(
        self,
        engine: InterviewEngine,
        decision: InterviewerDecision,
        violation: str,
    ) -> str:
        """
        Build an internal correction instruction for
        the interviewer agent.

        This message is never shown to the candidate.
        """

        target = (
            decision.target_competency
            if decision.target_competency
            is not None
            else "None"
        )

        remaining_sources = engine.required_question_sources_remaining()

        return f"""
Your previous proposed interview decision violated the
interview policy.

POLICY VIOLATION:
{violation}

REJECTED QUESTION TYPE:
{decision.question_type.value}

REJECTED SOURCE:
{decision.question_source.value}

REJECTED TARGET:
{target}

REJECTED QUESTION:
{decision.next_question}

CURRENT INTERVIEW PHASE:
{engine.get_phase().value}

CURRENT POLICY STATE:
{engine.format_policy_context()}

CURRENT REQUIRED/PREFERRED SOURCES:
{chr(10).join(f'- {source.value}' for source in remaining_sources) or '- None'}

CURRENT INTERVIEW PLAN:
{engine.format_plan_context()}

Generate a replacement interviewer decision.

Requirements:

1. Do not repeat the rejected action.

2. Respect the current interview phase.

3. Respect follow-up limits.

4. Do not target a competency that is already ASSESSED, except for the
   narrow missing-source coverage exception described by policy.

5. If a follow-up is unavailable, choose another eligible
   interview target when appropriate.

6. If behavioral questions are required and the interview
   is in the BEHAVIORAL phase, generate an eligible
   behavioral question.

7. Return only the structured interviewer decision.

8. Preserve the current source for a FOLLOW_UP and obey the required source
   for a NEW_TARGET.
""".strip()


def main():
    """
    Local DecisionResolver test.

    Scenario:

    Machine Learning has already consumed both allowed
    follow-ups.

    The first proposed decision incorrectly asks a third
    Machine Learning follow-up.

    The fake generator then proposes a valid SQL question.

    Gemini is NOT called.
    """

    from src.planning.interview_plan import (
        InterviewPlan,
        InterviewTarget,
    )
    from src.policy.interview_policy import (
        InterviewPolicyConfig,
    )

    print("=" * 80)
    print("DECISION RESOLVER TEST")
    print("=" * 80)

    plan = InterviewPlan(
        role="Data Scientist",
        targets=[
            InterviewTarget(
                competency="Machine Learning",
                priority="HIGH",
                reason=(
                    "Machine learning is required."
                ),
                resume_evidence=[
                    "Built machine learning models."
                ],
                evidence_expected=[
                    "model selection",
                    "validation",
                    "evaluation",
                ],
            ),
            InterviewTarget(
                competency="SQL",
                priority="HIGH",
                reason=(
                    "SQL is required for analysis."
                ),
                resume_evidence=[
                    "Built SQL data pipelines."
                ],
                evidence_expected=[
                    "joins",
                    "aggregation",
                    "query reasoning",
                ],
            ),
        ],
    )

    engine = InterviewEngine(
        role=plan.role,
        interview_targets=[
            target.competency
            for target in plan.targets
        ],
        opening_question=(
            "Tell me about yourself."
        ),
        interview_plan=plan,
        policy_config=(
            InterviewPolicyConfig(
                max_followups_per_target=2,
                behavioral_questions_required=3,
            )
        ),
    )

    engine.start_technical_phase()

    # Consume both ML follow-ups.
    engine.record_followup(
        "Machine Learning"
    )

    engine.record_followup(
        "Machine Learning"
    )

    print(
        "\nMachine Learning follow-ups used:",
        engine.get_followups_used(
            "Machine Learning"
        ),
    )

    # --------------------------------------------------------
    # Invalid proposal
    # --------------------------------------------------------

    invalid_decision = InterviewerDecision(
        next_question=(
            "Can you tell me one more thing about "
            "the machine learning model?"
        ),
        question_type=(
            QuestionType.FOLLOW_UP
        ),
        target_competency=(
            "Machine Learning"
        ),
        competency_updates={
            "Machine Learning":
                CompetencyState.EXPLORED,
        },
    )

    # --------------------------------------------------------
    # Fake replacement generator
    # --------------------------------------------------------

    generator_calls = []

    def fake_decision_generator(
        correction_message: str,
    ) -> InterviewerDecision:

        generator_calls.append(
            correction_message
        )

        return InterviewerDecision(
            next_question=(
                "How would you use SQL to combine "
                "customer and transaction data for "
                "an analysis?"
            ),
            question_type=(
                QuestionType.NEW_TARGET
            ),
            target_competency="SQL",
            competency_updates={
                "Machine Learning":
                    CompetencyState.EXPLORED,
            },
        )

    # --------------------------------------------------------
    # Resolve
    # --------------------------------------------------------

    resolver = DecisionResolver(
        max_attempts=3
    )

    result = resolver.resolve(
        engine=engine,
        initial_decision=invalid_decision,
        decision_generator=(
            fake_decision_generator
        ),
    )

    print("\nRESOLUTION RESULT")
    print("-" * 80)

    print(
        f"Attempts: {result.attempts}"
    )

    print(
        f"Question Type: "
        f"{result.decision.question_type.value}"
    )

    print(
        f"Target: "
        f"{result.decision.target_competency}"
    )

    print(
        "Question:"
    )

    print(
        result.decision.next_question
    )

    print(
        "\nCorrection messages generated:",
        len(
            result.correction_messages
        ),
    )

    print(
        "Replacement generator calls:",
        len(
            generator_calls
        ),
    )

    if result.correction_messages:

        print(
            "\nFIRST CORRECTION MESSAGE"
        )

        print("-" * 80)

        print(
            result.correction_messages[0]
        )

    print("\nGemini was NOT called.")


if __name__ == "__main__":
    main()
