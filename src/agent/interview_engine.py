from typing import Dict, List, Optional

from src.agent.interviewer_agent import (
    InterviewerDecision,
    QuestionType,
)
from src.agent.turn_processor import (
    TurnProcessor,
    TurnResult,
)
from src.memory.interview_memory import (
    InterviewMemory,
)
from src.planning.interview_plan import (
    InterviewPlan,
    InterviewTarget,
)
from src.policy.interview_policy import (
    InterviewPhase,
    InterviewPolicy,
    InterviewPolicyConfig,
)
from src.policy.question_source import QuestionSource
from src.state.competency_tracker import (
    CompetencyState,
    CompetencyTracker,
    STATE_RANK,
)


class InterviewPolicyViolation(Exception):
    """
    Raised when the interviewer agent proposes a next
    question that violates deterministic interview policy.
    """

    pass


class InterviewEngine:
    """
    Central controller for an active interview.

    The engine owns:

    - InterviewPlan
    - current question
    - current question metadata
    - interview memory
    - competency coverage
    - interview phase
    - follow-up budgets
    - behavioral requirements

    The LLM proposes interview actions.

    The engine decides whether those actions are allowed.
    """

    def __init__(
        self,
        role: str,
        interview_targets: List[str],
        opening_question: str,
        interview_plan: Optional[InterviewPlan] = None,
        policy_config: Optional[
            InterviewPolicyConfig
        ] = None,
    ):
        self.role = role

        self.interview_targets = list(
            interview_targets
        )

        self.interview_plan = interview_plan

        self.current_question = (
            opening_question.strip()
        )

        if not self.current_question:
            raise ValueError(
                "Opening question cannot be empty."
            )

        # The opening question is treated separately from
        # technical / behavioral / closing questions.
        self.current_question_type: Optional[
            QuestionType
        ] = None

        self.current_target_competency: Optional[
            str
        ] = None

        self.current_question_source = QuestionSource.OPENING

        self.memory = InterviewMemory()

        self.competency_tracker = (
            CompetencyTracker(
                competencies=self.interview_targets
            )
        )

        self.policy = InterviewPolicy(
            config=policy_config
        )
        self.policy.record_question_source(QuestionSource.OPENING)

        self.turn_processor = (
            TurnProcessor(
                memory=self.memory,
                competency_tracker=(
                    self.competency_tracker
                ),
            )
        )


    # ========================================================
    # Decision validation
    # ========================================================

    def validate_decision(
        self,
        decision: InterviewerDecision,
        phase: Optional[InterviewPhase] = None,
        competency_states: Optional[
            Dict[str, CompetencyState]
        ] = None,
    ) -> None:
        """
        Validate an interviewer-agent proposal against
        deterministic interview policy.

        This method does NOT mutate interview state.
        """

        question_type = (
            decision.question_type
        )

        target = (
            decision.target_competency
        )

        if phase is None:
            phase = self.get_phase()

        self._validate_question_source(
            decision, phase=phase, competency_states=competency_states
        )

        # ----------------------------------------------------
        # FOLLOW-UP
        # ----------------------------------------------------

        if question_type == QuestionType.FOLLOW_UP:

            if phase != InterviewPhase.TECHNICAL:
                raise InterviewPolicyViolation(
                    "FOLLOW_UP questions are only allowed "
                    "during the TECHNICAL phase."
                )

            if target is None:
                raise InterviewPolicyViolation(
                    "FOLLOW_UP requires a target competency."
                )

            if not self.can_ask_followup(
                target
            ):
                raise InterviewPolicyViolation(
                    "Follow-up limit reached for "
                    f"{target}."
                )

            target_state = (
                competency_states.get(target)
                if competency_states is not None
                else self.competency_tracker.get_state(target)
            )

            if target_state == CompetencyState.ASSESSED:
                raise InterviewPolicyViolation(
                    f"{target} is already ASSESSED. "
                    "Move to another target."
                )

            return

        # ----------------------------------------------------
        # NEW TARGET
        # ----------------------------------------------------

        if question_type == QuestionType.NEW_TARGET:

            if phase not in {
                InterviewPhase.INTRODUCTION,
                InterviewPhase.TECHNICAL,
            }:
                raise InterviewPolicyViolation(
                    "NEW_TARGET questions are only allowed "
                    "during INTRODUCTION or TECHNICAL."
                )

            if target is None:
                raise InterviewPolicyViolation(
                    "NEW_TARGET requires a target competency."
                )

            if self.get_target(target) is None:
                raise InterviewPolicyViolation(
                    "Unknown interview target: "
                    f"{target}."
                )

            target_state = (
                competency_states.get(target)
                if competency_states is not None
                else self.competency_tracker.get_state(target)
            )

            if (
                target_state == CompetencyState.ASSESSED
                and decision.question_source
                not in self.required_question_sources_remaining()
            ):
                raise InterviewPolicyViolation(
                    f"{target} is already ASSESSED."
                )

            return

        # ----------------------------------------------------
        # BEHAVIORAL
        # ----------------------------------------------------

        if question_type == QuestionType.BEHAVIORAL:

            if phase != InterviewPhase.BEHAVIORAL:
                raise InterviewPolicyViolation(
                    "BEHAVIORAL questions are only allowed "
                    "during the BEHAVIORAL phase."
                )

            if self.behavioral_complete():
                raise InterviewPolicyViolation(
                    "All required behavioral questions "
                    "have already been completed."
                )

            if target is None:
                raise InterviewPolicyViolation(
                    "BEHAVIORAL requires a target competency."
                )

            return

        # ----------------------------------------------------
        # CLOSING
        # ----------------------------------------------------

        if question_type == QuestionType.CLOSING:

            if phase != InterviewPhase.CLOSING:
                raise InterviewPolicyViolation(
                    "CLOSING questions are only allowed "
                    "during the CLOSING phase."
                )

            if target is not None:
                raise InterviewPolicyViolation(
                    "CLOSING must not have a target "
                    "competency."
                )

            return

        raise InterviewPolicyViolation(
            "Unsupported question type."
        )

    def _validate_question_source(
        self,
        decision: InterviewerDecision,
        *,
        phase: InterviewPhase,
        competency_states=None,
    ) -> None:
        source = decision.question_source
        target = decision.target_competency

        if not isinstance(source, QuestionSource):
            raise InterviewPolicyViolation("Unsupported question source.")

        phase_sources = {
            InterviewPhase.INTRODUCTION: {QuestionSource.OPENING},
            InterviewPhase.TECHNICAL: {
                QuestionSource.RESUME_VALIDATION,
                QuestionSource.JD_TECHNICAL,
                QuestionSource.JD_SCENARIO,
            },
            InterviewPhase.BEHAVIORAL: {QuestionSource.BEHAVIORAL},
            InterviewPhase.CLOSING: {QuestionSource.CLOSING},
            InterviewPhase.COMPLETE: set(),
        }
        if source not in phase_sources[phase]:
            raise InterviewPolicyViolation(
                f"{source.value} source is not allowed during {phase.value}."
            )

        expected = {
            QuestionType.BEHAVIORAL: QuestionSource.BEHAVIORAL,
            QuestionType.CLOSING: QuestionSource.CLOSING,
        }.get(decision.question_type)
        if expected is not None and source != expected:
            raise InterviewPolicyViolation(
                f"{decision.question_type.value} must use {expected.value}."
            )

        if decision.question_type == QuestionType.FOLLOW_UP:
            if (
                self.current_question_source != QuestionSource.OPENING
                and source != self.current_question_source
            ):
                raise InterviewPolicyViolation(
                    "FOLLOW_UP must preserve current question source "
                    f"{self.current_question_source.value}."
                )

        plan_target = self.get_target(target) if target is not None else None
        if source == QuestionSource.RESUME_VALIDATION and (
            plan_target is None or not plan_target.resume_evidence
        ):
            raise InterviewPolicyViolation(
                "RESUME_VALIDATION requires resume evidence for the target."
            )
        if source in {QuestionSource.JD_TECHNICAL, QuestionSource.JD_SCENARIO}:
            if plan_target is None or not plan_target.reason.strip():
                raise InterviewPolicyViolation(
                    f"{source.value} requires a job-relevant interview target."
                )

        remaining = self.required_question_sources_remaining()
        if (
            phase == InterviewPhase.TECHNICAL
            and decision.question_type == QuestionType.NEW_TARGET
            and remaining
            and source != remaining[0]
        ):
            raise InterviewPolicyViolation(
                f"{remaining[0].value} is currently required; rejected "
                f"source {source.value}."
            )


    # ========================================================
    # Decision acceptance
    # ========================================================

    def accept_next_question(
        self,
        decision: InterviewerDecision,
    ) -> None:
        """
        Validate and accept the next question proposed by
        the interviewer agent.

        This is where structural policy mutations occur.

        FOLLOW_UP:
            consumes one follow-up from the target's budget.

        BEHAVIORAL:
            does NOT increment the behavioral count yet.
            The question must first be answered.

        NEW_TARGET:
            does not consume follow-up budget.

        CLOSING:
            does not immediately complete the interview.
            The candidate still needs to answer it.
        """

        self.validate_decision(
            decision
        )

        if (
            decision.question_type
            == QuestionType.FOLLOW_UP
        ):
            recorded = self.record_followup(
                decision.target_competency
            )

            if not recorded:
                raise InterviewPolicyViolation(
                    "Unable to record follow-up for "
                    f"{decision.target_competency}."
                )

        self.current_question = (
            decision.next_question
        )

        self.current_question_type = (
            decision.question_type
        )

        self.current_target_competency = (
            decision.target_competency
        )

        self.current_question_source = decision.question_source
        self.policy.record_question_source(decision.question_source)


    # ========================================================
    # Turn processing
    # ========================================================

    def process_candidate_answer(
        self,
        candidate_answer: str,
        decision: InterviewerDecision,
        next_phase: Optional[InterviewPhase] = None,
    ) -> TurnResult:
        """
        Process the candidate's answer to the CURRENT
        question and then accept the proposed NEXT question.

        Important:

        competency_updates describe the question that was
        just answered.

        question_type and target_competency describe the
        NEXT question.
        """

        candidate_answer = candidate_answer.strip()

        if not candidate_answer:
            raise ValueError(
                "Candidate answer cannot be empty."
            )

        validation_phase = (
            next_phase
            if next_phase is not None
            else self.get_phase()
        )

        projected_states = self.get_competency_states()

        for competency, state in decision.competency_updates.items():
            if competency not in projected_states:
                continue
            current_state = projected_states[competency]
            if STATE_RANK[state] > STATE_RANK[current_state]:
                projected_states[competency] = state

        # Validate before TurnProcessor mutates memory or
        # competency state. This makes a rejected decision a
        # no-op from the engine's perspective.
        self.validate_decision(
            decision,
            phase=validation_phase,
            competency_states=projected_states,
        )

        answered_question_type = (
            self.current_question_type
        )

        # Process the completed question/answer first.
        result = (
            self.turn_processor.process_turn(
                current_question=(
                    self.current_question
                ),
                candidate_answer=(
                    candidate_answer
                ),
                decision=decision,
            )
        )

        self._sync_plan_states()

        # A behavioral question counts only after the
        # candidate has actually answered it.
        if (
            answered_question_type
            == QuestionType.BEHAVIORAL
        ):
            self.record_behavioral_question()

        if next_phase is not None:
            self.transition_to_phase(next_phase)

        # The next question proposed by the agent must
        # satisfy policy before becoming current.
        self.accept_next_question(
            decision
        )

        return result


    def record_final_answer(
        self,
        candidate_answer: str,
    ) -> TurnResult:
        """Commit the answer to the closing question."""

        if self.get_phase() != InterviewPhase.CLOSING:
            raise InterviewPolicyViolation(
                "A final answer can only be recorded "
                "during the CLOSING phase."
            )

        candidate_answer = candidate_answer.strip()

        if not candidate_answer:
            raise ValueError(
                "Candidate answer cannot be empty."
            )

        self.memory.add_turn(
            question=self.current_question,
            answer=candidate_answer,
        )

        self.complete_interview()

        return TurnResult(
            current_question=self.current_question,
            candidate_answer=candidate_answer,
            next_question="",
            competency_changes={},
        )


    # ========================================================
    # Interview phase
    # ========================================================

    def get_phase(
        self,
    ) -> InterviewPhase:
        """
        Return current interview phase.
        """

        return self.policy.get_phase()


    def start_technical_phase(
        self,
    ) -> None:
        """
        Move into technical interviewing.
        """

        self.policy.start_technical_phase()


    def start_behavioral_phase(
        self,
    ) -> None:
        """
        Move into behavioral interviewing.
        """

        self.policy.start_behavioral_phase()


    def start_closing_phase(
        self,
    ) -> None:
        """
        Move into closing.
        """

        self.policy.start_closing_phase()


    def complete_interview(
        self,
    ) -> None:
        """
        Mark interview complete.
        """

        self.policy.complete_interview()


    def transition_to_phase(
        self,
        phase: InterviewPhase,
    ) -> None:
        """Apply one of the policy's existing phase transitions."""

        if phase == self.get_phase():
            return

        if phase == InterviewPhase.TECHNICAL:
            self.start_technical_phase()
        elif phase == InterviewPhase.BEHAVIORAL:
            self.start_behavioral_phase()
        elif phase == InterviewPhase.CLOSING:
            self.start_closing_phase()
        elif phase == InterviewPhase.COMPLETE:
            self.complete_interview()
        else:
            raise InterviewPolicyViolation(
                f"Unsupported phase transition: {phase.value}."
            )

        if self.get_phase() != phase:
            raise InterviewPolicyViolation(
                "Interview policy rejected transition to "
                f"{phase.value}."
            )


    # ========================================================
    # Follow-up control
    # ========================================================

    def can_ask_followup(
        self,
        competency: str,
    ) -> bool:
        """
        Return whether another follow-up is allowed.
        """

        return self.policy.can_ask_followup(
            competency
        )


    def record_followup(
        self,
        competency: str,
    ) -> bool:
        """
        Consume one follow-up from a competency budget.
        """

        return self.policy.record_followup(
            competency
        )


    def get_followups_used(
        self,
        competency: str,
    ) -> int:
        """
        Return follow-ups already used.
        """

        return self.policy.get_followup_count(
            competency
        )


    def get_followups_remaining(
        self,
        competency: str,
    ) -> int:
        """
        Return follow-ups remaining.
        """

        return self.policy.remaining_followups(
            competency
        )


    # ========================================================
    # Behavioral control
    # ========================================================

    def record_behavioral_question(
        self,
    ) -> bool:
        """
        Record one completed behavioral question.
        """

        return (
            self.policy.record_behavioral_question()
        )


    def behavioral_complete(
        self,
    ) -> bool:
        """
        Return whether required behavioral questions
        have been completed.
        """

        return (
            self.policy.behavioral_complete()
        )


    def behavioral_questions_remaining(
        self,
    ) -> int:
        """
        Return behavioral questions remaining.
        """

        return (
            self.policy
            .behavioral_questions_remaining()
        )


    # ========================================================
    # Competency state
    # ========================================================

    def _sync_plan_states(
        self,
    ) -> None:
        """
        Synchronize InterviewPlan target states with the
        CompetencyTracker.
        """

        if self.interview_plan is None:
            return

        states = (
            self.competency_tracker.get_states()
        )

        for target in self.interview_plan.targets:

            if target.competency in states:

                target.state = states[
                    target.competency
                ]


    def get_competency_states(
        self,
    ) -> Dict[
        str,
        CompetencyState,
    ]:
        """
        Return current competency states.
        """

        return (
            self.competency_tracker.get_states()
        )


    def get_assessed_competencies(
        self,
    ) -> List[str]:
        """
        Return assessed competencies.
        """

        return (
            self.competency_tracker.get_assessed()
        )


    def get_unassessed_competencies(
        self,
    ) -> List[str]:
        """
        Return competencies still requiring evidence.
        """

        return (
            self.competency_tracker.get_unassessed()
        )


    # ========================================================
    # Interview plan
    # ========================================================

    def get_target(
        self,
        competency: str,
    ) -> Optional[InterviewTarget]:
        """
        Retrieve full target metadata.
        """

        if self.interview_plan is None:
            return None

        for target in self.interview_plan.targets:

            if target.competency == competency:
                return target

        return None


    def get_active_targets(
        self,
    ) -> List[InterviewTarget]:
        """
        Return targets not yet ASSESSED.
        """

        if self.interview_plan is None:
            return []

        return [
            target
            for target in self.interview_plan.targets
            if (
                target.state
                != CompetencyState.ASSESSED
            )
        ]


    def format_plan_context(
        self,
    ) -> str:
        """
        Format rich plan context for the interviewer agent.
        """

        if self.interview_plan is None:
            return (
                "No structured interview plan available."
            )

        sections = []

        for target in self.interview_plan.targets:

            evidence = (
                "\n".join(
                    f"    - {item}"
                    for item
                    in target.resume_evidence
                )
                if target.resume_evidence
                else "    - None"
            )

            expected = (
                "\n".join(
                    f"    - {item}"
                    for item
                    in target.evidence_expected
                )
                if target.evidence_expected
                else "    - None"
            )

            followups_used = (
                self.get_followups_used(
                    target.competency
                )
            )

            followups_remaining = (
                self.get_followups_remaining(
                    target.competency
                )
            )

            section = (
                f"Competency: {target.competency}\n"
                f"Priority: {target.priority}\n"
                f"State: {target.state.value}\n"
                f"Reason: {target.reason}\n"
                f"Follow-ups Used: {followups_used}\n"
                f"Follow-ups Remaining: "
                f"{followups_remaining}\n"
                f"Resume Evidence:\n"
                f"{evidence}\n"
                f"Evidence Expected:\n"
                f"{expected}"
            )

            sections.append(
                section
            )

        return "\n\n".join(
            sections
        )


    # ========================================================
    # Agent context
    # ========================================================

    def format_policy_context(
        self,
    ) -> str:
        """
        Return policy state for the interviewer-agent prompt.
        """

        remaining = self.required_question_sources_remaining()
        allowed = self.allowed_next_question_sources()
        return "\n".join([
            self.policy.format_policy_state(),
            f"Current Question Source: {self.current_question_source.value}",
            "Question Sources Still Required: "
            + (", ".join(source.value for source in remaining) or "None"),
            "Allowed/Preferred Next Sources: "
            + ", ".join(source.value for source in allowed),
        ])

    def required_question_sources_remaining(self):
        targets = self.interview_plan.targets if self.interview_plan else []
        return self.policy.required_question_sources_remaining(
            has_resume_evidence=any(target.resume_evidence for target in targets),
            has_technical_targets=bool(targets),
        )

    def allowed_next_question_sources(self):
        phase = self.get_phase()
        if phase == InterviewPhase.INTRODUCTION:
            # The next accepted question is technical after the opening answer.
            remaining = self.required_question_sources_remaining()
            return remaining[:1] or [QuestionSource.JD_TECHNICAL]
        if phase == InterviewPhase.TECHNICAL:
            remaining = self.required_question_sources_remaining()
            preferred = remaining[:1] or [
                QuestionSource.JD_TECHNICAL,
                QuestionSource.JD_SCENARIO,
                QuestionSource.RESUME_VALIDATION,
            ]
            if (
                self.current_question_source in {
                    QuestionSource.RESUME_VALIDATION,
                    QuestionSource.JD_TECHNICAL,
                    QuestionSource.JD_SCENARIO,
                }
                and self.current_target_competency is not None
                and self.can_ask_followup(self.current_target_competency)
                and self.current_question_source not in preferred
            ):
                preferred.append(self.current_question_source)
            return preferred
        if phase == InterviewPhase.BEHAVIORAL:
            return [QuestionSource.BEHAVIORAL]
        return [QuestionSource.CLOSING]


    # ========================================================
    # General state
    # ========================================================

    def get_current_question(
        self,
    ) -> str:
        """
        Return candidate-facing current question.
        """

        return self.current_question


    def get_current_question_type(
        self,
    ) -> Optional[QuestionType]:
        """
        Return metadata for current question.
        """

        return self.current_question_type


    def get_current_target_competency(
        self,
    ) -> Optional[str]:
        """
        Return primary target of current question.
        """

        return self.current_target_competency

    def get_current_question_source(self) -> QuestionSource:
        return self.current_question_source


    def get_interview_history(
        self,
    ) -> str:
        """
        Return formatted interview history.
        """

        return self.memory.format_history()


    def is_complete(
        self,
    ) -> bool:
        """
        Return whether interview has been explicitly
        completed.
        """

        return (
            self.get_phase()
            == InterviewPhase.COMPLETE
        )


    def format_state(
        self,
    ) -> str:
        """
        Development representation of interview state.
        """

        question_type = (
            self.current_question_type.value
            if self.current_question_type
            is not None
            else "OPENING"
        )

        target = (
            self.current_target_competency
            if self.current_target_competency
            is not None
            else "None"
        )

        lines = [
            f"Role: {self.role}",
            (
                f"Phase: "
                f"{self.get_phase().value}"
            ),
            "",
            "Current Question:",
            self.current_question,
            (
                f"Question Type: "
                f"{question_type}"
            ),
            (
                f"Target Competency: "
                f"{target}"
            ),
            "",
            "Competencies:",
            self.competency_tracker.format_status(),
            "",
            "Assessed:",
            str(
                self.get_assessed_competencies()
            ),
            "",
            "Still To Assess:",
            str(
                self.get_unassessed_competencies()
            ),
            "",
            "Behavioral Questions Remaining:",
            str(
                self.behavioral_questions_remaining()
            ),
        ]

        return "\n".join(
            lines
        )


def main():
    """
    Test deterministic enforcement of interviewer-agent
    decisions.

    Gemini is NOT called.
    """

    print("=" * 80)
    print("INTERVIEW ENGINE DECISION ENFORCEMENT TEST")
    print("=" * 80)

    plan = InterviewPlan(
        role="Data Scientist",
        targets=[
            InterviewTarget(
                competency="Machine Learning",
                priority="HIGH",
                reason=(
                    "Machine learning is a core "
                    "job requirement."
                ),
                resume_evidence=[
                    (
                        "Built predictive machine "
                        "learning models."
                    )
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
                    "SQL is required for data "
                    "extraction and analysis."
                ),
                resume_evidence=[
                    (
                        "Built SQL data pipelines."
                    )
                ],
                evidence_expected=[
                    "joins",
                    "aggregation",
                    "query reasoning",
                ],
            ),
            InterviewTarget(
                competency="Communication",
                priority="HIGH",
                reason=(
                    "The role requires communication "
                    "with technical and non-technical "
                    "audiences."
                ),
                resume_evidence=[
                    (
                        "Created technical documentation "
                        "and user guides."
                    )
                ],
                evidence_expected=[
                    "stakeholder communication",
                    "technical explanation",
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
            "Tell me about yourself and your "
            "experience relevant to this role."
        ),
        interview_plan=plan,
        policy_config=(
            InterviewPolicyConfig(
                max_followups_per_target=2,
                behavioral_questions_required=3,
            )
        ),
    )

    # --------------------------------------------------------
    # Start technical phase
    # --------------------------------------------------------

    engine.start_technical_phase()

    print("\nINITIAL")
    print("-" * 80)

    print(
        engine.format_state()
    )

    # --------------------------------------------------------
    # New technical target
    # --------------------------------------------------------

    technical_decision = InterviewerDecision(
        next_question=(
            "Tell me about a machine learning model "
            "you built and how you evaluated it."
        ),
        question_type=(
            QuestionType.NEW_TARGET
        ),
        target_competency=(
            "Machine Learning"
        ),
        competency_updates={},
    )

    engine.accept_next_question(
        technical_decision
    )

    print("\n" + "=" * 80)
    print("NEW TARGET ACCEPTED")
    print("=" * 80)

    print(
        engine.format_state()
    )

    # --------------------------------------------------------
    # First follow-up
    # --------------------------------------------------------

    followup_1 = InterviewerDecision(
        next_question=(
            "How did you validate the model and "
            "choose the final approach?"
        ),
        question_type=(
            QuestionType.FOLLOW_UP
        ),
        target_competency=(
            "Machine Learning"
        ),
        competency_updates={},
    )

    engine.accept_next_question(
        followup_1
    )

    print("\nFOLLOW-UP 1 ACCEPTED")

    print(
        "Used:",
        engine.get_followups_used(
            "Machine Learning"
        ),
    )

    # --------------------------------------------------------
    # Second follow-up
    # --------------------------------------------------------

    followup_2 = InterviewerDecision(
        next_question=(
            "What tradeoffs did you consider when "
            "selecting the final model?"
        ),
        question_type=(
            QuestionType.FOLLOW_UP
        ),
        target_competency=(
            "Machine Learning"
        ),
        competency_updates={},
    )

    engine.accept_next_question(
        followup_2
    )

    print("\nFOLLOW-UP 2 ACCEPTED")

    print(
        "Used:",
        engine.get_followups_used(
            "Machine Learning"
        ),
    )

    # --------------------------------------------------------
    # Third follow-up should fail
    # --------------------------------------------------------

    followup_3 = InterviewerDecision(
        next_question=(
            "What else did you learn from the model?"
        ),
        question_type=(
            QuestionType.FOLLOW_UP
        ),
        target_competency=(
            "Machine Learning"
        ),
        competency_updates={},
    )

    print("\nATTEMPTING FOLLOW-UP 3")

    try:

        engine.accept_next_question(
            followup_3
        )

        print(
            "ERROR: Follow-up 3 was incorrectly accepted."
        )

    except InterviewPolicyViolation as error:

        print(
            "Correctly rejected:"
        )

        print(
            error
        )

    # --------------------------------------------------------
    # Behavioral question during technical phase should fail
    # --------------------------------------------------------

    behavioral = InterviewerDecision(
        next_question=(
            "Tell me about a time you explained a "
            "technical result to a non-technical stakeholder."
        ),
        question_type=(
            QuestionType.BEHAVIORAL
        ),
        target_competency=(
            "Communication"
        ),
        competency_updates={},
    )

    print(
        "\nATTEMPTING BEHAVIORAL DURING TECHNICAL PHASE"
    )

    try:

        engine.accept_next_question(
            behavioral
        )

        print(
            "ERROR: Behavioral question was "
            "incorrectly accepted."
        )

    except InterviewPolicyViolation as error:

        print(
            "Correctly rejected:"
        )

        print(
            error
        )

    # --------------------------------------------------------
    # Start behavioral phase
    # --------------------------------------------------------

    engine.start_behavioral_phase()

    print("\n" + "=" * 80)
    print("BEHAVIORAL PHASE")
    print("=" * 80)

    engine.accept_next_question(
        behavioral
    )

    print(
        "Behavioral question accepted."
    )

    print(
        "Behavioral completed BEFORE answer:",
        engine.policy.behavioral_questions_completed,
    )

    # Simulate the candidate answering behavioral Q1.
    # The decision below proposes behavioral Q2.

    behavioral_2 = InterviewerDecision(
        next_question=(
            "Tell me about a time you had to work "
            "through an ambiguous analytical problem."
        ),
        question_type=(
            QuestionType.BEHAVIORAL
        ),
        target_competency=(
            "Communication"
        ),
        competency_updates={
            "Communication": (
                CompetencyState.EXPLORED
            )
        },
    )

    engine.process_candidate_answer(
        candidate_answer=(
            "I presented model results to an operations "
            "team and focused on the business impact "
            "instead of technical terminology."
        ),
        decision=behavioral_2,
    )

    print(
        "\nBehavioral completed AFTER answer:",
        engine.policy.behavioral_questions_completed,
    )

    print(
        "Behavioral remaining:",
        engine.behavioral_questions_remaining(),
    )

    print("\nCURRENT QUESTION")

    print(
        engine.get_current_question()
    )

    print(
        "Question Type:",
        engine.get_current_question_type().value,
    )

    # --------------------------------------------------------
    # Final state
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL STATE")
    print("=" * 80)

    print(
        engine.format_state()
    )

    print("\nGemini was NOT called.")


if __name__ == "__main__":
    main()
