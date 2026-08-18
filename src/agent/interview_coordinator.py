from dataclasses import dataclass
from typing import Callable, Optional

from src.agent.decision_resolver import DecisionResolver
from src.agent.engine_factory import create_interview_engine
from src.agent.interview_engine import InterviewEngine
from src.agent.interviewer_agent import InterviewerDecision
from src.agent.turn_processor import TurnResult
from src.job.job_profile import JobProfile
from src.planning.interview_plan import InterviewPlan
from src.policy.interview_policy import (
    InterviewPhase,
    InterviewPolicyConfig,
)
from src.profile.candidate_profile import CandidateProfile
from src.state.competency_tracker import CompetencyState


DecisionGenerator = Callable[
    [
        InterviewEngine,
        str,
        str,
        str,
        Optional[str],
    ],
    InterviewerDecision,
]


@dataclass
class CoordinatorTurnResult:
    """Public result of one atomic coordinator turn."""

    turn: TurnResult
    phase: InterviewPhase
    attempts: int
    is_complete: bool

    @property
    def next_question(self) -> Optional[str]:
        return self.turn.next_question or None


class InterviewCoordinator:
    """Connect the existing planning, agent, and engine layers."""

    def __init__(
        self,
        candidate_profile: CandidateProfile,
        job_profile: JobProfile,
        job_description: str,
        decision_generator: DecisionGenerator,
        interview_plan: Optional[InterviewPlan] = None,
        policy_config: Optional[InterviewPolicyConfig] = None,
        max_decision_attempts: int = 3,
    ):
        self.candidate_profile = candidate_profile
        self.job_profile = job_profile
        self.job_description = job_description
        if interview_plan is None:
            from src.planning.plan_builder import build_interview_plan

            interview_plan = build_interview_plan(
                candidate_profile,
                job_profile,
            )

        self.interview_plan = interview_plan
        self.engine = create_interview_engine(self.interview_plan)

        if policy_config is not None:
            # The factory preserves the plan; replace only the
            # policy configuration before the interview starts.
            self.engine.policy.config = policy_config

        self.decision_generator = decision_generator
        self.resolver = DecisionResolver(
            max_attempts=max_decision_attempts
        )
        self.resume_evidence = self._format_resume_evidence()

    @property
    def current_question(self) -> str:
        return self.engine.get_current_question()

    @property
    def is_complete(self) -> bool:
        return self.engine.is_complete()

    def submit_answer(
        self,
        candidate_answer: str,
    ) -> CoordinatorTurnResult:
        """Generate, resolve, and atomically commit one answer."""

        candidate_answer = candidate_answer.strip()

        if not candidate_answer:
            raise ValueError("Candidate answer cannot be empty.")

        if self.engine.is_complete():
            raise RuntimeError("The interview is already complete.")

        if self.engine.get_phase() == InterviewPhase.CLOSING:
            turn = self.engine.record_final_answer(candidate_answer)
            return CoordinatorTurnResult(
                turn=turn,
                phase=self.engine.get_phase(),
                attempts=0,
                is_complete=True,
            )

        def generate(
            correction_message: Optional[str] = None,
        ) -> InterviewerDecision:
            return self.decision_generator(
                self.engine,
                self.job_description,
                self.resume_evidence,
                candidate_answer,
                correction_message,
            )

        initial_decision = generate()

        resolution = self.resolver.resolve(
            engine=self.engine,
            initial_decision=initial_decision,
            decision_generator=generate,
            validation_phase=self._projected_phase,
            validation_states=self._projected_states,
        )

        next_phase = self._projected_phase(
            resolution.decision
        )

        turn = self.engine.process_candidate_answer(
            candidate_answer=candidate_answer,
            decision=resolution.decision,
            next_phase=next_phase,
        )

        return CoordinatorTurnResult(
            turn=turn,
            phase=self.engine.get_phase(),
            attempts=resolution.attempts,
            is_complete=self.engine.is_complete(),
        )

    def _projected_phase(
        self,
        decision: InterviewerDecision,
    ) -> InterviewPhase:
        """Determine the phase for the proposed next question."""

        phase = self.engine.get_phase()

        if phase == InterviewPhase.INTRODUCTION:
            return InterviewPhase.TECHNICAL

        if phase == InterviewPhase.TECHNICAL:
            states = self._projected_states(decision)

            if all(
                state == CompetencyState.ASSESSED
                for state in states.values()
            ) and not self.engine.required_question_sources_remaining():
                if self.engine.behavioral_questions_remaining() > 0:
                    return InterviewPhase.BEHAVIORAL
                return InterviewPhase.CLOSING

            return InterviewPhase.TECHNICAL

        if phase == InterviewPhase.BEHAVIORAL:
            remaining = self.engine.behavioral_questions_remaining()
            if remaining <= 1:
                return InterviewPhase.CLOSING
            return InterviewPhase.BEHAVIORAL

        return phase

    def _projected_states(
        self,
        decision: InterviewerDecision,
    ) -> dict[str, CompetencyState]:
        states = self.engine.get_competency_states()

        for competency, state in decision.competency_updates.items():
            if competency not in states:
                continue
            if state == CompetencyState.ASSESSED:
                states[competency] = state

        return states

    def _format_resume_evidence(self) -> str:
        claims = [claim.claim for claim in self.candidate_profile.claims]
        return "\n".join(f"- {claim}" for claim in claims)

    def to_session_dict(self) -> dict:
        """Return a versioned, JSON-compatible session snapshot."""

        from src.session.session_serialization import (
            serialize_interview_session,
        )

        return serialize_interview_session(self)

    @classmethod
    def from_session_dict(
        cls,
        session_data: dict,
        decision_generator: DecisionGenerator,
    ):
        """Restore a coordinator with injected runtime dependencies."""

        from src.session.session_serialization import (
            restore_interview_session,
        )

        return restore_interview_session(
            session_data,
            decision_generator,
            coordinator_factory=cls,
        )


def live_decision_generator(
    engine: InterviewEngine,
    job_description: str,
    resume_evidence: str,
    candidate_answer: str,
    correction_message: Optional[str],
) -> InterviewerDecision:
    """Adapter for the existing Gemini decision generator."""

    from src.agent.live_decision_generator import (
        generate_interviewer_decision,
    )

    return generate_interviewer_decision(
        engine=engine,
        job_description=job_description,
        resume_evidence=resume_evidence,
        candidate_answer=candidate_answer,
        correction_message=correction_message,
    )
