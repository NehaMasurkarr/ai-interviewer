import pytest

from src.agent.decision_resolver import DecisionResolutionError
from src.agent.interview_coordinator import InterviewCoordinator
from src.agent.interviewer_agent import InterviewerDecision, QuestionType
from src.job.job_profile import JobProfile, JobRequirement
from src.planning.interview_plan import InterviewPlan, InterviewTarget
from src.policy.interview_policy import InterviewPhase, InterviewPolicyConfig
from src.profile.candidate_profile import CandidateProfile, ResumeClaim
from src.state.competency_tracker import CompetencyState


def decision(
    question_type,
    target="Python",
    updates=None,
    question="What did you do?",
):
    return InterviewerDecision(
        next_question=question,
        question_type=question_type,
        target_competency=target,
        competency_updates=updates or {},
    )


def coordinator(generator, *, attempts=3, behavioral=1, followups=1):
    candidate = CandidateProfile(
        name="Candidate",
        claims=[
            ResumeClaim(
                claim="Built a Python service.",
                source="Project",
                technologies=["Python"],
            )
        ],
    )
    job = JobProfile(
        role="Engineer",
        requirements=[JobRequirement(name="Python", priority="HIGH")],
    )
    plan = InterviewPlan(
        role="Engineer",
        targets=[
            InterviewTarget(
                competency="Python",
                priority="HIGH",
                reason="Required by the role.",
            )
        ],
    )
    return InterviewCoordinator(
        candidate_profile=candidate,
        job_profile=job,
        job_description="Python engineer",
        decision_generator=generator,
        interview_plan=plan,
        policy_config=InterviewPolicyConfig(
            max_followups_per_target=followups,
            behavioral_questions_required=behavioral,
            require_jd_technical=False,
            require_jd_scenario=False,
            require_resume_validation_when_evidence=False,
        ),
        max_decision_attempts=attempts,
    )


class SequenceGenerator:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.corrections = []

    def __call__(
        self,
        engine,
        job_description,
        resume_evidence,
        candidate_answer,
        correction_message,
    ):
        self.corrections.append(correction_message)
        return next(self.decisions)


def test_successful_turn_commits_answer_and_next_question():
    generator = SequenceGenerator([
        decision(QuestionType.NEW_TARGET, question="Explain your Python service.")
    ])
    subject = coordinator(generator)

    result = subject.submit_answer("I build backend systems.")

    assert result.next_question == "Explain your Python service."
    assert result.phase == InterviewPhase.TECHNICAL
    assert result.attempts == 1
    assert len(subject.engine.memory) == 1


def test_invalid_decision_is_retried_and_resolved():
    generator = SequenceGenerator([
        decision(QuestionType.BEHAVIORAL),
        decision(QuestionType.NEW_TARGET, question="Describe your Python work."),
    ])
    subject = coordinator(generator)

    result = subject.submit_answer("Opening answer")

    assert result.attempts == 2
    assert result.next_question == "Describe your Python work."
    assert generator.corrections[0] is None
    assert "POLICY VIOLATION" in generator.corrections[1]


def test_failed_resolution_does_not_mutate_state():
    generator = SequenceGenerator([
        decision(
            QuestionType.BEHAVIORAL,
            updates={"Python": CompetencyState.ASSESSED},
        )
    ])
    subject = coordinator(generator, attempts=1)
    original_question = subject.current_question

    with pytest.raises(DecisionResolutionError):
        subject.submit_answer("Opening answer")

    assert len(subject.engine.memory) == 0
    assert subject.current_question == original_question
    assert subject.engine.get_phase() == InterviewPhase.INTRODUCTION
    assert subject.engine.get_competency_states()["Python"] == (
        CompetencyState.NOT_COVERED
    )


def test_projected_assessed_followup_is_rejected_without_mutation():
    generator = SequenceGenerator([
        decision(
            QuestionType.FOLLOW_UP,
            updates={"Python": CompetencyState.ASSESSED},
        )
    ])
    subject = coordinator(generator, attempts=1)

    with pytest.raises(DecisionResolutionError):
        subject.submit_answer("Opening answer")

    assert len(subject.engine.memory) == 0
    assert subject.engine.get_competency_states()["Python"] == (
        CompetencyState.NOT_COVERED
    )


def test_competency_updates_are_committed_to_tracker_and_plan():
    generator = SequenceGenerator([
        decision(
            QuestionType.NEW_TARGET,
            updates={"Python": CompetencyState.EXPLORED},
        )
    ])
    subject = coordinator(generator)

    subject.submit_answer("I described my Python experience.")

    assert subject.engine.get_competency_states()["Python"] == (
        CompetencyState.EXPLORED
    )
    assert subject.interview_plan.targets[0].state == CompetencyState.EXPLORED


def test_followup_limit_rejects_second_followup_and_retries():
    generator = SequenceGenerator([
        decision(QuestionType.FOLLOW_UP, question="First follow-up?"),
        decision(QuestionType.FOLLOW_UP, question="Second follow-up?"),
        decision(QuestionType.NEW_TARGET, question="New Python angle?"),
    ])
    subject = coordinator(generator, followups=1)

    subject.submit_answer("Opening answer")
    result = subject.submit_answer("First technical answer")

    assert result.attempts == 2
    assert result.next_question == "New Python angle?"
    assert subject.engine.get_followups_used("Python") == 1


def test_deterministic_phase_transitions():
    generator = SequenceGenerator([
        decision(QuestionType.NEW_TARGET, question="Explain Python."),
        decision(
            QuestionType.BEHAVIORAL,
            updates={"Python": CompetencyState.ASSESSED},
            question="Tell me about a conflict.",
        ),
        decision(
            QuestionType.CLOSING,
            target=None,
            question="Do you have any questions for us?",
        ),
    ])
    subject = coordinator(generator, behavioral=1)

    assert subject.submit_answer("Introduction").phase == InterviewPhase.TECHNICAL
    assert subject.submit_answer("Python evidence").phase == InterviewPhase.BEHAVIORAL
    assert subject.submit_answer("Behavioral evidence").phase == InterviewPhase.CLOSING


def test_closing_answer_completes_without_model_call():
    generator = SequenceGenerator([
        decision(QuestionType.NEW_TARGET, question="Explain Python."),
        decision(
            QuestionType.BEHAVIORAL,
            updates={"Python": CompetencyState.ASSESSED},
            question="Tell me about a conflict.",
        ),
        decision(
            QuestionType.CLOSING,
            target=None,
            question="Any questions for us?",
        ),
    ])
    subject = coordinator(generator, behavioral=1)
    subject.submit_answer("Introduction")
    subject.submit_answer("Python evidence")
    subject.submit_answer("Behavioral evidence")

    result = subject.submit_answer("No further questions, thank you.")

    assert result.is_complete
    assert result.phase == InterviewPhase.COMPLETE
    assert result.next_question is None
    assert result.attempts == 0
    assert len(subject.engine.memory) == 4
    assert len(generator.corrections) == 3

    with pytest.raises(RuntimeError, match="already complete"):
        subject.submit_answer("Another answer")
