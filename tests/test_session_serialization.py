import copy
import json

import pytest

from src.agent.interview_coordinator import InterviewCoordinator
from src.agent.interviewer_agent import InterviewerDecision, QuestionType
from src.job.job_profile import JobProfile, JobRequirement
from src.planning.interview_plan import InterviewPlan, InterviewTarget
from src.policy.interview_policy import InterviewPhase, InterviewPolicyConfig
from src.profile.candidate_profile import CandidateProfile, ResumeClaim
from src.session.session_serialization import (
    InterviewSessionError,
    restore_interview_session,
    restore_interview_session_json,
    session_to_json,
)
from src.state.competency_tracker import CompetencyState


def make_decision(question_type, target, question, updates=None):
    return InterviewerDecision(
        next_question=question,
        question_type=question_type,
        target_competency=target,
        competency_updates=updates or {},
    )


class SequenceGenerator:
    def __init__(self, decisions):
        self.decisions = iter(decisions)

    def __call__(self, engine, jd, resume, answer, correction):
        return next(self.decisions)


def make_coordinator(generator, behavioral=2, followups=2):
    candidate = CandidateProfile(
        name="Ada Candidate",
        education=["MS Computer Science"],
        skills=["Python", "Communication"],
        claims=[
            ResumeClaim(
                claim="Built a Python platform.",
                source="Platform project",
                technologies=["Python"],
            )
        ],
    )
    job = JobProfile(
        role="Backend Engineer",
        company="Example",
        summary="Build services.",
        requirements=[
            JobRequirement("Python", "HIGH", ["Production code"]),
            JobRequirement("Communication", "MEDIUM", ["Collaboration"]),
        ],
        responsibilities=["Build services"],
        preferred_qualifications=["Cloud experience"],
    )
    plan = InterviewPlan(
        role="Backend Engineer",
        targets=[
            InterviewTarget(
                competency="Python",
                priority="HIGH",
                reason="Required",
                resume_evidence=["Built a Python platform."],
                evidence_expected=["Production code"],
            ),
            InterviewTarget(
                competency="Communication",
                priority="MEDIUM",
                reason="Required",
                evidence_expected=["Collaboration"],
            ),
        ],
    )
    return InterviewCoordinator(
        candidate_profile=candidate,
        job_profile=job,
        job_description="Build backend services with Python.",
        decision_generator=generator,
        interview_plan=plan,
        policy_config=InterviewPolicyConfig(
            max_followups_per_target=followups,
            behavioral_questions_required=behavioral,
            require_jd_technical=False,
            require_jd_scenario=False,
            require_resume_validation_when_evidence=False,
        ),
        max_decision_attempts=4,
    )


def noop_generator(*args):
    raise AssertionError("Generator should not be called during restoration.")


def assert_equivalent(original, restored):
    assert restored.to_session_dict() == original.to_session_dict()
    assert restored.candidate_profile == original.candidate_profile
    assert restored.job_profile == original.job_profile
    assert restored.interview_plan == original.interview_plan


def test_fresh_coordinator_round_trip_and_json_compatibility():
    original = make_coordinator(noop_generator)

    session = original.to_session_dict()
    json_text = json.dumps(session)
    restored = InterviewCoordinator.from_session_dict(
        json.loads(json_text), noop_generator
    )

    assert session["schema_version"] == 2
    assert_equivalent(original, restored)
    assert restored.engine.get_phase() == InterviewPhase.INTRODUCTION
    assert restored.current_question == original.current_question
    assert restored.engine.get_current_question_type() is None
    assert restored.engine.get_current_target_competency() is None


def test_partial_technical_state_round_trip_preserves_all_engine_state():
    original = make_coordinator(
        SequenceGenerator([
            make_decision(
                QuestionType.FOLLOW_UP,
                "Python",
                "How was the platform deployed?",
                {"Python": CompetencyState.EXPLORED},
            )
        ])
    )
    original.submit_answer("I built a Python platform.")

    restored = restore_interview_session(
        original.to_session_dict(), noop_generator
    )

    assert_equivalent(original, restored)
    assert restored.engine.get_phase() == InterviewPhase.TECHNICAL
    assert restored.engine.get_current_question_type() == QuestionType.FOLLOW_UP
    assert restored.engine.get_current_target_competency() == "Python"
    assert restored.engine.get_followups_used("Python") == 1
    assert restored.engine.get_competency_states()["Python"] == (
        CompetencyState.EXPLORED
    )
    assert restored.interview_plan.targets[0].state == CompetencyState.EXPLORED
    assert restored.engine.memory.get_history() == original.engine.memory.get_history()
    assert restored.engine.policy.config.max_followups_per_target == 2
    assert restored.engine.policy.config.behavioral_questions_required == 2
    assert restored.resolver.max_attempts == 4


def test_behavioral_progress_round_trip():
    original = make_coordinator(
        SequenceGenerator([
            make_decision(QuestionType.NEW_TARGET, "Python", "Explain Python."),
            make_decision(
                QuestionType.BEHAVIORAL,
                "Communication",
                "Tell me about a conflict.",
                {
                    "Python": CompetencyState.ASSESSED,
                    "Communication": CompetencyState.ASSESSED,
                },
            ),
            make_decision(
                QuestionType.BEHAVIORAL,
                "Communication",
                "Tell me about ambiguity.",
            ),
        ])
    )
    original.submit_answer("Introduction")
    original.submit_answer("Technical evidence")
    original.submit_answer("Conflict example")

    restored = restore_interview_session(
        original.to_session_dict(), noop_generator
    )

    assert_equivalent(original, restored)
    assert restored.engine.get_phase() == InterviewPhase.BEHAVIORAL
    assert restored.engine.policy.behavioral_questions_completed == 1
    assert restored.engine.behavioral_questions_remaining() == 1


def test_closing_phase_round_trip():
    original = make_coordinator(
        SequenceGenerator([
            make_decision(QuestionType.NEW_TARGET, "Python", "Explain Python."),
            make_decision(
                QuestionType.BEHAVIORAL,
                "Communication",
                "Tell me about teamwork.",
                {
                    "Python": CompetencyState.ASSESSED,
                    "Communication": CompetencyState.ASSESSED,
                },
            ),
            make_decision(
                QuestionType.CLOSING,
                None,
                "Do you have any questions?",
            ),
        ]),
        behavioral=1,
    )
    original.submit_answer("Introduction")
    original.submit_answer("Technical evidence")
    original.submit_answer("Teamwork example")

    restored = restore_interview_session(
        original.to_session_dict(), noop_generator
    )

    assert_equivalent(original, restored)
    assert restored.engine.get_phase() == InterviewPhase.CLOSING
    assert restored.engine.get_current_question_type() == QuestionType.CLOSING
    assert restored.engine.get_current_target_competency() is None

    original.submit_answer("No questions, thank you.")
    completed = restore_interview_session(
        original.to_session_dict(), noop_generator
    )

    assert_equivalent(original, completed)
    assert completed.is_complete
    assert completed.engine.get_phase() == InterviewPhase.COMPLETE


def test_restored_coordinator_continues_and_matches_original():
    opening = make_decision(
        QuestionType.NEW_TARGET,
        "Python",
        "Explain the platform architecture.",
        {"Python": CompetencyState.MENTIONED},
    )
    original = make_coordinator(SequenceGenerator([opening]))
    original.submit_answer("Opening answer")
    session = original.to_session_dict()

    next_decision = make_decision(
        QuestionType.NEW_TARGET,
        "Communication",
        "How did you communicate the design?",
        {"Python": CompetencyState.EXPLORED},
    )
    original.decision_generator = SequenceGenerator([next_decision])
    restored = restore_interview_session(
        session, SequenceGenerator([next_decision])
    )

    original_result = original.submit_answer("Architecture evidence")
    restored_result = restored.submit_answer("Architecture evidence")

    assert original_result == restored_result
    assert restored.to_session_dict() == original.to_session_dict()


def test_json_helpers_are_deterministic_and_exclude_runtime_objects():
    generator = SequenceGenerator([])
    coordinator = make_coordinator(generator)

    first = coordinator.to_session_dict()
    second = coordinator.to_session_dict()
    json_text = session_to_json(coordinator)
    restored = restore_interview_session_json(json_text, noop_generator)

    assert first == second
    assert "decision_generator" not in json_text
    assert "SequenceGenerator" not in json_text
    assert restored.to_session_dict() == first


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.pop("engine"),
        lambda data: data["engine"].update(phase="UNKNOWN"),
        lambda data: data["engine"]["competency_states"].update(Python="BAD"),
        lambda data: data["engine"]["competency_states"].update(Unknown="ASSESSED"),
        lambda data: data["engine"]["policy"]["followup_counts"].update(Python=-1),
        lambda data: data["engine"].update(history=[{"question": "missing answer"}]),
        lambda data: data["interview_plan"]["targets"][0].update(state="ASSESSED"),
        lambda data: data["engine"].update(current_question=""),
    ],
)
def test_malformed_session_is_rejected_without_coordinator_creation(mutate):
    data = copy.deepcopy(make_coordinator(noop_generator).to_session_dict())
    mutate(data)
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        raise AssertionError("Invalid state must be rejected before construction.")

    with pytest.raises(InterviewSessionError):
        restore_interview_session(
            data,
            noop_generator,
            coordinator_factory=factory,
        )

    assert factory_calls == []


def test_unsupported_schema_version_is_rejected():
    data = make_coordinator(noop_generator).to_session_dict()
    data["schema_version"] = 3

    with pytest.raises(
        InterviewSessionError,
        match="Unsupported interview session schema version: 3",
    ):
        restore_interview_session(data, noop_generator)
