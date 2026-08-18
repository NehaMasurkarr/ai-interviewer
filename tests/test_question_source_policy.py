import copy
import json

import pytest

from src.agent.decision_resolver import DecisionResolutionError
from src.agent.interview_coordinator import InterviewCoordinator
from src.agent.interview_engine import InterviewPolicyViolation
from src.agent.interviewer_agent import (
    InterviewerDecision,
    QuestionType,
    parse_interviewer_decision,
)
from src.job.job_profile import JobProfile, JobRequirement
from src.planning.interview_plan import InterviewPlan, InterviewTarget
from src.policy.interview_policy import InterviewPhase, InterviewPolicyConfig
from src.policy.question_source import QuestionSource
from src.profile.candidate_profile import CandidateProfile, ResumeClaim
from src.session.session_serialization import InterviewSessionError
from src.state.competency_tracker import CompetencyState


def make_decision(source, *, kind=QuestionType.NEW_TARGET, target="Python", updates=None):
    return InterviewerDecision(
        next_question=f"A {source.value} question?",
        question_type=kind,
        target_competency=target,
        competency_updates=updates or {},
        question_source=source,
    )


class SequenceGenerator:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.corrections = []

    def __call__(self, engine, jd, resume, answer, correction):
        self.corrections.append(correction)
        return next(self.decisions)


def make_coordinator(generator, *, resume_evidence=True, attempts=3):
    evidence = ["Built a production Python service."] if resume_evidence else []
    candidate = CandidateProfile(
        name="Candidate",
        claims=(
            [ResumeClaim(evidence[0], "Project", ["Python"])]
            if evidence else []
        ),
    )
    job = JobProfile(
        role="Engineer",
        requirements=[JobRequirement("Python", "HIGH", ["Applied reasoning"])],
    )
    plan = InterviewPlan(
        role="Engineer",
        targets=[
            InterviewTarget(
                competency="Python",
                priority="HIGH",
                reason="Python is required by the job description.",
                resume_evidence=evidence,
                evidence_expected=["Production design and debugging"],
            )
        ],
    )
    return InterviewCoordinator(
        candidate_profile=candidate,
        job_profile=job,
        job_description="Build production Python services.",
        decision_generator=generator,
        interview_plan=plan,
        policy_config=InterviewPolicyConfig(behavioral_questions_required=1),
        max_decision_attempts=attempts,
    )


def test_parser_requires_known_explicit_question_source():
    payload = {
        "next_question": "Explain Python.",
        "question_type": "NEW_TARGET",
        "question_source": "UNKNOWN",
        "target_competency": "Python",
        "competency_updates": {},
    }
    with pytest.raises(ValueError, match="Invalid interviewer question source"):
        parse_interviewer_decision(json.dumps(payload), ["Python"])

    payload.pop("question_source")
    with pytest.raises(ValueError, match="Invalid interviewer question source"):
        parse_interviewer_decision(json.dumps(payload), ["Python"])


@pytest.mark.parametrize(
    "source",
    [QuestionSource.JD_TECHNICAL, QuestionSource.JD_SCENARIO],
)
def test_job_derived_sources_are_valid_for_job_target(source):
    subject = make_coordinator(SequenceGenerator([]))
    subject.engine.start_technical_phase()
    if source == QuestionSource.JD_SCENARIO:
        subject.engine.policy.record_question_source(QuestionSource.JD_TECHNICAL)
    subject.engine.validate_decision(make_decision(source))


def test_resume_validation_requires_target_evidence():
    valid = make_coordinator(SequenceGenerator([]), resume_evidence=True)
    valid.engine.start_technical_phase()
    valid.engine.policy.record_question_source(QuestionSource.JD_TECHNICAL)
    valid.engine.policy.record_question_source(QuestionSource.JD_SCENARIO)
    valid.engine.validate_decision(make_decision(QuestionSource.RESUME_VALIDATION))

    invalid = make_coordinator(SequenceGenerator([]), resume_evidence=False)
    invalid.engine.start_technical_phase()
    with pytest.raises(InterviewPolicyViolation, match="requires resume evidence"):
        invalid.engine.validate_decision(make_decision(QuestionSource.RESUME_VALIDATION))


@pytest.mark.parametrize(
    ("source", "kind", "phase"),
    [
        (QuestionSource.BEHAVIORAL, QuestionType.BEHAVIORAL, InterviewPhase.TECHNICAL),
        (QuestionSource.OPENING, QuestionType.NEW_TARGET, InterviewPhase.TECHNICAL),
        (QuestionSource.CLOSING, QuestionType.CLOSING, InterviewPhase.TECHNICAL),
    ],
)
def test_phase_rejects_ineligible_source(source, kind, phase):
    subject = make_coordinator(SequenceGenerator([]))
    target = None if kind == QuestionType.CLOSING else "Python"
    with pytest.raises(InterviewPolicyViolation):
        subject.engine.validate_decision(
            make_decision(source, kind=kind, target=target), phase=phase
        )


def test_rejected_proposal_does_not_increment_source_counts():
    generator = SequenceGenerator([
        make_decision(QuestionSource.RESUME_VALIDATION),
        make_decision(QuestionSource.JD_TECHNICAL),
    ])
    subject = make_coordinator(generator)
    result = subject.submit_answer("Introduction")

    assert result.attempts == 2
    assert subject.engine.policy.get_question_source_count(
        QuestionSource.RESUME_VALIDATION
    ) == 0
    assert subject.engine.policy.get_question_source_count(
        QuestionSource.JD_TECHNICAL
    ) == 1


def test_required_coverage_prevents_transition_and_uses_assessed_exception():
    generator = SequenceGenerator([
        make_decision(QuestionSource.JD_TECHNICAL),
        make_decision(
            QuestionSource.JD_SCENARIO,
            updates={"Python": CompetencyState.ASSESSED},
        ),
        make_decision(QuestionSource.RESUME_VALIDATION),
        make_decision(
            QuestionSource.BEHAVIORAL,
            kind=QuestionType.BEHAVIORAL,
        ),
        make_decision(
            QuestionSource.CLOSING,
            kind=QuestionType.CLOSING,
            target=None,
        ),
    ])
    subject = make_coordinator(generator)

    assert subject.submit_answer("Introduction").phase == InterviewPhase.TECHNICAL
    assert subject.submit_answer("Technical answer").phase == InterviewPhase.TECHNICAL
    assert subject.engine.get_competency_states()["Python"] == CompetencyState.ASSESSED
    assert subject.submit_answer("Scenario answer").phase == InterviewPhase.TECHNICAL
    assert subject.submit_answer("Resume evidence").phase == InterviewPhase.BEHAVIORAL

    counts = subject.engine.policy.question_source_counts
    assert counts[QuestionSource.JD_TECHNICAL] == 1
    assert counts[QuestionSource.JD_SCENARIO] == 1
    assert counts[QuestionSource.RESUME_VALIDATION] == 1


def test_resume_coverage_is_not_required_without_evidence():
    subject = make_coordinator(SequenceGenerator([]), resume_evidence=False)
    assert subject.engine.required_question_sources_remaining() == [
        QuestionSource.JD_TECHNICAL,
        QuestionSource.JD_SCENARIO,
    ]


@pytest.mark.parametrize(
    "source",
    [
        QuestionSource.RESUME_VALIDATION,
        QuestionSource.JD_TECHNICAL,
        QuestionSource.JD_SCENARIO,
    ],
)
def test_followup_preserves_source_and_relabel_is_rejected(source):
    subject = make_coordinator(SequenceGenerator([]))
    subject.engine.start_technical_phase()
    # Disable quota ordering here to isolate follow-up semantics.
    subject.engine.policy.config.require_jd_technical = False
    subject.engine.policy.config.require_jd_scenario = False
    subject.engine.policy.config.require_resume_validation_when_evidence = False
    subject.engine.accept_next_question(make_decision(source))

    subject.engine.validate_decision(
        make_decision(source, kind=QuestionType.FOLLOW_UP)
    )
    replacement = (
        QuestionSource.JD_SCENARIO
        if source != QuestionSource.JD_SCENARIO
        else QuestionSource.JD_TECHNICAL
    )
    with pytest.raises(InterviewPolicyViolation, match="must preserve"):
        subject.engine.validate_decision(
            make_decision(replacement, kind=QuestionType.FOLLOW_UP)
        )


def test_source_counts_and_configuration_survive_session_round_trip():
    subject = make_coordinator(
        SequenceGenerator([make_decision(QuestionSource.JD_TECHNICAL)])
    )
    subject.submit_answer("Introduction")
    restored = InterviewCoordinator.from_session_dict(
        subject.to_session_dict(), SequenceGenerator([])
    )

    assert restored.to_session_dict() == subject.to_session_dict()
    assert restored.engine.current_question_source == QuestionSource.JD_TECHNICAL
    assert restored.engine.policy.get_question_source_count(
        QuestionSource.JD_TECHNICAL
    ) == 1
    assert restored.engine.policy.config.require_jd_scenario is True


def test_malformed_source_state_is_rejected_before_factory_creation():
    data = make_coordinator(SequenceGenerator([])).to_session_dict()
    data["engine"]["current_question_source"] = "BAD"
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)

    with pytest.raises(InterviewSessionError):
        from src.session.session_serialization import restore_interview_session

        restore_interview_session(
            data, SequenceGenerator([]), coordinator_factory=factory
        )
    assert calls == []


def test_schema_v1_is_migrated_deterministically():
    subject = make_coordinator(
        SequenceGenerator([make_decision(QuestionSource.JD_TECHNICAL)])
    )
    subject.submit_answer("Introduction")
    data = subject.to_session_dict()
    data["schema_version"] = 1
    data["engine"].pop("current_question_source")
    policy = data["engine"]["policy"]
    policy.pop("question_source_counts")
    policy["config"].pop("require_jd_technical")
    policy["config"].pop("require_jd_scenario")
    policy["config"].pop("require_resume_validation_when_evidence")

    restored = InterviewCoordinator.from_session_dict(data, SequenceGenerator([]))

    assert restored.engine.current_question_source == QuestionSource.JD_TECHNICAL
    assert restored.engine.policy.get_question_source_count(
        QuestionSource.JD_TECHNICAL
    ) == 1


def test_failed_resolution_keeps_all_interview_state_atomic():
    subject = make_coordinator(
        SequenceGenerator([make_decision(QuestionSource.RESUME_VALIDATION)]),
        attempts=1,
    )
    before = copy.deepcopy(subject.to_session_dict())
    with pytest.raises(DecisionResolutionError):
        subject.submit_answer("Introduction")
    assert subject.to_session_dict() == before
