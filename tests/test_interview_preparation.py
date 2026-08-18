from pathlib import Path

import pytest

from src.agent.interview_coordinator import InterviewCoordinator
from src.job.job_profile import JobProfile, JobRequirement
from src.pipeline.interview_preparation import (
    InterviewPreparationError,
    prepare_interview_coordinator,
)
from src.planning.interview_plan import InterviewPlan, InterviewTarget
from src.profile.candidate_profile import CandidateProfile, ResumeClaim


def unused_decision_generator(*args, **kwargs):
    raise AssertionError("Decision generation must not run during preparation.")


@pytest.fixture
def resume_path(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.pdf"
    path.write_bytes(b"fake pdf; parsing is injected")
    return path


@pytest.fixture
def prepared_dependencies():
    candidate = CandidateProfile(
        name="Ada Candidate",
        skills=["Python"],
        claims=[
            ResumeClaim(
                claim="Built a Python service.",
                source="Platform project",
                technologies=["Python"],
            )
        ],
    )
    job = JobProfile(
        role="Backend Engineer",
        requirements=[
            JobRequirement(
                name="Python",
                priority="HIGH",
                evidence_expected=["Production Python experience"],
            )
        ],
    )
    plan = InterviewPlan(
        role="Backend Engineer",
        targets=[
            InterviewTarget(
                competency="Python",
                priority="HIGH",
                reason="Python is required.",
            )
        ],
    )
    calls = []

    def extract_resume(path):
        calls.append(("resume", path))
        return "Ada Candidate\nBuilt a Python service."

    def extract_candidate(text):
        calls.append(("candidate", text))
        return candidate

    def extract_job(text):
        calls.append(("job", text))
        return job

    def build_plan(candidate_arg, job_arg):
        calls.append(("plan", candidate_arg, job_arg))
        return plan

    return {
        "candidate": candidate,
        "job": job,
        "plan": plan,
        "calls": calls,
        "resume_text_extractor": extract_resume,
        "candidate_profile_extractor": extract_candidate,
        "job_profile_extractor": extract_job,
        "plan_builder": build_plan,
    }


def prepare(resume_path, dependencies, **overrides):
    arguments = {
        "resume_path": resume_path,
        "job_description": "  Build backend systems.\n  Strong Python required. ",
        "decision_generator": unused_decision_generator,
        "resume_text_extractor": dependencies["resume_text_extractor"],
        "candidate_profile_extractor": dependencies[
            "candidate_profile_extractor"
        ],
        "job_profile_extractor": dependencies["job_profile_extractor"],
        "plan_builder": dependencies["plan_builder"],
    }
    arguments.update(overrides)
    return prepare_interview_coordinator(**arguments)


def test_raw_inputs_create_ready_coordinator(
    resume_path,
    prepared_dependencies,
):
    result = prepare(resume_path, prepared_dependencies)

    assert isinstance(result, InterviewCoordinator)
    assert result.candidate_profile is prepared_dependencies["candidate"]
    assert result.job_profile is prepared_dependencies["job"]
    assert result.interview_plan is prepared_dependencies["plan"]
    assert result.engine.interview_plan is prepared_dependencies["plan"]
    assert result.job_description == (
        "Build backend systems.\nStrong Python required."
    )
    assert result.current_question
    assert "Backend Engineer" in result.current_question
    assert result.engine.get_unassessed_competencies() == ["Python"]


def test_injected_dependencies_run_in_order_without_model_calls(
    resume_path,
    prepared_dependencies,
):
    prepare(resume_path, prepared_dependencies)

    calls = prepared_dependencies["calls"]
    assert [call[0] for call in calls] == [
        "resume",
        "candidate",
        "job",
        "plan",
    ]
    assert calls[3][1] is prepared_dependencies["candidate"]
    assert calls[3][2] is prepared_dependencies["job"]


@pytest.mark.parametrize("job_description", ["", "  \n \t"])
def test_empty_job_description_is_rejected_before_extraction(
    resume_path,
    prepared_dependencies,
    job_description,
):
    with pytest.raises(ValueError, match="Job description is empty"):
        prepare(
            resume_path,
            prepared_dependencies,
            job_description=job_description,
        )

    assert prepared_dependencies["calls"] == []


def test_missing_resume_is_rejected(prepared_dependencies, tmp_path):
    missing = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="Resume file not found"):
        prepare(missing, prepared_dependencies)

    assert prepared_dependencies["calls"] == []


def test_invalid_resume_format_is_rejected(
    prepared_dependencies,
    tmp_path,
):
    invalid = tmp_path / "resume.txt"
    invalid.write_text("resume", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported resume format"):
        prepare(invalid, prepared_dependencies)

    assert prepared_dependencies["calls"] == []


@pytest.mark.parametrize(
    ("dependency_name", "message"),
    [
        ("candidate_profile_extractor", "Candidate profile extraction failed"),
        ("job_profile_extractor", "Job profile extraction failed"),
    ],
)
def test_extraction_failure_does_not_construct_coordinator(
    resume_path,
    prepared_dependencies,
    dependency_name,
    message,
):
    coordinator_calls = []

    def fail(*args):
        raise RuntimeError("fake extraction failure")

    def coordinator_factory(**kwargs):
        coordinator_calls.append(kwargs)
        raise AssertionError("Coordinator must not be constructed.")

    with pytest.raises(InterviewPreparationError, match=message):
        prepare(
            resume_path,
            prepared_dependencies,
            coordinator_factory=coordinator_factory,
            **{dependency_name: fail},
        )

    assert coordinator_calls == []


def test_empty_plan_is_rejected_before_coordinator_construction(
    resume_path,
    prepared_dependencies,
):
    coordinator_calls = []

    def empty_plan(candidate, job):
        return InterviewPlan(role=job.role, targets=[])

    def coordinator_factory(**kwargs):
        coordinator_calls.append(kwargs)
        raise AssertionError("Coordinator must not be constructed.")

    with pytest.raises(
        InterviewPreparationError,
        match="does not contain valid targets",
    ):
        prepare(
            resume_path,
            prepared_dependencies,
            plan_builder=empty_plan,
            coordinator_factory=coordinator_factory,
        )

    assert coordinator_calls == []
