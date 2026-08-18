from pathlib import Path
from typing import Callable, Optional, Union

from src.agent.interview_coordinator import (
    DecisionGenerator,
    InterviewCoordinator,
)
from src.job.job_extractor import build_job_extraction_prompt
from src.job.job_parser import (
    clean_job_description,
    extract_job_description,
)
from src.job.job_profile import JobProfile
from src.planning.interview_plan import InterviewPlan
from src.profile.profile_extractor import (
    build_profile_extraction_prompt,
)
from src.profile.candidate_profile import CandidateProfile
from src.resume.resume_parser import (
    SUPPORTED_EXTENSIONS,
    extract_resume_text,
)


RESUME_DIRECTORY = Path("data/resumes")

JOB_DESCRIPTION_PATH = Path(
    "data/jobs/job_description.txt"
)


class InterviewPreparationError(RuntimeError):
    """Raised when an extraction or planning stage fails."""


def prepare_interview_coordinator(
    resume_path: Union[str, Path],
    job_description: str,
    decision_generator: DecisionGenerator,
    *,
    resume_text_extractor: Optional[Callable[[str], str]] = None,
    candidate_profile_extractor: Optional[
        Callable[[str], CandidateProfile]
    ] = None,
    job_profile_extractor: Optional[
        Callable[[str], JobProfile]
    ] = None,
    plan_builder: Optional[
        Callable[[CandidateProfile, JobProfile], InterviewPlan]
    ] = None,
    coordinator_factory: Callable[..., InterviewCoordinator] = (
        InterviewCoordinator
    ),
    **coordinator_options,
) -> InterviewCoordinator:
    """Create a ready InterviewCoordinator from raw inputs."""

    path = _validate_resume_path(resume_path)
    cleaned_job_description = _validate_job_description(job_description)

    if resume_text_extractor is None:
        resume_text_extractor = extract_resume_text

    if candidate_profile_extractor is None:
        from src.profile.profile_extractor import (
            extract_candidate_profile_from_text,
        )

        candidate_profile_extractor = extract_candidate_profile_from_text

    if job_profile_extractor is None:
        from src.job.job_extractor import extract_job_profile

        job_profile_extractor = extract_job_profile

    if plan_builder is None:
        from src.planning.plan_builder import build_interview_plan

        plan_builder = build_interview_plan

    try:
        resume_text = resume_text_extractor(str(path))
    except Exception as error:
        raise InterviewPreparationError(
            "Resume text extraction failed."
        ) from error

    if not isinstance(resume_text, str) or not resume_text.strip():
        raise InterviewPreparationError(
            "Resume text extraction produced no readable text."
        )

    try:
        candidate_profile = candidate_profile_extractor(resume_text)
    except Exception as error:
        raise InterviewPreparationError(
            "Candidate profile extraction failed."
        ) from error

    _validate_candidate_profile(candidate_profile)

    try:
        job_profile = job_profile_extractor(cleaned_job_description)
    except Exception as error:
        raise InterviewPreparationError(
            "Job profile extraction failed."
        ) from error

    _validate_job_profile(job_profile)

    try:
        interview_plan = plan_builder(candidate_profile, job_profile)
    except Exception as error:
        raise InterviewPreparationError(
            "Interview plan creation failed."
        ) from error

    _validate_interview_plan(interview_plan)

    return coordinator_factory(
        candidate_profile=candidate_profile,
        job_profile=job_profile,
        job_description=cleaned_job_description,
        decision_generator=decision_generator,
        interview_plan=interview_plan,
        **coordinator_options,
    )


def _validate_resume_path(resume_path: Union[str, Path]) -> Path:
    if resume_path is None or not str(resume_path).strip():
        raise ValueError("Resume path is required.")

    path = Path(resume_path)

    if not path.is_file():
        raise FileNotFoundError(f"Resume file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported resume format. Please use PDF or DOCX."
        )

    return path


def _validate_job_description(job_description: str) -> str:
    if not isinstance(job_description, str):
        raise TypeError("Job description must be text.")

    cleaned = clean_job_description(job_description)

    if not cleaned:
        raise ValueError("Job description is empty.")

    return cleaned


def _validate_candidate_profile(profile: CandidateProfile) -> None:
    if not isinstance(profile, CandidateProfile):
        raise InterviewPreparationError(
            "Candidate profile extraction returned an invalid result."
        )

    has_content = any(
        [
            profile.name.strip(),
            profile.education,
            profile.experiences,
            profile.projects,
            profile.skills,
            profile.certifications,
            profile.claims,
        ]
    )

    if not has_content:
        raise InterviewPreparationError(
            "Candidate profile extraction returned an empty profile."
        )


def _validate_job_profile(profile: JobProfile) -> None:
    if not isinstance(profile, JobProfile):
        raise InterviewPreparationError(
            "Job profile extraction returned an invalid result."
        )

    if not profile.role.strip():
        raise InterviewPreparationError(
            "Job profile does not contain a role."
        )

    if not profile.requirements or any(
        not requirement.name.strip()
        for requirement in profile.requirements
    ):
        raise InterviewPreparationError(
            "Job profile does not contain valid requirements."
        )


def _validate_interview_plan(plan: InterviewPlan) -> None:
    if not isinstance(plan, InterviewPlan):
        raise InterviewPreparationError(
            "Plan builder returned an invalid interview plan."
        )

    if not plan.role.strip():
        raise InterviewPreparationError(
            "Interview plan does not contain a role."
        )

    if not plan.targets or any(
        not target.competency.strip()
        for target in plan.targets
    ):
        raise InterviewPreparationError(
            "Interview plan does not contain valid targets."
        )


def find_resume() -> Path:
    """
    Find the first supported resume.
    """

    supported_extensions = {
        ".pdf",
        ".docx",
    }

    resume_files = sorted(
        [
            path
            for path in RESUME_DIRECTORY.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in supported_extensions
            )
        ]
    )

    if not resume_files:
        raise FileNotFoundError(
            "No resume found in data/resumes/"
        )

    return resume_files[0]


def load_interview_inputs():
    """
    Load resume and JD text.

    No LLM calls.
    """

    resume_path = find_resume()

    resume_text = extract_resume_text(
        str(resume_path)
    )

    job_description = extract_job_description(
        str(JOB_DESCRIPTION_PATH)
    )

    return (
        resume_path,
        resume_text,
        job_description,
    )


def build_extraction_prompts(
    resume_text: str,
    job_description: str,
):
    """
    Prepare both LLM extraction prompts.

    Still no LLM calls.
    """

    candidate_prompt = (
        build_profile_extraction_prompt(
            resume_text
        )
    )

    job_prompt = (
        build_job_extraction_prompt(
            job_description
        )
    )

    return (
        candidate_prompt,
        job_prompt,
    )


def main():
    """
    Test preparation through prompt creation.

    This deliberately stops BEFORE Gemini.
    """

    print("=" * 80)
    print("INTERVIEW PREPARATION PIPELINE")
    print("=" * 80)

    (
        resume_path,
        resume_text,
        job_description,
    ) = load_interview_inputs()

    (
        candidate_prompt,
        job_prompt,
    ) = build_extraction_prompts(
        resume_text=resume_text,
        job_description=job_description,
    )

    print("\nINPUTS")

    print(
        f"- Resume: {resume_path.name}"
    )

    print(
        f"- Resume characters: "
        f"{len(resume_text):,}"
    )

    print(
        f"- JD characters: "
        f"{len(job_description):,}"
    )

    print("\nEXTRACTION PROMPTS")

    print(
        f"- Candidate prompt: "
        f"{len(candidate_prompt):,} characters"
    )

    print(
        f"- Job prompt: "
        f"{len(job_prompt):,} characters"
    )

    print("\nPIPELINE STATUS")

    print("1. Resume loaded                 ✓")
    print("2. Job description loaded        ✓")
    print("3. Candidate extraction prepared ✓")
    print("4. Job extraction prepared       ✓")

    print("\nNext stage:")
    print(
        "CandidateProfile + JobProfile "
        "-> InterviewPlan"
    )

    print(
        "\nGemini was NOT called."
    )


if __name__ == "__main__":
    main()
