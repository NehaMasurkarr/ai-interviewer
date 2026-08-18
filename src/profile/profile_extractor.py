import json
from pathlib import Path
from typing import Callable, Dict, List, Optional
from src.profile.candidate_profile import (
    CandidateProfile,
    Experience,
    Project,
    ResumeClaim,
)
from src.resume.resume_parser import (
    SUPPORTED_EXTENSIONS,
    extract_resume_text,
)


def build_profile_extraction_prompt(
    resume_text: str,
) -> str:
    """
    Build a prompt that converts raw resume text into
    structured candidate information.
    """

    return f"""
You are extracting structured information from a candidate resume.

Use ONLY information explicitly present in the resume.

Do not infer, exaggerate, or invent experience.

RESUME:
{resume_text}

Extract:

1. Candidate name.

2. Education.

3. Work experiences:
   - title
   - company
   - important description bullets

4. Projects:
   - project name
   - important description bullets

5. Skills.

6. Certifications.

7. Resume claims worth validating during a job interview.

A resume claim should be a meaningful technical or professional
claim that an interviewer may want the candidate to explain,
defend, or demonstrate.

Good claims include:
- built a production system
- improved a measurable metric
- designed an architecture
- deployed a model
- processed a large dataset
- implemented a pipeline
- reduced cost or processing time
- supported a large number of users

Do NOT create claims from simple skill-list entries such as
"Python", "SQL", or "TensorFlow".

Preserve important quantitative information from claims.

For every claim include:
- claim
- source
- technologies

The source should identify the experience or project where the
claim appears.

Return ONLY valid JSON in this structure:

{{
    "name": "Candidate Name",
    "education": [
        "Education entry"
    ],
    "experiences": [
        {{
            "title": "Job Title",
            "company": "Company",
            "description": [
                "Resume bullet"
            ]
        }}
    ],
    "projects": [
        {{
            "name": "Project Name",
            "description": [
                "Resume bullet"
            ]
        }}
    ],
    "skills": [
        "Python",
        "SQL"
    ],
    "certifications": [
        "Certification"
    ],
    "claims": [
        {{
            "claim": "Meaningful resume claim",
            "source": "Experience or Project",
            "technologies": [
                "Technology 1",
                "Technology 2"
            ]
        }}
    ]
}}
""".strip()


def clean_json_response(
    response_text: str,
) -> str:
    """
    Remove Markdown code fences if the model returns them.
    """

    text = response_text.strip()

    if text.startswith("```"):
        text = text.replace(
            "```json",
            "",
            1,
        )

        text = text.replace(
            "```",
            "",
        )

        text = text.strip()

    return text


def parse_candidate_profile(
    response_text: str,
) -> CandidateProfile:
    """
    Convert structured JSON into a CandidateProfile.
    """

    text = clean_json_response(
        response_text
    )

    data: Dict = json.loads(text)

    experiences: List[Experience] = []

    for item in data.get(
        "experiences",
        [],
    ):
        experiences.append(
            Experience(
                title=item.get(
                    "title",
                    "",
                ),
                company=item.get(
                    "company",
                    "",
                ),
                description=item.get(
                    "description",
                    [],
                ),
            )
        )

    projects: List[Project] = []

    for item in data.get(
        "projects",
        [],
    ):
        projects.append(
            Project(
                name=item.get(
                    "name",
                    "",
                ),
                description=item.get(
                    "description",
                    [],
                ),
            )
        )

    claims: List[ResumeClaim] = []

    for item in data.get(
        "claims",
        [],
    ):
        claims.append(
            ResumeClaim(
                claim=item.get(
                    "claim",
                    "",
                ),
                source=item.get(
                    "source",
                    "",
                ),
                technologies=item.get(
                    "technologies",
                    [],
                ),
            )
        )

    return CandidateProfile(
        name=data.get(
            "name",
            "",
        ),
        education=data.get(
            "education",
            [],
        ),
        experiences=experiences,
        projects=projects,
        skills=data.get(
            "skills",
            [],
        ),
        certifications=data.get(
            "certifications",
            [],
        ),
        claims=claims,
    )


def extract_candidate_profile(
    resume_path: str,
    content_generator: Optional[Callable[[str], str]] = None,
) -> CandidateProfile:
    """
    Complete resume ingestion flow:

    Resume file
        ->
    Resume text
        ->
    Gemini structured extraction
        ->
    CandidateProfile
    """

    resume_text = extract_resume_text(
        resume_path
    )

    return extract_candidate_profile_from_text(
        resume_text=resume_text,
        content_generator=content_generator,
    )


def extract_candidate_profile_from_text(
    resume_text: str,
    content_generator: Optional[Callable[[str], str]] = None,
) -> CandidateProfile:
    """Extract a CandidateProfile from already parsed resume text."""

    resume_text = resume_text.strip()

    if not resume_text:
        raise ValueError("Resume text is empty.")

    if content_generator is None:
        from src.generation.gemini_generator import (
            generate_content_with_retry,
        )

        content_generator = generate_content_with_retry

    prompt = build_profile_extraction_prompt(resume_text)
    response_text = content_generator(prompt)

    return parse_candidate_profile(response_text)


def find_resume() -> Path:
    """
    Find the first supported resume inside data/resumes.
    """

    resume_directory = Path(
        "data/resumes"
    )

    if not resume_directory.exists():
        raise FileNotFoundError(
            "Resume directory does not exist: "
            "data/resumes/"
        )

    resume_files = sorted(
        [
            path
            for path in resume_directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )

    if not resume_files:
        raise FileNotFoundError(
            "No PDF or DOCX resume found inside "
            "data/resumes/"
        )

    return resume_files[0]


def print_candidate_profile(
    profile: CandidateProfile,
) -> None:
    """
    Display the extracted profile for development testing.
    """

    print("\n" + "=" * 80)
    print("CANDIDATE PROFILE")
    print("=" * 80)

    print(f"\nName: {profile.name}")

    print("\nEDUCATION")

    for education in profile.education:
        print(f"- {education}")

    print("\nEXPERIENCE")

    for experience in profile.experiences:
        print(
            f"\n- {experience.title} "
            f"at {experience.company}"
        )

        for description in experience.description:
            print(f"  - {description}")

    print("\nPROJECTS")

    for project in profile.projects:
        print(f"\n- {project.name}")

        for description in project.description:
            print(f"  - {description}")

    print("\nSKILLS")

    for skill in profile.skills:
        print(f"- {skill}")

    print("\nCERTIFICATIONS")

    for certification in profile.certifications:
        print(f"- {certification}")

    print("\nRESUME CLAIMS")

    for index, claim in enumerate(
        profile.claims,
        start=1,
    ):
        print(
            f"\n{index}. {claim.claim}"
        )

        print(
            f"   Source: {claim.source}"
        )

        if claim.technologies:
            print(
                "   Technologies: "
                + ", ".join(
                    claim.technologies
                )
            )


def main():
    """
    Run the real resume -> CandidateProfile pipeline.
    """

    resume_path = find_resume()

    print("=" * 80)
    print("REAL RESUME PROFILE EXTRACTION")
    print("=" * 80)

    print(
        f"\nResume: {resume_path.name}"
    )

    print(
        "\nExtracting candidate profile..."
    )

    profile = extract_candidate_profile(
        str(resume_path)
    )

    print_candidate_profile(
        profile
    )


if __name__ == "__main__":
    main()
