import json
from typing import Callable, Dict, List, Optional

from src.job.job_profile import (
    JobProfile,
    JobRequirement,
)


VALID_PRIORITIES = {
    "HIGH",
    "MEDIUM",
    "LOW",
}


def build_job_extraction_prompt(
    job_description: str,
) -> str:
    """
    Build a prompt that converts a raw job description
    into a structured JobProfile.
    """

    return f"""
You are extracting structured interview requirements from a
job description.

Use ONLY information contained in the job description.

Do not invent requirements that are not present.

JOB DESCRIPTION:
{job_description}

Extract:

1. Role title.

2. Company name, if available.

3. A short summary of the role.

4. Core interview requirements.

Each requirement should represent a competency, skill,
knowledge area, or capability that would be useful to assess
during an interview.

For every requirement include:

- name
- priority
- evidence_expected

Priority must be exactly one of:

HIGH
MEDIUM
LOW

Use HIGH when the requirement appears central to performing
the role.

Use MEDIUM when it is relevant but not one of the main
requirements.

Use LOW for preferred, optional, or secondary qualifications.

"evidence_expected" should describe what useful interview
evidence would demonstrate the requirement.

For example:

Requirement:
RAG and LLM Systems

Evidence expected:
- experience designing retrieval pipelines
- understanding of retrieval strategy
- experience evaluating LLM outputs
- production implementation experience

Do not simply copy every technology into a separate requirement.
Group closely related technologies when they represent the same
competency.

5. Responsibilities.

6. Preferred qualifications.

Return ONLY valid JSON in this structure:

{{
    "role": "Role Title",
    "company": "Company Name",
    "summary": "Short role summary",
    "requirements": [
        {{
            "name": "Requirement",
            "priority": "HIGH",
            "evidence_expected": [
                "Evidence 1",
                "Evidence 2"
            ]
        }}
    ],
    "responsibilities": [
        "Responsibility"
    ],
    "preferred_qualifications": [
        "Preferred qualification"
    ]
}}
""".strip()


def parse_job_profile(
    response_text: str,
) -> JobProfile:
    """
    Convert structured JSON into a JobProfile.
    """

    text = response_text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    data: Dict = json.loads(text)

    requirements: List[JobRequirement] = []

    for item in data.get("requirements", []):

        priority = item.get(
            "priority",
            "MEDIUM",
        ).upper()

        if priority not in VALID_PRIORITIES:
            priority = "MEDIUM"

        requirements.append(
            JobRequirement(
                name=item.get("name", ""),
                priority=priority,
                evidence_expected=item.get(
                    "evidence_expected",
                    [],
                ),
            )
        )

    return JobProfile(
        role=data.get("role", ""),
        company=data.get("company", ""),
        summary=data.get("summary", ""),
        requirements=requirements,
        responsibilities=data.get(
            "responsibilities",
            [],
        ),
        preferred_qualifications=data.get(
            "preferred_qualifications",
            [],
        ),
    )


def extract_job_profile(
    job_description: str,
    content_generator: Optional[Callable[[str], str]] = None,
) -> JobProfile:
    """Extract a JobProfile from raw job-description text."""

    job_description = job_description.strip()

    if not job_description:
        raise ValueError("Job description is empty.")

    if content_generator is None:
        from src.generation.gemini_generator import (
            generate_content_with_retry,
        )

        content_generator = generate_content_with_retry

    prompt = build_job_extraction_prompt(job_description)
    response_text = content_generator(prompt)

    return parse_job_profile(response_text)


def main():
    """
    Test JSON parsing without making an LLM call.
    """

    test_response = """
    {
        "role": "Machine Learning Engineer",
        "company": "Example Company",
        "summary": "Build production machine learning and LLM systems.",
        "requirements": [
            {
                "name": "LLMs and RAG",
                "priority": "HIGH",
                "evidence_expected": [
                    "RAG architecture experience",
                    "retrieval strategy knowledge",
                    "LLM evaluation experience"
                ]
            },
            {
                "name": "Python",
                "priority": "HIGH",
                "evidence_expected": [
                    "Production Python experience"
                ]
            },
            {
                "name": "SQL",
                "priority": "MEDIUM",
                "evidence_expected": [
                    "Querying and data manipulation"
                ]
            }
        ],
        "responsibilities": [
            "Build production ML systems.",
            "Develop LLM-powered applications."
        ],
        "preferred_qualifications": [
            "Experience with cloud platforms."
        ]
    }
    """

    profile = parse_job_profile(
        test_response
    )

    print("=" * 80)
    print("JOB EXTRACTOR TEST")
    print("=" * 80)

    print(f"\nRole: {profile.role}")
    print(f"Company: {profile.company}")
    print(f"Summary: {profile.summary}")

    print("\nREQUIREMENTS")

    for requirement in profile.requirements:

        print(
            f"\n- {requirement.name} "
            f"[{requirement.priority}]"
        )

        for evidence in requirement.evidence_expected:
            print(f"  - {evidence}")

    print("\nRESPONSIBILITIES")

    for responsibility in profile.responsibilities:
        print(f"- {responsibility}")

    print("\nPREFERRED QUALIFICATIONS")

    for qualification in profile.preferred_qualifications:
        print(f"- {qualification}")


if __name__ == "__main__":
    main()
