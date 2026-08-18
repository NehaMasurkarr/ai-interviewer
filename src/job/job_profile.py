from dataclasses import dataclass, field
from typing import List


@dataclass
class JobRequirement:
    """
    One competency or requirement from a job description.
    """

    name: str
    priority: str = "MEDIUM"
    evidence_expected: List[str] = field(
        default_factory=list
    )


@dataclass
class JobProfile:
    """
    Structured representation of a job description.

    This will later be created automatically from the
    job description supplied for the interview.
    """

    role: str = ""

    company: str = ""

    summary: str = ""

    requirements: List[JobRequirement] = field(
        default_factory=list
    )

    responsibilities: List[str] = field(
        default_factory=list
    )

    preferred_qualifications: List[str] = field(
        default_factory=list
    )


def main():
    """
    Test the JobProfile data structure.
    """

    profile = JobProfile(
        role="Machine Learning Engineer",
        company="Example Company",
        summary=(
            "Build and deploy production machine "
            "learning and LLM systems."
        ),
        requirements=[
            JobRequirement(
                name="LLMs and RAG",
                priority="HIGH",
                evidence_expected=[
                    "RAG architecture",
                    "retrieval strategy",
                    "LLM evaluation",
                ],
            ),
            JobRequirement(
                name="Python",
                priority="HIGH",
                evidence_expected=[
                    "Production Python experience",
                ],
            ),
            JobRequirement(
                name="SQL",
                priority="MEDIUM",
                evidence_expected=[
                    "Querying and data manipulation",
                ],
            ),
        ],
        responsibilities=[
            "Build production ML systems.",
            "Develop LLM-powered applications.",
            "Work with large datasets.",
        ],
        preferred_qualifications=[
            "Experience with cloud platforms.",
            "Experience deploying ML models.",
        ],
    )

    print("=" * 80)
    print("JOB PROFILE TEST")
    print("=" * 80)

    print(f"\nRole: {profile.role}")
    print(f"Company: {profile.company}")

    print("\nREQUIREMENTS")

    for requirement in profile.requirements:

        print(
            f"- {requirement.name} "
            f"({requirement.priority})"
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