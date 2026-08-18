from dataclasses import dataclass, field
from typing import List


@dataclass
class Experience:
    """
    One work experience from the candidate's resume.
    """

    title: str
    company: str
    description: List[str] = field(default_factory=list)


@dataclass
class Project:
    """
    One project from the candidate's resume.
    """

    name: str
    description: List[str] = field(default_factory=list)


@dataclass
class ResumeClaim:
    """
    A resume claim that may be worth validating
    during the interview.
    """

    claim: str
    source: str
    technologies: List[str] = field(default_factory=list)


@dataclass
class CandidateProfile:
    """
    Structured representation of a candidate's resume.

    This profile will later be combined with the job
    description to create an interview plan.
    """

    name: str = ""

    education: List[str] = field(
        default_factory=list
    )

    experiences: List[Experience] = field(
        default_factory=list
    )

    projects: List[Project] = field(
        default_factory=list
    )

    skills: List[str] = field(
        default_factory=list
    )

    certifications: List[str] = field(
        default_factory=list
    )

    claims: List[ResumeClaim] = field(
        default_factory=list
    )


def main():
    """
    Test the candidate profile data structure.
    """

    profile = CandidateProfile(
        name="Test Candidate",
        education=[
            "MS Data Science",
        ],
        experiences=[
            Experience(
                title="Machine Learning Engineer Intern",
                company="Example Company",
                description=[
                    "Built an LLM-powered RAG system.",
                    "Developed NLP pipelines.",
                ],
            )
        ],
        projects=[
            Project(
                name="Recommendation System",
                description=[
                    "Built a hybrid recommendation system."
                ],
            )
        ],
        skills=[
            "Python",
            "SQL",
            "Machine Learning",
            "RAG",
        ],
        certifications=[
            "AWS Data Engineering",
        ],
        claims=[
            ResumeClaim(
                claim="Built an LLM-powered RAG system.",
                source="Machine Learning Engineer Intern",
                technologies=[
                    "LLM",
                    "RAG",
                ],
            )
        ],
    )

    print("=" * 80)
    print("CANDIDATE PROFILE TEST")
    print("=" * 80)

    print(f"\nName: {profile.name}")

    print("\nEXPERIENCE")

    for experience in profile.experiences:
        print(
            f"- {experience.title} "
            f"at {experience.company}"
        )

    print("\nPROJECTS")

    for project in profile.projects:
        print(f"- {project.name}")

    print("\nSKILLS")

    for skill in profile.skills:
        print(f"- {skill}")

    print("\nRESUME CLAIMS")

    for claim in profile.claims:
        print(f"- {claim.claim}")
        print(f"  Source: {claim.source}")

        if claim.technologies:
            print(
                "  Technologies: "
                + ", ".join(claim.technologies)
            )


if __name__ == "__main__":
    main()