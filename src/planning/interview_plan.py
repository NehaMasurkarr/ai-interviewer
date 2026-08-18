from dataclasses import dataclass, field
from typing import List

from src.state.competency_tracker import CompetencyState


@dataclass
class InterviewTarget:
    """
    One area that the AI interviewer should assess.

    A target is derived from the job description and may
    also be connected to evidence or claims found in the
    candidate's resume.
    """

    competency: str

    priority: str

    reason: str

    resume_evidence: List[str] = field(
        default_factory=list
    )

    evidence_expected: List[str] = field(
        default_factory=list
    )

    state: CompetencyState = (
        CompetencyState.NOT_COVERED
    )


@dataclass
class InterviewPlan:
    """
    Complete assessment plan for one candidate
    interviewing for one job.
    """

    role: str

    targets: List[InterviewTarget] = field(
        default_factory=list
    )


def main():
    """
    Test the interview plan structure.
    """

    plan = InterviewPlan(
        role="Machine Learning Engineer",
        targets=[
            InterviewTarget(
                competency="LLMs and RAG",
                priority="HIGH",
                reason=(
                    "The job requires production LLM and "
                    "RAG experience, and the candidate "
                    "claims relevant experience."
                ),
                resume_evidence=[
                    (
                        "Built an LLM-powered AI interviewer "
                        "using LangChain and RAG pipelines."
                    ),
                    (
                        "Built an agentic AI assistant using "
                        "LLM agents and LangChain."
                    ),
                ],
                evidence_expected=[
                    "RAG architecture",
                    "retrieval strategy",
                    "LLM evaluation",
                ],
            ),
            InterviewTarget(
                competency="Python",
                priority="HIGH",
                reason=(
                    "Python is a core requirement and appears "
                    "throughout the candidate's experience."
                ),
                resume_evidence=[
                    (
                        "Built data pipelines using Python, "
                        "SQL, Databricks, and AWS."
                    ),
                ],
                evidence_expected=[
                    "Production Python experience",
                ],
            ),
            InterviewTarget(
                competency="SQL",
                priority="MEDIUM",
                reason=(
                    "SQL is required by the job and is listed "
                    "in the candidate's experience and skills."
                ),
                resume_evidence=[
                    (
                        "Built data pipelines using Python, "
                        "SQL, Databricks, and AWS."
                    ),
                ],
                evidence_expected=[
                    "Querying and data manipulation",
                ],
            ),
        ],
    )

    print("=" * 80)
    print("INTERVIEW PLAN TEST")
    print("=" * 80)

    print(f"\nRole: {plan.role}")

    print("\nINTERVIEW TARGETS")

    for target in plan.targets:

        print(
            f"\n- {target.competency} "
            f"[{target.priority}]"
        )

        print(
            f"  State: {target.state.value}"
        )

        print(
            f"  Reason: {target.reason}"
        )

        if target.resume_evidence:

            print("  Resume Evidence:")

            for evidence in target.resume_evidence:
                print(f"    - {evidence}")

        if target.evidence_expected:

            print("  Evidence Expected:")

            for evidence in target.evidence_expected:
                print(f"    - {evidence}")


if __name__ == "__main__":
    main()