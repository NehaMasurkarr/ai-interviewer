from typing import List

from src.agent.interview_engine import (
    InterviewEngine,
)
from src.planning.interview_plan import (
    InterviewPlan,
    InterviewTarget,
)


def get_interview_targets(
    plan: InterviewPlan,
) -> List[str]:
    """
    Extract unique competency names from the plan.
    """

    targets = []

    for target in plan.targets:

        competency = target.competency.strip()

        if not competency:
            continue

        if competency not in targets:
            targets.append(competency)

    if not targets:
        raise ValueError(
            "Interview plan contains no valid targets."
        )

    return targets


def build_opening_question(
    plan: InterviewPlan,
) -> str:
    """
    Create the opening interview question.
    """

    role = plan.role.strip()

    if not role:
        role = "this position"

    return (
        "Tell me about yourself and the experience "
        f"you have that is most relevant to the {role} role."
    )


def create_interview_engine(
    plan: InterviewPlan,
) -> InterviewEngine:
    """
    Create an InterviewEngine while preserving the
    complete InterviewPlan.
    """

    interview_targets = (
        get_interview_targets(
            plan
        )
    )

    opening_question = (
        build_opening_question(
            plan
        )
    )

    return InterviewEngine(
        role=plan.role,
        interview_targets=interview_targets,
        opening_question=opening_question,
        interview_plan=plan,
    )


def main():
    """
    Local factory integration test.

    No Gemini calls.
    """

    print("=" * 80)
    print("INTERVIEW ENGINE FACTORY TEST")
    print("=" * 80)

    plan = InterviewPlan(
        role="Data Scientist",
        targets=[
            InterviewTarget(
                competency="Python",
                priority="HIGH",
                reason=(
                    "Strong Python proficiency is required."
                ),
                resume_evidence=[
                    (
                        "Built data pipelines using Python, "
                        "SQL, Databricks, and AWS."
                    )
                ],
                evidence_expected=[
                    "Python data processing",
                    "maintainable Python code",
                    "production-quality implementation",
                ],
            ),
            InterviewTarget(
                competency="SQL",
                priority="HIGH",
                reason=(
                    "SQL is required for data extraction "
                    "and analysis."
                ),
                resume_evidence=[
                    (
                        "Built data pipelines using Python, "
                        "SQL, Databricks, and AWS."
                    )
                ],
                evidence_expected=[
                    "query construction",
                    "joins",
                    "aggregation",
                    "data extraction",
                ],
            ),
            InterviewTarget(
                competency="Statistical Modeling",
                priority="HIGH",
                reason=(
                    "The role independently develops "
                    "statistical models."
                ),
                resume_evidence=[
                    (
                        "Applied regression, hypothesis "
                        "testing, and exploratory analysis."
                    )
                ],
                evidence_expected=[
                    "model assumptions",
                    "validation",
                    "interpretation",
                ],
            ),
            InterviewTarget(
                competency="Machine Learning",
                priority="HIGH",
                reason=(
                    "The role requires model training, "
                    "validation, evaluation, and "
                    "interpretation."
                ),
                resume_evidence=[
                    (
                        "Built predictive forecasting "
                        "models and multiple ML systems."
                    )
                ],
                evidence_expected=[
                    "model selection",
                    "training",
                    "evaluation",
                    "performance interpretation",
                ],
            ),
            InterviewTarget(
                competency="Communication",
                priority="HIGH",
                reason=(
                    "The role requires translating "
                    "technical results for wide audiences."
                ),
                resume_evidence=[
                    (
                        "Developed technical documentation, "
                        "ML demos, and user guides."
                    )
                ],
                evidence_expected=[
                    "technical communication",
                    "stakeholder communication",
                    "explaining analytical results",
                ],
            ),
            InterviewTarget(
                competency="Time Series Forecasting",
                priority="MEDIUM",
                reason=(
                    "Forecasting experience is preferred."
                ),
                resume_evidence=[
                    (
                        "Built and deployed predictive "
                        "forecasting models."
                    )
                ],
                evidence_expected=[
                    "forecasting methodology",
                    "time-aware validation",
                    "forecast evaluation",
                ],
            ),
            InterviewTarget(
                competency="MLOps",
                priority="MEDIUM",
                reason=(
                    "MLOps and CI/CD experience is preferred."
                ),
                resume_evidence=[
                    (
                        "Implemented CI/CD pipelines to "
                        "support MLOps workflows."
                    )
                ],
                evidence_expected=[
                    "CI/CD",
                    "model deployment",
                    "versioning",
                    "reproducibility",
                ],
            ),
        ],
    )

    engine = create_interview_engine(
        plan
    )

    print(
        f"\nRole: {engine.role}"
    )

    print("\nOPENING QUESTION")
    print(
        engine.get_current_question()
    )

    print("\nFULL INTERVIEW PLAN")
    print("-" * 80)

    print(
        engine.format_plan_context()
    )

    print("\nGemini was NOT called.")


if __name__ == "__main__":
    main()