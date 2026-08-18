from src.agent.engine_factory import (
    create_interview_engine,
)
from src.agent.live_decision_generator import (
    generate_interviewer_decision,
)
from src.planning.interview_plan import (
    InterviewPlan,
    InterviewTarget,
)


def main():
    """
    First real Gemini interviewer test.

    One candidate answer is sent to Gemini and we
    inspect the structured interviewer decision.
    """

    print("=" * 80)
    print("LIVE GEMINI INTERVIEWER TEST")
    print("=" * 80)

    # ========================================================
    # Test Job Description
    # ========================================================

    job_description = """
Data Scientist role requiring strong Python, SQL,
statistical modeling, machine learning, communication,
forecasting, and MLOps experience.
""".strip()

    # ========================================================
    # Test Resume Evidence
    # ========================================================

    resume_evidence = """
The candidate has experience building forecasting models
using Python, working with SQL data pipelines, performing
statistical analysis, and implementing CI/CD pipelines for
machine learning systems.
""".strip()

    # ========================================================
    # Interview Plan
    # ========================================================

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
                        "Built forecasting and ETL "
                        "pipelines using Python."
                    ),
                ],
                evidence_expected=[
                    "Python data processing",
                    "maintainable Python code",
                    "production implementation",
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
                    "Built SQL data pipelines.",
                ],
                evidence_expected=[
                    "joins",
                    "aggregation",
                    "query reasoning",
                ],
            ),
            InterviewTarget(
                competency="Statistical Modeling",
                priority="HIGH",
                reason=(
                    "Statistical modeling is required "
                    "for the role."
                ),
                resume_evidence=[
                    (
                        "Performed regression and "
                        "statistical analysis."
                    ),
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
                    "Machine learning is a core "
                    "job requirement."
                ),
                resume_evidence=[
                    (
                        "Built predictive machine "
                        "learning models."
                    ),
                ],
                evidence_expected=[
                    "model selection",
                    "training",
                    "evaluation",
                ],
            ),
            InterviewTarget(
                competency="Communication",
                priority="HIGH",
                reason=(
                    "The role requires communication "
                    "with technical and non-technical "
                    "audiences."
                ),
                resume_evidence=[
                    (
                        "Created technical documentation "
                        "and user guides."
                    ),
                ],
                evidence_expected=[
                    "stakeholder communication",
                    "technical explanation",
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
                    ),
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
                    "MLOps experience is preferred."
                ),
                resume_evidence=[
                    (
                        "Implemented CI/CD pipelines "
                        "for ML systems."
                    ),
                ],
                evidence_expected=[
                    "CI/CD",
                    "deployment",
                    "versioning",
                    "reproducibility",
                ],
            ),
        ],
    )

    # ========================================================
    # Create Engine
    # ========================================================

    engine = create_interview_engine(
        plan=plan
    )

    engine.start_technical_phase()

    print("\nCURRENT QUESTION")
    print("-" * 80)

    print(
        engine.get_current_question()
    )

    # ========================================================
    # Candidate Answer
    #
    # Later this comes from speech-to-text.
    # ========================================================

    candidate_answer = """
I worked on a forecasting project where I used Python to
build an ETL pipeline and prepare the data. I trained several
models and compared their performance using validation data.
I also automated parts of the pipeline so the forecasts could
be generated repeatedly.
""".strip()

    print("\nCANDIDATE ANSWER")
    print("-" * 80)

    print(
        candidate_answer
    )

    print("\nCalling Gemini...")

    # ========================================================
    # REAL GEMINI CALL
    # ========================================================

    decision = generate_interviewer_decision(
        engine=engine,
        job_description=job_description,
        resume_evidence=resume_evidence,
        candidate_answer=candidate_answer,
    )

    # ========================================================
    # Result
    # ========================================================

    print("\n" + "=" * 80)
    print("GEMINI DECISION")
    print("=" * 80)

    print(
        "\nQuestion Type:",
        decision.question_type.value,
    )

    print(
        "Target:",
        decision.target_competency,
    )

    print(
        "\nNext Question:"
    )

    print(
        decision.next_question
    )

    print(
        "\nCompetency Updates:"
    )

    if decision.competency_updates:

        for competency, state in (
            decision.competency_updates.items()
        ):

            print(
                f"- {competency}: "
                f"{state.value}"
            )

    else:

        print(
            "- No competency updates"
        )

    print(
        "\nReal Gemini call completed."
    )


if __name__ == "__main__":
    main()