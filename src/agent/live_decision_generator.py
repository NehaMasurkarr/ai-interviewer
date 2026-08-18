import os
from typing import Optional

from dotenv import load_dotenv

from src.agent.interviewer_agent import (
    InterviewerDecision,
    build_interviewer_agent_prompt,
    parse_interviewer_decision,
)
from src.agent.interview_engine import (
    InterviewEngine,
)
from src.generation.gemini_generator import (
    generate_content_with_retry,
)


load_dotenv()


def generate_interviewer_decision(
    engine: InterviewEngine,
    job_description: str,
    resume_evidence: str,
    candidate_answer: str,
    correction_message: Optional[str] = None,
) -> InterviewerDecision:
    """
    Generate one real interviewer decision using Gemini.

    Gemini receives:

    - role
    - job description
    - resume evidence
    - full interview plan
    - policy state
    - interview history
    - current question
    - candidate answer

    If the DecisionResolver rejected a previous decision,
    correction_message is also supplied.
    """

    plan_context = (
        engine.format_plan_context()
    )

    policy_context = (
        engine.format_policy_context()
    )

    prompt = build_interviewer_agent_prompt(
        role=engine.role,
        job_description=job_description,
        resume_evidence=resume_evidence,
        interview_plan_context=(
            plan_context
        ),
        interview_history=(
            engine.get_interview_history()
        ),
        current_question=(
            engine.get_current_question()
        ),
        candidate_answer=(
            candidate_answer
        ),
        retrieved_examples="",
    )

    prompt += f"""

============================================================
DETERMINISTIC INTERVIEW POLICY
============================================================

{policy_context}

You MUST obey this policy.

The Python InterviewEngine will validate your decision.
"""

    if correction_message:

        prompt += f"""

============================================================
PREVIOUS DECISION REJECTED
============================================================

{correction_message}

Generate a DIFFERENT decision that satisfies the policy.
"""

    response_text = (
        generate_content_with_retry(
            prompt
        )
    )

    allowed_competencies = [
        target.competency
        for target in (
            engine.interview_plan.targets
        )
    ]

    return parse_interviewer_decision(
        response_text=response_text,
        allowed_competencies=(
            allowed_competencies
        ),
    )


def main():
    """
    Import test only.

    Gemini is NOT called here.
    """

    print("=" * 80)
    print("LIVE DECISION GENERATOR")
    print("=" * 80)

    print(
        "\nModule imported successfully."
    )

    print(
        "\nThe generator is ready to connect "
        "Gemini to InterviewEngine."
    )

    print(
        "\nGemini was NOT called."
    )


if __name__ == "__main__":
    main()