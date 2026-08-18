import json
from typing import Dict, List

from src.generation.gemini_generator import (
    generate_content_with_retry,
)
from src.state.competency_tracker import CompetencyState


VALID_STATES = {
    "NOT_COVERED",
    "MENTIONED",
    "EXPLORED",
    "ASSESSED",
}


def build_assessment_prompt(
    question: str,
    answer: str,
    competencies: List[str],
) -> str:
    """
    Build the competency coverage assessment prompt.
    """

    competency_text = "\n".join(
        f"- {competency}"
        for competency in competencies
    )

    return f"""
You are tracking interview COVERAGE, not candidate performance.

Your job is to determine whether the interviewer has gathered
enough evidence about each competency from THIS interview turn.

INTERVIEW QUESTION:
{question}

CANDIDATE ANSWER:
{answer}

COMPETENCIES:
{competency_text}

Assign exactly one state to each competency:

NOT_COVERED:
The competency is unrelated to this question and answer.

MENTIONED:
The competency is only referenced or named. It has not been
meaningfully investigated.

EXPLORED:
The candidate provides meaningful detail about the competency,
but the interviewer has not yet gathered enough evidence to
consider it sufficiently assessed.

ASSESSED:
The question directly tests the competency and the candidate's
response gives meaningful evidence about their knowledge,
experience, reasoning, OR lack of knowledge.

IMPORTANT RULES:

1. These states measure COVERAGE, not candidate quality.

2. A candidate can perform poorly and still have a competency
   marked ASSESSED.

Example:

Question:
"Explain how you would perform hypothesis testing."

Answer:
"I know the term, but I don't know how to perform it."

Statistics should be ASSESSED because the interviewer directly
tested it and learned that the candidate cannot explain it.

3. Mentioning a technology does not mean the competency has
   been assessed.

Example:

"I have used TensorFlow."

Deep Learning should normally be MENTIONED.

4. If the interviewer directly asks the candidate to explain
   a project, method, technical decision, architecture, or
   reasoning related to a competency and the candidate provides
   substantive detail, that competency should normally be
   ASSESSED.

5. Do NOT infer unrelated competencies merely because the
   candidate communicates an answer.

Communication should only be MENTIONED, EXPLORED, or ASSESSED
when communication itself is discussed or tested, such as
presenting findings, explaining technical concepts to
stakeholders, collaboration, or handling communication
challenges.

6. Evaluation metrics such as precision, recall, F1-score,
   or ROC-AUC do not automatically mean Statistics is assessed.
   Determine whether statistical reasoning itself was actually
   explored.

7. Do not score the candidate.

8. Evaluate only evidence from THIS question and answer.

Return ONLY valid JSON using exactly these states:

NOT_COVERED
MENTIONED
EXPLORED
ASSESSED

Example:

{{
    "Machine Learning": "NOT_COVERED",
    "Deep Learning": "ASSESSED",
    "SQL": "NOT_COVERED",
    "Statistics": "MENTIONED",
    "Communication": "NOT_COVERED"
}}
""".strip()


def parse_assessment(
    response_text: str,
    competencies: List[str],
) -> Dict[str, CompetencyState]:
    """
    Convert the model response into CompetencyState values.
    """

    text = response_text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    data = json.loads(text)

    results = {}

    for competency in competencies:

        state = data.get(
            competency,
            "NOT_COVERED",
        )

        if state not in VALID_STATES:
            state = "NOT_COVERED"

        results[competency] = CompetencyState(state)

    return results


def assess_competencies(
    question: str,
    answer: str,
    competencies: List[str],
) -> Dict[str, CompetencyState]:
    """
    Determine competency coverage for one interview turn.
    """

    prompt = build_assessment_prompt(
        question=question,
        answer=answer,
        competencies=competencies,
    )

    response_text = generate_content_with_retry(prompt)

    return parse_assessment(
        response_text=response_text,
        competencies=competencies,
    )