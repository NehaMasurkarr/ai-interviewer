from typing import List, Dict


# ============================================================
# RAG Prompt Builder
# ============================================================

def build_interviewer_prompt(
    role: str,
    job_description: str,
    current_question: str,
    candidate_answer: str,
    retrieved_examples: List[Dict],
) -> str:
    """
    Build the prompt used by the LLM to generate the next
    interviewer question.

    Retrieved interview sequences are used as examples of
    how real/simulated interviews progressed in similar
    situations. They should guide the model, not be copied.
    """

    examples_text = format_retrieved_examples(
        retrieved_examples
    )

    prompt = f"""
You are an AI interviewer conducting a professional job interview.

Your task is to generate the single best next interview question.

ROLE:
{role}

JOB DESCRIPTION:
{job_description}

CURRENT INTERVIEW QUESTION:
{current_question}

CANDIDATE ANSWER:
{candidate_answer}

SIMILAR HISTORICAL INTERVIEW EXAMPLES:
{examples_text}

INSTRUCTIONS:

1. Carefully analyze the candidate's answer.

2. Decide whether the best next question should:
   - ask a relevant follow-up,
   - probe for more detail,
   - test technical depth,
   - ask for a concrete example,
   - clarify something vague,
   - or move naturally to another important competency.

3. Use the historical examples only as context for how similar
   interviews progressed.

4. Do NOT blindly copy a historical next question.

5. The next question must be appropriate for THIS candidate's
   actual answer.

6. Keep the question relevant to the role and job description.

7. Avoid repeating a question that has effectively already
   been answered.

8. Ask only ONE interview question.

9. Do not provide feedback, scoring, commentary, or explanation.

10. Return only the next interview question.

NEXT INTERVIEW QUESTION:
"""

    return prompt.strip()


# ============================================================
# Format retrieved examples
# ============================================================

def format_retrieved_examples(
    retrieved_examples: List[Dict],
) -> str:

    if not retrieved_examples:
        return "No historical examples available."

    formatted = []

    for index, example in enumerate(
        retrieved_examples,
        start=1,
    ):

        question = example.get(
            "question",
            "",
        )

        answer = example.get(
            "answer",
            "",
        )

        next_question = example.get(
            "next_question",
            "",
        )

        role = example.get(
            "role",
            "",
        )

        formatted.append(
            f"""
Example {index}

Role:
{role}

Question:
{question}

Candidate Answer:
{answer}

Next Question:
{next_question}
""".strip()
        )

    return "\n\n".join(formatted)


# ============================================================
# Manual test
# ============================================================

def main():

    example_results = [
        {
            "role": "Data Scientist",
            "question": (
                "Can you walk me through your "
                "experience with machine learning?"
            ),
            "answer": (
                "I've worked with basic machine learning "
                "algorithms like logistic regression and "
                "decision trees."
            ),
            "next_question": (
                "Can you tell me about a time when you "
                "had to communicate complex technical "
                "ideas to a non-technical audience?"
            ),
        },
        {
            "role": "Data Scientist",
            "question": (
                "What machine learning libraries or "
                "frameworks have you used?"
            ),
            "answer": (
                "I've worked with scikit-learn and "
                "TensorFlow for classification and "
                "regression."
            ),
            "next_question": (
                "Have you worked with big data tools "
                "like Hadoop or Spark?"
            ),
        },
    ]

    prompt = build_interviewer_prompt(
        role="Data Scientist",
        job_description=(
            "We are looking for a Data Scientist with "
            "experience in machine learning, Python, "
            "statistical modeling, and communicating "
            "insights to stakeholders."
        ),
        current_question=(
            "Tell me about your experience "
            "with machine learning."
        ),
        candidate_answer=(
            "I built classification models using Python "
            "and scikit-learn. I also worked with "
            "TensorFlow on a deep learning project."
        ),
        retrieved_examples=example_results,
    )

    print(prompt)


if __name__ == "__main__":
    main()