from typing import List, Dict, Optional


# ============================================================
# RAG Prompt Builder
# ============================================================

def build_interviewer_prompt(
    role: str,
    job_description: str,
    current_question: str,
    candidate_answer: str,
    retrieved_examples: List[Dict],
    interview_history: Optional[str] = None,
) -> str:
    """
    Build the prompt used by the LLM to generate the next
    interviewer question.

    The prompt combines:
    - role and job description
    - previous interview history
    - current question and candidate answer
    - retrieved historical interview examples
    """

    examples_text = format_retrieved_examples(
        retrieved_examples
    )

    if not interview_history:
        interview_history = (
            "No previous interview questions have been asked."
        )

    prompt = f"""
You are an AI interviewer conducting a professional job interview.

Your task is to generate the single best next interview question.

ROLE:
{role}

JOB DESCRIPTION:
{job_description}

PREVIOUS INTERVIEW HISTORY:
{interview_history}

CURRENT INTERVIEW QUESTION:
{current_question}

CANDIDATE ANSWER:
{candidate_answer}

SIMILAR HISTORICAL INTERVIEW EXAMPLES:
{examples_text}

INSTRUCTIONS:

1. Carefully analyze the candidate's current answer.

2. Consider the entire interview history before deciding
   what to ask next.

3. Do not repeat a question, topic, or competency that has
   already been sufficiently covered in the interview.

4. Decide whether the best next question should:
   - ask a relevant follow-up,
   - probe for more detail,
   - test technical depth,
   - ask for a concrete example,
   - clarify something vague,
   - or move naturally to another important competency.

5. Prioritize useful follow-up questions when the candidate
   mentions something that deserves deeper exploration.

6. Use the historical examples only as context for how
   similar interviews progressed.

7. Do NOT blindly copy a historical next question.

8. The next question must be appropriate for THIS candidate's
   actual answer and previous interview responses.

9. Keep the question relevant to the role and job description.

10. Ask only ONE interview question.

11. Do not praise, evaluate, score, or provide feedback on
    the candidate's answer.

12. Do not include introductory phrases such as:
    "That's great", "Good answer", or
    "That sounds like great experience."

13. Return ONLY the interview question. Do not provide
    commentary or explanation.

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

    interview_history = """
Turn 1

Question:
Tell me about your experience with machine learning.

Candidate Answer:
I have built classification models using Python and
scikit-learn.

Turn 2

Question:
Tell me about one of those classification projects.

Candidate Answer:
I built a customer churn prediction model and evaluated
it using precision, recall, and F1-score.
""".strip()

    prompt = build_interviewer_prompt(
        role="Data Scientist",
        job_description=(
            "We are looking for a Data Scientist with "
            "experience in machine learning, Python, "
            "statistical modeling, and communicating "
            "insights to stakeholders."
        ),
        current_question=(
            "Have you worked with deep learning?"
        ),
        candidate_answer=(
            "Yes. I built an image classification model "
            "using TensorFlow and a convolutional "
            "neural network."
        ),
        retrieved_examples=example_results,
        interview_history=interview_history,
    )

    print(prompt)


if __name__ == "__main__":
    main()