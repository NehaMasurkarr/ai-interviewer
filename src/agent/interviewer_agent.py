import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from src.state.competency_tracker import (
    CompetencyState,
)
from src.policy.question_source import QuestionSource


VALID_STATES = {
    "NOT_COVERED",
    "MENTIONED",
    "EXPLORED",
    "ASSESSED",
}


class QuestionType(Enum):
    """
    Type of question proposed by the interviewer agent.

    The InterviewEngine and InterviewPolicy can use this
    metadata to decide whether the proposed question is
    structurally allowed.
    """

    FOLLOW_UP = "FOLLOW_UP"
    NEW_TARGET = "NEW_TARGET"
    BEHAVIORAL = "BEHAVIORAL"
    CLOSING = "CLOSING"


VALID_QUESTION_TYPES = {
    question_type.value
    for question_type in QuestionType
}

VALID_QUESTION_SOURCES = {source.value for source in QuestionSource}


@dataclass
class InterviewerDecision:
    """
    Structured result produced by the interviewer agent
    after evaluating one candidate answer.

    next_question:
        The actual candidate-facing question.

    question_type:
        Whether the question is a follow-up, new technical
        target, behavioral question, or closing question.

    target_competency:
        The primary competency the next question is intended
        to assess.

        This may be None for a closing question.

    competency_updates:
        Coverage evidence gathered from the CURRENT completed
        question and candidate answer.
    """

    next_question: str

    question_type: QuestionType

    target_competency: Optional[str]

    competency_updates: Dict[
        str,
        CompetencyState,
    ]

    # Kept after existing fields so older injected test/client decisions remain
    # source-compatible. Structured model output must always provide this field.
    question_source: Optional[QuestionSource] = None

    def __post_init__(self) -> None:
        """Preserve older injected callers while keeping decisions explicit."""

        if self.question_source is not None:
            return
        if self.question_type == QuestionType.BEHAVIORAL:
            self.question_source = QuestionSource.BEHAVIORAL
        elif self.question_type == QuestionType.CLOSING:
            self.question_source = QuestionSource.CLOSING
        else:
            self.question_source = QuestionSource.JD_TECHNICAL


def build_interviewer_agent_prompt(
    role: str,
    job_description: str,
    resume_evidence: str,
    interview_plan_context: str,
    interview_history: str,
    current_question: str,
    candidate_answer: str,
    policy_context: str = "",
    retrieved_examples: str = "",
) -> str:
    """
    Build the prompt for one interviewer-agent turn.

    The model must:

    1. Evaluate evidence gathered from the current turn.
    2. Decide what kind of question should come next.
    3. Identify the primary target of that question.
    4. Generate exactly one next question.

    Structural enforcement is performed later by the
    InterviewEngine / InterviewPolicy.
    """

    if not interview_history.strip():
        interview_history = (
            "No previous interview turns."
        )

    if not resume_evidence.strip():
        resume_evidence = (
            "No relevant resume evidence available."
        )

    if not interview_plan_context.strip():
        interview_plan_context = (
            "No structured interview plan available."
        )

    if not policy_context.strip():
        policy_context = (
            "No additional interview policy context available."
        )

    if not retrieved_examples.strip():
        retrieved_examples = (
            "No historical examples available."
        )

    return f"""
You are an AI interviewer conducting a professional job interview.

Your responsibility is to gather useful evidence about the
candidate for the specific role while maintaining a natural,
structured interview.

After every candidate response, perform FIVE tasks:

1. Determine what competency evidence was gathered from the
   CURRENT question and candidate answer.

2. Decide what TYPE of question should come next.

3. Select the required QUESTION SOURCE shown by policy.

4. Identify the PRIMARY COMPETENCY targeted by that next
   question.

5. Generate exactly ONE next interview question.


ROLE:
{role}


JOB DESCRIPTION:
{job_description}


CANDIDATE RESUME:
{resume_evidence}


INTERVIEW PLAN:
{interview_plan_context}


INTERVIEW POLICY:
{policy_context}


PREVIOUS INTERVIEW HISTORY:
{interview_history}


CURRENT QUESTION:
{current_question}


CANDIDATE ANSWER:
{candidate_answer}


SIMILAR HISTORICAL INTERVIEW EXAMPLES:
{retrieved_examples}


============================================================
COVERAGE STATES
============================================================

Each interview competency has one of four coverage states.


NOT_COVERED:

The interview has not gathered meaningful evidence about the
competency.


MENTIONED:

The candidate referenced the competency, technology, method,
or experience, but meaningful evidence has not yet been
gathered.

Mentioning something does NOT automatically mean the candidate
demonstrated competency.

Example:

Candidate:
"I have used TensorFlow."

This may justify MENTIONED for a relevant competency.

It does NOT by itself justify EXPLORED or ASSESSED.


EXPLORED:

The candidate provided meaningful relevant detail, but
additional evidence would still be useful before considering
the competency sufficiently assessed.


ASSESSED:

The interview has gathered enough meaningful evidence to
understand the candidate's level for that competency.

ASSESSED is a COVERAGE state.

It does NOT mean the candidate performed well.

A weak, incomplete, or incorrect answer can still produce
ASSESSED if the question directly tested the competency and
revealed the candidate's actual level.


============================================================
QUESTION TYPES
============================================================

Every next question must have exactly ONE question type.


FOLLOW_UP

Use FOLLOW_UP when the next question continues investigating
the same primary competency or claim from the current
discussion.

Example:

Current question:
"Tell me about the forecasting model you built."

Candidate:
"I trained several models and selected XGBoost."

Next question:
"How did you validate the models and determine that XGBoost
was the best choice?"

Question type:
FOLLOW_UP

Target competency:
Time Series Forecasting


NEW_TARGET

Use NEW_TARGET when intentionally moving to another technical
or job-related competency.

Example:

Current discussion:
Forecasting

Next question:
"How would you use SQL to prepare customer-level features for
this analysis?"

Question type:
NEW_TARGET

Target competency:
SQL


BEHAVIORAL

Use BEHAVIORAL for a dedicated behavioral or situational
question intended to gather evidence about areas such as:

- communication
- collaboration
- ownership
- ambiguity
- conflict
- stakeholder management
- prioritization
- leadership
- adaptability

Behavioral questions should be relevant to the job
description.

Example:

"Tell me about a time you had to explain a complex analytical
result to a non-technical stakeholder. How did you approach
it?"

Question type:
BEHAVIORAL

Target competency:
Communication


CLOSING

Use CLOSING only when the interview policy indicates that the
interview should enter its closing stage.

A closing question does not need a target competency.

For CLOSING:

target_competency must be null.


============================================================
QUESTION SOURCES
============================================================

Every next question must explicitly use one source:

- RESUME_VALIDATION: validate actual resume evidence for the target. Vary
  phrasing; do not repeatedly begin with "Your resume mentions".
- JD_TECHNICAL: directly test role-required knowledge, independent of resume
  claims. Never introduce it as a resume claim.
- JD_SCENARIO: pose a realistic hypothetical problem derived from the role.
- BEHAVIORAL: a dedicated behavioral question only.
- CLOSING: the closing question only.

The application controls source strategy. Obey CURRENT REQUIRED/PREFERRED
SOURCES in INTERVIEW POLICY; do not default to resume validation. FOLLOW_UP
must preserve the current question source.


============================================================
EVIDENCE RULES
============================================================

1. Resume claims are NOT verified facts.

They are claims that may be investigated.

For example:

"Built and deployed predictive forecasting models."

does not automatically prove forecasting competency.


2. Evidence Expected describes useful evidence the interview
should attempt to collect.


3. Do not require every evidence item to appear literally.

Use professional judgment about whether sufficient evidence
has been gathered.


4. Tool names alone are weak evidence.

"I used Python."

does NOT demonstrate production Python ability.

"I used TensorFlow."

does NOT automatically demonstrate deep learning expertise.

"I used SQL."

does NOT demonstrate query construction ability.


5. Evaluation metrics such as precision, recall, F1-score,
ROC-AUC, MAE, or RMSE do not automatically demonstrate
statistical reasoning.


6. competency_updates describe evidence gathered from the
CURRENT completed turn.

They do NOT describe what the next question might assess.


============================================================
FOLLOW-UP RULES
============================================================

Follow-ups should be purposeful.

Do NOT ask a follow-up simply because one is allowed.


Consider a FOLLOW_UP when:

- an important claim needs validation
- the answer is relevant but lacks important detail
- reasoning behind a decision is unclear
- evidence expected by the interview plan is still missing
- one focused probe could substantially improve the evidence


Do NOT continue probing merely because more questions could
theoretically be asked.


If the interview policy says that no follow-ups remain for a
competency, do NOT return FOLLOW_UP for that competency.


If a competency is already ASSESSED, normally move to another
important target.


============================================================
NEW TARGET RULES
============================================================

When moving to a new technical competency:

1. Prefer important HIGH-priority targets.

2. Consider targets that remain:
   NOT_COVERED,
   MENTIONED,
   or EXPLORED.

3. Use resume evidence when useful.

4. Maintain natural conversational flow.

5. Do not simply walk through the interview plan from top to
   bottom.

6. Do not repeatedly return to competencies that are already
   sufficiently ASSESSED.


============================================================
BEHAVIORAL RULES
============================================================

The interview requires dedicated behavioral questions.

Use the INTERVIEW POLICY to determine how many remain.

Behavioral questions should be adapted to the role.

Do not ask generic behavioral questions when the job
description suggests a more relevant situation.

Good behavioral areas may include:

- explaining technical findings
- working through ambiguity
- handling disagreement
- receiving feedback
- working with stakeholders
- taking ownership
- prioritizing competing work
- collaborating across teams


A behavioral question counts as one PRIMARY behavioral
question.

Do not combine several unrelated behavioral situations into
one question.


============================================================
QUESTION QUALITY
============================================================

1. Ask questions appropriate to the specific role.

2. Adapt questions to THIS candidate.

3. Consider both the job description and resume.

4. Do not simply turn the job description into a checklist.

5. Do not simply walk through the resume line by line.

6. Do not ask the same question twice using different wording.

7. Prefer questions that gather useful evidence rather than
   trivia or definitions.

8. Ask exactly ONE candidate-facing question.

9. Do not praise the candidate.

10. Do not provide feedback.

11. Do not score the candidate.

12. Do not provide hints.

13. Do not tell the candidate what answer is expected.

14. Do not say:

"That's great"
"Good answer"
"Great experience"
"Excellent"

15. Historical interview examples are context only.

Never blindly copy their next questions.

16. Update only competencies for which the CURRENT completed
turn provides relevant evidence.

17. Never intentionally downgrade a competency state.


============================================================
RETURN FORMAT
============================================================

Return ONLY valid JSON.

Do not use Markdown.

Do not include explanations outside the JSON.

For FOLLOW_UP, NEW_TARGET, or BEHAVIORAL:

{{
    "next_question": "The next interview question",
    "question_type": "FOLLOW_UP",
    "question_source": "JD_TECHNICAL",
    "target_competency": "Competency Name",
    "competency_updates": {{
        "Competency Name": "STATE"
    }}
}}


For CLOSING:

{{
    "next_question": "The closing interview question",
    "question_type": "CLOSING",
    "question_source": "CLOSING",
    "target_competency": null,
    "competency_updates": {{
    }}
}}


Allowed question types:

FOLLOW_UP
NEW_TARGET
BEHAVIORAL
CLOSING


Allowed question sources:

RESUME_VALIDATION
JD_TECHNICAL
JD_SCENARIO
BEHAVIORAL
OPENING
CLOSING


Allowed competency states:

NOT_COVERED
MENTIONED
EXPLORED
ASSESSED
""".strip()


def clean_json_response(
    response_text: str,
) -> str:
    """
    Remove optional Markdown JSON fences.
    """

    text = response_text.strip()

    if text.startswith("```"):

        text = text.replace(
            "```json",
            "",
            1,
        )

        text = text.replace(
            "```",
            "",
        )

        text = text.strip()

    return text


def parse_interviewer_decision(
    response_text: str,
    allowed_competencies: List[str],
) -> InterviewerDecision:
    """
    Parse and validate the interviewer-agent response.

    Invalid question types or invalid target competencies
    raise errors because they affect interview control.

    Invalid competency updates are ignored because they
    should not break the interview.
    """

    text = clean_json_response(
        response_text
    )

    data = json.loads(
        text
    )

    # --------------------------------------------------------
    # Next question
    # --------------------------------------------------------

    next_question = str(
        data.get(
            "next_question",
            "",
        )
    ).strip()

    if not next_question:
        raise ValueError(
            "Interviewer agent returned no next question."
        )

    # --------------------------------------------------------
    # Question type
    # --------------------------------------------------------

    question_type_name = str(
        data.get(
            "question_type",
            "",
        )
    ).strip().upper()

    if (
        question_type_name
        not in VALID_QUESTION_TYPES
    ):
        raise ValueError(
            "Invalid interviewer question type: "
            f"{question_type_name}"
        )

    question_type = QuestionType[
        question_type_name
    ]

    question_source_name = str(data.get("question_source", "")).strip().upper()
    if question_source_name not in VALID_QUESTION_SOURCES:
        raise ValueError(
            "Invalid interviewer question source: " f"{question_source_name}"
        )
    question_source = QuestionSource[question_source_name]

    # --------------------------------------------------------
    # Target competency
    # --------------------------------------------------------

    raw_target = data.get(
        "target_competency"
    )

    target_competency: Optional[str]

    if raw_target is None:

        target_competency = None

    else:

        target_competency = str(
            raw_target
        ).strip()

        if not target_competency:
            target_competency = None

    allowed = set(
        allowed_competencies
    )

    if question_type == QuestionType.CLOSING:

        if target_competency is not None:
            raise ValueError(
                "CLOSING question must not have "
                "a target competency."
            )

    else:

        if target_competency is None:
            raise ValueError(
                f"{question_type.value} question must "
                "have a target competency."
            )

        if target_competency not in allowed:
            raise ValueError(
                "Unknown target competency: "
                f"{target_competency}"
            )

    # --------------------------------------------------------
    # Competency updates
    # --------------------------------------------------------

    raw_updates = data.get(
        "competency_updates",
        {},
    )

    if not isinstance(
        raw_updates,
        dict,
    ):
        raise ValueError(
            "competency_updates must be a JSON object."
        )

    updates: Dict[
        str,
        CompetencyState,
    ] = {}

    for competency, state in (
        raw_updates.items()
    ):

        if competency not in allowed:
            continue

        state_name = str(
            state
        ).strip().upper()

        if state_name not in VALID_STATES:
            continue

        updates[competency] = (
            CompetencyState[
                state_name
            ]
        )

    return InterviewerDecision(
        next_question=next_question,
        question_type=question_type,
        target_competency=target_competency,
        competency_updates=updates,
        question_source=question_source,
    )


def main():
    """
    Test question-control metadata and parsing.

    Gemini is NOT called.
    """

    competencies = [
        "Python",
        "SQL",
        "Statistical Modeling",
        "Machine Learning",
        "Communication",
        "Time Series Forecasting",
        "MLOps",
    ]

    interview_plan_context = """
Competency: Python
Priority: HIGH
State: EXPLORED
Follow-ups Used: 0
Follow-ups Remaining: 2
Evidence Expected:
    - Python data processing
    - maintainable Python code

Competency: SQL
Priority: HIGH
State: NOT_COVERED
Follow-ups Used: 0
Follow-ups Remaining: 2
Evidence Expected:
    - joins
    - aggregation
    - query construction

Competency: Time Series Forecasting
Priority: MEDIUM
State: EXPLORED
Follow-ups Used: 1
Follow-ups Remaining: 1
Evidence Expected:
    - forecasting methodology
    - time-aware validation
    - forecast evaluation
""".strip()

    policy_context = """
Interview Phase: TECHNICAL
Maximum Follow-ups Per Target: 2
Behavioral Questions Required: 3
Behavioral Questions Completed: 0
Behavioral Questions Remaining: 3
""".strip()

    prompt = build_interviewer_agent_prompt(
        role="Data Scientist",
        job_description=(
            "The role requires Python, SQL, statistical "
            "modeling, machine learning, forecasting, "
            "and communication."
        ),
        resume_evidence=(
            "The candidate built predictive forecasting "
            "models using Python and worked with SQL "
            "data pipelines."
        ),
        interview_plan_context=(
            interview_plan_context
        ),
        interview_history=(
            "Turn 1\n\n"
            "Question:\n"
            "Tell me about a forecasting project.\n\n"
            "Candidate Answer:\n"
            "I built a forecasting system using Python."
        ),
        current_question=(
            "How did you evaluate the forecasting model?"
        ),
        candidate_answer=(
            "I used time-based validation and compared "
            "MAE and RMSE across different periods."
        ),
        policy_context=(
            policy_context
        ),
        retrieved_examples="",
    )

    print("=" * 80)
    print("INTERVIEWER AGENT CONTROL TEST")
    print("=" * 80)

    print("\nPROMPT CREATED")
    print(
        f"Prompt characters: {len(prompt):,}"
    )

    # --------------------------------------------------------
    # FOLLOW-UP test
    # --------------------------------------------------------

    followup_response = """
{
    "next_question": "What made you choose those evaluation metrics for this forecasting problem?",
    "question_type": "FOLLOW_UP",
    "target_competency": "Time Series Forecasting",
    "competency_updates": {
        "Time Series Forecasting": "EXPLORED",
        "Statistical Modeling": "MENTIONED"
    }
}
"""

    followup = parse_interviewer_decision(
        response_text=followup_response,
        allowed_competencies=competencies,
    )

    print("\nFOLLOW-UP DECISION")
    print("-" * 80)

    print(
        f"Question Type: "
        f"{followup.question_type.value}"
    )

    print(
        f"Target: "
        f"{followup.target_competency}"
    )

    print(
        f"Question: "
        f"{followup.next_question}"
    )

    # --------------------------------------------------------
    # NEW TARGET test
    # --------------------------------------------------------

    new_target_response = """
{
    "next_question": "How would you use SQL to prepare the data required for this analysis?",
    "question_type": "NEW_TARGET",
    "target_competency": "SQL",
    "competency_updates": {
        "Time Series Forecasting": "ASSESSED"
    }
}
"""

    new_target = parse_interviewer_decision(
        response_text=new_target_response,
        allowed_competencies=competencies,
    )

    print("\nNEW TARGET DECISION")
    print("-" * 80)

    print(
        f"Question Type: "
        f"{new_target.question_type.value}"
    )

    print(
        f"Target: "
        f"{new_target.target_competency}"
    )

    print(
        f"Question: "
        f"{new_target.next_question}"
    )

    # --------------------------------------------------------
    # BEHAVIORAL test
    # --------------------------------------------------------

    behavioral_response = """
{
    "next_question": "Tell me about a time you had to explain a complex analytical result to a non-technical stakeholder.",
    "question_type": "BEHAVIORAL",
    "target_competency": "Communication",
    "competency_updates": {}
}
"""

    behavioral = parse_interviewer_decision(
        response_text=behavioral_response,
        allowed_competencies=competencies,
    )

    print("\nBEHAVIORAL DECISION")
    print("-" * 80)

    print(
        f"Question Type: "
        f"{behavioral.question_type.value}"
    )

    print(
        f"Target: "
        f"{behavioral.target_competency}"
    )

    print(
        f"Question: "
        f"{behavioral.next_question}"
    )

    # --------------------------------------------------------
    # CLOSING test
    # --------------------------------------------------------

    closing_response = """
{
    "next_question": "Is there anything else about your experience that you would like us to know?",
    "question_type": "CLOSING",
    "target_competency": null,
    "competency_updates": {}
}
"""

    closing = parse_interviewer_decision(
        response_text=closing_response,
        allowed_competencies=competencies,
    )

    print("\nCLOSING DECISION")
    print("-" * 80)

    print(
        f"Question Type: "
        f"{closing.question_type.value}"
    )

    print(
        f"Target: "
        f"{closing.target_competency}"
    )

    print(
        f"Question: "
        f"{closing.next_question}"
    )

    print("\nGemini was NOT called.")


if __name__ == "__main__":
    main()
