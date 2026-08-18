from pathlib import Path

path = Path("src/agent/interviewer_agent.py")
text = path.read_text()


# ============================================================
# 1. Import QuestionSource
# ============================================================

old = """from src.state.competency_tracker import (
    CompetencyState,
)
"""

new = """from src.state.competency_tracker import (
    CompetencyState,
)
from src.policy.question_source import (
    QuestionSource,
)
"""

if old not in text:
    raise RuntimeError("Could not find competency_tracker import.")

text = text.replace(old, new, 1)


# ============================================================
# 2. Add valid question sources
# ============================================================

old = """VALID_QUESTION_TYPES = {
    question_type.value
    for question_type in QuestionType
}
"""

new = """VALID_QUESTION_TYPES = {
    question_type.value
    for question_type in QuestionType
}


VALID_QUESTION_SOURCES = {
    source.value
    for source in QuestionSource
}
"""

if old not in text:
    raise RuntimeError("Could not find VALID_QUESTION_TYPES.")

text = text.replace(old, new, 1)


# ============================================================
# 3. Add question_source to InterviewerDecision
# ============================================================

old = """    question_type: QuestionType

    target_competency: Optional[str]

    competency_updates: Dict[
"""

new = """    question_type: QuestionType

    question_source: QuestionSource

    target_competency: Optional[str]

    competency_updates: Dict[
"""

if old not in text:
    raise RuntimeError("Could not find InterviewerDecision fields.")

text = text.replace(old, new, 1)


# ============================================================
# 4. Update prompt task count
# ============================================================

old = """After every candidate response, perform FOUR tasks:

1. Determine what competency evidence was gathered from the
   CURRENT question and candidate answer.

2. Decide what TYPE of question should come next.

3. Identify the PRIMARY COMPETENCY targeted by that next
   question.

4. Generate exactly ONE next interview question.
"""

new = """After every candidate response, perform FIVE tasks:

1. Determine what competency evidence was gathered from the
   CURRENT question and candidate answer.

2. Decide what TYPE of question should come next.

3. Decide what SOURCE should drive the next question.

4. Identify the PRIMARY COMPETENCY targeted by that next
   question.

5. Generate exactly ONE next interview question.
"""

if old not in text:
    raise RuntimeError("Could not find prompt task section.")

text = text.replace(old, new, 1)


# ============================================================
# 5. Insert question-source instructions before evidence rules
# ============================================================

marker = """============================================================
EVIDENCE RULES
============================================================
"""

question_source_section = """============================================================
QUESTION SOURCES
============================================================

Every next question must have exactly ONE question source.

The source describes what primarily motivated the question.


RESUME_VALIDATION

Use this when validating a specific claim, project,
technology, responsibility, or achievement from the
candidate's resume.

Example:

Resume:
"Built predictive forecasting models."

Question:
"Walk me through the forecasting model you built and how you
evaluated it."

Source:
RESUME_VALIDATION


JD_TECHNICAL

Use this to directly test technical knowledge, reasoning, or
skills required by the JOB DESCRIPTION.

The question must NOT depend on the candidate having mentioned
the topic on their resume.

Example:

Job requirement:
Model training and validation.

Question:
"Suppose your model performs significantly better on training
data than validation data. How would you diagnose the problem
and what approaches might you use to address it?"

Source:
JD_TECHNICAL


JD_SCENARIO

Use this for a realistic technical or analytical situation
derived from the job description.

The candidate should explain how they would approach the
problem.

Example:

Job requirement:
Maintain and update analytical solutions.

Question:
"A production model's performance begins declining several
months after deployment. How would you investigate the cause
and decide whether the model needs to be retrained?"

Source:
JD_SCENARIO


BEHAVIORAL

Use this for dedicated behavioral questions.

Example:

"Tell me about a time you had to explain a complex analytical
result to a non-technical stakeholder."

Source:
BEHAVIORAL


OPENING

Reserved for the opening interview question.


CLOSING

Reserved for the closing interview question.


============================================================
SOURCE BALANCE RULES
============================================================

The interview must NOT become a resume walkthrough.

Resume evidence and the job description serve different
purposes.

RESUME_VALIDATION questions validate whether claimed
experience is genuine and sufficiently understood.

JD_TECHNICAL questions independently test whether the
candidate has knowledge required by the role.

JD_SCENARIO questions independently test whether the
candidate can apply that knowledge to realistic situations.


For HIGH-priority technical competencies:

Do not rely exclusively on RESUME_VALIDATION.

When useful, gather independent evidence using JD_TECHNICAL
or JD_SCENARIO questions.


IMPORTANT:

A candidate may have strong resume evidence but still need
independent job-based assessment.

A candidate may also have little or no resume evidence for a
job requirement. That is NOT a reason to skip the requirement.

Instead, use JD_TECHNICAL or JD_SCENARIO to assess it.


When selecting NEW_TARGET questions:

Do not automatically begin with:

"In your resume, you mentioned..."

Consider whether a JD_TECHNICAL or JD_SCENARIO question would
provide stronger independent evidence.


Maintain a natural mix of:

- resume validation
- JD-based technical assessment
- JD-based scenarios

Do not mechanically alternate sources.

Choose the source that produces the strongest useful evidence
while preventing the interview from becoming dominated by
resume questions.


"""

if marker not in text:
    raise RuntimeError("Could not find EVIDENCE RULES marker.")

text = text.replace(
    marker,
    question_source_section + marker,
    1,
)


# ============================================================
# 6. Update return JSON examples
# ============================================================

old = '''{
    "next_question": "The next interview question",
    "question_type": "FOLLOW_UP",
    "target_competency": "Competency Name",
    "competency_updates": {
        "Competency Name": "STATE"
    }
}'''

new = '''{
    "next_question": "The next interview question",
    "question_type": "FOLLOW_UP",
    "question_source": "RESUME_VALIDATION",
    "target_competency": "Competency Name",
    "competency_updates": {
        "Competency Name": "STATE"
    }
}'''

if old not in text:
    raise RuntimeError("Could not find normal return JSON example.")

text = text.replace(old, new, 1)


old = '''{
    "next_question": "The closing interview question",
    "question_type": "CLOSING",
    "target_competency": null,
    "competency_updates": {
    }
}'''

new = '''{
    "next_question": "The closing interview question",
    "question_type": "CLOSING",
    "question_source": "CLOSING",
    "target_competency": null,
    "competency_updates": {
    }
}'''

if old not in text:
    raise RuntimeError("Could not find closing return JSON example.")

text = text.replace(old, new, 1)


# ============================================================
# 7. Add allowed question sources to prompt
# ============================================================

old = """Allowed competency states:

NOT_COVERED
MENTIONED
EXPLORED
ASSESSED
"""

new = """Allowed question sources:

RESUME_VALIDATION
JD_TECHNICAL
JD_SCENARIO
BEHAVIORAL
OPENING
CLOSING


Source/type consistency:

BEHAVIORAL question type must use BEHAVIORAL source.

CLOSING question type must use CLOSING source.

FOLLOW_UP and NEW_TARGET technical questions should normally
use RESUME_VALIDATION, JD_TECHNICAL, or JD_SCENARIO.


Allowed competency states:

NOT_COVERED
MENTIONED
EXPLORED
ASSESSED
"""

if old not in text:
    raise RuntimeError("Could not find allowed competency states.")

text = text.replace(old, new, 1)


# ============================================================
# 8. Parse question_source
# ============================================================

marker = """    # --------------------------------------------------------
    # Target competency
    # --------------------------------------------------------
"""

source_parser = """    # --------------------------------------------------------
    # Question source
    # --------------------------------------------------------

    question_source_name = str(
        data.get(
            "question_source",
            "",
        )
    ).strip().upper()

    if (
        question_source_name
        not in VALID_QUESTION_SOURCES
    ):
        raise ValueError(
            "Invalid interviewer question source: "
            f"{question_source_name}"
        )

    question_source = QuestionSource[
        question_source_name
    ]

    if (
        question_type == QuestionType.BEHAVIORAL
        and question_source
        != QuestionSource.BEHAVIORAL
    ):
        raise ValueError(
            "BEHAVIORAL question must use "
            "BEHAVIORAL question source."
        )

    if (
        question_type == QuestionType.CLOSING
        and question_source
        != QuestionSource.CLOSING
    ):
        raise ValueError(
            "CLOSING question must use "
            "CLOSING question source."
        )

    if (
        question_type
        in {
            QuestionType.FOLLOW_UP,
            QuestionType.NEW_TARGET,
        }
        and question_source
        in {
            QuestionSource.BEHAVIORAL,
            QuestionSource.OPENING,
            QuestionSource.CLOSING,
        }
    ):
        raise ValueError(
            f"{question_type.value} cannot use "
            f"{question_source.value} question source."
        )

"""

if marker not in text:
    raise RuntimeError("Could not find target competency parser marker.")

text = text.replace(
    marker,
    source_parser + marker,
    1,
)


# ============================================================
# 9. Add question_source to returned decision
# ============================================================

old = """    return InterviewerDecision(
        next_question=next_question,
        question_type=question_type,
        target_competency=target_competency,
        competency_updates=updates,
    )
"""

new = """    return InterviewerDecision(
        next_question=next_question,
        question_type=question_type,
        question_source=question_source,
        target_competency=target_competency,
        competency_updates=updates,
    )
"""

if old not in text:
    raise RuntimeError("Could not find InterviewerDecision return.")

text = text.replace(old, new, 1)


# ============================================================
# 10. Update local test JSON
# ============================================================

text = text.replace(
    '"question_type": "FOLLOW_UP",\n'
    '    "target_competency": "Time Series Forecasting",',
    '"question_type": "FOLLOW_UP",\n'
    '    "question_source": "JD_TECHNICAL",\n'
    '    "target_competency": "Time Series Forecasting",',
)

text = text.replace(
    '"question_type": "NEW_TARGET",\n'
    '    "target_competency": "SQL",',
    '"question_type": "NEW_TARGET",\n'
    '    "question_source": "JD_TECHNICAL",\n'
    '    "target_competency": "SQL",',
)

text = text.replace(
    '"question_type": "BEHAVIORAL",\n'
    '    "target_competency": "Communication",',
    '"question_type": "BEHAVIORAL",\n'
    '    "question_source": "BEHAVIORAL",\n'
    '    "target_competency": "Communication",',
)

text = text.replace(
    '"question_type": "CLOSING",\n'
    '    "target_competency": null,',
    '"question_type": "CLOSING",\n'
    '    "question_source": "CLOSING",\n'
    '    "target_competency": null,',
)


# ============================================================
# 11. Print source in local tests
# ============================================================

old = """    print(
        f"Target: "
        f"{followup.target_competency}"
    )
"""

new = """    print(
        f"Question Source: "
        f"{followup.question_source.value}"
    )

    print(
        f"Target: "
        f"{followup.target_competency}"
    )
"""

if old in text:
    text = text.replace(old, new, 1)


old = """    print(
        f"Target: "
        f"{new_target.target_competency}"
    )
"""

new = """    print(
        f"Question Source: "
        f"{new_target.question_source.value}"
    )

    print(
        f"Target: "
        f"{new_target.target_competency}"
    )
"""

if old in text:
    text = text.replace(old, new, 1)


old = """    print(
        f"Target: "
        f"{behavioral.target_competency}"
    )
"""

new = """    print(
        f"Question Source: "
        f"{behavioral.question_source.value}"
    )

    print(
        f"Target: "
        f"{behavioral.target_competency}"
    )
"""

if old in text:
    text = text.replace(old, new, 1)


old = """    print(
        f"Target: "
        f"{closing.target_competency}"
    )
"""

new = """    print(
        f"Question Source: "
        f"{closing.question_source.value}"
    )

    print(
        f"Target: "
        f"{closing.target_competency}"
    )
"""

if old in text:
    text = text.replace(old, new, 1)


# ============================================================
# SAVE
# ============================================================

path.write_text(text)

print("=" * 80)
print("QUESTION SOURCE UPGRADE COMPLETE")
print("=" * 80)
print()
print("Updated:")
print("src/agent/interviewer_agent.py")
print()
print("Backup:")
print("src/agent/interviewer_agent_backup.py")
