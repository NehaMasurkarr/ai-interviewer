from src.data.transcript_parser import (
    parse_transcript,
    create_qa_sequences,
)


# ============================================================
# 1. Standard interviewer + named candidate
# ============================================================

def test_standard_transcript():

    transcript = """
Interviewer: Tell me about your experience with machine learning.
Neha: I have worked with NLP and LLM applications.

Interviewer: Tell me about an NLP project.
Neha: I built an interview transcript processing pipeline.
"""

    turns = parse_transcript(
        transcript=transcript,
        candidate_name="Neha",
    )

    assert len(turns) == 4
    assert turns[0]["speaker"] == "interviewer"
    assert turns[1]["speaker"] == "candidate"

    sequences = create_qa_sequences(turns)

    assert len(sequences) == 2
    assert sequences[0]["next_question"] == "Tell me about an NLP project."
    assert sequences[1]["next_question"] is None


# ============================================================
# 2. Markdown heading speaker labels
# ============================================================

def test_markdown_speaker_labels():

    transcript = """
Interviewer: Tell me about yourself.
# Kristina Patrick: I am a software engineer.

# Interviewer: Tell me about Python.
Kristina Patrick: I use Python for backend development.
"""

    turns = parse_transcript(
        transcript=transcript,
        candidate_name="Kristina Patrick",
    )

    assert len(turns) == 4
    assert turns[0]["speaker"] == "interviewer"
    assert turns[1]["speaker"] == "candidate"
    assert turns[2]["speaker"] == "interviewer"
    assert turns[3]["speaker"] == "candidate"


# ============================================================
# 3. Shortened candidate name
# ============================================================

def test_shortened_candidate_name():

    transcript = """
Interviewer: Tell me about your background.
Mrs. George: I have eight years of data engineering experience.

Interviewer: Tell me about data warehousing.
Mrs. George: I have worked extensively with Redshift.
"""

    turns = parse_transcript(
        transcript=transcript,
        candidate_name="Mrs. Brittany George",
    )

    assert turns[1]["speaker"] == "candidate"
    assert turns[3]["speaker"] == "candidate"


# ============================================================
# 4. Named interviewer
# ============================================================

def test_named_interviewer():

    transcript = """
Interviewer: Rachel Lee, Senior Software Engineer
Candidate: Elizabeth Smith, UI Engineer candidate

Rachel Lee: Tell me about your background.
Elizabeth Smith: I have five years of frontend experience.

Rachel Lee: Tell me about Git.
Elizabeth Smith: I have used Git in several projects.
"""

    turns = parse_transcript(
        transcript=transcript,
        candidate_name="Elizabeth Smith",
    )

    sequences = create_qa_sequences(turns)

    assert len(sequences) == 2
    assert turns[0]["speaker"] == "interviewer"
    assert turns[1]["speaker"] == "candidate"


# ============================================================
# 5. Shortened interviewer name
# ============================================================

def test_shortened_interviewer_name():

    transcript = """
Interviewer: Dr. Rachel Kim, Lead Data Scientist
Candidate: Sharon Brooks

Dr. Kim: Tell me about machine learning.
Sharon Brooks: I have worked with regression models.

Dr. Kim: Tell me about Python.
Sharon Brooks: I use Python for data analysis.
"""

    turns = parse_transcript(
        transcript=transcript,
        candidate_name="Sharon Brooks",
    )

    sequences = create_qa_sequences(turns)

    assert len(sequences) == 2
    assert turns[0]["speaker"] == "interviewer"
    assert turns[1]["speaker"] == "candidate"


# ============================================================
# 6. Candidate label used as dialogue
# ============================================================

def test_candidate_label_dialogue():

    transcript = """
Interviewer: Dr. Rachel Lee, Data Science Team Lead
Candidate: David Fields, Data Scientist Candidate

Interviewer: Tell me about your background.
Candidate: I have five years of experience working with data.

Interviewer: Tell me about statistical modeling.
Candidate: I have worked with regression and classification.
"""

    turns = parse_transcript(
        transcript=transcript,
        candidate_name="David Fields",
    )

    sequences = create_qa_sequences(turns)

    assert len(sequences) == 2
    assert sequences[0]["answer"] == (
        "I have five years of experience working with data."
    )


# ============================================================
# 7. Candidate-name typo
# ============================================================

def test_candidate_name_typo():

    transcript = """
Interviewer: Tell me about yourself.
Michael Daniel: I have eight years of experience.

Interviewer: Tell me about data architecture.
Michael Daniel: I specialize in database architecture.
"""

    turns = parse_transcript(
        transcript=transcript,
        candidate_name="Micheal Daniel",
    )

    sequences = create_qa_sequences(turns)

    assert len(sequences) == 2


# ============================================================
# 8. Panel interview
# ============================================================

def test_panel_interview():

    transcript = """
Interviewees:
- Dr. Lisa Nguyen, Hiring Manager
- Mr. John Lee, Data Science Team Lead
- Ms. Sophia Patel, Data Engineer

Candidate: Randy Murphy

Dr. Lisa Nguyen: Tell us about your background.
Randy Murphy: I have five years of data analysis experience.

Mr. John Lee: Tell us about Tableau.
Randy Murphy: I have created dashboards using Tableau.

Ms. Sophia Patel: Tell us about data pipelines.
Randy Murphy: I have worked with Apache Beam.
"""

    turns = parse_transcript(
        transcript=transcript,
        candidate_name="Randy Murphy",
    )

    sequences = create_qa_sequences(turns)

    assert len(sequences) == 3
    assert sequences[0]["question"] == "Tell us about your background."
    assert sequences[1]["question"] == "Tell us about Tableau."
    assert sequences[2]["question"] == "Tell us about data pipelines."


# ============================================================
# 9. Q -> A -> Next-Q relationship
# ============================================================

def test_next_question_sequence():

    transcript = """
Interviewer: What is overfitting?
Neha: Overfitting occurs when a model learns the training data too closely.

Interviewer: How would you reduce overfitting?
Neha: I could use regularization or cross-validation.

Interviewer: What is cross-validation?
Neha: It evaluates a model across multiple train-validation splits.
"""

    turns = parse_transcript(
        transcript=transcript,
        candidate_name="Neha",
    )

    sequences = create_qa_sequences(turns)

    assert sequences[0]["next_question"] == (
        "How would you reduce overfitting?"
    )

    assert sequences[1]["next_question"] == (
        "What is cross-validation?"
    )

    assert sequences[2]["next_question"] is None


# ============================================================
# 10. Markdown bold speaker labels
# ============================================================

def test_markdown_bold_speaker_labels():

    transcript = """
**Interviewer:** Dr. Rachel Kim, Hiring Manager
**Candidate:** Amelia, Senior Data Scientist

**Dr. Kim:** Tell me about your machine learning experience.
**Amelia:** I have worked with supervised and unsupervised learning.

**Dr. Kim:** Tell me about large-scale data processing.
**Amelia:** I have used Spark to process large datasets.
"""

    turns = parse_transcript(
        transcript=transcript,
        candidate_name="Amelia",
    )

    sequences = create_qa_sequences(turns)

    assert len(sequences) == 2

    assert sequences[0]["question"] == (
        "Tell me about your machine learning experience."
    )

    assert sequences[0]["answer"] == (
        "I have worked with supervised and unsupervised learning."
    )


# ============================================================
# 11. Hiring Manager used as interviewer label
# ============================================================

def test_hiring_manager_label():

    transcript = """
**Hiring Manager:** Good morning, Ravi. Tell me about your HR experience.
**Ravi:** I have five years of experience in HR.

**Hiring Manager:** How would you handle an employee relations issue?
**Ravi:** I would first speak with the employee to understand the situation.

**Hiring Manager:** Tell me about your experience developing training programs.
**Ravi:** I have designed training programs using needs assessments.
"""

    turns = parse_transcript(
        transcript=transcript,
        candidate_name="Ravi",
    )

    sequences = create_qa_sequences(turns)

    assert len(sequences) == 3

    assert sequences[0]["question"] == (
        "Good morning, Ravi. Tell me about your HR experience."
    )

    assert sequences[0]["answer"] == (
        "I have five years of experience in HR."
    )