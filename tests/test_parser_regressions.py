from src.data.transcript_parser import (
    parse_transcript,
    create_qa_sequences,
)


def test_interviewer_dialogue_with_comma_is_not_metadata():

    transcript = """
Interviewer: Good morning, Jason. It's great to meet you. Welcome to the interview.

Jason Jones: Good morning. Thank you for having me.

Interviewer: Before we begin, I want to let you know that this interview will cover several topics. Can you start by telling me about your background?

Jason Jones: I have three years of experience in e-commerce.

Interviewer: Great. Let's dive into customer service. Can you give me an example of a difficult customer issue?

Jason Jones: I once worked with an upset customer whose preferred shipping option was unavailable.
"""

    turns = parse_transcript(
        transcript=transcript,
        candidate_name="Jason Jones",
    )

    sequences = create_qa_sequences(turns)

    assert len(sequences) == 3

    assert sequences[0]["next_question"] == (
        "Before we begin, I want to let you know that this interview "
        "will cover several topics. Can you start by telling me about "
        "your background?"
    )


def test_numbered_interviewer_labels():

    transcript = """
Interviewer 1: Tell me about your background.
Rohit: I have three years of data science experience.

Interviewer 2: Tell me about Spark.
Rohit: I have used Spark for ETL pipelines.

Interviewer 3: Tell me about machine learning.
Rohit: I have worked with classification models.
"""

    turns = parse_transcript(
        transcript=transcript,
        candidate_name="Rohit",
    )

    sequences = create_qa_sequences(turns)

    assert len(sequences) == 3

    assert all(
        turn["speaker"] == "interviewer"
        for turn in turns
        if turn["speaker_name"].startswith("interviewer")
    )


def test_hr_manager_label():

    transcript = """
HR Manager: Tell me about your HR background.
Usha: I have five years of HR experience.

HR Manager: How would you handle an employee relations issue?
Usha: I would first gather context from everyone involved.
"""

    turns = parse_transcript(
        transcript=transcript,
        candidate_name="Usha",
    )

    sequences = create_qa_sequences(turns)

    assert len(sequences) == 2


def test_interviewee_label_as_candidate():

    transcript = """
Interviewer: Tell me about your project management experience.
Interviewee: I have managed logistics and operations projects.

Interviewer: Tell me about Agile.
Interviewee: I have limited Agile experience.
"""

    turns = parse_transcript(
        transcript=transcript,
        candidate_name="Candidate Name",
    )

    sequences = create_qa_sequences(turns)

    assert len(sequences) == 2


def test_named_interviewer_without_metadata():

    transcript = """
Alex: Tell me about your design background.
Rahul: I have three years of graphic design experience.

Alex: Tell me about Illustrator.
Rahul: I use Illustrator for vector graphics.

Alex: Tell me about branding.
Rahul: Branding is an area I am still developing.
"""

    turns = parse_transcript(
        transcript=transcript,
        candidate_name="Rahul",
    )

    sequences = create_qa_sequences(turns)

    assert len(sequences) == 3


def test_split_line_interviewer_name():

    transcript = """
Interviewer: Dr.
Smith, AI Research Lead

Dr.
Smith: Tell me about your background.
Sita: I have a background in computer science and AI.

Dr.
Smith: Tell me about NLP.
Sita: I have worked with text classification and NER.
"""

    turns = parse_transcript(
        transcript=transcript,
        candidate_name="Sita",
    )

    sequences = create_qa_sequences(turns)

    assert len(sequences) == 2