from src.state.competency_assessor import (
    build_assessment_prompt,
    parse_assessment,
)
from src.state.competency_tracker import CompetencyState


COMPETENCIES = [
    "Machine Learning",
    "Deep Learning",
    "SQL",
    "Statistics",
    "Communication",
]


def test_parse_valid_assessment():

    response = """
    {
        "Machine Learning": "ASSESSED",
        "Deep Learning": "MENTIONED",
        "SQL": "NOT_COVERED",
        "Statistics": "EXPLORED",
        "Communication": "NOT_COVERED"
    }
    """

    result = parse_assessment(
        response_text=response,
        competencies=COMPETENCIES,
    )

    assert (
        result["Machine Learning"]
        == CompetencyState.ASSESSED
    )

    assert (
        result["Deep Learning"]
        == CompetencyState.MENTIONED
    )

    assert (
        result["SQL"]
        == CompetencyState.NOT_COVERED
    )

    assert (
        result["Statistics"]
        == CompetencyState.EXPLORED
    )


def test_invalid_state_defaults_to_not_covered():

    response = """
    {
        "Machine Learning": "EXPERT",
        "Deep Learning": "MENTIONED"
    }
    """

    result = parse_assessment(
        response_text=response,
        competencies=COMPETENCIES,
    )

    assert (
        result["Machine Learning"]
        == CompetencyState.NOT_COVERED
    )

    assert (
        result["Deep Learning"]
        == CompetencyState.MENTIONED
    )


def test_missing_competency_defaults_to_not_covered():

    response = """
    {
        "Machine Learning": "ASSESSED"
    }
    """

    result = parse_assessment(
        response_text=response,
        competencies=COMPETENCIES,
    )

    assert (
        result["Machine Learning"]
        == CompetencyState.ASSESSED
    )

    assert (
        result["SQL"]
        == CompetencyState.NOT_COVERED
    )

    assert (
        result["Communication"]
        == CompetencyState.NOT_COVERED
    )


def test_markdown_json_is_parsed():

    response = """
    ```json
    {
        "Machine Learning": "EXPLORED",
        "Deep Learning": "MENTIONED"
    }
    ```
    """

    result = parse_assessment(
        response_text=response,
        competencies=COMPETENCIES,
    )

    assert (
        result["Machine Learning"]
        == CompetencyState.EXPLORED
    )

    assert (
        result["Deep Learning"]
        == CompetencyState.MENTIONED
    )


def test_prompt_distinguishes_mention_from_assessment():

    prompt = build_assessment_prompt(
        question="Tell me about your experience.",
        answer="I have used TensorFlow.",
        competencies=COMPETENCIES,
    )

    assert "Mentioning a technology does not mean" in prompt
    assert "MENTIONED" in prompt
    assert "ASSESSED" in prompt