from enum import Enum


class QuestionSource(Enum):
    """
    Describes where an interview question primarily
    comes from.
    """

    RESUME_VALIDATION = "RESUME_VALIDATION"
    JD_TECHNICAL = "JD_TECHNICAL"
    JD_SCENARIO = "JD_SCENARIO"
    BEHAVIORAL = "BEHAVIORAL"
    OPENING = "OPENING"
    CLOSING = "CLOSING"


def describe_question_source(
    source: QuestionSource,
) -> str:
    """
    Human-readable description of each question source.
    """

    descriptions = {
        QuestionSource.RESUME_VALIDATION: (
            "Validate a specific claim or experience "
            "from the candidate's resume."
        ),
        QuestionSource.JD_TECHNICAL: (
            "Directly assess knowledge or reasoning "
            "required by the job description without "
            "depending on a resume claim."
        ),
        QuestionSource.JD_SCENARIO: (
            "Give the candidate a realistic problem "
            "derived from the job requirements and "
            "assess how they would approach it."
        ),
        QuestionSource.BEHAVIORAL: (
            "Assess behavior such as communication, "
            "ownership, collaboration, ambiguity, "
            "conflict, or decision making."
        ),
        QuestionSource.OPENING: (
            "Opening interview question."
        ),
        QuestionSource.CLOSING: (
            "Closing interview question."
        ),
    }

    return descriptions[source]


def main():
    """
    Local test.
    """

    print("=" * 80)
    print("QUESTION SOURCE TEST")
    print("=" * 80)

    for source in QuestionSource:
        print(f"\n{source.value}")
        print(describe_question_source(source))


if __name__ == "__main__":
    main()