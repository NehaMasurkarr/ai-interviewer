from src.policy.question_source import QuestionSource


def describe_question_source(
    source: QuestionSource,
) -> str:
    """
    Human-readable description used in prompts.
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
            "or situation derived from the job "
            "requirements and assess how they would "
            "approach it."
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

        print(
            f"\n{source.value}"
        )

        print(
            describe_question_source(source)
        )


if __name__ == "__main__":
    main()
