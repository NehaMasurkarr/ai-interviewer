from src.state.competency_assessor import assess_competencies


COMPETENCIES = [
    "Machine Learning",
    "Deep Learning",
    "SQL",
    "Statistics",
    "Communication",
]


def run_test(
    name: str,
    question: str,
    answer: str,
) -> None:

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    result = assess_competencies(
        question=question,
        answer=answer,
        competencies=COMPETENCIES,
    )

    print("\nASSESSMENT:")

    for competency, state in result.items():
        print(f"- {competency}: {state.value}")


def main():

    # --------------------------------------------------------
    # Test 1: Tool is only mentioned
    # --------------------------------------------------------

    run_test(
        name="TEST 1 - SIMPLE MENTION",
        question="Tell me about your technical experience.",
        answer="I have used TensorFlow before.",
    )

    # --------------------------------------------------------
    # Test 2: Deep learning is meaningfully explored
    # --------------------------------------------------------

    run_test(
        name="TEST 2 - DEEP LEARNING DETAIL",
        question=(
            "Can you describe the deep learning project "
            "you built with TensorFlow?"
        ),
        answer=(
            "I built an image classification model using "
            "a convolutional neural network. I used transfer "
            "learning and data augmentation and evaluated the "
            "model using precision, recall, and F1-score."
        ),
    )

    # --------------------------------------------------------
    # Test 3: SQL is directly tested
    # --------------------------------------------------------

    run_test(
        name="TEST 3 - SQL ASSESSMENT",
        question=(
            "How would you write a SQL query to calculate "
            "total sales by customer?"
        ),
        answer=(
            "I would group the records by customer_id and "
            "use SUM on the sales amount. If customer details "
            "were stored in another table, I would join the "
            "tables using customer_id before aggregating."
        ),
    )

    # --------------------------------------------------------
    # Test 4: Candidate does not know the answer
    # --------------------------------------------------------

    run_test(
        name="TEST 4 - WEAK ANSWER",
        question=(
            "Can you explain how you would perform "
            "hypothesis testing?"
        ),
        answer=(
            "I'm familiar with the term, but I haven't "
            "really used hypothesis testing and I'm not "
            "sure how I would approach it."
        ),
    )


if __name__ == "__main__":
    main()