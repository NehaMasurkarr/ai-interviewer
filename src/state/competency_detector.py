from typing import List


# ============================================================
# Competency Keyword Map
# ============================================================

COMPETENCY_KEYWORDS = {
    "Machine Learning": [
        "machine learning",
        "classification",
        "regression",
        "scikit-learn",
        "sklearn",
        "random forest",
        "decision tree",
        "logistic regression",
        "predictive model",
    ],

    "Deep Learning": [
        "deep learning",
        "tensorflow",
        "pytorch",
        "neural network",
        "cnn",
        "convolutional neural network",
        "transformer",
    ],

    "SQL": [
        "sql",
        "query",
        "queries",
        "join",
        "joins",
        "database",
        "mysql",
        "postgresql",
    ],

    "Statistics": [
        "statistics",
        "statistical",
        "hypothesis testing",
        "confidence interval",
        "p-value",
        "precision",
        "recall",
        "f1-score",
        "roc-auc",
        "correlation",
    ],

    "Communication": [
        "stakeholder",
        "stakeholders",
        "communicate",
        "communication",
        "presentation",
        "presented",
        "dashboard",
        "power bi",
        "tableau",
        "business users",
    ],
}


# ============================================================
# Detect competencies
# ============================================================

def detect_competencies(
    question: str,
    answer: str,
) -> List[str]:
    """
    Detect competencies discussed in the current interview
    question and candidate answer.
    """

    text = f"{question} {answer}".lower()

    detected = []

    for competency, keywords in COMPETENCY_KEYWORDS.items():

        if any(keyword in text for keyword in keywords):
            detected.append(competency)

    return detected


# ============================================================
# Manual test
# ============================================================

def main():

    question = (
        "Can you walk me through a classification project "
        "you built using scikit-learn?"
    )

    answer = (
        "I built a customer churn model using Python and "
        "scikit-learn. I evaluated it using precision, recall, "
        "F1-score, and ROC-AUC. I presented the results to "
        "stakeholders through a Power BI dashboard."
    )

    detected = detect_competencies(
        question=question,
        answer=answer,
    )

    print("=" * 80)
    print("COMPETENCY DETECTOR TEST")
    print("=" * 80)

    print("\nDETECTED COMPETENCIES:")

    for competency in detected:
        print(f"- {competency}")


if __name__ == "__main__":
    main()