import re
from typing import List

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.job.job_profile import (
    JobProfile,
    JobRequirement,
)
from src.planning.interview_plan import (
    InterviewPlan,
    InterviewTarget,
)
from src.profile.candidate_profile import (
    CandidateProfile,
    ResumeClaim,
)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SIMILARITY_THRESHOLD = 0.35

_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model once and reuse it.
    """

    global _embedding_model

    if _embedding_model is None:
        print(
            f"Loading semantic matching model: "
            f"{MODEL_NAME}"
        )

        _embedding_model = SentenceTransformer(
            MODEL_NAME
        )

    return _embedding_model


def normalize_term(text: str) -> str:
    """
    Normalize a technology or requirement name for
    explicit matching.
    """

    text = text.lower().strip()

    text = re.sub(
        r"[^a-z0-9+#./-]+",
        " ",
        text,
    )

    return " ".join(text.split())


def build_requirement_text(
    requirement: JobRequirement,
) -> str:
    """
    Create semantic text representing a job requirement.
    """

    parts = [
        requirement.name,
        *requirement.evidence_expected,
    ]

    return ". ".join(
        part.strip()
        for part in parts
        if part.strip()
    )


def build_claim_text(
    claim: ResumeClaim,
) -> str:
    """
    Create semantic text representing a resume claim.
    """

    parts = [
        claim.claim,
        claim.source,
        *claim.technologies,
    ]

    return ". ".join(
        part.strip()
        for part in parts
        if part.strip()
    )


def has_explicit_technology_match(
    requirement: JobRequirement,
    claim: ResumeClaim,
) -> bool:
    """
    Check whether a technology explicitly listed in the
    resume claim matches the job requirement.

    This handles requirements such as Python, SQL,
    TensorFlow, AWS, PyTorch, etc. without relying only
    on embedding similarity.
    """

    requirement_name = normalize_term(
        requirement.name
    )

    for technology in claim.technologies:

        technology_name = normalize_term(
            technology
        )

        if not technology_name:
            continue

        if (
            requirement_name == technology_name
            or technology_name in requirement_name
        ):
            return True

    return False


def find_matching_claims(
    requirement: JobRequirement,
    claims: List[ResumeClaim],
    threshold: float = SIMILARITY_THRESHOLD,
) -> List[ResumeClaim]:
    """
    Find resume claims related to a job requirement.

    Matching uses two signals:

    1. Explicit technology matching for named tools
       and technologies.

    2. Semantic similarity for broader concepts such
       as model deployment, RAG architecture, MLOps,
       experimentation, or communication.
    """

    if not claims:
        return []

    model = get_embedding_model()

    requirement_text = build_requirement_text(
        requirement
    )

    claim_texts = [
        build_claim_text(claim)
        for claim in claims
    ]

    requirement_embedding = model.encode(
        [requirement_text],
        normalize_embeddings=True,
    )

    claim_embeddings = model.encode(
        claim_texts,
        normalize_embeddings=True,
    )

    similarities = cosine_similarity(
        requirement_embedding,
        claim_embeddings,
    )[0]

    matches = []

    for claim, score in zip(
        claims,
        similarities,
    ):

        explicit_match = (
            has_explicit_technology_match(
                requirement=requirement,
                claim=claim,
            )
        )

        semantic_match = (
            float(score) >= threshold
        )

        if explicit_match or semantic_match:

            matches.append(
                (
                    claim,
                    float(score),
                    explicit_match,
                )
            )

    matches.sort(
        key=lambda item: (
            item[2],
            item[1],
        ),
        reverse=True,
    )

    return [
        claim
        for claim, _, _ in matches
    ]


def build_reason(
    requirement: JobRequirement,
    matching_claims: List[ResumeClaim],
) -> str:
    """
    Explain why a requirement should be assessed.
    """

    if matching_claims:
        return (
            f"The job requires {requirement.name}, and "
            "the candidate has related resume evidence "
            "that should be validated during the interview."
        )

    return (
        f"The job requires {requirement.name}, but the "
        "candidate's resume does not provide strong direct "
        "evidence for this requirement, so it should be "
        "assessed during the interview."
    )


def build_interview_plan(
    candidate: CandidateProfile,
    job: JobProfile,
) -> InterviewPlan:
    """
    Build an interview plan by aligning job requirements
    with candidate resume evidence.
    """

    targets = []

    for requirement in job.requirements:

        matching_claims = find_matching_claims(
            requirement=requirement,
            claims=candidate.claims,
        )

        resume_evidence = [
            claim.claim
            for claim in matching_claims
        ]

        target = InterviewTarget(
            competency=requirement.name,
            priority=requirement.priority,
            reason=build_reason(
                requirement=requirement,
                matching_claims=matching_claims,
            ),
            resume_evidence=resume_evidence,
            evidence_expected=(
                requirement.evidence_expected
            ),
        )

        targets.append(target)

    priority_order = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2,
    }

    targets.sort(
        key=lambda target: priority_order.get(
            target.priority,
            1,
        )
    )

    return InterviewPlan(
        role=job.role,
        targets=targets,
    )


def main():
    """
    Test hybrid resume/JD alignment locally.
    """

    candidate = CandidateProfile(
        name="Test Candidate",
        claims=[
            ResumeClaim(
                claim=(
                    "Built an LLM-powered AI interviewer "
                    "using LangChain and RAG pipelines."
                ),
                source="Machine Learning Engineer Intern",
                technologies=[
                    "LLM",
                    "LangChain",
                    "RAG",
                ],
            ),
            ResumeClaim(
                claim=(
                    "Built data pipelines using Python, "
                    "SQL, Databricks, and AWS."
                ),
                source="Data Scientist Intern",
                technologies=[
                    "Python",
                    "SQL",
                    "Databricks",
                    "AWS",
                ],
            ),
            ResumeClaim(
                claim=(
                    "Implemented CI/CD pipelines for "
                    "machine learning deployment."
                ),
                source="Machine Learning Engineer Intern",
                technologies=[
                    "CI/CD",
                    "MLOps",
                ],
            ),
        ],
    )

    job = JobProfile(
        role="Machine Learning Engineer",
        company="Example Company",
        requirements=[
            JobRequirement(
                name="LLMs and RAG",
                priority="HIGH",
                evidence_expected=[
                    "RAG architecture",
                    "retrieval strategy",
                    "LLM evaluation",
                ],
            ),
            JobRequirement(
                name="Python",
                priority="HIGH",
                evidence_expected=[
                    "Production Python experience",
                ],
            ),
            JobRequirement(
                name="SQL",
                priority="MEDIUM",
                evidence_expected=[
                    "Querying and data manipulation",
                ],
            ),
            JobRequirement(
                name="Model Deployment",
                priority="MEDIUM",
                evidence_expected=[
                    "Deploying machine learning models",
                    "Production model lifecycle",
                ],
            ),
        ],
    )

    plan = build_interview_plan(
        candidate=candidate,
        job=job,
    )

    print("=" * 80)
    print("HYBRID INTERVIEW PLAN TEST")
    print("=" * 80)

    print(f"\nCandidate: {candidate.name}")
    print(f"Role: {plan.role}")

    print("\nINTERVIEW TARGETS")

    for target in plan.targets:

        print(
            f"\n- {target.competency} "
            f"[{target.priority}]"
        )

        print(
            f"  State: {target.state.value}"
        )

        print(
            f"  Reason: {target.reason}"
        )

        if target.resume_evidence:

            print("  Matching Resume Evidence:")

            for evidence in target.resume_evidence:
                print(f"    - {evidence}")

        else:
            print(
                "  Matching Resume Evidence: None"
            )

        if target.evidence_expected:

            print("  Evidence Expected:")

            for evidence in target.evidence_expected:
                print(f"    - {evidence}")


if __name__ == "__main__":
    main()