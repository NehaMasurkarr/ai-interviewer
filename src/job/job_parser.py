from pathlib import Path


def extract_job_description(
    file_path: str,
) -> str:
    """
    Read and clean a job description from a text file.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Job description not found: {path}"
        )

    if path.suffix.lower() != ".txt":
        raise ValueError(
            "Job description must currently be "
            "provided as a .txt file."
        )

    text = path.read_text(
        encoding="utf-8"
    )

    cleaned_text = clean_job_description(
        text
    )

    if not cleaned_text:
        raise ValueError(
            "Job description is empty."
        )

    return cleaned_text


def clean_job_description(
    text: str,
) -> str:
    """
    Clean unnecessary whitespace while preserving
    paragraph structure.
    """

    cleaned_lines = []

    for line in text.splitlines():

        line = " ".join(
            line.split()
        )

        if line:
            cleaned_lines.append(line)

    return "\n".join(
        cleaned_lines
    )


def main():
    """
    Test using the development job description.
    """

    job_path = (
        "data/jobs/job_description.txt"
    )

    print("=" * 80)
    print("JOB DESCRIPTION PARSER TEST")
    print("=" * 80)

    job_description = (
        extract_job_description(
            job_path
        )
    )

    print("\nEXTRACTED JOB DESCRIPTION")
    print("-" * 80)

    print(job_description)

    print("\n" + "=" * 80)

    print(
        "Characters extracted: "
        f"{len(job_description):,}"
    )


if __name__ == "__main__":
    main()