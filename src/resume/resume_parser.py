from pathlib import Path

from docx import Document
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
}


def extract_pdf_text(file_path: Path) -> str:
    """
    Extract text from a PDF resume.
    """

    reader = PdfReader(str(file_path))

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def extract_docx_text(file_path: Path) -> str:
    """
    Extract text from a DOCX resume.
    """

    document = Document(str(file_path))

    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


def clean_resume_text(text: str) -> str:
    """
    Perform basic cleanup while preserving resume structure.
    """

    lines = []

    for line in text.splitlines():
        cleaned_line = " ".join(line.split())

        if cleaned_line:
            lines.append(cleaned_line)

    return "\n".join(lines)


def extract_resume_text(file_path: str) -> str:
    """
    Extract and clean text from a supported resume file.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Resume file not found: {path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported resume format. "
            "Please use PDF or DOCX."
        )

    if extension == ".pdf":
        text = extract_pdf_text(path)

    else:
        text = extract_docx_text(path)

    cleaned_text = clean_resume_text(text)

    if not cleaned_text:
        raise ValueError(
            "No readable text could be extracted "
            "from the resume."
        )

    return cleaned_text


def main():
    """
    Test the parser using a resume placed in data/resumes/.
    """

    resume_directory = Path("data/resumes")

    resume_files = [
        path
        for path in resume_directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not resume_files:
        print(
            "No resume found.\n"
            "Place a PDF or DOCX resume inside:\n"
            "data/resumes/"
        )
        return

    resume_path = resume_files[0]

    print("=" * 80)
    print("RESUME PARSER TEST")
    print("=" * 80)

    print(f"\nResume: {resume_path.name}")

    text = extract_resume_text(
        str(resume_path)
    )

    print("\nEXTRACTED TEXT")
    print("-" * 80)
    print(text)

    print("\n" + "=" * 80)
    print(
        f"Characters extracted: {len(text):,}"
    )


if __name__ == "__main__":
    main()