import os

from dotenv import load_dotenv
from google import genai


# Load variables from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Make sure it exists in your .env file."
    )


# Create Gemini client
client = genai.Client(api_key=api_key)


def generate_next_question(prompt: str) -> str:
    """
    Send the RAG interviewer prompt to Gemini
    and return the generated next interview question.
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response.text.strip()


def main():
    """
    Simple test to verify that Gemini is connected correctly.
    """

    test_prompt = """
You are conducting a Data Scientist interview.

The candidate was asked:
Tell me about your experience with machine learning.

The candidate answered:
I built classification models using Python and scikit-learn.
I also worked with TensorFlow on a deep learning project.

Generate ONE relevant follow-up interview question.

Return only the question.
"""

    print("Testing Gemini connection...\n")

    question = generate_next_question(test_prompt)

    print("Generated question:")
    print(question)


if __name__ == "__main__":
    main()
