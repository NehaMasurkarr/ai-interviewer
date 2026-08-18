import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Make sure it exists in your .env file."
    )


client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.5-flash"


def generate_content_with_retry(
    prompt: str,
    max_retries: int = 3,
) -> str:
    """
    Send a prompt to Gemini with retry handling for
    temporary server errors such as 503 UNAVAILABLE.
    """

    for attempt in range(max_retries):

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            if not response.text:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            return response.text.strip()

        except ServerError as error:

            if attempt == max_retries - 1:
                raise

            wait_seconds = 2 ** attempt

            print(
                f"Gemini temporarily unavailable. "
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        "Gemini request failed after retries."
    )


def generate_next_question(prompt: str) -> str:
    """
    Generate the next interview question.
    """

    return generate_content_with_retry(prompt)


def main():

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