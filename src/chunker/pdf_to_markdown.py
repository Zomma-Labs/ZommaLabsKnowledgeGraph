"""
PDF to Markdown converter using Gemini.

Takes a PDF file and converts it to clean, well-structured markdown
with proper header hierarchy using # notation.
"""

import argparse
import base64
import os
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure API key from environment
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


CONVERSION_PROMPT = """Convert the contents of this PDF into well-formatted Markdown.
Preserve all structural elements like headings, lists, tables, and paragraphs.
Maintain the original formatting and hierarchy. Ensure the output is clean.
You can remove the headers and footers.
All headers should use # where the number of # indicates their level (# for h1, ## for h2, etc.).
All tables MUST be formatted as HTML tables using <table>, <tr>, <th>, and <td> tags. Do NOT use markdown pipe tables."""


def pdf_to_markdown(
    pdf_path: str | Path,
    output_path: str | Path | None = None,
    model_name: str = "gemini-2.5-pro",
) -> str:
    """
    Convert a PDF to markdown using Gemini.

    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save the markdown (defaults to same name with .md)
        model_name: Gemini model to use

    Returns:
        The markdown content as a string
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if output_path is None:
        output_path = pdf_path.with_suffix('.md')
    else:
        output_path = Path(output_path)

    # Read PDF as bytes
    pdf_bytes = pdf_path.read_bytes()

    # Initialize Gemini
    model = genai.GenerativeModel(model_name)

    # Create the content with PDF
    response = model.generate_content(
        [
            CONVERSION_PROMPT,
            {
                "mime_type": "application/pdf",
                "data": base64.standard_b64encode(pdf_bytes).decode("utf-8"),
            },
        ],
        generation_config=genai.GenerationConfig(
            temperature=0.1,  # Low temperature for faithful conversion
            max_output_tokens=100000,  # Large output for full documents
        ),
    )

    markdown_content = response.text

    # Save to file
    output_path.write_text(markdown_content, encoding='utf-8')
    print(f"Saved markdown to: {output_path}")

    return markdown_content


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown using Gemini"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the PDF file to convert"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for the markdown file (defaults to same name with .md)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
        help="Gemini model to use (default: gemini-2.5-pro)"
    )

    args = parser.parse_args()

    markdown = pdf_to_markdown(
        pdf_path=args.pdf_path,
        output_path=args.output,
        model_name=args.model,
    )

    print(f"\nConverted {len(markdown)} characters of markdown")


if __name__ == "__main__":
    main()
