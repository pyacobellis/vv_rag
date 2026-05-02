"""
Daily comment agent — reads the codebase and bylaws, asks Claude for one
specific observation, and appends it (with today's date) to daily_notes.py.
Run this before daily_commit_agent.py so the note gets picked up and pushed.
"""

from datetime import date
from pathlib import Path

import anthropic
from dotenv import load_dotenv

REPO_DIR = Path(__file__).parent


def main():
    load_dotenv(REPO_DIR / ".env")
    client = anthropic.Anthropic()

    code = (REPO_DIR / "utils.py").read_text(encoding="utf-8")
    bylaws_sample = (REPO_DIR / "bylaws_clean.txt").read_text(encoding="utf-8")[:2000]

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                "You are a thoughtful reviewer of a RAG system that queries "
                "Victorian Volunteer bylaws using LlamaIndex and Claude.\n\n"
                f"Utility code:\n{code}\n\n"
                f"Bylaws sample:\n{bylaws_sample}\n\n"
                "Write ONE specific, insightful observation about the code, the bylaws, "
                "or a concrete improvement idea. Format it as 1-3 Python comment lines "
                "(starting with #). Be specific — no generic praise."
            ),
        }],
    )

    observation = response.content[0].text.strip()
    today = date.today().isoformat()

    notes_file = REPO_DIR / "daily_notes.py"
    with notes_file.open("a", encoding="utf-8") as f:
        f.write(f"\n# [{today}]\n{observation}\n")

    print(f"Observation added for {today}.")


if __name__ == "__main__":
    main()
