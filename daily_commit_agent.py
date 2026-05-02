"""
Daily commit agent — stages all changes, commits with a Claude-generated message,
pushes to 'develop', and opens a weekly PR to 'main' on Fridays.
"""

import json
import os
import subprocess
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import anthropic
from dotenv import load_dotenv

REPO_DIR = Path(__file__).parent
GITHUB_OWNER = "pyacobellis"
GITHUB_REPO = "vv_rag"
BRANCH = "develop"
BASE_BRANCH = "main"


def git(args: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args, cwd=REPO_DIR, capture_output=capture, text=True
    )


def staged_diff() -> str:
    git(["add", "."])
    return git(["diff", "--cached"]).stdout


def ask_claude(prompt: str, client: anthropic.Anthropic, max_tokens: int = 300) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def github(path: str, method: str = "GET", data: dict = None, token: str = "") -> dict:
    url = f"https://api.github.com{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def main():
    load_dotenv(REPO_DIR / ".env")
    token = os.environ["GITHUB_TOKEN"]
    client = anthropic.Anthropic()

    diff = staged_diff()
    if not diff.strip():
        print("Nothing to commit.")
        return

    # Commit
    commit_msg = ask_claude(
        f"Write a concise git commit message (subject line + optional short body) "
        f"for this diff. Reply with the message only:\n\n{diff[:4000]}"
    )
    git(["commit", "-m", commit_msg], capture=False)

    # Push
    git(["push", "origin", BRANCH], capture=False)
    print(f"Pushed to {BRANCH}.")

    # Open a PR on Fridays only (weekday 4 = Friday)
    if date.today().weekday() != 4:
        print("Not Friday — skipping PR.")
        return

    open_prs = github(
        f"/repos/{GITHUB_OWNER}/{GITHUB_REPO}/pulls"
        f"?state=open&head={GITHUB_OWNER}:{BRANCH}&base={BASE_BRANCH}",
        token=token,
    )
    if open_prs:
        print(f"PR already open: {open_prs[0]['html_url']}")
        return

    week = date.today().strftime("%Y-W%V")
    pr_body = ask_claude(
        f"Write a short pull request description (2-3 bullet points) summarising "
        f"this week's changes. Reply with the description only:\n\n{diff[:4000]}",
        client,
        max_tokens=400,
    )
    pr = github(
        f"/repos/{GITHUB_OWNER}/{GITHUB_REPO}/pulls",
        method="POST",
        data={
            "title": f"Weekly update {week}",
            "body": pr_body,
            "head": BRANCH,
            "base": BASE_BRANCH,
        },
        token=token,
    )
    print(f"PR created: {pr.get('html_url', pr)}")


if __name__ == "__main__":
    main()
