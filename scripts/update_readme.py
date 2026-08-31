from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER = "JESUSROYETH"

OWN_ACCOUNTS = {USER.lower()}

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "README.template.md"
OUTPUT = ROOT / "README.md"

API = "https://api.github.com"
TOKEN = os.environ.get("GITHUB_TOKEN", "")


def api(path: str) -> list | dict:
    url = path if path.startswith("http") else f"{API}{path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": f"{USER}-profile-readme",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"

    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429) and attempt < 4:
                wait = 2 ** (attempt + 3)
                print(f"  rate limited, waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"giving up on {url}")


def search_total(query: str) -> int:
    q = urllib.parse.quote(query)
    data = api(f"/search/issues?q={q}&per_page=1")
    time.sleep(2)
    return int(data["total_count"])


def discover_upstream_repos() -> list[str]:
    repos: dict[str, None] = {}
    for page in range(1, 11):
        q = urllib.parse.quote(f"author:{USER} type:pr is:merged")
        data = api(f"/search/issues?q={q}&per_page=100&page={page}")
        items = data.get("items", [])
        for item in items:
            repo = item["repository_url"].removeprefix(f"{API}/repos/")
            if repo.split("/")[0].lower() not in OWN_ACCOUNTS:
                repos.setdefault(repo, None)
        if len(items) < 100:
            break
        time.sleep(2)
    return list(repos)


def build_table(counts: list[tuple[str, int]]) -> str:
    lines = ["| | Merged PRs |", "|---|---|"]
    for repo, count in counts:
        proof = (
            f"https://github.com/{repo}/pulls"
            f"?q=is%3Apr+author%3A{USER}+is%3Amerged"
        )
        lines.append(f"| [{repo}](https://github.com/{repo}) | [{count}]({proof}) |")
    return "\n".join(lines)


MENTION = re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(USER)}(?![A-Za-z0-9_-])", re.IGNORECASE)


def release_credits(repos: list[str]) -> int:
    total = 0
    for repo in repos:
        for page in range(1, 51):
            data = api(f"/repos/{repo}/releases?per_page=100&page={page}")
            if not data:
                break
            total += sum(1 for rel in data if MENTION.search(rel.get("body") or ""))
            if len(data) < 100:
                break
        else:
            raise RuntimeError(f"{repo}: more than 5000 releases, pagination cap hit")
    return total


def main() -> int:
    if not TOKEN:
        print("warning: no GITHUB_TOKEN, low rate limits apply", file=sys.stderr)

    print("discovering upstream repositories...")
    repos = discover_upstream_repos()
    if not repos:
        print("error: no upstream repos found, refusing to write an empty table", file=sys.stderr)
        return 1
    print(f"  found {len(repos)}: {', '.join(repos)}")

    print("counting merged PRs...")
    counts: list[tuple[str, int]] = []
    for repo in repos:
        n = search_total(f"author:{USER} type:pr is:merged repo:{repo}")
        print(f"  {repo}: {n}")
        counts.append((repo, n))
    counts.sort(key=lambda pair: (-pair[1], pair[0]))

    print("counting release credits...")
    credits = release_credits(repos)
    print(f"  {credits} releases")

    readme = TEMPLATE.read_text(encoding="utf-8")
    for name, value in (
        ("{{CONTRIB_TABLE}}", build_table(counts)),
        ("{{RELEASE_CREDITS}}", str(credits)),
    ):
        found = readme.count(name)
        if found != 1:
            print(f"error: {name} appears {found} times, expected exactly 1", file=sys.stderr)
            return 1
        readme = readme.replace(name, value)

    if "{{" in readme:
        leftover = readme[readme.index("{{"):][:40]
        print(f"error: unreplaced placeholder near {leftover!r}", file=sys.stderr)
        return 1

    OUTPUT.write_text(readme, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
