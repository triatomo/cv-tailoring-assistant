"""
Load raw achievement bullets from a JSON file (full-line // comments allowed). Each achievement is already
an atomic unit (one bullet = one chunk), so no text-splitting is needed here
-- this module is about loading + validating structure, not chunking logic.
"""
import json
import sys
from pathlib import Path
from typing import List
from pydantic import BaseModel


class Achievement(BaseModel):
    id: str
    company: str
    role: str
    date_range: str
    text: str
    tags: List[str]
    highlights: List[str] = []  # phrases rendered in bold on the CV, kept verbatim by rewrites


_warned = set()  # (id, phrase) pairs already reported; achievements are reloaded per requirement


def load_achievements(path: str = "data/achievements/achievements.json") -> List[Achievement]:
    """Load and validate achievements from a JSON file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Achievements file not found: {path}")

    # Full-line "//" comments are allowed (used as role group headers in the file).
    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.loads("".join(line for line in f if not line.lstrip().startswith("//")))

    achievements = [Achievement(**item) for item in raw]
    # The CV bolds a highlight only where it appears verbatim, so a stale one fails silently.
    for a in achievements:
        for phrase in a.highlights:
            if phrase.lower() not in a.text.lower() and (a.id, phrase) not in _warned:
                _warned.add((a.id, phrase))
                print(f"Warning: highlight not found in {a.id}'s text, it won't be bold: '{phrase}'", file=sys.stderr)
    return achievements


if __name__ == "__main__":
    items = load_achievements()
    print(f"Loaded {len(items)} achievements.")
    for a in items[:3]:
        print(f"  [{a.id}] {a.company} / {a.role}: {a.text[:80]}...")
