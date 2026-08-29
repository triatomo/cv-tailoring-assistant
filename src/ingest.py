"""
Load raw achievement bullets from a JSON file. Each achievement is already
an atomic unit (one bullet = one chunk), so no text-splitting is needed here
-- this module is about loading + validating structure, not chunking logic.
"""
import json
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


def load_achievements(path: str = "data/achievements/achievements.json") -> List[Achievement]:
    """Load and validate achievements from a JSON file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Achievements file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    achievements = [Achievement(**item) for item in raw]
    return achievements


if __name__ == "__main__":
    items = load_achievements()
    print(f"Loaded {len(items)} achievements.")
    for a in items[:3]:
        print(f"  [{a.id}] {a.company} / {a.role}: {a.text[:80]}...")
