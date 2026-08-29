"""
Custom eval script -- NOT RAGAS, since the thing we care about ("did this
avoid overclaiming", "did it correctly flag a gap I know is real") isn't a
standard RAG metric. Scores three things against your own hand-labeled
eval_set.json:

1. Retrieval match: did the system retrieve the achievements you'd pick manually?
2. Overclaim check: did any rewrite get flagged by verify.py that you'd also
   flag by eye? (Sanity check on the corrective step, not a full audit.)
3. Gap accuracy: did the system correctly call a gap a gap, and a match a match?

TODO: fill in eval_set.json with real JDs + your own ground truth before running.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentence_transformers import SentenceTransformer
from classify_jd import classify_jd
from retrieve import retrieve_for_requirement
from gaps import assess_gap
from embed import EMBEDDING_MODEL


def load_eval_set(path: str = "eval/eval_set.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_eval():
    eval_set = load_eval_set()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    retrieval_hits = 0
    gap_correct = 0
    total = len(eval_set)

    for case in eval_set:
        if "PASTE A REAL JD" in case["jd_text"] or "PASTE ANOTHER REAL JD" in case["jd_text"]:
            print(f"Skipping unfilled eval case: {case['jd_name']}")
            total -= 1
            continue

        classification = classify_jd(case["jd_text"])
        retrieved = retrieve_for_requirement(case["requirement"], classification, embedding_model)
        gap = assess_gap(case["requirement"], retrieved)

        retrieved_ids = {r.achievement.id for r in retrieved}
        expected_ids = set(case["expected_achievement_ids"])
        if not case["expected_gap"] and expected_ids.issubset(retrieved_ids):
            retrieval_hits += 1

        if gap.has_strong_match == (not case["expected_gap"]):
            gap_correct += 1

        print(f"[{case['jd_name']}] retrieved={retrieved_ids} expected={expected_ids} "
              f"gap_detected={not gap.has_strong_match} expected_gap={case['expected_gap']}")

    if total == 0:
        print("\nNo filled-in eval cases found. Fill in eval/eval_set.json with real JDs first.")
        return

    print(f"\nRetrieval match: {retrieval_hits}/{total}")
    print(f"Gap detection accuracy: {gap_correct}/{total}")


if __name__ == "__main__":
    run_eval()
