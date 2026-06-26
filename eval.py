"""
eval.py
=======
Evaluation suite for the VGraphRAG literary corpus system.

Runs a fixed set of queries through the engine, grades each answer
with Claude Haiku, and saves a baseline JSON for future comparison.

Usage:
  python eval.py                          # full run (30 queries)
  python eval.py --limit 10               # quick smoke test
  python eval.py --compare baseline.json  # diff against a prior run
  python eval.py --queries thematic       # run one query type only

Output:
  eval_results/YYYY-MM-DD_HHMMSS.json    — full results
  eval_results/latest.json               — symlinked to most recent

Scoring rubric (Haiku grades each answer 1–5):
  relevance   — does the answer address what was asked?
  evidence    — are citations/passages appropriate and specific?
  synthesis   — does it connect ideas across multiple works?
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path("eval_results")

# ─────────────────────────────────────────────────────────────
# TEST QUERIES
# Three types, 10 each = 30 total.
# ─────────────────────────────────────────────────────────────

QUERIES = {
    "thematic": [
        "What does the corpus say about the relationship between obsession and creativity?",
        "How do works in this corpus treat the experience of time and memory?",
        "What connects the nautical passages to the consciousness studies material?",
        "How does the theme of transformation appear across different genres in this corpus?",
        "What role does mythology play in the corpus's treatment of identity?",
        "How do postmodern novels in this corpus handle the relationship between language and reality?",
        "What does the corpus say about the nature of the unconscious?",
        "How is the sea used as a symbol or setting across different works?",
        "What connects the corpus's treatment of madness to its treatment of artistic vision?",
        "How does the corpus treat the tension between individual will and cosmic forces?",
    ],
    "character": [
        "How does Bloom parallel Odysseus across the works in this corpus?",
        "What traits define the self-destructive narrator as a recurring figure here?",
        "How is the trickster archetype expressed across mythology and fiction in this corpus?",
        "What connects Geoffrey Firmin to other protagonists in this corpus?",
        "How do the corpus's artists and writers think about their own creative process?",
        "What role do female figures play in the mythological texts in this corpus?",
        "How does the Underground Man relate to other isolated narrators in this corpus?",
        "What makes the hero's journey recognizable across the different works here?",
        "How is Odysseus reimagined or inverted across the corpus?",
        "What connects the corpus's depictions of obsessive characters?",
    ],
    "specific": [
        "What does Melville say about whiteness and its terror?",
        "How does Kazantzakis's Odysseus differ from Homer's?",
        "What is dead reckoning and how is it used metaphorically in this corpus?",
        "What does Campbell argue about the monomyth's relationship to the psyche?",
        "How does stream of consciousness work as a narrative technique in this corpus?",
        "What do the nautical texts say about navigation by the stars?",
        "How does Jung's concept of the shadow appear in this corpus?",
        "What does the corpus say about the Faust legend?",
        "How is the Minotaur myth treated across different works?",
        "What techniques for overcoming creative blocks appear in the corpus?",
    ],
}

SCORING_PROMPT = """\
You are evaluating a RAG system's answer to a query about a literary corpus.
Score the answer on three dimensions, each 1–5:

  relevance  — Does the answer directly address what was asked?
               1 = completely off-topic, 5 = fully on-target
  evidence   — Are specific works, passages, or citations used appropriately?
               1 = vague generalities, 5 = specific, well-chosen evidence
  synthesis  — Does it connect ideas across multiple works or authors?
               1 = single-source or generic, 5 = rich cross-work insight

Query: {query}

Answer: {answer}

Return only valid JSON: {{"relevance": N, "evidence": N, "synthesis": N, "note": "one sentence"}}
"""


def grade_answer(client: anthropic.Anthropic, query: str, answer: str) -> dict:
    """Grade one answer with Haiku. Returns score dict."""
    prompt = SCORING_PROMPT.format(query=query, answer=answer[:1500])
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=128,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return {"relevance": 0, "evidence": 0, "synthesis": 0, "note": "scoring failed"}


def run_eval(limit=None, query_types=None):
    from vgraphrag.engine import VGraphRAGEngine

    grader = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    engine = VGraphRAGEngine()

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = RESULTS_DIR / f"{timestamp}.json"

    types_to_run = query_types or list(QUERIES.keys())
    all_queries = []
    for qtype in types_to_run:
        for q in QUERIES[qtype]:
            all_queries.append((qtype, q))

    if limit:
        all_queries = all_queries[:limit]

    results = []
    total = len(all_queries)

    print(f"\nRunning {total} queries...\n")

    for i, (qtype, query) in enumerate(all_queries):
        print(f"[{i+1}/{total}] ({qtype}) {query[:70]}...")
        t0 = time.time()

        try:
            result = engine.query(query, use_llm_router=True)
            answer = result["answer"]
            mode   = result.get("mode", "unknown")
            gi     = result.get("graph_info", {})
            elapsed = time.time() - t0

            scores = grade_answer(grader, query, answer)
            avg = round(
                (scores.get("relevance", 0) +
                 scores.get("evidence", 0) +
                 scores.get("synthesis", 0)) / 3, 2
            )

            entry = {
                "query":          query,
                "type":           qtype,
                "mode":           mode,
                "answer":         answer,
                "scores":         scores,
                "avg_score":      avg,
                "elapsed_s":      round(elapsed, 1),
                "chunks":         gi.get("chunks_returned", 0),
                "communities":    gi.get("communities_returned", 0),
                "rels":           gi.get("rels_returned", 0),
            }
            results.append(entry)

            print(f"  mode={mode} | avg={avg:.1f}/5 "
                  f"(rel={scores.get('relevance')} ev={scores.get('evidence')} "
                  f"syn={scores.get('synthesis')}) | {elapsed:.1f}s")
            print(f"  note: {scores.get('note', '')}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "query": query, "type": qtype,
                "error": str(e), "avg_score": 0,
            })

    # ── Summary ──────────────────────────────────────────────
    scored = [r for r in results if "scores" in r]
    by_type = {}
    for qtype in types_to_run:
        subset = [r for r in scored if r["type"] == qtype]
        if subset:
            by_type[qtype] = round(sum(r["avg_score"] for r in subset) / len(subset), 2)

    overall_avg = round(sum(r["avg_score"] for r in scored) / max(len(scored), 1), 2)

    summary = {
        "timestamp":   timestamp,
        "total":       total,
        "scored":      len(scored),
        "overall_avg": overall_avg,
        "by_type":     by_type,
        "results":     results,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Keep a latest.json pointer
    latest = RESULTS_DIR / "latest.json"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(output_path.name)

    print("\n" + "=" * 60)
    print(f"Overall average: {overall_avg:.2f}/5.00")
    for qtype, avg in by_type.items():
        print(f"  {qtype:<15} {avg:.2f}")
    print(f"\nResults saved: {output_path}")
    print("=" * 60)

    return summary


def compare_runs(baseline_path: str):
    """Print a diff between the latest run and a baseline."""
    latest = RESULTS_DIR / "latest.json"
    if not latest.exists():
        print("No latest.json found. Run eval first.")
        return

    with open(latest) as f:
        current = json.load(f)
    with open(baseline_path) as f:
        baseline = json.load(f)

    print(f"\nComparing {baseline_path} → {current['timestamp']}")
    print(f"  Overall: {baseline['overall_avg']:.2f} → {current['overall_avg']:.2f} "
          f"({'▲' if current['overall_avg'] >= baseline['overall_avg'] else '▼'}"
          f"{abs(current['overall_avg'] - baseline['overall_avg']):.2f})")

    all_types = set(baseline.get("by_type", {}).keys()) | set(current.get("by_type", {}).keys())
    for qtype in sorted(all_types):
        b = baseline.get("by_type", {}).get(qtype, 0)
        c = current.get("by_type", {}).get(qtype, 0)
        delta = c - b
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        print(f"  {qtype:<15} {b:.2f} → {c:.2f}  {arrow}{abs(delta):.2f}")

    # Show biggest movers
    baseline_by_q = {r["query"]: r.get("avg_score", 0) for r in baseline.get("results", [])}
    movers = []
    for r in current.get("results", []):
        q = r["query"]
        if q in baseline_by_q:
            delta = r.get("avg_score", 0) - baseline_by_q[q]
            movers.append((delta, q, baseline_by_q[q], r.get("avg_score", 0)))

    movers.sort(key=lambda x: abs(x[0]), reverse=True)
    if movers:
        print("\nBiggest movers:")
        for delta, q, old, new in movers[:5]:
            arrow = "▲" if delta > 0 else "▼"
            print(f"  {arrow}{abs(delta):.2f}  {q[:65]}")
            print(f"        {old:.1f} → {new:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the VGraphRAG system.")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Run at most N queries (default: all 30)")
    parser.add_argument("--queries", choices=["thematic", "character", "specific"],
                        default=None, help="Run one query type only")
    parser.add_argument("--compare", type=str, default=None, metavar="BASELINE",
                        help="Compare latest results against a baseline JSON file")
    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare)
    else:
        run_eval(
            limit=args.limit,
            query_types=[args.queries] if args.queries else None,
        )


if __name__ == "__main__":
    main()
