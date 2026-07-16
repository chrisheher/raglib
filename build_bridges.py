"""
build_bridges.py
=================
Finds "unexpected connections" — entities whose normalized name is shared
between the literary graph (vgraphrag/, built from clean_text/) and the
technical graph (vgraphrag_technical/, built from templates/pdfs/).

Since entity extraction runs independently per graph with different
ontologies, a shared name is either:
  (a) a genuine bridge — the same real-world referent treated literally in
      one corpus and symbolically/thematically in the other (e.g. "buoy":
      a navigational component in a nautical manual, a symbol of human
      approximation in The New Bowditch), or
  (b) a coincidental homonym — two unrelated things that happen to share a
      name (e.g. "governor": a political figure in JR vs. an engine's fuel
      governor; "bloom": Leopold Bloom vs. an LLM's personality after
      fine-tuning).

Stage 1 (cheap, no API calls): intersect normalized node-name keys across
both graphs, then drop pairs whose types are proper-noun-prone on either
side (literary CHARACTER/AUTHOR/MYTHOLOGICAL_FIGURE, technical
ORGANIZATION) — these are the most common source of (b).

Stage 2 (one batched Claude call): for the remaining candidates, ask Claude
to judge genuine vs. coincidental and write a one-sentence note on what
makes the genuine ones interesting. This catches generic-word collisions
("dream", "fog", "light") that survive the type filter.

Output: vgraphrag_technical_db/bridges.json
  [{"name", "lit_type", "lit_desc", "lit_work", "tech_type", "tech_desc",
    "tech_work", "combined_degree", "note"}, ...]

Usage:
  python build_bridges.py
  python build_bridges.py --dry-run   # stage 1 only, print candidates
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from vgraphrag.graph_builder import load_graph as load_lit_graph, GRAPH_PATH as LIT_GRAPH_PATH
from vgraphrag_technical.graph_builder import load_graph as load_tech_graph, GRAPH_PATH as TECH_GRAPH_PATH

load_dotenv()

OUT_PATH = Path("vgraphrag_technical_db/bridges.json")
MODEL = "claude-sonnet-4-6"

# Types prone to accidental name collision rather than genuine thematic
# overlap — proper nouns (people, orgs) collide by coincidence far more
# than common nouns/concepts do.
LIT_EXCLUDE_TYPES = {"CHARACTER", "AUTHOR", "MYTHOLOGICAL_FIGURE"}
TECH_EXCLUDE_TYPES = {"ORGANIZATION"}

JUDGE_SYSTEM = """\
You judge whether a shared name between two independently-built knowledge \
graphs is a genuine thematic bridge or a coincidental homonym.

Graph A is literary/mythological criticism. Graph B is technical — nautical \
engineering manuals and AI/RAG research papers. Both graphs extracted \
entities independently, so a shared name can mean:
  - GENUINE: the same real-world referent, treated literally in one corpus \
and symbolically/thematically in the other (e.g. "buoy" as a navigational \
component in a manual, and as a symbol of human approximation in a novel \
about the sea).
  - COINCIDENCE: two unrelated things that happen to share a name (e.g. \
"governor" as a political figure vs. an engine's fuel governor).

Return ONLY a JSON array, no prose, no markdown fences."""

JUDGE_USER_TMPL = """\
Judge each of the following {n} candidates. For each, return an object:
{{"name": "...", "genuine": true|false, "note": "one sentence, present tense, \
naming the specific connection for genuine cases (blank string if not genuine)"}}

CANDIDATES:
{candidates_json}
"""


def normalize_type_pair(lit_type: str, tech_type: str) -> bool:
    """True if this type pair is worth sending to the LLM judge."""
    if lit_type in LIT_EXCLUDE_TYPES:
        return False
    if tech_type in TECH_EXCLUDE_TYPES:
        return False
    return True


def find_candidates() -> list[dict]:
    G_lit = load_lit_graph(LIT_GRAPH_PATH)
    G_tech = load_tech_graph(TECH_GRAPH_PATH)

    shared = set(G_lit.nodes()) & set(G_tech.nodes())
    print(f"literary graph: {G_lit.number_of_nodes()} nodes")
    print(f"technical graph: {G_tech.number_of_nodes()} nodes")
    print(f"{len(shared)} exact normalized-name matches")

    candidates = []
    for key in shared:
        ld = G_lit.nodes[key]
        td = G_tech.nodes[key]
        lit_type = ld.get("type", "CONCEPT")
        tech_type = td.get("type", "CONCEPT")
        if not normalize_type_pair(lit_type, tech_type):
            continue
        candidates.append({
            "name":       ld.get("name", key),
            "lit_type":   lit_type,
            "lit_desc":   ld.get("description", "")[:300],
            "lit_work":   (ld.get("work_titles") or ["?"])[0],
            "tech_type":  tech_type,
            "tech_desc":  td.get("description", "")[:300],
            "tech_work":  (td.get("work_titles") or ["?"])[0],
            "combined_degree": G_lit.degree(key) + G_tech.degree(key),
        })

    candidates.sort(key=lambda c: c["combined_degree"], reverse=True)
    print(f"{len(candidates)} candidates survive the type-pair filter")
    return candidates


def judge_candidates(candidates: list[dict]) -> list[dict]:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    judge_input = [{"name": c["name"], "lit_type": c["lit_type"], "lit_desc": c["lit_desc"],
                     "tech_type": c["tech_type"], "tech_desc": c["tech_desc"]} for c in candidates]

    user_prompt = JUDGE_USER_TMPL.format(
        n=len(judge_input),
        candidates_json=json.dumps(judge_input, ensure_ascii=False, indent=2),
    )

    msg = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
    verdicts = {v["name"]: v for v in json.loads(raw)}

    # Safety net: the judge occasionally marks genuine=true while its own
    # note explains the two sides are actually different referents (seen
    # once in practice, on "transmission" — Propp's narrative mechanism vs.
    # the mechanical marine transmission). Catch that self-contradiction.
    CONTRADICTION = re.compile(
        r'different referent|unrelated|coincidenc|distinct meaning|not (the )?same|no (real |genuine )?connection',
        re.I,
    )

    genuine = []
    for c in candidates:
        v = verdicts.get(c["name"])
        if not v or not v.get("genuine"):
            continue
        note = v.get("note", "")
        if CONTRADICTION.search(note):
            continue
        c["note"] = note
        genuine.append(c)

    print(f"{len(genuine)} of {len(candidates)} candidates judged genuine")
    return genuine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Stage 1 only — print candidates, no API call")
    args = parser.parse_args()

    candidates = find_candidates()

    if args.dry_run:
        for c in candidates[:60]:
            print(f"  {c['name']!r:30} deg={c['combined_degree']:<4} {c['lit_type']:<15} / {c['tech_type']}")
        return

    bridges = judge_candidates(candidates)
    bridges.sort(key=lambda c: c["combined_degree"], reverse=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(bridges, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(bridges)} bridges to {OUT_PATH}")
    for b in bridges[:10]:
        print(f"  {b['name']!r:25} {b['note']}")


if __name__ == "__main__":
    main()
