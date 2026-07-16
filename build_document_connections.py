"""
build_document_connections.py
==============================
Finds "interesting connections between documents" within the literary
corpus (clean_text/) — entities that recur across multiple, genre-distant
works, surfacing echoes a reader wouldn't expect (e.g. "Metafiction"
linking Don Juan, Hamlet on the Holodeck, and DFW essays) rather than the
canonical references any literary critic would reach for (Freud,
Shakespeare, God — which appear so broadly they aren't a "discovery").

Unlike build_bridges.py (which compares two independently-built graphs),
this works within the single literary graph, using each node's
source_chunks to reconstruct which works actually mention it — the
graph's own `work_titles` field is stale (only set at node creation, never
appended to on merge), so it can't be trusted for this.

Stage 1 (cheap, no API calls):
  - Reconstruct each entity's true distinct-work list from source_chunks.
  - Drop seeded bridge entities (not organic discoveries), THEME/CONCEPT
    types (too abstract — thematic overlap is expected in literary
    criticism, not surprising), and AUTHOR/LITERARY_WORK types (a shared
    author or title is structurally a citation, not a vehicle — an entity
    has to be usable AS a symbol for something else, which names and
    titles generally aren't). Calibrated against a user-reviewed sample:
    every accepted "metaphorical" connection was SYMBOL, MYTHOLOGICAL_FIGURE,
    CHARACTER, or PLACE; every rejected "citation" connection involved a
    named authority, biographical fact, or shared subject matter.
  - SYMBOL and MYTHOLOGICAL_FIGURE get a relaxed work-count ceiling (12 vs.
    7) since they were the strongest performers in that calibration — a
    symbol recurring often can still be doing real metaphorical work in
    each specific pairing, unlike a citation, which just gets more
    canonical the more it recurs.
  - Keep entities appearing in >=2 distinct works whose works span >=2
    different top-level genres (NAUTICAL/STORIES/AI/HUMANITY, from
    ChromaDB's taxonomy_leaf metadata) — same-genre recurrence is
    expected, cross-genre is the "unexpected" signal.
  - Additionally compute how many distinct Leiden communities (from the
    literary graph's own vgraphrag_db/communities.json) the involved works
    touch. This is a finer-grained version of the genre check — the graph's
    own community-detection algorithm didn't cluster these works together,
    which is stronger evidence of a real structural surprise than a coarse
    4-bucket genre label. Used to prioritize candidates and to give the
    judge model concrete context instead of a bare entity name.

Stage 2 (one batched Claude call): judge which survivors are genuine
specific echoes worth surfacing vs. still-too-canonical, and write a
one-sentence note naming the actual juxtaposition.

Output: vgraphrag_db/document_connections.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import anthropic
import chromadb
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from vgraphrag.graph_builder import load_graph, GRAPH_PATH, strip_metadata_header, parse_filename

CLEAN_TEXT_DIR = Path("clean_text")

load_dotenv()

OUT_PATH = Path("vgraphrag_db/document_connections.json")
MODEL = "claude-sonnet-4-6"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "literary_documents"

TAXONOMY_TOP = {"1": "NAUTICAL", "2": "STORIES", "3": "AI", "4": "HUMANITY"}
EXCLUDE_TYPES = {"THEME", "CONCEPT", "AUTHOR", "LITERARY_WORK"}
HIGH_PRIOR_TYPES = {"SYMBOL", "MYTHOLOGICAL_FIGURE"}
MIN_WORKS = 2
MAX_WORKS_DEFAULT = 7
MAX_WORKS_HIGH_PRIOR = 12
COMMUNITIES_PATH = Path("vgraphrag_db/communities.json")

JUDGE_SYSTEM = """\
You judge whether an entity recurring across several literary/critical \
works is a genuine METAPHORICAL connection — the same image, symbol, or \
concept independently used as a VEHICLE mapped onto an abstract TENOR in \
two different works — as opposed to a CITATION connection, where the \
entity is merely a shared reference: a quoted rule, a biographical fact, \
a name-drop, an authority both authors cite, or two works discussing the \
same historical/mythological figure as their literal subject matter \
rather than borrowing it to characterize something else.

ACCEPT (metaphorical) — calibration examples:
- Caliban: Koestler uses Caliban to personify "lumbering emotion crashing \
into thought"; another book echoes the same metaphor. The mythological \
figure is a VEHICLE for a psychological process.
- Burning Bush: used in two unrelated books as a symbol of bodily \
luminosity / transformation-without-destruction — an image mapped onto an \
experiential state in both.
- Vulture / Kite: a symbolic bird-image (Freud's Leonardo analysis) recurs \
as a psychoanalytic symbol in an unrelated work of fiction criticism.
- Scylla and Charybdis: the myth is used as a metaphor for two entirely \
different concrete dilemmas (malpractice vs. firm collapse; a \
psychoanalytic double-bind) — same vehicle, different but resonant tenors.
- John von Neumann: his wave-function analysis is borrowed as a conceptual \
metaphor into an unrelated field (anthropology of myth) — a concept \
TRANSFERRED to explain something else, not just cited as physics.

REJECT (citation/name-recognition) — calibration examples:
- Two books both quote Ford Madox Ford's craft rule as a writing \
principle — a citation of authority, not a symbol standing in for \
something else.
- Two books mention Rhode Island's charter as a historical fact about \
religious freedom — a shared historical reference, not a metaphor.
- Two books cite Diderot's articulation of the Oedipal wish — an idea \
attributed to a thinker, not an image doing symbolic work.
- Two books both discuss Neoptolemus's moral dilemma as their literal \
subject (both are ABOUT Philoctetes) — shared subject matter, not a \
symbol borrowed to characterize something unrelated.
- A biographical anecdote (e.g. Frank Lloyd Wright's ivy-planting advice, \
Sherwood Anderson receiving a false story from Faulkner) repeated in two \
books — a shared fact, not a metaphor.

Each candidate includes n_communities and comm_context: the number of \
distinct Leiden communities (thematic clusters the graph's own community-\
detection algorithm found) the involved works belong to, and short \
summaries of those communities. n_communities >= 2 means the graph itself \
did NOT already cluster these works together — treat that as evidence \
toward genuine surprise. n_communities <= 1 means the works were already \
in the same thematic cluster, so lean more skeptical unless the metaphor \
itself is still clearly specific and non-obvious.

Return ONLY a JSON array, no prose, no markdown fences."""

JUDGE_USER_TMPL = """\
Judge each of the following {n} candidates. For each, return an object:
{{"name": "...", "metaphorical": true|false, "note": "under 25 words, \
present tense, MUST name at least two of the specific works by title (not \
'multiple works' or 'both works') and the shared vehicle/tenor (blank \
string if not metaphorical)"}}

If more than two works are listed, pick the two clearest instances of the \
metaphor and name both — never omit a second title.

Keep notes terse — this is a large batch and every response must fit.

CANDIDATES:
{candidates_json}
"""


def work_from_chunk(cid: str) -> str:
    s = cid
    if "__" in s:
        s = s.split("__")[0]
    s = re.sub(r'_\d+$', '', s)
    return s.replace("_", " ").strip()


def load_title_to_genre() -> dict:
    """Map work title -> top-level genre (NAUTICAL/STORIES/AI/HUMANITY)
    via ChromaDB's taxonomy_leaf metadata, matched against clean_text's
    'source' field (work title extraction mirrors work_from_chunk)."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(COLLECTION_NAME)
    res = col.get(limit=col.count(), include=["metadatas"])

    title_to_top = {}
    for m in res["metadatas"]:
        leaf = m.get("taxonomy_leaf", "")
        src = m.get("source", "") or m.get("title", "")
        if not leaf or not src:
            continue
        src = src.split("/")[-1]
        if src.endswith(".txt"):
            src = src[:-4]
        title_to_top[work_from_chunk(src)] = TAXONOMY_TOP.get(leaf[0], "OTHER")
    return title_to_top


def load_work_to_communities(G) -> tuple[dict[str, set[int]], dict[int, str]]:
    """
    Map work title -> set of Leiden community ids whose member entities were
    extracted from that work, plus community id -> its existing LLM summary
    (first 150 chars, for judge context). Built from communities.json's
    node_keys against the same source_chunks derivation used everywhere
    else here, since work_titles is unreliable (see module docstring).
    """
    if not COMMUNITIES_PATH.exists():
        print("  no communities.json found — skipping community-crossing signal")
        return {}, {}

    communities = json.loads(COMMUNITIES_PATH.read_text())
    work_to_comms: dict[str, set[int]] = {}
    comm_summaries: dict[int, str] = {}

    for comm in communities:
        comm_id = comm["id"]
        comm_summaries[comm_id] = (comm.get("summary") or "")[:150]
        for node_key in comm.get("node_keys", []):
            if not G.has_node(node_key):
                continue
            for cid in G.nodes[node_key].get("source_chunks", []):
                work_to_comms.setdefault(work_from_chunk(cid), set()).add(comm_id)

    return work_to_comms, comm_summaries


def find_candidates() -> list[dict]:
    G = load_graph(GRAPH_PATH)
    title_to_top = load_title_to_genre()
    work_to_comms, comm_summaries = load_work_to_communities(G)
    print(f"literary graph: {G.number_of_nodes()} nodes")
    print(f"{len(title_to_top)} works mapped to a top-level genre")
    print(f"{len(work_to_comms)} works mapped to Leiden communities")

    candidates = []
    for key, d in G.nodes(data=True):
        if d.get("work_titles") == ["__seed__"]:
            continue
        etype = d.get("type", "CONCEPT")
        if etype in EXCLUDE_TYPES:
            continue

        chunk_ids_by_work: dict[str, list[str]] = {}
        for cid in d.get("source_chunks", []):
            chunk_ids_by_work.setdefault(work_from_chunk(cid), []).append(cid)

        works = sorted(chunk_ids_by_work)
        max_works = MAX_WORKS_HIGH_PRIOR if etype in HIGH_PRIOR_TYPES else MAX_WORKS_DEFAULT
        if not (MIN_WORKS <= len(works) <= max_works):
            continue
        genres = {title_to_top.get(w) for w in works if w in title_to_top}
        genres.discard(None)
        if len(genres) < 2:
            continue

        touched_comms: set[int] = set()
        for w in works:
            touched_comms |= work_to_comms.get(w, set())
        comm_context = "; ".join(
            f"community {cid}: {comm_summaries.get(cid, '')}"
            for cid in sorted(touched_comms)[:3]
        )

        candidates.append({
            "name":              d.get("name", key),
            "type":              etype,
            "desc":              d.get("description", "")[:300],
            "works":             works,
            "genres":            sorted(genres),
            "n_communities":     len(touched_comms),
            "comm_context":      comm_context,
            "chunk_ids_by_work": chunk_ids_by_work,
        })

    candidates.sort(key=lambda c: (len(c["genres"]), c["n_communities"]), reverse=True)
    print(f"{len(candidates)} candidates survive filtering")
    return candidates


def judge_candidates(candidates: list[dict]) -> list[dict]:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    judge_input = [{"name": c["name"], "type": c["type"], "desc": c["desc"],
                     "works": c["works"], "n_communities": c["n_communities"],
                     "comm_context": c["comm_context"]} for c in candidates]

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

    if msg.stop_reason == "max_tokens":
        # Response was cut off mid-array — salvage complete objects only.
        raw = raw[:raw.rfind('}') + 1] + ']'
        print("  response truncated at max_tokens, salvaging complete entries...")

    verdicts = {v["name"]: v for v in json.loads(raw)}

    interesting = []
    for c in candidates:
        v = verdicts.get(c["name"])
        if v and v.get("metaphorical"):
            c["note"] = v.get("note", "")
            interesting.append(c)

    print(f"{len(interesting)} of {len(candidates)} candidates judged interesting")
    return interesting


REWRITE_NOTE_SYSTEM = """\
You rewrite one-line notes describing a metaphorical connection between \
literary/critical works so every note names at least two specific works \
by title. Return ONLY a JSON array, no prose, no markdown fences."""

REWRITE_NOTE_USER_TMPL = """\
Rewrite the note for each of the following {n} connections. Requirements:
- Under 25 words, present tense
- MUST name at least two of the specific works by title (never "multiple \
works" or "both works")
- If more than two works are listed, pick the two clearest instances of \
the metaphor
- Keep naming the same vehicle/tenor the existing note already identified —
  you're fixing specificity, not re-judging the connection

Return: [{{"name": "...", "note": "..."}}, ...]

CONNECTIONS:
{connections_json}
"""


def regenerate_notes(connections: list[dict]) -> list[dict]:
    """
    Rewrite just the 'note' field on an already-curated connections list,
    without re-running accept/reject judgment — cheaper than a full
    judge_candidates() pass, and doesn't risk changing which 72 survived.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    rewrite_input = [{"name": c["name"], "note": c["note"], "works": c["works"]}
                      for c in connections]

    user_prompt = REWRITE_NOTE_USER_TMPL.format(
        n=len(rewrite_input),
        connections_json=json.dumps(rewrite_input, ensure_ascii=False, indent=2),
    )

    msg = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=REWRITE_NOTE_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
    if msg.stop_reason == "max_tokens":
        raw = raw[:raw.rfind('}') + 1] + ']'
        print("  response truncated at max_tokens, salvaging complete entries...")

    rewritten = {v["name"]: v["note"] for v in json.loads(raw)}
    for c in connections:
        if c["name"] in rewritten:
            c["note"] = rewritten[c["name"]]

    return connections


CITED_WORKS_SYSTEM = """\
For each connection, identify exactly which of the listed works its note \
actually names. Return ONLY a JSON array, no prose, no markdown fences."""

CITED_WORKS_USER_TMPL = """\
For each of the following {n} connections, return the subset of "works" \
that its "note" explicitly names (verbatim strings copied from "works" — \
do not paraphrase or invent titles). A note naming two works should map to \
exactly two entries.

Return: [{{"name": "...", "cited_works": ["...", "..."]}}, ...]

CONNECTIONS:
{connections_json}
"""


def filter_passages_to_note(connections: list[dict]) -> list[dict]:
    """
    Some connections have more works (and thus passages) than their note
    names — up to 12 for SYMBOL/MYTHOLOGICAL_FIGURE candidates, but every
    note was written to name just the two clearest instances. Prune
    'passages' down to only the works the note actually discusses, so the
    modal never shows more sources than the note accounts for.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    to_check = [c for c in connections if len(c["passages"]) > 2]
    if not to_check:
        return connections

    check_input = [{"name": c["name"], "note": c["note"], "works": c["works"]}
                    for c in to_check]

    user_prompt = CITED_WORKS_USER_TMPL.format(
        n=len(check_input),
        connections_json=json.dumps(check_input, ensure_ascii=False, indent=2),
    )

    msg = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=CITED_WORKS_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
    if msg.stop_reason == "max_tokens":
        raw = raw[:raw.rfind('}') + 1] + ']'
        print("  response truncated at max_tokens, salvaging complete entries...")

    cited = {v["name"]: set(v.get("cited_works", [])) for v in json.loads(raw)}

    pruned = 0
    for c in connections:
        works = cited.get(c["name"])
        if not works:
            continue
        kept = [p for p in c["passages"] if p["work"] in works]
        if kept:  # only prune if the mapping actually matched something
            pruned += len(c["passages"]) - len(kept)
            c["passages"] = kept

    print(f"  pruned {pruned} passages not named in their connection's note")
    return connections


def _reflow_text(text: str) -> str:
    """
    clean_text/ preserves the source book's original line-wrap boundaries
    as single '\\n' characters (an OCR/extraction artifact), with real
    paragraph breaks as '\\n\\n'. Rendered as-is (white-space: pre-wrap),
    every mid-sentence line-wrap becomes a forced break, producing choppy
    short lines instead of natural reflow. Collapse single newlines within
    a paragraph into spaces; keep paragraph breaks intact.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    reflowed = [re.sub(r'\s*\n\s*', ' ', p).strip() for p in paragraphs]
    return '\n\n'.join(p for p in reflowed if p)


def _crop_to_mentions(text: str, name: str, pad: int = 250) -> str:
    """
    Crop passage text down to the span from the first mention of `name` to
    the last — the chunk itself is a fixed-size window from graph_builder
    and often starts well before (or ends well after) the entity is
    actually discussed. Falls back to the untouched text if the name isn't
    found verbatim (surface form in this chunk may differ from the graph's
    canonical name).
    """
    matches = list(re.finditer(re.escape(name), text, re.IGNORECASE))
    if not matches:
        return text

    start, end = matches[0].start(), matches[-1].end()

    # A single (or tightly clustered) mention would crop to almost nothing —
    # center a readable window instead of a bare word or two.
    if end - start < pad:
        start, end = max(0, start - pad), min(len(text), end + pad)
    else:
        start, end = max(0, start - 120), min(len(text), end + 120)

    # Trim to the nearest word boundary so we don't cut mid-word.
    if start > 0:
        sp = text.find(' ', start)
        if 0 <= sp - start < 40:
            start = sp + 1
    if end < len(text):
        sp = text.rfind(' ', end - 40, end)
        if sp != -1:
            end = sp

    cropped = text[start:end].strip()
    prefix = '…' if start > 0 else ''
    suffix = '…' if end < len(text) else ''
    return f"{prefix}{cropped}{suffix}"


def resolve_passages(connections: list[dict]) -> None:
    """
    For each surviving connection, read the actual passage text for one
    representative chunk per involved work — this is what the UI modal
    shows when a user clicks through, so it needs real text, not just the
    LLM's paraphrase. Cropped to the span between the entity's first and
    last mention in that chunk, since the chunk itself is a fixed-size
    window that often runs well past where the entity is actually
    discussed. Mutates each connection in place, replacing
    chunk_ids_by_work with a 'passages' list.
    """
    for c in connections:
        passages = []
        for work, chunk_ids in c.pop("chunk_ids_by_work", {}).items():
            chunk_id = chunk_ids[0]
            path = CLEAN_TEXT_DIR / f"{chunk_id}.txt"
            if not path.exists():
                continue
            raw = path.read_text(encoding="utf-8", errors="ignore")
            text = _reflow_text(strip_metadata_header(raw))
            text = _crop_to_mentions(text, c["name"])
            author = parse_filename(f"{chunk_id}.txt")["author"]
            author = None if author in ("unknown", "") else author
            passages.append({"work": work, "chunk_id": chunk_id, "author": author, "text": text})
        c["passages"] = passages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Stage 1 only — print candidates, no API call")
    parser.add_argument("--regenerate-notes", action="store_true",
                         help="Rewrite notes on the existing OUT_PATH connections only — no re-judging, cheap")
    parser.add_argument("--prune-passages", action="store_true",
                         help="Prune passages down to the works each note actually names — no re-judging, cheap")
    args = parser.parse_args()

    if args.regenerate_notes:
        connections = json.loads(OUT_PATH.read_text())
        connections = regenerate_notes(connections)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(connections, f, indent=2, ensure_ascii=False)
        print(f"Rewrote notes for {len(connections)} connections in {OUT_PATH}")
        for c in connections[:10]:
            print(f"  {c['name']!r:25} {c['note']}")
        return

    if args.prune_passages:
        connections = json.loads(OUT_PATH.read_text())
        connections = filter_passages_to_note(connections)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(connections, f, indent=2, ensure_ascii=False)
        print(f"Pruned passages for {len(connections)} connections in {OUT_PATH}")
        return

    candidates = find_candidates()

    if args.dry_run:
        for c in candidates[:60]:
            print(f"  {c['name']!r:30} genres={c['genres']} works={c['works'][:4]}")
        return

    connections = judge_candidates(candidates)
    resolve_passages(connections)
    connections.sort(key=lambda c: len(c["genres"]), reverse=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(connections, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(connections)} connections to {OUT_PATH}")
    for c in connections[:10]:
        print(f"  {c['name']!r:25} {c['note']}")


if __name__ == "__main__":
    main()
