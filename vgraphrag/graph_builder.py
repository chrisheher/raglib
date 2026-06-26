"""
graph_builder.py
================
Stage 1: Corpus → Rich Knowledge Graph (RKG)

Reads every .txt file from clean_text/, strips the metadata header,
calls Claude to extract entities and relationships, and builds a
NetworkX DiGraph.  Progress is checkpointed to disk so the build
can be interrupted and resumed safely.

RKG node attributes:  name, type, description, source_chunks (list)
RKG edge attributes:  name, keywords, description, weight, source_chunk
"""

import json
import re
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import anthropic
import networkx as nx
from dotenv import load_dotenv

from .prompts import EXTRACTION_SYSTEM, EXTRACTION_USER

load_dotenv()

# ─── paths ────────────────────────────────────────────────────
CLEAN_TEXT_DIR  = Path("clean_text")
DB_DIR          = Path("vgraphrag_db")
GRAPH_PATH      = DB_DIR / "rkg.json"
PROGRESS_PATH   = DB_DIR / "extraction_progress.json"

# ─── model ────────────────────────────────────────────────────
MODEL        = "claude-sonnet-4-6"

# ─── corpus focus ─────────────────────────────────────────────
# Works excluded from graph extraction — technical instruction
# manuals whose entities (COLREGS, bilge pump, clove hitch) have
# no edges into the literary/mythological graph and distort PPR
# and Leiden community detection.
# These files remain in the ChromaDB chunk collection and are
# still retrievable by direct vector search.
SKIP_WORKS = {
    "Boating Skills and Seamanship",
    "Boatowners Manual",
    "Piloting",
}

def _should_skip(file_info: dict) -> bool:
    """Return True if this file's work title is in the skip list."""
    return file_info["work_title"] in SKIP_WORKS
MAX_TOKENS   = 4096
RATE_PAUSE   = 0.6    # seconds between API calls (avoid rate limits)
BATCH_SAVE   = 25     # save checkpoint every N files


# ─────────────────────────────────────────────────────────────
# FILENAME PARSING
# Extracts work title, author, section from the naming convention:
#   {Work Title}__{Author}_{section}.txt   (double underscore separator)
#   {Work Title}_{page_or_section}.txt     (no author)
# ─────────────────────────────────────────────────────────────

def parse_filename(filename: str) -> dict:
    stem = Path(filename).stem

    if "__" in stem:
        title_part, rest = stem.split("__", 1)
        parts = rest.split("_", 1)
        author  = parts[0] if parts else "unknown"
        section = parts[1] if len(parts) > 1 else ""
    else:
        # No author separator — treat whole stem as title + section
        parts   = stem.rsplit("_", 1)
        title_part = parts[0]
        author     = "unknown"
        section    = parts[1] if len(parts) > 1 else ""

    return {
        "work_title": title_part.replace("_", " ").strip(),
        "author":     author.replace("_", " ").strip(),
        "section":    section.replace("_", " ").strip(),
    }


def strip_metadata_header(text: str) -> str:
    if "### END METADATA ###" in text:
        return text.split("### END METADATA ###", 1)[1].strip()
    return text.strip()


def looks_garbled(text: str) -> bool:
    """Heuristic: reject text that is mostly OCR garbage."""
    if not text:
        return True
    words = text.split()
    if len(words) < 30:
        return True
    # Ratio of non-ASCII-printable characters
    non_ascii = sum(1 for c in text if ord(c) > 127 or (ord(c) < 32 and c not in "\n\t"))
    if non_ascii / max(len(text), 1) > 0.08:
        return True
    return False


# ─────────────────────────────────────────────────────────────
# LLM EXTRACTION
# ─────────────────────────────────────────────────────────────

def _call_claude(client: anthropic.Anthropic, system: str, user: str,
                 retries: int = 3) -> Optional[str]:
    """Call Claude with exponential backoff on errors."""
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            print(f"  Rate limit, waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None


def _repair_truncated_json(raw: str) -> Optional[dict]:
    """
    Attempt to salvage a truncated JSON response by walking back from the
    last successfully-closed list item in 'entities' or 'relationships'.
    Returns a partial result dict, or None if repair fails.
    """
    result = {"entities": [], "relationships": []}
    for key in ("entities", "relationships"):
        # Find the opening bracket for this array
        pattern = rf'"{key}"\s*:\s*\['
        m = re.search(pattern, raw)
        if not m:
            continue
        start = m.end() - 1  # position of '['
        # Walk backwards from end of string to find last complete {...}
        segment = raw[start:]
        depth = 0
        last_good = start
        i = 0
        while i < len(segment):
            c = segment[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    last_good = i
            i += 1
        trimmed = segment[:last_good + 1] + ']'
        try:
            parsed_list = json.loads(trimmed)
            result[key] = parsed_list
        except json.JSONDecodeError:
            pass
    if result["entities"] or result["relationships"]:
        print(f"  Repair recovered {len(result['entities'])} entities, "
              f"{len(result['relationships'])} relationships")
        return result
    return None


def extract_entities_and_relations(
    client: anthropic.Anthropic,
    text: str,
    file_info: dict,
) -> dict:
    """
    Call Claude to extract entities and relationships from one chunk.
    Returns {"entities": [...], "relationships": [...]} or empty dicts on failure.
    """
    user_prompt = EXTRACTION_USER.format(
        work_title=file_info["work_title"],
        author=file_info["author"],
        section=file_info["section"],
        text=text[:6000],  # cap to avoid very long prompts
    )

    raw = _call_claude(client, EXTRACTION_SYSTEM, user_prompt)
    if not raw:
        return {"entities": [], "relationships": []}

    # Strip markdown code fences if Claude added them
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw.strip(), flags=re.MULTILINE)

    try:
        parsed = json.loads(raw)
        return {
            "entities":      parsed.get("entities", []),
            "relationships": parsed.get("relationships", []),
        }
    except json.JSONDecodeError as e:
        print(f"  JSON parse failed: {e}, attempting repair...")
        repaired = _repair_truncated_json(raw)
        if repaired:
            return repaired
        return {"entities": [], "relationships": []}


# ─────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Canonical entity key: lowercase, strip leading 'the ', collapse spaces."""
    n = name.lower().strip()
    n = re.sub(r'^the\s+', '', n)
    n = re.sub(r'\s+', ' ', n)
    return n


def add_extraction_to_graph(
    G: nx.DiGraph,
    extraction: dict,
    chunk_id: str,
    work_title: str,
) -> None:
    """
    Merge one chunk's extraction results into the growing graph.
    Nodes are keyed by normalised name; duplicates are merged by
    extending source_chunks and, if the new description is longer,
    updating the description.
    """
    name_to_key = {}

    for ent in extraction.get("entities", []):
        raw_name = ent.get("name", "").strip()
        if not raw_name:
            continue
        key = _normalize_name(raw_name)
        name_to_key[raw_name] = key

        if G.has_node(key):
            # Merge: accumulate source chunks, keep longest description
            existing = G.nodes[key]
            existing["source_chunks"].append(chunk_id)
            if len(ent.get("description", "")) > len(existing.get("description", "")):
                existing["description"] = ent["description"]
        else:
            G.add_node(key, **{
                "name":          raw_name,
                "type":          ent.get("type", "CONCEPT"),
                "description":   ent.get("description", ""),
                "source_chunks": [chunk_id],
                "work_titles":   [work_title],
            })

    for rel in extraction.get("relationships", []):
        src_raw = rel.get("source", "").strip()
        tgt_raw = rel.get("target", "").strip()
        if not src_raw or not tgt_raw:
            continue

        src_key = name_to_key.get(src_raw, _normalize_name(src_raw))
        tgt_key = name_to_key.get(tgt_raw, _normalize_name(tgt_raw))

        # Ensure nodes exist even if not listed in entities (defensive)
        for key, raw in [(src_key, src_raw), (tgt_key, tgt_raw)]:
            if not G.has_node(key):
                G.add_node(key, name=raw, type="CONCEPT",
                           description="", source_chunks=[chunk_id],
                           work_titles=[work_title])

        # Allow multiple edges between the same pair (one per chunk)
        edge_key = (src_key, tgt_key, chunk_id)
        if not G.has_edge(src_key, tgt_key):
            G.add_edge(src_key, tgt_key, **{
                "name":         rel.get("name", "related_to"),
                "keywords":     rel.get("keywords", []),
                "description":  rel.get("description", ""),
                "weight":       float(rel.get("weight", 0.5)),
                "source_chunk": chunk_id,
                "work_title":   work_title,
            })
        else:
            # Strengthen existing edge if this chunk reinforces it
            existing_w = G[src_key][tgt_key]["weight"]
            new_w      = float(rel.get("weight", 0.5))
            G[src_key][tgt_key]["weight"] = min(existing_w + new_w * 0.2, 1.0)


# ─────────────────────────────────────────────────────────────
# SERIALIZATION
# NetworkX graph → JSON (node-link format, human-readable)
# ─────────────────────────────────────────────────────────────

def save_graph(G: nx.DiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(G)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_graph(path: Path) -> nx.DiGraph:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # NetworkX 3.4+ uses 'edges' key; older versions used 'links'
    if "links" in data and "edges" not in data:
        data["edges"] = data.pop("links")
    return nx.node_link_graph(data, directed=True, multigraph=False)


# ─────────────────────────────────────────────────────────────
# PROGRESS CHECKPOINTING
# ─────────────────────────────────────────────────────────────

def _load_progress() -> set:
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return set(json.load(f))
    return set()


def _save_progress(done: set) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(list(done), f)


# ─────────────────────────────────────────────────────────────
# MAIN BUILD FUNCTION
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# SEED ENTITIES
# Canonical anchor nodes added before extraction runs.
# These ensure key hub entities exist even if Claude misses them
# in OCR-corrupted or low-density chunks, and bridge disconnected
# genre clusters (nautical ↔ mythological ↔ psychological).
# ─────────────────────────────────────────────────────────────

SEED_ENTITIES = [
    # ── Mythological / archetypal hubs ───────────────────────
    {
        "name": "Odysseus",
        "type": "MYTHOLOGICAL_FIGURE",
        "description": (
            "The archetypal wandering hero — cunning, restless, homeward-bound. "
            "Central figure in Homer, radically reimagined by Kazantzakis as a "
            "Nietzschean seeker who sails past Ithaca. Paralleled by Bloom in Joyce, "
            "echoed in every questing narrator who cannot stop moving."
        ),
        "relationships": [
            ("the sea voyage as archetype", "enacts", ["journey", "nostos", "wandering"], 0.9),
            ("the trickster", "embodies", ["cunning", "disguise", "survival"], 0.7),
            ("Leopold Bloom", "parallels", ["modern odyssey", "urban wandering"], 0.85),
            ("the hero's journey", "completes", ["departure", "ordeal", "return"], 0.9),
        ],
    },
    {
        "name": "the trickster",
        "type": "MYTHOLOGICAL_FIGURE",
        "description": (
            "The archetypal figure who violates rules to expose their arbitrariness — "
            "Hermes, Loki, Coyote, and their literary descendants. Appears as the "
            "disruptive narrator, the con-artist protagonist, the child who names the "
            "emperor's nakedness. Campbell and Hyde both trace this figure."
        ),
        "relationships": [
            ("the hero's journey", "disrupts", ["chaos", "boundary-crossing"], 0.7),
            ("J.R. Vansant", "enacts", ["capitalism", "naive disruption"], 0.75),
            ("creativity", "requires", ["rule-breaking", "play"], 0.7),
        ],
    },
    {
        "name": "the hero's journey",
        "type": "CONCEPT",
        "description": (
            "Campbell's monomyth: departure, initiation, return. The deep structure "
            "underlying myth, fairy tale, and the modern novel. In this corpus it "
            "manifests literally in mythology texts and structurally in every "
            "protagonist who leaves home, descends, and resurfaces changed — or fails to."
        ),
        "relationships": [
            ("Odysseus", "archetypal_instance_of", ["nostos", "ordeal"], 0.95),
            ("the sea voyage as archetype", "maps_onto", ["departure", "return"], 0.85),
            ("the self-destructive narrator", "inverts", ["failed return", "no homecoming"], 0.75),
            ("creativity", "mirrors", ["incubation", "breakthrough", "integration"], 0.7),
        ],
    },

    # ── The bridge: nautical ↔ mythological ↔ psychological ──
    {
        "name": "the sea voyage as archetype",
        "type": "THEME",
        "description": (
            "The maritime journey operating simultaneously as literal seamanship "
            "and symbolic inner passage. Dead reckoning as intuition. The storm as "
            "initiation. The harbor as threshold. Bridges Chapman and Kazantzakis, "
            "Steinbeck's Sea of Cortez and the Odyssey, nautical technique and "
            "mythological descent."
        ),
        "relationships": [
            ("dead reckoning", "is_literal_form_of", ["navigation", "intuition"], 0.8),
            ("the unconscious", "symbolized_by", ["depth", "dissolution", "the deep"], 0.85),
            ("Odysseus", "enacted_by", ["wandering", "nostos"], 0.9),
            ("Log from the Sea of Cortez", "exemplifies", ["scientific voyage", "wonder"], 0.8),
            ("storm at sea", "functions_as", ["initiation", "ordeal"], 0.8),
        ],
    },
    {
        "name": "dead reckoning",
        "type": "CONCEPT",
        "description": (
            "The navigational practice of estimating position from last known point, "
            "heading, speed, and elapsed time — without external reference. "
            "In this corpus carries metaphorical weight: the mind navigating without "
            "landmarks, creative intuition as a form of dead reckoning, "
            "consciousness projecting forward from accumulated experience."
        ),
        "relationships": [
            ("the sea voyage as archetype", "grounds", ["navigation", "practice"], 0.8),
            ("intuition", "parallels", ["knowing without landmarks"], 0.75),
            ("consciousness", "analogous_to", ["self-positioning", "temporal projection"], 0.7),
        ],
    },

    # ── Psychological / consciousness hubs ───────────────────
    {
        "name": "consciousness",
        "type": "CONCEPT",
        "description": (
            "The central subject of the corpus's psychological and phenomenological "
            "strand. Treated technically in Van Dusen and Tillich, experientially in "
            "stream-of-consciousness fiction, mythologically as the light that "
            "descends into matter. Connects Jungian depth psychology, Buddhist "
            "mindfulness, postmodern narrative fragmentation, and nautical attentiveness."
        ),
        "relationships": [
            ("stream of consciousness", "expressed_through", ["interiority", "technique"], 0.9),
            ("the unconscious", "contains", ["shadow", "depth"], 0.85),
            ("the sea voyage as archetype", "maps_onto", ["depth", "surface", "navigation"], 0.7),
            ("creativity", "arises_from", ["liminal states", "flow"], 0.75),
            ("dead reckoning", "analogous_to", ["self-positioning"], 0.65),
        ],
    },
    {
        "name": "the unconscious",
        "type": "CONCEPT",
        "description": (
            "The substrate beneath conscious thought — Jung's collective unconscious, "
            "Freud's repressed, the mythic underworld as psychological map. "
            "In this corpus it surfaces in Aion, Art and the Creative Unconscious, "
            "dream interpretation texts, and symbolically as the sea, the underworld, "
            "the basement, the hold of the ship."
        ),
        "relationships": [
            ("consciousness", "underlies", ["depth", "substrate"], 0.9),
            ("the sea", "symbolizes", ["depth", "dissolution"], 0.85),
            ("creativity", "draws_from", ["primary process", "dream"], 0.8),
            ("the hero's journey", "descent_into", ["underworld", "initiation"], 0.8),
        ],
    },

    # ── Literary character hub: the self-destructive narrator ─
    {
        "name": "the self-destructive narrator",
        "type": "CHARACTER",
        "description": (
            "A recurring structural figure across the corpus: the first-person narrator "
            "whose self-awareness is total but self-corrective capacity is nil. "
            "Exley watching the Giants, the Underground Man raging at the wall, "
            "the Consul drinking toward oblivion, Oblomov refusing to rise. "
            "Each is a failed Odysseus — the wanderer who cannot return because "
            "he has confused his obsession with his identity."
        ),
        "relationships": [
            ("Frederick Exley", "instantiated_by", ["football", "fame", "failure"], 0.85),
            ("the Underground Man", "instantiated_by", ["resentment", "isolation"], 0.85),
            ("Geoffrey Firmin", "instantiated_by", ["mezcal", "the volcano", "paralysis"], 0.85),
            ("Oblomov", "instantiated_by", ["inertia", "refusal"], 0.8),
            ("Odysseus", "inverts", ["failed return", "no homecoming"], 0.75),
            ("obsession", "driven_by", ["compulsion", "substitution"], 0.85),
        ],
    },

    # ── Creativity hub ────────────────────────────────────────
    {
        "name": "creativity",
        "type": "CONCEPT",
        "description": (
            "The central practical concern of the corpus's art-theory strand — "
            "McKee on story structure, Art and Fear on the making of art, "
            "May on the courage to create, the Zen of Creativity on presence. "
            "Also the implicit subject of every postmodern novel in the corpus: "
            "what does it mean to make something when forms are exhausted?"
        ),
        "relationships": [
            ("the unconscious", "draws_from", ["primary process", "incubation"], 0.8),
            ("the hero's journey", "mirrors", ["departure", "return with gift"], 0.7),
            ("obsession", "related_to", ["monomania", "the cost of making"], 0.7),
            ("the trickster", "requires", ["rule-breaking", "play"], 0.65),
            ("flow state", "optimal_condition_for", ["absorption", "effortlessness"], 0.8),
        ],
    },
]


def _add_seed_entities(G: nx.DiGraph) -> None:
    """
    Add curated seed entities to the graph before extraction runs.
    Seeds that already exist (from a resumed build) are left unchanged.
    """
    added = 0
    for seed in SEED_ENTITIES:
        key = _normalize_name(seed["name"])

        if not G.has_node(key):
            G.add_node(key, **{
                "name":          seed["name"],
                "type":          seed["type"],
                "description":   seed["description"],
                "source_chunks": [],
                "work_titles":   ["__seed__"],
            })
            added += 1

        # Add seed relationships (skip if edge already exists)
        for tgt_name, rel_name, keywords, weight in seed.get("relationships", []):
            tgt_key = _normalize_name(tgt_name)
            if not G.has_node(tgt_key):
                G.add_node(tgt_key, name=tgt_name, type="CONCEPT",
                           description="", source_chunks=[], work_titles=["__seed__"])
            if not G.has_edge(key, tgt_key):
                G.add_edge(key, tgt_key,
                           name=rel_name,
                           keywords=keywords,
                           description="",
                           weight=weight,
                           source_chunk="__seed__",
                           work_title="__seed__")

    if added:
        print(f"  Seeded {added} canonical anchor entities into graph")


def build_graph(
    clean_text_dir: Path = CLEAN_TEXT_DIR,
    limit: Optional[int] = None,
    resume: bool = True,
) -> nx.DiGraph:
    """
    Build or resume building the Rich Knowledge Graph from the corpus.

    Args:
        clean_text_dir:  Path to folder containing pre-chunked .txt files
        limit:           Process at most this many files (None = all)
        resume:          If True, skip files already processed (default)

    Returns:
        A NetworkX DiGraph — the Rich Knowledge Graph.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Load or start fresh graph, then seed canonical anchor entities
    G = load_graph(GRAPH_PATH) if (resume and GRAPH_PATH.exists()) else nx.DiGraph()
    _add_seed_entities(G)
    done = _load_progress() if resume else set()

    files = sorted(clean_text_dir.glob("*.txt"))
    if limit:
        files = files[:limit]

    pending = [f for f in files if f.name not in done]
    total   = len(files)
    print(f"Corpus: {total} files | Already done: {len(done)} | To process: {len(pending)}")

    WORKERS = 8   # concurrent Claude calls
    lock    = threading.Lock()
    counter = [len(done)]  # mutable counter shared across threads

    def process_file(filepath: Path):
        """Worker: read → skip-check → Claude call. Returns (filepath, extraction|None)."""
        raw       = filepath.read_text(encoding="utf-8", errors="ignore")
        text      = strip_metadata_header(raw)
        file_info = parse_filename(filepath.name)

        if _should_skip(file_info):
            return filepath, None, "skipped (excluded work)"
        if looks_garbled(text):
            return filepath, None, "skipped (garbled/short)"

        extraction = extract_entities_and_relations(client, text, file_info)
        return filepath, extraction, file_info

    completed_count = [0]

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_file, fp): fp for fp in pending}

        for future in as_completed(futures):
            filepath, extraction, info = future.result()

            with lock:
                counter[0] += 1
                label = f"[{counter[0]}/{total}] {filepath.name}"

                if extraction is None:
                    print(f"{label} — {info}")
                else:
                    n_ents = len(extraction["entities"])
                    n_rels = len(extraction["relationships"])
                    print(f"{label} — {n_ents} entities, {n_rels} relations")
                    add_extraction_to_graph(
                        G,
                        extraction,
                        chunk_id=filepath.stem,
                        work_title=info["work_title"],
                    )

                done.add(filepath.name)
                completed_count[0] += 1

                if completed_count[0] % BATCH_SAVE == 0:
                    save_graph(G, GRAPH_PATH)
                    _save_progress(done)
                    print(f"  ✓ Checkpoint — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    save_graph(G, GRAPH_PATH)
    _save_progress(done)

    print(f"\n{'='*50}")
    print(f"Graph complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Files processed: {len(done)}/{total}")
    return G


# ─────────────────────────────────────────────────────────────
# GRAPH STATS (for inspection)
# ─────────────────────────────────────────────────────────────

def graph_stats(G: nx.DiGraph) -> dict:
    type_counts: dict = {}
    for _, d in G.nodes(data=True):
        t = d.get("type", "UNKNOWN")
        type_counts[t] = type_counts.get(t, 0) + 1

    rel_counts: dict = {}
    for _, _, d in G.edges(data=True):
        r = d.get("name", "unknown")
        rel_counts[r] = rel_counts.get(r, 0) + 1

    top_entities = sorted(
        [(n, G.degree(n), d.get("type", "?"), d.get("name", n))
         for n, d in G.nodes(data=True)],
        key=lambda x: x[1], reverse=True
    )[:20]

    return {
        "nodes":       G.number_of_nodes(),
        "edges":       G.number_of_edges(),
        "node_types":  dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True)),
        "rel_types":   dict(sorted(rel_counts.items(),  key=lambda x: x[1], reverse=True)[:20]),
        "top_entities": [{"key": e[0], "degree": e[1], "type": e[2], "name": e[3]}
                          for e in top_entities],
    }
