import os
import re
import time
import json
import chromadb
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
claude = Anthropic()

# Best quality embedding model at maximum dimensions
EMBEDDING_MODEL = "text-embedding-3-large"
DIMENSIONS = 3072
BATCH_SIZE = 50

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(
    name="literary_documents",
    metadata={"hnsw:space": "cosine"}  # cosine similarity is best for literary text
)

input_folder = "clean_text"


def strip_metadata_header(text):
    if "### END METADATA ###" in text:
        return text.split("### END METADATA ###")[1].strip()
    return text


TAGGING_PROMPT = """\
You are a literary metadata tagger. Read the passage and return a JSON object with exactly these fields.
Choose only from the allowed values listed — do not invent new ones.

GENRES (pick exactly one):
  postmodern_fiction, consciousness_studies, mythology, creative_thinking,
  nautical, poetry, drama, prose

THEMES (pick all that genuinely apply, 1–4 max):
  identity, memory, dream_and_vision, mortality, journey_and_quest,
  language_and_narrative, nature_and_cosmos, transformation,
  madness_and_obsession, time

TONES (pick all that genuinely apply, may be empty):
  ironic, melancholic, sublime, dark, playful

NARRATIVE_STYLE (pick one or omit if unclear):
  first_person, second_person, third_person

MYTH_TRADITION (pick one only if the passage explicitly engages that tradition, otherwise omit):
  greek, norse, egyptian, celtic

NAUTICAL_CONTEXT (only if nautical content is present, otherwise omit):
  storm_at_sea, navigation, port_and_harbor, general_maritime

CONSCIOUSNESS_TECHNIQUE (only if clearly present, otherwise omit):
  stream_of_consciousness, contemplative

Return only valid JSON, no explanation. Example:
{"genre":"prose","themes":["identity","time"],"tone":["melancholic"],"narrative_style":"first_person"}

Passage:
"""


def extract_literary_metadata(text, filename):
    """Tag a passage using Claude Haiku for accurate concept extraction."""
    metadata = {"source": filename}

    # Basic stats (no LLM needed)
    metadata["word_count"] = str(len(text.split()))
    year_match = re.search(r'\b(1[5-9]\d{2}|20[0-2]\d)\b', text)
    if year_match:
        metadata["year"] = year_match.group()

    # Truncate to ~600 words to control token cost
    words = text.split()
    excerpt = " ".join(words[:600])

    for attempt in range(3):
        try:
            resp = claude.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{"role": "user", "content": TAGGING_PROMPT + excerpt}]
            )
            raw = resp.content[0].text.strip()
            # strip markdown code fences if present
            raw = re.sub(r'^```json\s*|```$', '', raw, flags=re.MULTILINE).strip()
            tags = json.loads(raw)
            break
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                # Fall back to empty tags rather than crashing
                tags = {}

    if tags.get("genre"):
        metadata["genre"] = tags["genre"]
    else:
        metadata["genre"] = "prose"

    themes = tags.get("themes", [])
    if isinstance(themes, list) and themes:
        metadata["themes"] = ", ".join(themes)

    tones = tags.get("tone", [])
    if isinstance(tones, list) and tones:
        metadata["tone"] = ", ".join(tones)

    def _scalar(val):
        """Ensure single-value fields are always strings, not lists."""
        if isinstance(val, list):
            return val[0] if val else ""
        return val or ""

    for field in ("narrative_style", "myth_tradition", "nautical_context", "consciousness_technique"):
        val = _scalar(tags.get(field))
        if val:
            metadata[field] = val

    return metadata


def enrich_text_for_embedding(text, metadata):
    """Prepend metadata context to improve embedding quality"""
    prefix = ""
    if "genre" in metadata:
        prefix += f"Genre: {metadata['genre']}. "
    if "themes" in metadata:
        prefix += f"Themes: {metadata['themes']}. "
    if "tone" in metadata:
        prefix += f"Tone: {metadata['tone']}. "
    if "myth_tradition" in metadata:
        prefix += f"Mythological tradition: {metadata['myth_tradition']}. "
    if "consciousness_technique" in metadata:
        prefix += f"Consciousness technique: {metadata['consciousness_technique']}. "
    if "nautical_context" in metadata:
        prefix += f"Nautical context: {metadata['nautical_context']}. "
    return f"{prefix}\n\n{text}".strip()


def get_embedding(text):
    """Get embedding with retry logic for rate limits"""
    for attempt in range(3):
        try:
            response = client.embeddings.create(
                input=text,
                model=EMBEDDING_MODEL,
                dimensions=DIMENSIONS
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < 2:
                print(f"  Retry {attempt + 1} after error: {e}")
                time.sleep(2 ** attempt)
            else:
                raise e


if __name__ == "__main__":
    # Load already embedded files to allow safe re-runs
    already_embedded = set(collection.get()["ids"])

    files = [f for f in os.listdir(input_folder) if f.endswith(".txt") and not f.startswith("__")]
    files_to_process = [f for f in files if f.replace(".txt", "") not in already_embedded]

    print(f"Total files:       {len(files)}")
    print(f"Already embedded:  {len(already_embedded)}")
    print(f"To process:        {len(files_to_process)}")
    print()

    success_count = 0
    error_count = 0
    skipped_count = 0
    total = len(files_to_process)
    start_time = time.time()

    def _bar(done, total, width=30):
        filled = int(width * done / max(total, 1))
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    def _print_progress(i, filename, status, detail=""):
        done = i + 1
        pct = done / max(total, 1)
        elapsed = time.time() - start_time
        rate = done / max(elapsed, 0.1)
        eta = int((total - done) / rate) if rate > 0 else 0
        eta_str = f"{eta//60}m{eta%60:02d}s" if eta >= 60 else f"{eta}s"
        bar = _bar(done, total)
        name = filename[:35].ljust(35)
        line = (f"\r{bar} {done}/{total} ({pct:.0%})  "
                f"{status} {name}  eta {eta_str}  err {error_count}")
        if detail:
            line += f"  [{detail}]"
        print(line, end="", flush=True)

    for i, filename in enumerate(files_to_process):
        doc_id = filename.replace(".txt", "")
        input_path = os.path.join(input_folder, filename)

        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        text = strip_metadata_header(raw)

        if not text.strip():
            skipped_count += 1
            _print_progress(i, filename, "skip")
            continue

        if len(text.split()) < 20:
            skipped_count += 1
            _print_progress(i, filename, "skip")
            continue

        try:
            metadata = extract_literary_metadata(text, filename)
            enriched_text = enrich_text_for_embedding(text, metadata)
            embedding = get_embedding(enriched_text)

            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata]
            )

            success_count += 1
            genre = metadata.get("genre", "?")
            _print_progress(i, filename, "ok  ", genre)

            if i > 0 and i % BATCH_SIZE == 0:
                time.sleep(1)

        except Exception as e:
            error_count += 1
            _print_progress(i, filename, "ERR ")
            with open("embed_errors.log", "a") as log:
                log.write(f"{filename}: {e}\n")

    print()  # newline after final bar update

    print(f"""
=============================
Done!
✓ Embedded:  {success_count}
⏭  Skipped:  {skipped_count}
✗ Errors:    {error_count}
Total in DB: {collection.count()}
=============================
""")