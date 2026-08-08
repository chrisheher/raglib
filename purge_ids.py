"""
Deletes known-bad chunk/community ids from an existing chroma_db by id,
same rationale as sync_taxonomy.py: production's ChromaDB volume is a
persistent snapshot that doesn't pick up local corpus edits on its own, so
a manifest of ids-to-remove gets shipped and applied on every boot instead
of requiring a full DB snapshot re-download or volume wipe. Idempotent —
already-deleted ids are silently skipped.

Usage: python purge_ids.py <path-to-purge_manifest.json>
Manifest format: {"<collection_name>": ["<id>", ...], ...}
"""
import sys
import json
import chromadb

CHROMA_PATH = "chroma_db"


def main(manifest_path):
    manifest = json.loads(open(manifest_path).read())

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    for collection_name, ids in manifest.items():
        try:
            collection = client.get_collection(collection_name)
        except Exception as e:
            print(f"purge_ids: collection '{collection_name}' not found ({e}) — skipping")
            continue

        existing = collection.get(ids=ids, include=[])["ids"]
        if existing:
            collection.delete(ids=existing)
        print(f"purge_ids: {collection_name} — {len(existing)} of {len(ids)} id(s) deleted "
              f"({len(ids) - len(existing)} already gone).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python purge_ids.py <purge_manifest.json>")
        sys.exit(1)
    main(sys.argv[1])
