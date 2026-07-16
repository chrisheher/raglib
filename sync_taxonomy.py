"""
Applies taxonomy_leaf metadata (written locally by the one-off
build_taxonomy.py classification pass) onto an existing chroma_db's
chunks by id. Chunk ids are deterministic (source filename), so this
patches production's older/smaller chroma_db without needing a full
DB snapshot re-download — same rationale as start.sh's unconditional
document_connections.json refresh.

Usage: python sync_taxonomy.py <path-to-taxonomy_tags.json>
"""
import sys
import json
import chromadb

CHROMA_PATH     = "chroma_db"
COLLECTION_NAME = "literary_documents"
BATCH_SIZE      = 500


def main(tags_path):
    tags = json.loads(open(tags_path).read())

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    existing = collection.get(ids=list(tags.keys()), include=["metadatas"])
    to_update_ids = []
    to_update_metas = []
    for id_, meta in zip(existing["ids"], existing["metadatas"]):
        leaf = tags.get(id_)
        if leaf and meta.get("taxonomy_leaf") != leaf:
            to_update_ids.append(id_)
            to_update_metas.append({**meta, "taxonomy_leaf": leaf})

    for i in range(0, len(to_update_ids), BATCH_SIZE):
        collection.update(
            ids=to_update_ids[i:i + BATCH_SIZE],
            metadatas=to_update_metas[i:i + BATCH_SIZE],
        )

    print(f"taxonomy sync: {len(to_update_ids)} chunk(s) updated "
          f"({len(existing['ids'])} matched of {len(tags)} known tags).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python sync_taxonomy.py <taxonomy_tags.json>")
        sys.exit(1)
    main(sys.argv[1])
