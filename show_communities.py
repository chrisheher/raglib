import chromadb, json

c = chromadb.PersistentClient(path="chroma_db")
col = c.get_collection("literary_communities")
r = col.get(include=["metadatas", "documents"])

print(f"{len(r['documents'])} communities\n")
for i, (doc, meta) in enumerate(zip(r["documents"], r["metadatas"])):
    works = json.loads(meta.get("works", "[]"))
    print(f"{'='*60}")
    print(f"Community {i+1}  ({len(works)} works)")
    print(f"Works: {', '.join(works[:6])}" + (" …" if len(works) > 6 else ""))
    print()
    print(doc)
    print()
