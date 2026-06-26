"""
rebuild_indexes.py
==================
Run this after embed.py finishes to rebuild both index layers from the
freshly-tagged corpus.

  Stage A — hybrid_pipeline TKG + community index (ChromaDB metadata → graph)
  Stage B — vgraphrag RKG (Claude Sonnet entity/relation extraction from text)

Usage:
  python rebuild_indexes.py              # rebuild both
  python rebuild_indexes.py --tkg-only   # hybrid indexes only (fast, ~1 min)
  python rebuild_indexes.py --rkg-only   # vgraphrag RKG only  (slow, ~60 min)
  python rebuild_indexes.py --rkg-resume # resume interrupted RKG build
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def rebuild_tkg():
    print("\n" + "=" * 60)
    print("STAGE A — Hybrid TKG + Community Index")
    print("=" * 60)
    from hybrid_pipeline import build_indexes
    import chromadb

    # Wipe stale community collection so it rebuilds cleanly
    try:
        c = chromadb.PersistentClient(path="chroma_db")
        c.delete_collection("literary_communities")
        print("  Cleared stale community index.")
    except Exception:
        pass

    # Also remove stale TKG pickle
    tkg_path = Path("chroma_db/tkg.gpickle")
    if tkg_path.exists():
        tkg_path.unlink()
        print("  Cleared stale TKG pickle.")

    t0 = time.time()
    G = build_indexes(force_rebuild=True)
    elapsed = time.time() - t0

    import pickle
    with open(str(tkg_path), "rb") as f:
        G = pickle.load(f)

    print(f"\n✓ TKG built in {elapsed:.0f}s")
    print(f"  Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}")

    # Print community count
    try:
        c = chromadb.PersistentClient(path="chroma_db")
        comm_col = c.get_collection("literary_communities")
        print(f"  Communities indexed: {comm_col.count()}")
    except Exception:
        pass


def rebuild_rkg(resume=True):
    print("\n" + "=" * 60)
    print("STAGE B — VGraphRAG Rich Knowledge Graph (RKG)")
    print(f"  Mode: {'resume' if resume else 'full rebuild'}")
    print("=" * 60)

    from vgraphrag.graph_builder import (
        build_graph, graph_stats, GRAPH_PATH, PROGRESS_PATH
    )

    if not resume:
        # Wipe existing graph and progress to force full rebuild
        if GRAPH_PATH.exists():
            GRAPH_PATH.unlink()
            print("  Cleared existing RKG.")
        if PROGRESS_PATH.exists():
            PROGRESS_PATH.unlink()
            print("  Cleared extraction progress.")

    from vgraphrag.index_builder import build_all_indexes

    t0 = time.time()
    G = build_graph(resume=resume)
    elapsed_graph = time.time() - t0
    stats = graph_stats(G)

    print(f"\n✓ RKG built in {elapsed_graph:.0f}s")
    print(f"  Nodes: {stats['nodes']}  |  Edges: {stats['edges']}")
    print("\n  Node types:")
    for t, count in list(stats["node_types"].items())[:8]:
        print(f"    {t:<25} {count}")

    print("\nBuilding vgraphrag indexes (node, relationship, community)...")
    t1 = time.time()
    build_all_indexes(graph_path=GRAPH_PATH, force=True)
    print(f"✓ Indexes built in {time.time() - t1:.0f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild graph indexes after corpus changes."
    )
    parser.add_argument("--tkg-only",  action="store_true",
                        help="Rebuild hybrid TKG and community index only")
    parser.add_argument("--rkg-only",  action="store_true",
                        help="Rebuild vgraphrag RKG only")
    parser.add_argument("--rkg-resume", action="store_true",
                        help="Resume an interrupted RKG build (default when rebuilding RKG)")
    args = parser.parse_args()

    if args.tkg_only:
        rebuild_tkg()
    elif args.rkg_only:
        rebuild_rkg(resume=not args.rkg_resume is False)
    else:
        rebuild_tkg()
        rebuild_rkg(resume=True)

    print("\n" + "=" * 60)
    print("All indexes ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
