"""
build_vgraphrag.py
==================
CLI script to build the VGraphRAG pipeline from scratch.

Usage:
  # Full build (all stages):
  python build_vgraphrag.py

  # Limit extraction to first N files (for testing):
  python build_vgraphrag.py --limit 50

  # Resume interrupted graph build, then rebuild indexes:
  python build_vgraphrag.py --skip-graph --rebuild-indexes

  # Force-rebuild everything:
  python build_vgraphrag.py --force

  # Show graph stats for an existing graph:
  python build_vgraphrag.py --stats

  # Test a query against a built index:
  python build_vgraphrag.py --query "How does Bloom parallel Odysseus?"
  python build_vgraphrag.py --query "What themes connect Gaddis and Wallace?" --mode abstract

Stages:
  1. Graph Building   → vgraphrag_db/rkg.json
  2. Node Index       → vgraphrag_db/chroma/vgraphrag_nodes
  3. Relationship Index → vgraphrag_db/chroma/vgraphrag_relationships
  4. Community Index  → vgraphrag_db/chroma/vgraphrag_communities
                      + vgraphrag_db/communities.json
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure we can import vgraphrag as a package
sys.path.insert(0, str(Path(__file__).parent))

from vgraphrag.graph_builder import (
    build_graph, load_graph, graph_stats,
    CLEAN_TEXT_DIR, GRAPH_PATH,
)
from vgraphrag.index_builder import build_all_indexes
from vgraphrag.engine import VGraphRAGEngine


def cmd_build(args):
    """Stage 1: Build or resume the RKG."""
    if args.skip_graph:
        print("Skipping graph build (--skip-graph).")
    else:
        print("=" * 60)
        print("STAGE 1 — Graph Building")
        print(f"  Source: {CLEAN_TEXT_DIR}")
        print(f"  Output: {GRAPH_PATH}")
        if args.limit:
            print(f"  Limit:  {args.limit} files")
        print("=" * 60)

        G = build_graph(
            clean_text_dir=CLEAN_TEXT_DIR,
            limit=args.limit,
            resume=not args.force,
        )
        print(f"\nGraph ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n" + "=" * 60)
    print("STAGES 2–4 — Index Building")
    print("=" * 60)
    build_all_indexes(graph_path=GRAPH_PATH, force=args.force or args.rebuild_indexes)


def cmd_stats(args):
    """Print stats about the existing graph."""
    if not GRAPH_PATH.exists():
        print(f"No graph found at {GRAPH_PATH}. Run build first.")
        sys.exit(1)

    G = load_graph(GRAPH_PATH)
    stats = graph_stats(G)

    print(f"\nGraph: {stats['nodes']} nodes, {stats['edges']} edges")
    print("\nNode types:")
    for t, count in stats["node_types"].items():
        print(f"  {t:<30} {count}")

    print("\nTop relationship types:")
    for r, count in list(stats["rel_types"].items())[:15]:
        print(f"  {r:<35} {count}")

    print("\nTop 20 entities by degree:")
    for e in stats["top_entities"]:
        print(f"  [{e['type']:<25}] deg={e['degree']:<4} {e['name']}")


def cmd_query(args):
    """Run a single query through the engine and print the result."""
    if not GRAPH_PATH.exists():
        print("No graph found. Run build first.")
        sys.exit(1)

    engine = VGraphRAGEngine()
    result = engine.query(
        query=args.query,
        mode=args.mode,
        n_chunks=args.n_chunks,
        use_llm_router=True,
    )

    print("\n" + "=" * 60)
    print(f"Query: {args.query}")
    print(f"Mode:  {result['mode'].upper()}")
    print("=" * 60)

    gi = result["graph_info"]
    print(f"\nGraph info:")
    print(f"  Chunks returned:      {gi['chunks_returned']}")
    print(f"  Communities returned: {gi['communities_returned']}")
    print(f"  Relationships:        {gi['rels_returned']}")
    if gi.get("seed_entities"):
        print(f"  Seed entities:        {gi['seed_entities'][:6]}")
    if gi.get("top_ppr"):
        print(f"  Top PPR nodes:        {gi['top_ppr'][:5]}")

    if result.get("retrieval", {}).get("communities"):
        print("\nCommunity summaries retrieved:")
        for i, c in enumerate(result["retrieval"]["communities"][:2]):
            print(f"\n  [{i+1}] Score={c['score']:.3f} | "
                  f"Works: {', '.join(c.get('works', [])[:4])}")
            print(f"  {c['summary'][:300]}...")

    print("\nAnswer:")
    print("-" * 60)
    print(result["answer"])


def main():
    parser = argparse.ArgumentParser(
        description="Build and query the VGraphRAG literary knowledge graph pipeline."
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── build ─────────────────────────────────────────────
    build_p = subparsers.add_parser("build", help="Build the RKG and all indexes")
    build_p.add_argument("--limit",   type=int,  default=None,
                         help="Process at most N files (for testing)")
    build_p.add_argument("--force",   action="store_true",
                         help="Force rebuild everything from scratch")
    build_p.add_argument("--skip-graph", action="store_true",
                         help="Skip graph extraction, rebuild indexes only")
    build_p.add_argument("--rebuild-indexes", action="store_true",
                         help="Force-rebuild indexes without re-extracting the graph")

    # ── stats ─────────────────────────────────────────────
    stats_p = subparsers.add_parser("stats", help="Show graph statistics")

    # ── query ─────────────────────────────────────────────
    query_p = subparsers.add_parser("query", help="Run a single query")
    query_p.add_argument("query", type=str, help="The question to ask")
    query_p.add_argument("--mode", choices=["specific", "abstract"], default=None,
                         help="Force a pipeline mode (default: auto-route)")
    query_p.add_argument("--n-chunks", type=int, default=6,
                         help="Number of passage chunks to retrieve (default: 6)")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "query":
        cmd_query(args)
    else:
        # Default: run full build
        class DefaultArgs:
            limit = None
            force = False
            skip_graph = False
            rebuild_indexes = False
        print("No command specified — running full build.")
        cmd_build(DefaultArgs())


if __name__ == "__main__":
    main()
