"""
vgraphRAG — Literary Knowledge Graph Retrieval Pipeline
========================================================
Implements the VGraphRAG operator combination from:
  "In-depth Analysis of Graph-based RAG in a Unified Framework"
  Zhou et al., VLDB 2025 (arXiv:2503.04338)

Modules:
  graph_builder   Corpus → Rich Knowledge Graph (RKG) via LLM extraction
  index_builder   RKG → Node / Relationship / Community indexes in ChromaDB
  retriever       Modular operators: Entity.Link, Entity.PPR,
                  Relationship.Aggregator, Chunk.Aggregator,
                  Community.VDB, Node.VDB, Relationship.Onehop
  query_router    Claude-based specific vs abstract query classification
  engine          Orchestration: route → retrieve → generate
"""
