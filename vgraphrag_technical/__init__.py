"""
vgraphRAG Technical — Nautical + AI-Research Knowledge Graph Pipeline
=======================================================================
A second, independent VGraphRAG pipeline covering templates/pdfs/ (nautical
engineering manuals) and templates/pdfs/ai/ (RAG/GraphRAG research papers).

Kept separate from the vgraphrag/ package, which builds a literary knowledge
graph from clean_text/ using a CHARACTER/MYTHOLOGICAL_FIGURE/LITERARY_WORK
ontology unsuited to technical content — see vgraphrag/graph_builder.py's
SKIP_WORKS for why mixing the two degrades PPR and Leiden community
detection for both.

Modules:
  graph_builder   PDFs (templates/pdfs/, templates/pdfs/ai/) → Rich
                  Knowledge Graph (RKG) via LLM extraction with a
                  CONCEPT/METHOD/SYSTEM/PAPER_OR_MANUAL ontology
  index_builder   RKG → Node / Relationship / Community indexes in ChromaDB
  retriever       Modular operators: Entity.Link, Entity.PPR,
                  Relationship.Aggregator, Chunk.Aggregator,
                  Community.VDB, Node.VDB, Relationship.Onehop
  engine          Orchestration: route → retrieve → generate

Generic graph-algorithm code (PPR, Leiden, JSON repair, node merging,
BGE-M3 embedding) is imported directly from vgraphrag/ rather than
duplicated — only the domain-specific pieces (PDF ingestion, prompts,
corpus/DB paths) are reimplemented here.
"""
