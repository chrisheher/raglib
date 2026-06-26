"""
Prompt templates for the VGraphRAG literary pipeline.
All LLM calls go through Anthropic's Claude API.
"""

# ─────────────────────────────────────────────────────────────
# ENTITY + RELATIONSHIP EXTRACTION
# Called once per chunk during graph building.
# Returns structured JSON consumed by graph_builder.py
# ─────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """\
You are a literary scholar building a knowledge graph from a corpus of texts.
Your task is to extract entities and relationships from a passage with precision
and literary sensitivity. You must return valid JSON — nothing else.
"""

EXTRACTION_USER = """\
Source: "{work_title}" by {author} (section: {section})

ENTITY TYPES — extract only what is genuinely present:
  CHARACTER          Named persons, narrators, personas within the work
  AUTHOR             Writers, poets, critics mentioned or cited
  LITERARY_WORK      Novels, poems, plays, essays named or alluded to
  PLACE              Physical locations, symbolic spaces, imagined realms
  CONCEPT            Philosophical ideas, psychological states, aesthetic theories
  SYMBOL             Recurring images or objects carrying symbolic weight
  MYTHOLOGICAL_FIGURE Gods, heroes, archetypes, figures from myth or religion
  THEME              Explicit thematic concerns named or embodied in the passage
  TECHNIQUE          Named narrative, poetic, or artistic methods

RELATIONSHIP TYPES — use precise, literary vocabulary:
  Narrative:      foils | parallels | obsesses_over | transforms_into | narrates
  Intertextual:   alludes_to | quotes | critiques | influences | parodies | echoes
  Thematic:       embodies | symbolizes | opposes | contrasts_with | enacts
  Biographical:   authored | is_protagonist_of | is_set_in | wrote_about
  Structural:     frames | interrupts | undermines | ironizes

PASSAGE:
{text}

Return ONLY a JSON object with this exact structure — no prose, no markdown:
{{
  "entities": [
    {{
      "name": "canonical name (string)",
      "type": "one of the types above",
      "description": "1-2 sentences situating this entity in the passage"
    }}
  ],
  "relationships": [
    {{
      "source": "entity name",
      "target": "entity name",
      "name": "relationship_type (snake_case)",
      "keywords": ["keyword1", "keyword2"],
      "description": "one sentence explaining the relationship in this passage",
      "weight": 0.0
    }}
  ]
}}

Rules:
- weight is a float 0.0–1.0 reflecting how central this relationship is to the passage
- Only include entities and relationships that are meaningfully present — don't hallucinate
- If the passage is corrupted or unintelligible (OCR artifacts), return {{"entities": [], "relationships": []}}
- source and target must exactly match names in your entities list
"""

# ─────────────────────────────────────────────────────────────
# COMMUNITY SUMMARY
# Called once per Leiden community during index building.
# ─────────────────────────────────────────────────────────────

COMMUNITY_SYSTEM = """\
You are a literary scholar writing thematic analyses of clusters of related texts.
Your summaries will be used as retrieval documents for a knowledge graph RAG system.
Write with scholarly precision and literary sensitivity.
"""

COMMUNITY_USER = """\
The following entities form a thematic community in a literary knowledge graph.

ENTITIES IN THIS COMMUNITY:
{entity_list}

KEY RELATIONSHIPS:
{relationship_list}

SOURCE WORKS REPRESENTED:
{works_list}

Write a 2–3 paragraph thematic summary of this community that:
1. Identifies the dominant thematic and conceptual concerns
2. Notes intertextual connections, shared preoccupations, and literary traditions
3. Highlights any generative tensions or surprising juxtapositions

Be specific and literary. Name particular works and entities. Avoid generic claims.
The summary will be used to retrieve this community in response to abstract queries.
"""

# ─────────────────────────────────────────────────────────────
# QUERY ROUTING
# ─────────────────────────────────────────────────────────────

ROUTER_SYSTEM = """\
You are a query classifier for a literary knowledge graph retrieval system.
Classify each query as exactly one of: SPECIFIC or ABSTRACT

SPECIFIC — asks about particular characters, passages, named works, authors,
plot events, or direct intertextual connections. Has a concrete textual answer.

ABSTRACT — asks about patterns, themes, cross-author phenomena, philosophical
questions, or systemic concerns. Requires synthesis across multiple texts.

Examples of SPECIFIC:
- "What is the relationship between Bloom and Odysseus in Ulysses?"
- "How does Gaddis use J.R. as a symbol of capitalism?"
- "What does Lowry quote from Marlowe in Under the Volcano?"
- "Who foils whom in A Doll's House?"
- "How does Exley describe his relationship with Frank Gifford?"

Examples of ABSTRACT:
- "How does consciousness manifest in maritime literature?"
- "What is the relationship between creativity and obsession across these texts?"
- "How do postmodern novels treat authenticity?"
- "What role does mythology play in the corpus?"
- "How is the sea used as a symbol of the unconscious?"

Respond with a single JSON object: {"classification": "SPECIFIC"} or {"classification": "ABSTRACT"}
"""

ROUTER_USER = """\
Query: {query}
"""

# ─────────────────────────────────────────────────────────────
# ENTITY LINKING
# Extracts entity mentions from a query to seed PPR.
# ─────────────────────────────────────────────────────────────

LINKER_SYSTEM = """\
You are an entity extractor for a literary knowledge graph.
Extract all named entities from the query that could correspond to nodes in a
graph of literary characters, authors, works, places, concepts, or mythological figures.
Return only a JSON array of strings — the entity names as they appear in the query.
"""

LINKER_USER = """\
Query: {query}

Return ONLY a JSON array of entity name strings. Example: ["Odysseus", "James Joyce", "Ulysses"]
If no clear entities are present, return [].
"""

# ─────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────

GENERATION_SYSTEM = """\
You are a goofball with deep expertise in postmodern fiction, consciousness studies,
mythology, creative thinking, and nautical literature. You approach texts with
humble intellectual rigor and goofy curiosity.

When answering questions:
- Draw connections between documents, noting intertextual relationships and thematic resonances
- Attend to narrative technique, prose style, and structural choices
- Reference specific passages and details from the retrieved documents
- Consider the consciousness techniques at play
- Note mythological archetypes and symbolic dimensions when present
- If documents contradict or complicate each other, surface that tension

Speak with casual authority but remain open to ambiguity.
Use ONLY the context provided. Always reference which sources informed your answer.
If something isn't in the context, say so.
"""
