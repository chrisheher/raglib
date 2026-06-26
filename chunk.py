import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

input_folder = "clean_text"
output_folder = "chunks"

os.makedirs(output_folder, exist_ok=True)

# Initialize the semantic chunker
embeddings = OpenAIEmbeddings()

chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # options: percentile, standard_deviation, interquartile
    breakpoint_threshold_amount=95           # higher = bigger chunks, lower = smaller chunks
)

def strip_metadata(text):
    """Remove the metadata header we added in the cleaning step"""
    if "### END METADATA ###" in text:
        return text.split("### END METADATA ###")[1].strip()
    return text

# Process every cleaned text file
for filename in os.listdir(input_folder):
    if not filename.endswith(".txt"):
        continue

    input_path = os.path.join(input_folder, filename)
    print(f"Chunking: {filename}")

    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()

    text = strip_metadata(raw)

    # Skip empty files
    if not text.strip():
        print(f"  Skipped (empty): {filename}")
        continue

    # Create semantic chunks
    chunks = chunker.create_documents([text])

    # Save each chunk as its own file
    base_name = filename.replace(".txt", "")
    for i, chunk in enumerate(chunks):
        chunk_filename = f"{base_name}_chunk_{i+1}.txt"
        chunk_path = os.path.join(output_folder, chunk_filename)

        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(f"### CHUNK METADATA ###\n")
            f.write(f"# source: {filename}\n")
            f.write(f"# chunk: {i+1} of {len(chunks)}\n")
            f.write(f"### END CHUNK METADATA ###\n\n")
            f.write(chunk.page_content)

    print(f"  → {len(chunks)} chunks created")

print("Done!")