#!/bin/sh
DATA_DIR="${RAILWAY_VOLUME_MOUNT_PATH:-/data}"

# On first boot, download the database snapshot from GitHub releases.
DB_RELEASE_URL="${DB_RELEASE_URL:-https://github.com/chrisheher/raglib/releases/download/v1.0-dbs/raglib-dbs.tar.gz}"

if [ -d "$DATA_DIR" ] && [ ! -d "$DATA_DIR/chroma_db" ]; then
  echo "No databases found in $DATA_DIR — downloading snapshot..."
  curl -fsSL -o "$DATA_DIR/raglib-dbs.tar.gz" "$DB_RELEASE_URL"
  tar -xzf "$DATA_DIR/raglib-dbs.tar.gz" -C "$DATA_DIR"
  rm -f "$DATA_DIR/raglib-dbs.tar.gz"
  echo "Databases ready."
fi

# Symlink volume databases into the working directory.
if [ -d "$DATA_DIR" ]; then
  [ ! -e chroma_db ]    && ln -sf "$DATA_DIR/chroma_db"    chroma_db
  [ ! -e vgraphrag_db ] && ln -sf "$DATA_DIR/vgraphrag_db" vgraphrag_db
fi

# Refresh the document-connections feed on every boot — small (<1MB), so an
# unconditional re-fetch keeps it current without needing a volume wipe or
# a new full DB snapshot each time it's regenerated.
CONNECTIONS_URL="${CONNECTIONS_URL:-https://github.com/chrisheher/raglib/releases/download/v1.0-dbs/document_connections.json}"
if [ -d "$DATA_DIR/vgraphrag_db" ]; then
  curl -fsSL -o "$DATA_DIR/vgraphrag_db/document_connections.json" "$CONNECTIONS_URL" \
    && echo "document_connections.json refreshed." \
    || echo "document_connections.json refresh failed — keeping existing copy."
fi

# Clean stale ChromaDB Rust socket/lock files left behind on crash.
find chroma_db -name "*.sock" -o -name "*.lock" 2>/dev/null | xargs rm -f 2>/dev/null || true

exec python graphrag_app.py
