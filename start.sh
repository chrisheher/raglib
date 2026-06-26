#!/bin/sh
# Symlink volume-mounted databases into the working directory on first boot.
# Set RAILWAY_VOLUME_MOUNT_PATH (default /data) in Railway environment.
DATA_DIR="${RAILWAY_VOLUME_MOUNT_PATH:-/data}"
if [ -d "$DATA_DIR" ]; then
  [ ! -e chroma_db ]    && ln -sf "$DATA_DIR/chroma_db"    chroma_db
  [ ! -e vgraphrag_db ] && ln -sf "$DATA_DIR/vgraphrag_db" vgraphrag_db
fi

# ChromaDB Rust bindings leave stale socket/lock files behind on crash.
# Clean them before starting so restarts don't hit EEXIST.
find chroma_db -name "*.sock" -o -name "*.lock" 2>/dev/null | xargs rm -f 2>/dev/null || true

exec python graphrag_app.py
