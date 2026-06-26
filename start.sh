#!/bin/sh
# Symlink volume-mounted databases into the working directory on first boot.
# Set RAILWAY_VOLUME_MOUNT_PATH (default /data) in Railway environment.
DATA_DIR="${RAILWAY_VOLUME_MOUNT_PATH:-/data}"
if [ -d "$DATA_DIR" ]; then
  [ ! -e chroma_db ]    && ln -sf "$DATA_DIR/chroma_db"    chroma_db
  [ ! -e vgraphrag_db ] && ln -sf "$DATA_DIR/vgraphrag_db" vgraphrag_db
fi
exec python graphrag_app.py
