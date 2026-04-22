#!/bin/sh
set -e

SOURCES_DIR="${DATA_DIR:-/app/data/sources}"

if [ -z "$(ls -A "$SOURCES_DIR" 2>/dev/null)" ]; then
    echo "[entrypoint] $SOURCES_DIR vacío; sembrando CV por defecto"
    cp /opt/seed/*.pdf "$SOURCES_DIR/"
fi

exec "$@"
