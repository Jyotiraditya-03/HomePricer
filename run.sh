#!/bin/bash
cd "$(dirname "$0")"
echo "[1/2] Starting HomePricer API..."
uvicorn backend.main:app --reload --port 8000 &
sleep 2
echo "[2/2] Opening browser..."
xdg-open http://localhost:8000 2>/dev/null || open http://localhost:8000
