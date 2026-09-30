#!/usr/bin/env bash
# Script para executar o Algoritmo Genético do TSP
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

if [ ! -d "venv" ]; then
    echo "Criando ambiente virtual..."
    python3 -m venv venv
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r requirements.txt
fi

source venv/bin/activate
python3 main.py "$@"
