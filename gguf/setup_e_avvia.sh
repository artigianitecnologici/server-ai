#!/bin/bash

echo "== Creazione ambiente virtuale =="
python3 -m venv llm-env
source llm-env/bin/activate

echo "== Installazione dipendenze =="
pip install -r requirements.txt

echo "== Avvio creazione modello GGUF =="
python crea_modello_gguf.py

echo "== Modello GGUF creato! =="
echo "Ora copia 'modello_finetuned.gguf' nella cartella di Ollama."
