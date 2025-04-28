#!/bin/bash

# Directory di lavoro
mkdir -p ~/ollama_modelli_italiani
cd ~/ollama_modelli_italiani

echo "### Scaricamento modelli italiani funzionanti ###"

# 1. Scarica Mistral 7B Instruct GGUF
echo "Scaricamento Mistral 7B Instruct GGUF..."
wget -O mistral-7b-instruct.Q4_K_M.gguf https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf

# Aggiungi modello ad Ollama
echo "### Importazione in Ollama ###"

ollama create mistral-ita -f ./mistral-7b-instruct.Q4_K_M.gguf

echo "### Installazione completata! ###"
echo "Modello disponibile: mistral-ita"

