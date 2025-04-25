#!/bin/bash

# Nome del modello
MODEL_NAME="llamantino-q5"
MODEL_FILE="LLaMAntino-2-7b-hf-ITA-Q5_K_M.gguf"
MODEL_URL="https://huggingface.co/tensorblock/LLaMAntino-2-7b-hf-ITA-GGUF/resolve/main/${MODEL_FILE}"
OLLAMA_DIR="$HOME/.ollama/models/$MODEL_NAME"

echo "🔧 Inizio installazione di $MODEL_NAME..."

# Step 1: Installa git-lfs se necessario
if ! command -v git-lfs &> /dev/null; then
    echo "🔍 git-lfs non trovato. Lo installo..."
    sudo apt update && sudo apt install -y git-lfs
    git lfs install
else
    echo "✅ git-lfs già installato."
fi

# Step 2: Crea cartella per il modello
mkdir -p "$OLLAMA_DIR"

# Step 3: Scarica il file GGUF
echo "⬇️ Scarico il modello Q5_K_M da Hugging Face..."
wget -O "$OLLAMA_DIR/$MODEL_FILE" "$MODEL_URL"

# Step 4: Crea il Modelfile per Ollama
echo "📝 Creo il Modelfile..."
cat <<EOL > "$OLLAMA_DIR/Modelfile"
FROM ./$MODEL_FILE
TEMPLATE """{{ .Prompt }}"""
PARAMETER temperature 0.7
EOL

# Step 5: Registra il modello su Ollama
echo "📦 Registro il modello su Ollama..."
ollama create $MODEL_NAME -f "$OLLAMA_DIR/Modelfile"

# Step 6: Avvia il modello
echo "🚀 Avvio $MODEL_NAME su Ollama..."
ollama run $MODEL_NAME

echo "🎉 Modello $MODEL_NAME installato e avviato con successo!"
