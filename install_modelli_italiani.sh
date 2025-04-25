#!/bin/bash

# Funzione per scaricare e installare un modello
install_model() {
    MODEL_NAME=$1
    MODEL_FILE=$2
    MODEL_URL=$3
    OLLAMA_DIR="$HOME/.ollama/models/$MODEL_NAME"

    echo "🔍 Verifica modello: $MODEL_NAME..."

    if [ -f "$OLLAMA_DIR/$MODEL_FILE" ]; then
        echo "✅ Modello $MODEL_NAME già installato."
    else
        echo "⬇️ Scarico $MODEL_NAME..."
        mkdir -p "$OLLAMA_DIR"
        wget -O "$OLLAMA_DIR/$MODEL_FILE" "$MODEL_URL"

        echo "📝 Creo Modelfile per $MODEL_NAME..."
        cat <<EOL > "$OLLAMA_DIR/Modelfile"
FROM ./$MODEL_FILE
TEMPLATE """{{ .Prompt }}"""
PARAMETER temperature 0.7
EOL

        echo "📦 Registro $MODEL_NAME su Ollama..."
        ollama create $MODEL_NAME -f "$OLLAMA_DIR/Modelfile"

        echo "🎉 Modello $MODEL_NAME installato con successo!"
    fi
}

# Verifica git-lfs
if ! command -v git-lfs &> /dev/null; then
    echo "🔧 Installo git-lfs..."
    sudo apt update && sudo apt install -y git-lfs
    git lfs install
fi

# Modelli da installare
install_model "llamantino" "LLaMAntino-2-7b-hf-ITA-Q4_K_M.gguf" \
"https://huggingface.co/tensorblock/LLaMAntino-2-7b-hf-ITA-GGUF/resolve/main/LLaMAntino-2-7b-hf-ITA-Q4_K_M.gguf"

install_model "nous-hermes" "nous-hermes-2-mistral-7b-dpo.Q4_K_M.gguf" \
"https://huggingface.co/TheBloke/Nous-Hermes-2-Mistral-7B-DPO-GGUF/resolve/main/nous-hermes-2-mistral-7b-dpo.Q4_K_M.gguf"

install_model "mistral-instruct" "mistral-7b-instruct-v0.2.Q4_K_M.gguf" \
"https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

echo "🚀 Tutti i modelli richiesti sono pronti! Puoi eseguirli con: ollama run <nome_modello>"
