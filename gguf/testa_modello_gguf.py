import ollama

NOME_MODELLO = "modello-mio"

def chat_con_modello():
    print(f"Test del modello '{NOME_MODELLO}' con Ollama.")
    while True:
        domanda = input("Tu: ")
        if domanda.lower() in ["exit", "quit"]:
            break
        risposta = ollama.chat(model=NOME_MODELLO, messages=[{"role": "user", "content": domanda}])
        print("Modello:", risposta['message']['content'])

if __name__ == "__main__":
    chat_con_modello()
