import requests
import time
import json

OLLAMA_URL = "http://localhost:11434/api/chat"

def chat_with_model(model_name, messages):
    response = requests.post(OLLAMA_URL, json={
        "model": model_name,
        "messages": messages,
        "stream": False
    })
    response.raise_for_status()
    return response.json()['message']['content'].strip()

def scegli_modello(prompt):
    print(prompt)
    lista_modelli = requests.get("http://localhost:11434/api/tags").json().get("models", [])
    for i, modello in enumerate(lista_modelli):
        print(f"{i+1}) {modello['name']}")
    scelta_raw = input("Seleziona il numero del modello: ")
    print(f"Hai digitato: {repr(scelta_raw)}")  # Mostra i caratteri reali
    scelta = int(scelta_raw) - 1
    return lista_modelli[scelta]['name']


def main():
    print("=== CONVERSAZIONE TRA DUE LLM ===")

    modello1 = scegli_modello("Scegli il PRIMO modello:")
    modello2 = scegli_modello("Scegli il SECONDO modello:")

    n_turni = int(input("Quanti turni vuoi farli parlare? (es. 6): "))

    # Inizializza contesto
    system_msg = "Parla con un altro assistente AI. Il dialogo deve essere coerente e interessante."
    user_msg = "Ciao! Iniziamo a parlare di intelligenza artificiale?"

    storia = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    transcript = [
        {"turn": 0, "model": "system", "message": system_msg},
        {"turn": 0, "model": "user", "message": user_msg}
    ]

    dialogo_testuale = [
        f"[system]: {system_msg}",
        f"[user]: {user_msg}"
    ]

    for i in range(n_turni):
        modello_corrente = modello1 if i % 2 == 0 else modello2
        print(f"\n🧠 Turno {i+1} - {modello_corrente}")
        try:
            inizio = time.time()
            risposta = chat_with_model(modello_corrente, storia)
            durata = time.time() - inizio
            print(f"⏱️ Risposta in {durata:.2f}s:\n{risposta}")
            storia.append({"role": "assistant", "content": risposta})

            transcript.append({
                "turn": i + 1,
                "model": modello_corrente,
                "time": round(durata, 2),
                "message": risposta
            })

            dialogo_testuale.append(f"[{modello_corrente}]: {risposta}")

        except Exception as e:
            print(f"Errore nel turno {i+1} ({modello_corrente}): {e}")
            break

    # Salva in JSON
    with open("dialogo_llm.json", "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=4, ensure_ascii=False)

    # Salva in TXT leggibile
    with open("dialogo_llm.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(dialogo_testuale))

    print("\n✅ Conversazione salvata in:")
    print("📁 dialogo_llm.json (dettagliato)")
    print("📁 dialogo_llm.txt (formato leggibile)")

if __name__ == "__main__":
    main()
