import os
os.environ["USE_FLASH_ATTENTION"] = "0"
os.environ["USE_XFORMERS"] = "0"


import fitz
import json
from unsloth import FastLanguageModel
from datasets import load_dataset



def estrai_testo_da_cartella_pdf(pdf_folder="pdf", output_file="vault.txt"):
    print(f"Estrazione del testo da tutti i PDF nella cartella '{pdf_folder}'...")
    if not os.path.exists(pdf_folder):
        print(f"Cartella '{pdf_folder}' non trovata.")
        return None

    with open(output_file, "w", encoding="utf-8") as f_out:
        for nome_file in os.listdir(pdf_folder):
            if nome_file.endswith(".pdf"):
                percorso = os.path.join(pdf_folder, nome_file)
                print(f"Leggo: {nome_file}")
                doc = fitz.open(percorso)
                for pagina in doc:
                    f_out.write(pagina.get_text())
                    f_out.write("\n---\n")
    print(f"Testo combinato salvato in {output_file}")
    return output_file

def crea_dataset_da_vault(vault_file, output_dataset="dataset.jsonl"):
    print("Creazione del dataset JSONL...")
    with open(vault_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(output_dataset, "w", encoding="utf-8") as out_file:
        for line in lines:
            if line.strip():
                item = {
                    "prompt": f"Domanda: {line.strip()}\nRisposta:",
                    "response": line.strip()
                }
                out_file.write(json.dumps(item) + "\n")
    print(f"Dataset salvato in {output_dataset}")
    return output_dataset

def fine_tuning(dataset_file, modello_output="modello_finetuned.gguf"):
    print("Avvio del fine-tuning con Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained("unsloth/mistral-7b", max_seq_length=2048)
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    model = FastLanguageModel.get_peft_model(model)

    FastLanguageModel.finetune(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir="modello_finetuned_dir",
        epochs=1,
        batch_size=1,
        lr=2e-5
    )

    print(f"Salvataggio del modello in formato GGUF: {modello_output}")
    model.save_pretrained_gguf(modello_output)
    print("Fine-tuning completato!")

if __name__ == "__main__":
    pdf_folder = "pdf"
    vault_file = estrai_testo_da_cartella_pdf(pdf_folder)
    
    if vault_file is None or not os.path.exists(vault_file):
        print("Errore nella lettura dei PDF.")
        exit(1)

    dataset_file = crea_dataset_da_vault(vault_file)
    fine_tuning(dataset_file)
