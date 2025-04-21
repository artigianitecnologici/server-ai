# Importa le librerie necessarie
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import json

# Descrizione: Questo script permette di eseguire il fine-tuning del modello GePpeTto utilizzando un dataset personalizzato.
# Il modello GePpeTto è pre-addestrato sulla lingua italiana, e l'obiettivo è adattarlo a rispondere a domande specifiche
# sulla storia di Roma, il Colosseo, i gladiatori e il Papa.

# 1. Configurazione del Modello Pre-Addestrato
model_name = "LorenzoDeMatteis/GePpeTto"

# Caricamento del tokenizer e del modello senza autenticazione
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. Caricamento del Dataset dal File JSON
# Il file 'colosseo.json' contiene coppie di domande e risposte specifiche per l'addestramento del modello.
with open('colosseo.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# Preparazione del dataset per il training
dataset = Dataset.from_dict({
    "text": [f'{item["prompt"]} {item["response"]}' for item in train_data]
})

# 3. Tokenizzazione del Dataset
# Questa funzione converte il testo in input e le etichette (labels) per l'addestramento
def tokenize(batch):
    tokens = tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

# Applica la tokenizzazione al dataset
dataset = dataset.map(tokenize, batched=True)

# 4. Impostazione dei Parametri di Addestramento
training_args = TrainingArguments(
    output_dir='./results_geppetto',
    num_train_epochs=10,  # Aumenta le epoche per migliorare l'apprendimento
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-5
)

# 5. Configurazione del Trainer per l'Addestramento
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# 6. Avvio dell'Addestramento del Modello
trainer.train()

# 7. Funzione per Generare Risposte con il Modello Addestrato
# Questa funzione permette di testare il modello generando risposte a partire da un prompt
def genera_testo(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            repetition_penalty=1.2
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 8. Esempi di Interazione con il Modello Addestrato
print(genera_testo("Ciao, come ti chiami?"))
print(genera_testo("Conosci il Colosseo di Roma?"))
print(genera_testo("Parlami della storia del Colosseo"))
print(genera_testo("Dimmi qualcosa sul Colosseo"))
print(genera_testo("Chi erano i gladiatori?"))
