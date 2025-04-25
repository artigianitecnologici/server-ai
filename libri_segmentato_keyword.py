# -*- coding: utf-8 -*-
import pandas as pd
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from datetime import datetime
import os, shutil

print("📦 Inizializzazione...")

df = pd.read_csv("libri20full_genere.csv", delimiter=",", encoding="latin-1")
colonne_attese = ["ISBN", "Titolo", "Autore", "Descrizione"]
if not all(col in df.columns for col in colonne_attese):
    raise ValueError(f"❌ Il file deve contenere le colonne: {colonne_attese}, trovate: {df.columns.tolist()}")

docs = []
for _, row in df.iterrows():
    testo = f"""ISBN: {row['ISBN']}
Titolo: {row['Titolo']}
Autore: {row['Autore']}
Descrizione: {row['Descrizione']}"""    
    docs.append(Document(page_content=testo, metadata={
        "source": "libri30full.csv",
        "isbn": str(row['ISBN']).strip().lower(),
        "titolo": str(row['Titolo']).strip().lower(),
        "autore": str(row['Autore']).strip().lower(),
        "descrizione": str(row['Descrizione']).strip().lower(),
    }))

print(f"✅ Documenti caricati: {len(docs)}")

embeddings = OllamaEmbeddings(model="mistral")
db_path = "./dbmistral"
if os.path.exists(db_path):
    print("🧹 Pulizia della vector DB precedente...")
    shutil.rmtree(db_path)

vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=db_path)
vectordb.persist()
print("📚 Embedding completati e salvati in dbmistral")

qa = RetrievalQA.from_chain_type(
    llm=OllamaLLM(model="mistral"),
    retriever=vectordb.as_retriever(search_kwargs={"k": 10}),
    return_source_documents=True
)

# Ricerca guidata: "isbn xxx", "titolo yyy", ecc.
while True:
    ora = datetime.now().strftime("%H:%M:%S")
    query = input(f"\n📚 ({ora}) Fai una domanda sui libri (esci per uscire): ")
    if query.lower() in ["exit", "quit", "esci"]:
        break

    query_lower = query.lower().strip()
    result_docs = []

    tokens = query_lower.split()
    if tokens and tokens[0] in ["isbn", "titolo", "autore", "descrizione"]:
        campo = tokens[0]
        valore = " ".join(tokens[1:]).strip()
        for doc in docs:
            if valore in doc.metadata.get(campo, ""):
                result_docs.append(doc)
    else:
        result = qa.invoke({"query": query})
        print("\n📖 Risposta generata dal modello:")
        print(result["result"])
        print("🔎 Fonti:", [doc.metadata.get('source', 'N/A') for doc in result['source_documents']])
        print("\n📄 Dettagli documenti recuperati:")
        for doc in result['source_documents']:
            print(doc.page_content)
            print("-" * 80)
        continue

    if result_docs:
        print(f"🔍 Risultato trovato direttamente in {len(result_docs)} documento/i:")
        for doc in result_docs:
            print("\n📘 Documento corrispondente:")
            print(doc.page_content)
            print("-" * 80)
    else:
        print("❌ Nessun risultato trovato nei campi specifici.")

    ora_attuale = datetime.now().strftime("%H:%M:%S")
    print("⏰ Orario attuale risposta:", ora_attuale)
