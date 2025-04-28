#!/usr/bin/python3
import os
import json
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from time import time
import subprocess
import sys
from ollama import Client
import requests

# Load config from JSON
DEFAULT_CONFIG_PATH = os.path.expandvars("$HOME/github/server-ai/")
config_file = os.path.join(DEFAULT_CONFIG_PATH, "config.json")

if os.path.exists(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
else:
    raise FileNotFoundError(f"Configuration file not found at {config_file}")

PATH = os.path.expandvars(config.get("main_path", "$HOME/github/server-ai/"))
LOG_PATH = os.path.join(PATH, "log")
os.makedirs(LOG_PATH, exist_ok=True)

CURRENT_MODEL = config.get("default_model", "llama3:latest")
PROMPT_SYSTEM = config.get("prompt_system", "Rispondi sempre in italiano. Ti chiami MARRtino.")

# Load FAQ dataset
json_path = os.path.join("data", "faq_marrtino_en_keys.json")
if not os.path.exists(json_path):
    os.makedirs("data", exist_ok=True)
    with open(json_path, "w") as f:
        f.write('[{"question": "How do I turn on MARRtino?", "answer": "Press the red button on the back until the head lights up."}]')

faq_df = pd.read_json(json_path)
vectorizer = TfidfVectorizer().fit(faq_df['question'])
faq_vectors = vectorizer.transform(faq_df['question'])

ollama_client = Client(host='http://localhost:11434')

app = Flask(__name__)
app.static_folder = 'static'

def log_to_file(question, bot_answer, model=None):
    now = datetime.now()
    data_ora = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(os.path.join(LOG_PATH, "log.txt"), "a") as log_file:
        report = data_ora + "\n" + \
                 "[QUESTION]:   " + question + ";" + \
                 f"[MODEL]: {model}; [OLLAMA]: " + bot_answer
        log_file.write(report + "\n")

    if bot_answer != "":
        with open(os.path.join(LOG_PATH, "user.txt"), "a") as bot_file:
            bot_file.write("user: " + question + "\n")
            bot_file.write("bot: " + bot_answer + "\n")

def split_string(msg):
    print(f"[DEBUG] Raw model output: {msg}", file=sys.stderr)
    if not isinstance(msg, str):
        return "(model error)"
    return msg  # Mostra tutto senza troncare


def get_response(messages: list, model_name):
    print(f"[DEBUG] Messages sent to model: {messages}", file=sys.stderr)
    try:
        start_time = time()
        response = ollama_client.chat(
            model=model_name,
            messages=messages,
            options={"num_predict": 100, "temperature": 0.7}
        )

        elapsed_time = time() - start_time
        print(f"[DEBUG] Full model response: {response}", file=sys.stderr)
        print(f"[DEBUG] Model response time: {elapsed_time:.2f} seconds", file=sys.stderr)
        return response['message']
    except Exception as e:
        print(f"[DEBUG] Ollama error: {e}", file=sys.stderr)
        return {"content": f"(error: {str(e)})"}

def get_ollama_models():
    try:
        models_data = ollama_client.list()
        print(f"[DEBUG] Raw models_data: {models_data}", file=sys.stderr)

        models = []
        for m in models_data.get("models", []):
            model_name = m.get("model") or m.get("name")
            if model_name:
                models.append(model_name)

        return models or ["llama3:latest"]
    except Exception as e:
        print(f"[DEBUG] Errore nel recupero modelli Ollama: {e}", file=sys.stderr)
        return ["llama3:latest"]

def send_to_ros2(text, url="http://192.168.1.19:5001/send"):
    try:
        params = {"text": text}
        requests.get(url, params=params)
        print(f"[DEBUG] ✅ Inviato al nodo ROS: {text}", file=sys.stderr)
    except Exception as e:
        print(f"[DEBUG] ❌ Errore nell'invio al nodo ROS: {e}", file=sys.stderr)

@app.route("/")
def home():
    models = get_ollama_models()
    print(f"[DEBUG] Modelli disponibili: {models}", file=sys.stderr)
    return render_template("indexchatbot.html", models=models)

@app.route("/models")
def models_page():
    models = get_ollama_models()
    return render_template("models.html", models=models, current_model=CURRENT_MODEL)

@app.route("/getmodel")
def set_model():
    global CURRENT_MODEL
    selected_model = request.args.get('model')
    print(f"[DEBUG] Richiesta ricevuta per cambiare modello a: {selected_model}", file=sys.stderr)
    if selected_model:
        CURRENT_MODEL = selected_model
        print(f"[DEBUG] Modello attivo impostato su: {CURRENT_MODEL}", file=sys.stderr)
        return f"Modello impostato su: {CURRENT_MODEL}"
    else:
        print(f"[DEBUG] Nessun modello specificato nella richiesta.", file=sys.stderr)
        return "Nessun modello specificato."

@app.route("/get")
def get_bot_response():
    global CURRENT_MODEL
    myquery = request.args.get('msg')
    model = CURRENT_MODEL

    print(f"[DEBUG] Message received: {myquery}", file=sys.stderr)
    print(f"[DEBUG] Using model: {model}", file=sys.stderr)

    input_vector = vectorizer.transform([myquery])
    sim_scores = cosine_similarity(input_vector, faq_vectors)
    best_idx = sim_scores.argmax()
    best_score = sim_scores[0, best_idx]

    print(f"[DEBUG] Best match index: {best_idx} - Score: {best_score}", file=sys.stderr)

    if best_score > 0.6:
        response = faq_df.iloc[best_idx]['answer']
        log_to_file(myquery, response, model=model)
        return response
    else:
        messages = [
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": myquery}
        ]
        new_message = get_response(messages, model_name=model)
        msgout = split_string(new_message['content'])
        log_to_file(myquery, msgout, model=model)
        return msgout
    
@app.route("/faq", methods=["GET", "POST"])
def faq_manager():
    faq_file = os.path.join("data", "faq_marrtino_en_keys.json")
    with open(faq_file, "r") as f:
        faqs = json.load(f)

    if request.method == "POST":
        action = request.form.get("action")
        if action == "add":
            question = request.form.get("question")
            answer = request.form.get("answer")
            faqs.append({"question": question, "answer": answer})
        elif action == "edit":
            idx = int(request.form.get("index"))
            faqs[idx]["question"] = request.form.get("question")
            faqs[idx]["answer"] = request.form.get("answer")
        elif action == "delete":
            idx = int(request.form.get("index"))
            faqs.pop(idx)

        with open(faq_file, "w") as f:
            json.dump(faqs, f, indent=4, ensure_ascii=False)

        return render_template("faq.html", faqs=faqs, message="FAQ aggiornata!")

    return render_template("faq.html", faqs=faqs)

@app.route('/bot')
def bot():
    global CURRENT_MODEL
    myquery = request.args.get('query')
    model = CURRENT_MODEL
    messages = [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": myquery}
    ]
    new_message = get_response(messages, model_name=model)
    msgout = split_string(new_message['content'])
    log_to_file(myquery, msgout, model=model)
    send_to_ros2(msgout)
    return msgout

@app.route('/json')
def json_response():
    global CURRENT_MODEL
    myquery = request.args.get('query')
    model = CURRENT_MODEL
    messages = [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": myquery}
    ]
    new_message = get_response(messages, model_name=model)
    msg = new_message['content']
    msgjson = {
        "response": msg,
        "action": "ok"
    }
    return jsonify(msgjson)

@app.route("/config", methods=["GET", "POST"])
def config_page():
    global CURRENT_MODEL, PROMPT_SYSTEM, config
    config_file = os.path.join(PATH, "config.json")

    if request.method == "POST":
        new_model = request.form.get("default_model")
        new_prompt = request.form.get("prompt_system")

        CURRENT_MODEL = new_model
        PROMPT_SYSTEM = new_prompt

        config["default_model"] = new_model
        config["prompt_system"] = new_prompt

        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)

        return render_template("config.html", config=config, message="Configurazione aggiornata!")

    return render_template("config.html", config=config)

if __name__ == '__main__':
    print(f"ChatBot with Ollama v.1.00 - Modello di default: {CURRENT_MODEL}")
    app.run(host='0.0.0.0', debug=True, port=8060)
