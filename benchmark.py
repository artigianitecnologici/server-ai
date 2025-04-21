import time
import psutil
import platform
import socket
import csv
import subprocess
import matplotlib.pyplot as plt
from ollama import Client

PROMPT = "Spiegami la fotosintesi clorofilliana in modo semplice."
CSV_FILENAME = "benchmark_results.csv"
PLOT_FILENAME = "benchmark_plot.png"

client = Client()

def format_gb(value):
    return value / (1024**3)

def get_gpu_info():
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL
        )
        gpu_name = output.decode().strip()
        return gpu_name if gpu_name else "GPU NVIDIA non rilevata"
    except Exception:
        # macOS Apple Silicon
        try:
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                return "Apple Silicon GPU (Metal)"
        except:
            pass
        return "Nessuna GPU rilevata"

def get_system_info():
    return {
        "host": socket.gethostname(),
        "cpu": platform.processor(),
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(format_gb(psutil.virtual_memory().total), 2),
        "os": platform.system() + " " + platform.release(),
        "architecture": platform.machine(),
        "gpu": get_gpu_info()
    }

def get_installed_models():
    try:
        models_data = client.list()
        models = []
        for m in models_data.get("models", []):
            model_name = m.get("model") or m.get("name")
            if model_name:
                models.append(model_name)
        return models
    except Exception as e:
        print(f"❌ Errore nel recupero dei modelli: {e}")
        return []

def benchmark_model(model_name):
    print(f"\n📌 TEST: {model_name}")
    try:
        cpu_before = psutil.cpu_percent(interval=1)
        ram_before = psutil.virtual_memory().used

        start_time = time.time()
        response = client.chat(model=model_name, messages=[
            {"role": "user", "content": PROMPT}
        ])
        end_time = time.time()

        cpu_after = psutil.cpu_percent(interval=1)
        ram_after = psutil.virtual_memory().used

        result = {
            "model": model_name,
            "time_sec": round(end_time - start_time, 2),
            "cpu_before": cpu_before,
            "cpu_after": cpu_after,
            "ram_before": round(format_gb(ram_before), 2),
            "ram_after": round(format_gb(ram_after), 2),
            "response_sample": response["message"]["content"][:80] + "..."
        }

        return result

    except Exception as e:
        return {
            "model": model_name,
            "error": str(e)
        }

def save_to_csv(results, system_info, filename=CSV_FILENAME):
    keys = list(results[0].keys()) + list(system_info.keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            r.update(system_info)
            writer.writerow(r)

    print(f"\n📁 Risultati salvati in {filename}")

def plot_results(results, system_info):
    filtered = [r for r in results if "error" not in r]

    models = [r["model"] for r in filtered]
    times = [r["time_sec"] for r in filtered]
    cpu_after = [r["cpu_after"] for r in filtered]
    ram_used = [r["ram_after"] - r["ram_before"] for r in filtered]

    plt.figure(figsize=(12, 6))
    plt.suptitle(
        f"Benchmark - {system_info['host']} ({system_info['cpu']})\nGPU: {system_info['gpu']}",
        fontsize=14
    )

    # Tempo di risposta
    plt.subplot(1, 3, 1)
    bars1 = plt.bar(models, times)
    plt.ylabel("Secondi")
    plt.title("⏱ Tempo di Risposta")
    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}",
                 ha='center', va='bottom', fontsize=8)

    # Uso CPU
    plt.subplot(1, 3, 2)
    bars2 = plt.bar(models, cpu_after)
    plt.ylabel("% CPU dopo")
    plt.title("💻 Uso CPU")
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.1f}%",
                 ha='center', va='bottom', fontsize=8)

    # RAM usata
    plt.subplot(1, 3, 3)
    bars3 = plt.bar(models, ram_used)
    plt.ylabel("RAM usata (GB)")
    plt.title("🧠 RAM Utilizzata")
    for bar in bars3:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f} GB",
                 ha='center', va='bottom', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(PLOT_FILENAME)
    print(f"\n🖼️ Grafico salvato come '{PLOT_FILENAME}'")
    plt.show()

if __name__ == "__main__":
    print("📊 Avvio benchmark per modelli Ollama\n")

    system_info = get_system_info()

    print("🔍 INFO SISTEMA:")
    for k, v in system_info.items():
        print(f"   {k}: {v}")

    models = get_installed_models()
    if not models:
        print("❌ Nessun modello trovato. Usa 'ollama run <modello>' prima di eseguire il benchmark.")
        exit(1)

    all_results = []
    for model in models:
        result = benchmark_model(model)
        all_results.append(result)

    print("\n🧾 RISULTATI FINALI:")
    for res in all_results:
        if "error" in res:
            print(f"❌ {res['model']}: ERRORE → {res['error']}")
        else:
            print(f"✅ {res['model']}: {res['time_sec']} sec, CPU {res['cpu_before']}%→{res['cpu_after']}%, "
                  f"RAM {res['ram_before']}→{res['ram_after']} GB")

    save_to_csv(all_results, system_info)
    plot_results(all_results, system_info)
