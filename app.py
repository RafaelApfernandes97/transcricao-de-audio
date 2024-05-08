from flask import Flask, request, jsonify
import requests
import whisper
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
import traceback

app = Flask(__name__)

os.environ["PATH"] += os.pathsep + "/usr/bin/"

# Modelo da transcrição do Whisper - ajuste o modelo conforme sua capacidade de hardware
modelo = whisper.load_model("large-v2")

# Configurar um pool de threads para lidar com as transcrições
executor = ThreadPoolExecutor(max_workers=5)  # Ajuste max_workers conforme necessário

def download_and_transcribe(audio_url):
    try:
        # Baixar o arquivo de áudio com um nome único
        audio_path = f"audio_{uuid.uuid4()}.ogg"
        response = requests.get(audio_url)
        if response.status_code != 200:
            return {"error": "Falha ao realizar o download do áudio"}

        with open(audio_path, "wb") as f:
            f.write(response.content)

        # Transcrever o arquivo baixado
        resposta = modelo.transcribe(audio_path)
        transcription = resposta["text"]

        # Remover o arquivo baixado após a transcrição
        os.remove(audio_path)

        return {"Transcrição": transcription}
    except Exception as e:
        # Tratamento de exceção com traceback para diagnóstico detalhado
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    data = request.get_json()
    audio_url = data.get('url')
    if not audio_url:
        return jsonify({"error": "Falha na URL"}), 400

    # Submeter a tarefa para o executor e retornar uma resposta imediatamente
    future = executor.submit(download_and_transcribe, audio_url)

    # Neste ponto, poderíamos retornar uma resposta imediata (e.g., um ID de tarefa),
    # mas para manter a consistência com o exemplo anterior, vamos esperar pelo resultado aqui.
    result = future.result()

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)