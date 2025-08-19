import vosk
import sounddevice as sdd
from resemblyzer import VoiceEncoder, preprocess_wav
import queue
import numpy as np
import logging
from pathlib import Path
import json
from sklearn.metrics.pairwise import cosine_similarity
import threading


SAMPLERATE = 16000
BLOCKSIZE = 8000
CHUNK = 1.0
THRESHOLD = 0.1
SILENT_THRESHOLD = 0.01
MODEL = vosk.Model('vosk-model-small-ru-0.22')

logging.basicConfig(
    filename='convo',
    filemode='a',
    format='%(asctime)s - %(message)s',
    encoding='utf-8',
    level=logging.INFO
)
logger = logging.getLogger('ConversationLogger')


# загружает базу данных голосов из папки
def load_speakers_db(folder_path, encoder):
    speaker_db = {}
    for file in Path(folder_path).glob("*.wav"):
        name = file.stem
        wav = preprocess_wav(file)
        emb = encoder.embed_utterance(wav)
        speaker_db[name] = emb
    return speaker_db


# находит голос из базы данных голосов
def find_speaker_in_db(embedding, speaker_db, threshold=THRESHOLD):
    # 1) Плохой файл НЕТ!
    # 2) Плохо преобразуем звук в векторы/или по разному
    # 3) Плохо сравниваем

    best_score = -1
    best_name = 'Unknown'
    for name, emb in speaker_db.items():
        score = cosine_similarity([emb], [embedding])[0][0]
        # score = 1 - cosine(emb, embedding)
        if score > best_score and score >= threshold:
                best_name = name
                best_score = score
    return best_name, best_score


def listening(encoder, speaker_db, device_id, name):
    q_audio = queue.Queue()
    rec = vosk.KaldiRecognizer(MODEL, SAMPLERATE)

    def audio_callback(indata, frames, time_info, status):
        q_audio.put(indata.copy())
    
    with sdd.InputStream(samplerate=SAMPLERATE, blocksize=BLOCKSIZE, dtype='int16', channels=1, 
                            device=device_id, callback=audio_callback):
        print("Recording started...")
        while True:
            wav = q_audio.get()
            # wav = (chunk * 32767).astype(np.int16)
            # chunk = chunk.flatten()
            if np.max(np.abs(wav)) < SILENT_THRESHOLD:
                continue
            # emb = encoder.embed_utterance(chunk)
            # speaker_name, speaker_score = find_speaker_in_db(emb, speaker_db)
            if rec.AcceptWaveform(wav.tobytes()):
                result = json.loads(rec.Result())
                if result['text']:
                    tagged = f"{name}: {result['text']}"
                    print(tagged)
                    logger.info(tagged)
            

if __name__ == '__main__':

    # === Ask user to pick devices ===
    print("\nAvailable devices:")
    devices = sdd.query_devices()
    for i, dev in enumerate(devices):
        print(f"{i}: {dev['name']} {dev['max_input_channels']}")

    mic_id = int(input("\nEnter your microphone device ID: "))
    mic_name = input("Enter your microphone speaker name: ")
    sys_id = int(input("Enter your system audio (loopback/stereo mix) device ID: "))
    sys_name = input("Enter your system audio speaker name: ")

    encoder = VoiceEncoder()
    speaker_db = load_speakers_db('speaker_db', encoder)


    # === Start both streams in parallel threads ===
    mic_thread = threading.Thread(target=listening, args=(encoder, speaker_db, mic_id, mic_name))
    sys_thread = threading.Thread(target=listening, args=(encoder, speaker_db, sys_id, sys_name))

    mic_thread.start()
    sys_thread.start()

    # Keep the main thread alive
    mic_thread.join()
    sys_thread.join()