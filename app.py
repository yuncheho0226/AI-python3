import os
import zipfile
import io
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from flask import Flask, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff=80.0, fs=44100, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

@app.route('/api/process', methods=['POST'])
def process_voicebank():
    name = request.form.get('voicebank_name', 'Voicebank')
    file1 = request.files.get('file1')
    file2 = request.files.get('file2')

    def load_audio(f):
        data, sr = sf.read(f)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        data = highpass_filter(data, cutoff=80.0, fs=sr)
        return data, sr

    data1, sr1 = load_audio(file1) if file1 else (None, 44100)
    data2, sr2 = load_audio(file2) if file2 else (None, 44100)

    if data1 is not None and data2 is not None:
        max_len = max(len(data1), len(data2))
        d1 = np.pad(data1, (0, max_len - len(data1)))
        d2 = np.pad(data2, (0, max_len - len(data2)))
        audio_data = (d1 + d2) * 0.5
        sample_rate = sr1
    elif data1 is not None:
        audio_data, sample_rate = data1, sr1
    elif data2 is not None:
        audio_data, sample_rate = data2, sr2
    else:
        return "No audio provided", 400

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        target_audio_name = f"{name}_fused.wav"
        
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        zip_file.writestr(target_audio_name, wav_io.getvalue())

        oto = ""
        # 1. Japanese
        for c in ["", "k", "s", "t", "n", "h", "m", "y", "r", "w", "g", "z", "d", "b", "p"]:
            for v in ["a", "i", "u", "e", "o"]:
                oto += f"{c+v}={target_audio_name},50,0,100,25,25\n- {c+v}={target_audio_name},50,0,100,25,25\n"
        
        # 2. Mandarin Chinese
        for init in ["", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h"]:
            for fin in ["a", "o", "e", "i", "u", "ai", "ei", "ao", "ou", "an", "en"]:
                oto += f"{init+fin}={target_audio_name},50,0,100,25,25\n"

        # 3. English
        for v in ["AA", "AE", "AH", "AO", "AY", "EH", "EY", "IH", "IY", "OW", "UW"]:
            oto += f"{v}={target_audio_name},50,0,100,25,25\n"

        # 4. Korean
        for init in ["g", "n", "d", "r", "m", "b", "s", "", "j", "ch", "k", "t", "p", "h"]:
            for v in ["a", "ae", "eo", "e", "o", "u", "eu", "i"]:
                oto += f"{init+v}={target_audio_name},50,0,100,25,25\n"

        # 5. Brazilian Portuguese
        for c in ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "m", "n", "l", "r"]:
            for v in ["a", "e", "i", "o", "u", "an", "en", "in"]:
                oto += f"{c+v}={target_audio_name},50,0,100,25,25\n"

        zip_file.writestr("oto.ini", oto)

        if file1:
            file1.seek(0)
            d, sr = load_audio(file1)
            io_w = io.BytesIO()
            sf.write(io_w, d, sr, format='WAV', subtype='PCM_16')
            zip_file.writestr(file1.filename if file1.filename.endswith('.wav') else "file1.wav", io_w.getvalue())

        if file2:
            file2.seek(0)
            d, sr = load_audio(file2)
            io_w = io.BytesIO()
            sf.write(io_w, d, sr, format='WAV', subtype='PCM_16')
            zip_file.writestr(file2.filename if file2.filename.endswith('.wav') else "file2.wav", io_w.getvalue())

        zip_file.writestr("character.yaml", f"name: {name}\nauthor: SPER\n")
        zip_file.writestr("info.txt", f"Voicebank: {name}\nLanguages: JA, ZH, EN, KO, PT-BR")

    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name=f"{name}_Voicebank.zip")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
