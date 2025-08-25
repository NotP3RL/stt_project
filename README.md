# Speech-to-Text Logger with Speaker Recognition

This project records audio (microphone and system audio) and converts it to text in real time using [Vosk](https://alphacephei.com/vosk/).  
It also includes basic speaker recognition with [Resemblyzer](https://github.com/resemble-ai/Resemblyzer), tagging transcriptions with speaker names from a reference database.

## Installation

Requirements: Python 3.8+

Install dependencies:
```bash
pip install -r requirements.txt
```

Download a Vosk model (e.g., [vosk-model-small-ru-0.22](https://alphacephei.com/vosk/models))

For capturing system audio, install [VB-CABLE Virtual Audio Device](https://vb-audio.com/Cable/).

Set it up as the default recording device or select it when running the script.

Project structure:
```bash
/project
  ├── vosk-model-small-ru-0.22
  ├── speaker_db/        # folder with .wav reference voices
  ├── main.py
  └── README.md
```

## Usage

1. Put `.wav` files of known speakers in `speaker_db/` (filename = speaker name).
2. Run:
```bash
python sampler.py
```
3. Select audio devices when prompted (microphone and VB-CABLE/system audio).
4. Transcriptions will be printed and saved in the file `convo`.

## Notes

* Speaker recognition works best with clean, long reference samples.
* Currently configured for Russian recognition.
