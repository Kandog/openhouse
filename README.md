# Openhouse AI Assistant

A local AI host assistant for openhouse events. The app uses a webcam for face detection, stores visitor records in SQLite, greets new and returning visitors with text-to-speech, listens through the microphone using `sounddevice`, and uses a local LLM Studio-compatible API for conversational responses.

## Features

- Embedded live camera feed in the main window
- Face detection and simple visitor recognition
- New visitor registration with face encoding
- Returning visitor greetings
- Warm first-time greeting and name request
- Microphone-based speech capture using `sounddevice`
- LLM-generated host responses through LLM Studio
- Text-to-speech host replies using `pyttsx3`
- Conversation panel with host and visitor messages in different colors
- Visitor count on the main screen
- Visitor dashboard with daily totals and hourly breakdown
- SQLite-backed visitor database

## Project Structure

```text
.
├── main.py              # Entry point
├── app.py               # Main Tkinter app and visitor flow
├── face.py              # Camera, face detection, and face comparison
├── database.py          # SQLite visitor/event storage
├── llm.py               # LLM Studio API client
├── stt.py               # Speech-to-text using sounddevice + SpeechRecognition
├── tts.py               # Text-to-speech using pyttsx3
├── dashboard.py         # In-memory visitor dashboard stats
├── config.py            # Environment/config loading
├── requirements.txt     # Python dependencies
└── .env                 # Local configuration
```

## Requirements

- Windows 11 or compatible desktop environment
- Python 3.14 or your configured Python version
- Webcam
- Microphone
- LLM Studio running a local model with the OpenAI-compatible API enabled

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`PyAudio` is intentionally not required because it does not currently install cleanly on Python 3.14. This project uses `sounddevice` for microphone capture instead.

## Configuration

Create or update `.env` in the project root:

```env
LLM_BASE_URL=http://127.0.0.1:1234
LLM_API_KEY=nokey
LLM_MODEL=qwen2.5-0.5b-instruct

FACE_CAMERA_INDEX=0
FACE_THRESHOLD=0.5
FACE_JPEG_QUALITY=95

COOLDOWN_SECONDS=45

TTS_VOICE_INDEX=0
TTS_RATE=150
TTS_VOLUME=1.0

STT_LANGUAGE=en
STT_TIMEOUT=10
STT_PHONEME_THRESHOLD=0.6

LOG_LEVEL=INFO
LOG_DIR=.openhouse/logs
```

### LLM Studio Setup

This app expects LLM Studio's OpenAI-compatible endpoint:

```text
POST /v1/chat/completions
```

Make sure LLM Studio is running, a model is loaded, and the local server is enabled. The `LLM_BASE_URL` value should point to the LLM Studio server address, for example:

```env
LLM_BASE_URL=http://127.0.0.1:1234
```

If LLM Studio logs show requests to `/api/generate`, the app is using an Ollama-style endpoint and `llm.py` should be checked.

### Changing the Host Speaker

The host voice is controlled by `TTS_VOICE_INDEX` in `.env`:

```env
TTS_VOICE_INDEX=0
```

List the voices installed on your Windows system with:

```bash
python -c "import pyttsx3; e=pyttsx3.init(); [print(i, v.name, v.id) for i, v in enumerate(e.getProperty('voices'))]"
```

Example output:

```text
0 Microsoft David Desktop
1 Microsoft Zira Desktop
```

Set `TTS_VOICE_INDEX` to the voice index you want, then restart the app:

```env
TTS_VOICE_INDEX=1
```

You can also tune speech speed and volume:

```env
TTS_RATE=150
TTS_VOLUME=1.0
```

## Running the App

```bash
python main.py
```

The app will:

1. Initialize the SQLite database at `.openhouse/visitors.db`
2. Load known visitors
3. Open the camera
4. Display the live camera feed in the main window
5. Watch for visitors entering the frame
6. Greet visitors and start a short voice conversation

## Visitor Flow

### New Visitor

1. The visitor enters the camera frame.
2. The app speaks a warm greeting.
3. The app asks what name the visitor would like to be called.
4. The app records the name and face encoding.
5. The app generates a personalized greeting through LLM Studio.
6. The app listens for visitor speech and speaks AI-generated responses back.

### Returning Visitor

1. The visitor enters the camera frame.
2. The app recognizes the face if it matches a saved visitor.
3. The app generates a welcome-back message.
4. The app speaks the message and starts a short conversation.

## Dashboard

Click **View Dashboard** in the app to see:

- Today's total visits
- Today's unique visitors
- Hourly breakdown
- Visitor names per hour
- New vs returning visitor type

Dashboard stats are currently in-memory for the running session. Visitor records and events are stored in SQLite.

## Notes on Camera Behavior

The app uses motion-style detection logic so a visitor is counted when they enter the camera view, not repeatedly while they remain in front of the camera. A visitor must leave the frame and re-enter to trigger a new visit event.

If camera index `0` does not work, update `.env`:

```env
FACE_CAMERA_INDEX=1
```

Try `0`, `1`, or `2` depending on your webcam setup.

## Troubleshooting

### Camera opens but app says "Opening camera..."

This usually means camera setup is blocking or the GUI loop is stuck. Confirm `face.open_camera()` returns successfully and that `app.py` starts camera initialization in a background thread.

### LLM Studio says "Unexpected endpoint or method"

The app must call:

```text
/v1/chat/completions
```

not:

```text
/api/generate
```

Check `llm.py` and `.env`.

### Microphone says PyAudio is missing

This project should use `sounddevice`, not PyAudio. Confirm `requirements.txt` includes:

```text
sounddevice>=0.4.5
```

and `stt.py` imports `sounddevice`.

### Host only speaks the first response

`tts.py` creates a fresh `pyttsx3` engine for each utterance to avoid Windows SAPI getting stuck after the first call. If speech stops working, restart the app and check the terminal for `[tts] Speech error` messages.

## Data Storage

Visitor data is stored in:

```text
.openhouse/visitors.db
```

Tables:

- `visitors` — visitor name, face embedding, timestamps, visit count
- `events` — visit and registration event history

## Development

Run a syntax check:

```bash
python -m py_compile main.py app.py face.py database.py llm.py stt.py tts.py dashboard.py
```

Run the app:

```bash
python main.py
```

## Privacy Notice

This app stores visitor names and face embeddings locally in SQLite. If used in a real openhouse or public setting, make sure visitors are informed and consent to being recorded or recognized according to local privacy laws and event policies.
