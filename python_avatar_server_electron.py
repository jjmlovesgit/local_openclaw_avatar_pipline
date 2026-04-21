from fastapi import FastAPI, UploadFile, File, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from imtalker_core import InferenceAgent
import os
import uuid
import asyncio
import time
import shutil
import re
import subprocess
import struct
import wave
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
from PIL import Image
import base64
import httpx
import torch

# WebRTC VAD
import webrtcvad

# Wake Word - Using livekit-wakeword for better accuracy
try:
    from livekit.wakeword import WakeWordModel
    LIVEKIT_WAKEWORD_AVAILABLE = True
    print("✅ livekit-wakeword loaded")
except ImportError:
    LIVEKIT_WAKEWORD_AVAILABLE = False
    print("⚠️ livekit-wakeword not installed. Run: pip install livekit-wakeword")

# Whisper for speech-to-text
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ faster-whisper not installed. Run: pip install faster-whisper")

# ============================================
# CONFIGURATION - DUAL LLM SUPPORT
# ============================================
DEFAULT_AVATAR = "Johnny-Depp.png"
OUTPUT_DIR = "generated_clips"
UPLOAD_DIR = "uploads"
CHUNK_DIR = "chunks"
AVATAR_SIZE = (256, 256)

# LM Studio (Default - Conversational)
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_MODEL = "gemma-4-e4b-it"

OPENCLAW_URL = "http://127.0.0.1:18789/v1/chat/completions"
OPENCLAW_TOKEN = os.getenv("OPENCLAW_TOKEN", "")
if not OPENCLAW_TOKEN:
    print("⚠️ Warning: OPENCLAW_TOKEN environment variable not set. OpenClaw will not work.")
OPENCLAW_MODEL = "openclaw"

# PocketTTS
POCKET_TTS_URL = "http://localhost:8000"
POCKET_TTS_VOICE = "jonnydepp"

# Whisper configuration
WHISPER_MODEL_SIZE = "small"
WHISPER_COMPUTE_TYPE = "float16"

# Wake Word Configuration
WAKE_WORD_ENABLED = True
WAKE_WORD_MODEL = "hey_jarvis_v0.1.onnx"
WAKE_WORD_THRESHOLD = 0.5
WAKE_WORD_DEBOUNCE = 2.0
WAKE_WORD_TIMEOUT = 10.0  # Auto-deactivate after 10 seconds of silence

# VAD Configuration
VAD_AGGRESSIVENESS = 2  # 0-3, higher = more aggressive
VAD_FRAME_SIZE = 480  # 30ms at 16kHz
VAD_MIN_SPEECH_FRAMES = 10  # ~300ms minimum speech
VAD_SILENCE_TIMEOUT_FRAMES = 30  # ~900ms silence timeout
VAD_NOISE_GATE_THRESHOLD = 150

# GPU Sequential Processing
GPU_SEQUENTIAL_MODE = True  # Set to False to disable GPU queue

# System Prompts for each LLM
SYSTEM_PROMPT_LM_STUDIO = """You are a helpful AI assistant speaking through an avatar.

CRITICAL RULES:
1. Keep responses SHORT - 1 to 2 sentences maximum
2. ALWAYS end each sentence with a period, question mark, or exclamation point
3. Use simple, direct language
4. Be conversational and friendly
5. does not use special characters or emojis in your response

Keep it short and natural."""

SYSTEM_PROMPT_OPENCLAW = """You are a helpful voice assistant. Answer concisely in 1-2 short sentences.

Question: {prompt}

Answer:"""

# ============================================
# GLOBAL STATE
# ============================================
app = FastAPI()
agent = None
avatar_image = None
whisper_model = None
warmup_complete = False
wakeword_model = None

# GPU Management
gpu_lock = asyncio.Lock()
gpu_task_queue = asyncio.Queue()
gpu_worker_active = False

# Active sessions
chunk_status: Dict[str, Dict] = {}
active_sessions: Dict[str, 'RealtimeSession'] = {}

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/clips", StaticFiles(directory=OUTPUT_DIR), name="clips")
app.mount("/chunks", StaticFiles(directory=CHUNK_DIR), name="chunks")

# ============================================
# GPU WORKER FOR SEQUENTIAL PROCESSING
# ============================================
async def gpu_worker():
    """Worker that processes GPU tasks sequentially"""
    global gpu_worker_active
    
    print("🎮 GPU Worker started")
    
    while True:
        try:
            # Get next task from queue
            task = await gpu_task_queue.get()
            if task is None:  # Shutdown signal
                break
            
            async with gpu_lock:
                print(f"🔒 GPU lock acquired for: {task['name']}")
                
                # Execute the task
                result = await task['func'](*task.get('args', []), **task.get('kwargs', {}))
                
                # Set result in future if provided
                if 'future' in task:
                    task['future'].set_result(result)
                
                # Clear GPU cache after task
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                print(f"🔓 GPU lock released for: {task['name']}")
                
        except Exception as e:
            print(f"❌ GPU Worker error: {e}")
            if 'future' in task:
                task['future'].set_exception(e)
    
    print("🎮 GPU Worker stopped")

async def queue_gpu_task(name: str, func, *args, **kwargs):
    """Queue a GPU task and wait for completion"""
    if not GPU_SEQUENTIAL_MODE:
        # Run immediately without queue
        return await func(*args, **kwargs)
    
    future = asyncio.Future()
    
    await gpu_task_queue.put({
        'name': name,
        'func': func,
        'args': args,
        'kwargs': kwargs,
        'future': future
    })
    
    return await future

async def start_gpu_worker():
    global gpu_worker_active
    if not gpu_worker_active:
        gpu_worker_active = True
        asyncio.create_task(gpu_worker())

async def stop_gpu_worker():
    global gpu_worker_active
    if gpu_worker_active:
        await gpu_task_queue.put(None)  # Shutdown signal
        gpu_worker_active = False


# ============================================
# PERFORMANCE LOGGER
# ============================================
class PerformanceLogger:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()
        self.markers = {}
        self.chunk_times = []
        self.first_audio_time = None
        self.ttfa = None
        
    def mark(self, name: str):
        elapsed = time.time() - self.start_time
        self.markers[name] = elapsed
        print(f"⏱️  [{self.session_id}] {name}: {elapsed:.3f}s")
        return elapsed
    
    def mark_first_audio(self, chunk_index: int):
        if self.first_audio_time is None:
            self.first_audio_time = time.time() - self.start_time
            self.ttfa = self.first_audio_time
            print(f"\n{'='*60}")
            print(f"🔊 [{self.session_id}] 🎵 TIME TO FIRST AUDIO: {self.ttfa:.3f}s (Chunk {chunk_index + 1})")
            print(f"{'='*60}\n")
    
    def chunk_mark(self, chunk_index: int, name: str):
        elapsed = time.time() - self.start_time
        self.chunk_times.append({
            "chunk": chunk_index,
            "step": name,
            "time": elapsed
        })
        print(f"⏱️  [{self.session_id}] Chunk {chunk_index} - {name}: {elapsed:.3f}s")
    
    def summary(self):
        print(f"\n{'='*60}")
        print(f"📊 PERFORMANCE SUMMARY [{self.session_id}]")
        print(f"{'='*60}")
        
        if self.ttfa:
            print(f"\n🎯 CRITICAL METRIC - TIME TO FIRST AUDIO: {self.ttfa:.3f}s")
            print(f"{'─'*50}")
        
        prev_time = 0
        for name, time_val in sorted(self.markers.items()):
            duration = time_val - prev_time
            print(f"   {name:35s}: {duration:6.3f}s (total: {time_val:6.3f}s)")
            prev_time = time_val
        
        print(f"{'='*60}\n")


# ============================================
# THINKING MODEL RESPONSE PARSER (FINAL ANSWER ONLY)
# ============================================
def extract_final_answer(raw_text: str) -> str:
    """
    Extract the final answer from a thinking model response.
    Always returns only the final answer (what the avatar should speak).
    """
    if not raw_text:
        return ""
    
    final_answer = raw_text.strip()
    
    # Look for end-of-thinking markers
    markers = ['<channel|>', '<|channel|>', '</channel|>']
    
    for marker in markers:
        if marker in raw_text:
            parts = raw_text.split(marker)
            final_answer = parts[-1].strip()
            break
    
    # Clean up any remaining tags
    final_answer = re.sub(r'<[^>]+>', '', final_answer).strip()
    final_answer = re.sub(r'<\|[^>]+\|>', '', final_answer).strip()
    
    # If final answer is empty after cleaning, use original
    if not final_answer:
        final_answer = raw_text.strip()
    
    return final_answer


# ============================================
# WAKE WORD DETECTOR CLASS (livekit-wakeword)
# ============================================
class JarvisWakeWordDetector:
    """Wake word detector for 'Hey Jarvis' using livekit-wakeword"""
    
    def __init__(self, model_path=WAKE_WORD_MODEL, threshold=WAKE_WORD_THRESHOLD):
        self.threshold = threshold
        self.model_path = model_path
        
        if LIVEKIT_WAKEWORD_AVAILABLE:
            try:
                self.model = WakeWordModel(models=[model_path])
                print(f"✅ Wake word model loaded with livekit: {model_path}")
            except Exception as e:
                print(f"❌ Failed to load wake word model: {e}")
                self.model = None
        else:
            self.model = None
        
        self.buffer_duration = 2.0
        self.buffer_size = int(16000 * self.buffer_duration)
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.last_detection_time = 0
        self.debounce_seconds = WAKE_WORD_DEBOUNCE
        self.detection_count = 0
        self.frame_counter = 0
        
    def process_audio_chunk(self, audio_chunk):
        if self.model is None:
            return {"detected": False, "confidence": 0.0}
        
        if isinstance(audio_chunk, bytes):
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
        else:
            audio_np = audio_chunk
            
        self.audio_buffer.extend(audio_np)
        
        self.frame_counter += 1
        if self.frame_counter % 5 != 0:
            return {"detected": False, "confidence": 0.0}
        
        if len(self.audio_buffer) < self.buffer_size:
            return {"detected": False, "confidence": 0.0}
        
        audio_window = np.array(list(self.audio_buffer))[-self.buffer_size:]
        audio_float = audio_window.astype(np.float32) / 32768.0
        scores = self.model.predict(audio_float)
        score = scores.get("hey_jarvis_v0.1", 0.0)
        
        current_time = datetime.now().timestamp()
        if score > self.threshold:
            if current_time - self.last_detection_time > self.debounce_seconds:
                self.last_detection_time = current_time
                self.detection_count += 1
                return {
                    "detected": True,
                    "word": "hey_jarvis",
                    "confidence": float(score),
                    "count": self.detection_count
                }
        
        return {"detected": False, "confidence": float(score)}


# ============================================
# VAD SESSION WITH WAKE WORD INTEGRATION
# ============================================
class RealtimeSession:
    def __init__(self, session_id: str, voice: str, websocket: WebSocket):
        self.session_id = session_id
        self.voice = voice
        self.websocket = websocket
        self.audio_buffer = bytearray()
        self.processing_queue = asyncio.Queue()
        self.is_active = True
        self.current_utterance_id = 0
        self.use_openclaw = False      # LLM toggle
        self.skip_avatar = False       # Skip avatar video, play TTS only
        
        # VAD settings
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.min_speech_frames = VAD_MIN_SPEECH_FRAMES
        self.silence_timeout_frames = VAD_SILENCE_TIMEOUT_FRAMES
        self.frame_size = VAD_FRAME_SIZE
        self.noise_gate_threshold = VAD_NOISE_GATE_THRESHOLD
        
        # Speech state
        self.is_speaking = False
        self.speech_frames = []
        self.silence_frames = 0
        self.last_audio_level = 0
        
        # Wake word state
        self.wakeword_detector = wakeword_model if WAKE_WORD_ENABLED else None
        self.is_wakeword_activated = not WAKE_WORD_ENABLED
        self.last_activation_time = 0
        self.wakeword_timeout = WAKE_WORD_TIMEOUT
        self.wakeword_buffer = deque(maxlen=32000)
        
        # Worker task
        self.worker_task = None
        
    def get_audio_level(self, frame):
        try:
            samples = struct.unpack(f'{len(frame)//2}h', frame)
            rms = sum(s*s for s in samples) / len(samples)
            return rms ** 0.5
        except:
            return 0
    
    def is_voice_with_noise_gate(self, frame):
        if len(frame) != self.frame_size * 2:
            return False
            
        level = self.get_audio_level(frame)
        self.last_audio_level = level
        if level < self.noise_gate_threshold:
            return False
        
        try:
            return self.vad.is_speech(frame, 16000)
        except Exception as e:
            print(f"VAD error: {e}")
            return False
    
    def check_wakeword(self, frame):
        if self.wakeword_detector is None:
            return False
        
        audio_np = np.frombuffer(frame, dtype=np.int16)
        self.wakeword_buffer.extend(audio_np)
        
        if len(self.wakeword_buffer) >= 32000:
            buffer_array = np.array(list(self.wakeword_buffer))
            result = self.wakeword_detector.process_audio_chunk(buffer_array)
            if result.get("detected", False):
                self.wakeword_buffer.clear()
                return True
        
        return False
    
    async def activate_wakeword(self):
        self.is_wakeword_activated = True
        self.last_activation_time = time.time()
        self.speech_frames = []
        self.silence_frames = 0
        self.current_utterance_id = 0
        
        await self.websocket.send_json({
            "type": "wakeword_activated",
            "status": "active",
            "message": "🎯 Hey Jarvis detected! I'm listening...",
            "timestamp": time.time(),
            "timeout": self.wakeword_timeout
        })
        print(f"🎯 [{self.session_id}] Wake word activated!")
    
    async def deactivate_wakeword(self):
        self.is_wakeword_activated = False
        self.speech_frames = []
        self.current_utterance_id = 0
        
        await self.websocket.send_json({
            "type": "wakeword_deactivated",
            "status": "idle",
            "message": "😴 Listening mode ended. Say 'Hey Jarvis' to wake me again.",
            "timestamp": time.time()
        })
        print(f"⏰ [{self.session_id}] Wake word deactivated (timeout)")
    
    def check_timeout(self):
        if self.is_wakeword_activated and not self.is_speaking:
            if time.time() - self.last_activation_time > self.wakeword_timeout:
                return True
        return False
    
    async def stop(self):
        self.is_active = False
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except:
                pass


# ============================================
# UTILITY FUNCTIONS
# ============================================
def resize_avatar(image: Image.Image, size: Tuple[int, int] = (256, 256)) -> Image.Image:
    return image.resize(size, Image.Resampling.LANCZOS)

def split_text_into_chunks(text: str, target_duration_seconds: float = 3.0) -> List[str]:
    if not text:
        return []
    
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        sentences = [text]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if sentence and sentence[-1] not in '.!?':
            sentence += '.'
        
        sentence_duration = len(sentence) / 20
        current_duration = len(current_chunk) / 20 if current_chunk else 0
        
        if current_chunk and (current_duration + sentence_duration) > target_duration_seconds:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    merged_chunks = []
    for chunk in chunks:
        chunk_duration = len(chunk) / 20
        if chunk_duration < 1.0 and merged_chunks:
            merged_chunks[-1] += " " + chunk
        else:
            merged_chunks.append(chunk)
    
    return merged_chunks if merged_chunks else [text]

async def transcribe_audio(audio_path: str, perf: PerformanceLogger) -> str:
    global whisper_model
    
    if whisper_model is None:
        return ""
    
    try:
        segments, info = whisper_model.transcribe(
            audio_path,
            beam_size=5,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, threshold=0.5)
        )
        
        transcript = " ".join([segment.text for segment in segments])
        return transcript.strip()
        
    except Exception as e:
        print(f"Whisper error: {e}")
        return ""

async def query_llm(prompt: str, use_openclaw: bool = False, perf: PerformanceLogger = None) -> Dict[str, str]:
    """Query LLM and return TTS text (final answer only)"""
    
    if use_openclaw:
        # Use OpenClaw (Task-Oriented)
        print(f"   🔷 Using OpenClaw...")
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    OPENCLAW_URL,
                    json={
                        "model": OPENCLAW_MODEL,
                        "messages": [
                            {"role": "user", "content": SYSTEM_PROMPT_OPENCLAW.format(prompt=prompt)}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 150,
                        "stream": False
                    },
                    headers={
                        "Authorization": f"Bearer {OPENCLAW_TOKEN}",
                        "Content-Type": "application/json",
                        "Cache-Control": "no-cache"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    raw_result = data['choices'][0]['message']['content'].strip()
                    final_answer = extract_final_answer(raw_result)
                    print(f"   🆕 OpenClaw Response: {final_answer[:50]}...")
                    return {"tts_text": final_answer, "display_text": final_answer}
                else:
                    error_msg = "I'm having trouble connecting to OpenClaw right now."
                    return {"tts_text": error_msg, "display_text": error_msg}
        except Exception as e:
            print(f"OpenClaw error: {e}")
            error_msg = "I'm having trouble connecting to OpenClaw right now."
            return {"tts_text": error_msg, "display_text": error_msg}
    else:
        # Use LM Studio (Conversational - Default)
        print(f"   🤖 Using LM Studio...")
        try:
            # Longer timeout for thinking models
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    LM_STUDIO_URL,
                    json={
                        "model": LM_STUDIO_MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT_LM_STUDIO},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 512,
                        "stream": False
                    },
                    headers={"Cache-Control": "no-cache"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    raw_result = data['choices'][0]['message']['content'].strip()
                    final_answer = extract_final_answer(raw_result)
                    print(f"   🆕 LM Studio Response: {final_answer[:50]}...")
                    return {"tts_text": final_answer, "display_text": final_answer}
                else:
                    error_msg = "I'm having trouble connecting to LM Studio right now."
                    return {"tts_text": error_msg, "display_text": error_msg}
        except Exception as e:
            print(f"LM Studio error: {e}")
            error_msg = "I'm having trouble connecting to LM Studio right now."
            return {"tts_text": error_msg, "display_text": error_msg}

async def generate_speech_pockettts(text: str, output_path: str, voice: str, perf: PerformanceLogger, chunk_idx: int = None) -> bool:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{POCKET_TTS_URL}/v1/audio/speech",
                json={"input": text, "voice": voice, "response_format": "wav"}
            )
            
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return True
            return False
    except Exception as e:
        print(f"PocketTTS error: {e}")
        return False

async def generate_chunk_video(text: str, chunk_index: int, session_id: str, voice: str, perf: PerformanceLogger) -> Optional[str]:
    chunk_filename = f"stream_{session_id}_{chunk_index}.mp4"
    chunk_path = os.path.join(CHUNK_DIR, chunk_filename)
    
    if os.path.exists(chunk_path):
        return chunk_path
    
    temp_wav = os.path.join(CHUNK_DIR, f"temp_{session_id}_{chunk_index}.wav")
    
    success = await generate_speech_pockettts(text, temp_wav, voice, perf, chunk_index)
    
    if not success:
        return None
    
    await run_in_threadpool(
        agent.generate,
        img_pil=avatar_image,
        aud_path=temp_wav,
        output_path=chunk_path,
        nfe=3  # Reduced from 4 for faster generation
    )
    
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    
    # Clear GPU cache after video generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if chunk_index == 0:
        perf.mark_first_audio(chunk_index)
    
    return chunk_path

async def warmup_pockettts_voice(voice: str):
    print(f"\n🔥 Warming up PocketTTS voice: {voice}...")
    try:
        test_text = "Hello."
        test_wav = os.path.join(CHUNK_DIR, f"warmup_{voice}.wav")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{POCKET_TTS_URL}/v1/audio/speech",
                json={"input": test_text, "voice": voice, "response_format": "wav"}
            )
            
            if response.status_code == 200:
                with open(test_wav, "wb") as f:
                    f.write(response.content)
                print(f"✅ Voice '{voice}' warmed up successfully")
                if os.path.exists(test_wav):
                    os.remove(test_wav)
                return True
    except Exception as e:
        print(f"⚠️ Voice warmup error: {e}")
        return False

async def warmup_wakeword():
    global wakeword_model
    if WAKE_WORD_ENABLED and LIVEKIT_WAKEWORD_AVAILABLE:
        print("\n🔥 Warming up Wake Word detector (livekit-wakeword)...")
        try:
            wakeword_model = JarvisWakeWordDetector()
            dummy_audio = np.zeros(32000, dtype=np.int16)
            wakeword_model.process_audio_chunk(dummy_audio)
            print("✅ Wake Word detector ready! (livekit-wakeword)")
        except Exception as e:
            print(f"⚠️ Wake Word warmup failed: {e}")


# ============================================
# QUEUE WORKER FOR PROCESSING UTTERANCES
# ============================================
async def process_utterance(session: RealtimeSession, audio_data: bytes, utterance_id: int):
    start_time = time.time()
    unique_id = f"{session.session_id}_{utterance_id}_{int(start_time * 1000)}"
    perf = PerformanceLogger(session.session_id)
    
    try:
        if not session.is_active:
            return
        
        print(f"\n🔄 Processing utterance {utterance_id} [{unique_id}] - LLM: {'OpenClaw' if session.use_openclaw else 'LM Studio'} | Skip Avatar: {session.skip_avatar}")
        
        await session.websocket.send_json({
            "type": "status_update",
            "status": "transcribing",
            "message": "📝 Transcribing...",
            "utterance": utterance_id,
            "timestamp": time.time()
        })
        
        temp_audio = os.path.join(CHUNK_DIR, f"audio_{unique_id}.wav")
        with wave.open(temp_audio, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(audio_data)
        
        user_input = await transcribe_audio(temp_audio, perf)
        
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        
        # VALIDATION: Check if speech is too short
        if not user_input or len(user_input.strip()) < 2:
            print(f"⚠️ [{utterance_id}] Speech too short - NOT interrupting playback")
            await session.websocket.send_json({
                "type": "error",
                "status": "error",
                "message": "❌ Speech too short",
                "timestamp": time.time()
            })
            session.is_speaking = False
            session.speech_frames = []
            session.silence_frames = 0
            return
        
        # VALID SPEECH - NOW clear video and process
        print(f"✅ [{utterance_id}] Valid speech detected - interrupting playback")
        
        await session.websocket.send_json({
            "type": "clear_video",
            "utterance": utterance_id,
            "timestamp": time.time()
        })
        
        print(f"📝 [{utterance_id}] User said: {user_input}")
        
        # --- IMMEDIATE UI UPDATE WITH USER QUESTION ---
        await session.websocket.send_json({
            "type": "transcript",
            "utterance": utterance_id,
            "text": user_input,
            "response": "",           # empty placeholder
            "num_chunks": 0,
            "partial": True           # indicates this is just the user part
        })
        
        await session.websocket.send_json({
            "type": "status_update",
            "status": "generating",
            "message": f"💬 Generating response... (via {'OpenClaw' if session.use_openclaw else 'LM Studio'})",
            "utterance": utterance_id,
            "text": user_input,
            "timestamp": time.time()
        })
        
        # Queue LLM query through GPU worker (sequential processing)
        llm_response = await queue_gpu_task(
            f"LLM_query_utt{utterance_id}",
            query_llm,
            user_input,
            use_openclaw=session.use_openclaw,
            perf=perf
        )
        
        response_text = llm_response["tts_text"]
        display_text = llm_response["display_text"]
        
        print(f"🤖 [{utterance_id}] TTS Response: {response_text[:50]}...")
        
        utterance_session_id = f"{session.session_id}_utt{utterance_id}"
        
        # --- SKIP AVATAR MODE: Generate TTS only and send audio URL ---
        if session.skip_avatar:
            tts_filename = f"tts_{utterance_session_id}.wav"
            tts_path = os.path.join(CHUNK_DIR, tts_filename)
            success = await generate_speech_pockettts(response_text, tts_path, session.voice, perf)
            if success:
                await session.websocket.send_json({
                    "type": "audio_response",
                    "utterance": utterance_id,
                    "url": f"/chunks/{tts_filename}",
                    "text": display_text
                })
                # Also send complete transcript for UI
                await session.websocket.send_json({
                    "type": "transcript",
                    "utterance": utterance_id,
                    "text": user_input,
                    "response": display_text,
                    "partial": False
                })
            else:
                await session.websocket.send_json({"type": "error", "message": "TTS failed"})
            
            session.is_speaking = False
            session.speech_frames = []
            session.silence_frames = 0
            session.last_activation_time = time.time()
            return
        
        # --- AVATAR MODE: Generate video chunks as usual ---
        chunks = split_text_into_chunks(response_text, target_duration_seconds=3.0)
        
        # --- SEND COMPLETE TRANSCRIPT ---
        await session.websocket.send_json({
            "type": "transcript",
            "utterance": utterance_id,
            "text": user_input,
            "response": display_text,
            "num_chunks": len(chunks),
            "partial": False
        })
        
        # Generate and send video chunks (each queued through GPU worker)
        for chunk_idx, sentence in enumerate(chunks):
            if not session.is_active:
                break
            
            chunk_path = await queue_gpu_task(
                f"Video_chunk_{chunk_idx}_utt{utterance_id}",
                generate_chunk_video,
                sentence,
                chunk_idx,
                utterance_session_id,
                session.voice,
                perf
            )
            
            if chunk_path and os.path.exists(chunk_path):
                with open(chunk_path, 'rb') as f:
                    await session.websocket.send_bytes(f.read())
                
                await session.websocket.send_json({
                    "type": "chunk_complete",
                    "utterance": utterance_id,
                    "chunk": chunk_idx,
                    "total": len(chunks),
                    "duration": round(time.time() - start_time, 2)
                })
                
                if chunk_idx == 0:
                    perf.mark_first_audio(chunk_idx)
                
                try:
                    os.remove(chunk_path)
                except:
                    pass
        
        total_time = time.time() - start_time
        
        await session.websocket.send_json({
            "type": "status_update",
            "status": "active" if session.is_wakeword_activated else "idle",
            "message": "🟢 Listening..." if session.is_wakeword_activated else "💤 Say 'Hey Jarvis'",
            "total_time": round(total_time, 2),
            "timestamp": time.time()
        })
        
        await session.websocket.send_json({
            "type": "complete",
            "utterance": utterance_id,
            "total_time": round(total_time, 2)
        })
        
        print(f"✅ [{utterance_id}] Complete in {total_time:.2f}s with {len(chunks)} chunks")
        
        session.is_speaking = False
        session.speech_frames = []
        session.silence_frames = 0
        session.last_activation_time = time.time()
        
    except Exception as e:
        print(f"❌ Error processing utterance {utterance_id}: {e}")
        session.is_speaking = False
        session.speech_frames = []
        session.silence_frames = 0
        try:
            await session.websocket.send_json({
                "type": "error",
                "status": "error",
                "message": f"❌ Error: {str(e)}",
                "timestamp": time.time()
            })
        except:
            pass


async def queue_worker(session: RealtimeSession):
    """Worker that processes utterances from the queue sequentially"""
    utterance_counter = 0
    try:
        while session.is_active:
            try:
                audio_data = await asyncio.wait_for(session.processing_queue.get(), timeout=1.0)
                utterance_counter += 1
                session.current_utterance_id = utterance_counter
                
                await process_utterance(session, audio_data, utterance_counter)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Queue worker error: {e}")
                break
    finally:
        session.is_active = False
        print(f"🔚 Queue worker ended for session {session.session_id}")


# ============================================
# WEBSOCKET ENDPOINT WITH WAKE WORD & VAD
# ============================================
# ============================================
# WEBSOCKET ENDPOINT WITH WAKE WORD & VAD (BASELINE + DISCONNECT FIX)
# ============================================
@app.websocket("/ws/realtime/{voice}")
async def websocket_realtime(websocket: WebSocket, voice: str):
    await websocket.accept()
    
    session_id = str(uuid.uuid4())[:8]
    wakeword_status = "enabled" if WAKE_WORD_ENABLED else "disabled"
    print(f"🔊 Connected: {session_id} (Wake Word: {wakeword_status})")
    
    session = RealtimeSession(session_id, voice, websocket)
    active_sessions[session_id] = session
    session.worker_task = asyncio.create_task(queue_worker(session))
    
    await websocket.send_json({
        "type": "ready",
        "status": "idle",
        "message": "🎤 Say 'Hey Jarvis' to wake me up" if WAKE_WORD_ENABLED else "🟢 Always listening...",
        "wakeword_enabled": WAKE_WORD_ENABLED,
        "wakeword": "hey_jarvis" if WAKE_WORD_ENABLED else None,
        "timestamp": time.time()
    })
    
    async def send_status_updates():
        while session.is_active:
            await asyncio.sleep(0.5)
            
            if not session.is_active:
                break
                
            if WAKE_WORD_ENABLED:
                if session.is_wakeword_activated:
                    if session.is_speaking:
                        status = "speaking"
                        message = "🎙️ Listening to you..."
                        time_left = int(session.wakeword_timeout - (time.time() - session.last_activation_time))
                        extra = {"time_left": max(0, time_left)}
                    else:
                        time_left = int(session.wakeword_timeout - (time.time() - session.last_activation_time))
                        if time_left <= 3:
                            status = "warning"
                            message = f"⚠️ Listening mode ends in {time_left}s"
                            extra = {"time_left": time_left, "warning": True}
                        else:
                            status = "active"
                            message = f"🟢 Active - listening... ({time_left}s)"
                            extra = {"time_left": time_left}
                else:
                    status = "idle"
                    message = "💤 Sleeping - Say 'Hey Jarvis'"
                    extra = {"sleeping": True}
            else:
                if session.is_speaking:
                    status = "speaking"
                    message = "🎙️ Listening..."
                    extra = {}
                else:
                    status = "listening"
                    message = "🟢 Always listening..."
                    extra = {}
            
            try:
                await websocket.send_json({
                    "type": "status_update",
                    "status": status,
                    "message": message,
                    "audio_level": session.last_audio_level,
                    "is_speaking": session.is_speaking,
                    "is_activated": session.is_wakeword_activated if WAKE_WORD_ENABLED else True,
                    "timestamp": time.time(),
                    **extra
                })
            except:
                break
    
    status_task = asyncio.create_task(send_status_updates())
    
    try:
        while True:
            # Receive either text (JSON) or bytes (audio)
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                print(f"🔇 Disconnected: {session_id}")
                break
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    print(f"🔇 Disconnected (RuntimeError): {session_id}")
                    break
                raise
            
            # Handle JSON control messages
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "set_llm":
                        session.use_openclaw = data.get("use_openclaw", False)
                        print(f"🔧 [{session_id}] LLM switched to: {'OpenClaw' if session.use_openclaw else 'LM Studio'}")
                        await websocket.send_json({
                            "type": "llm_updated",
                            "use_openclaw": session.use_openclaw,
                            "message": f"LLM: {'OpenClaw' if session.use_openclaw else 'LM Studio'}"
                        })
                    elif data.get("type") == "set_skip_avatar":
                        session.skip_avatar = data.get("skip_avatar", False)
                        print(f"🔧 [{session_id}] Skip Avatar: {session.skip_avatar}")
                        await websocket.send_json({
                            "type": "skip_avatar_updated",
                            "skip_avatar": session.skip_avatar
                        })
                    continue
                except:
                    pass
            
            # Handle binary audio data
            if "bytes" in message:
                data = message["bytes"]
            else:
                continue
            
            session.audio_buffer.extend(data)
            
            while len(session.audio_buffer) >= session.frame_size * 2:
                frame_bytes = session.frame_size * 2
                frame = bytes(session.audio_buffer[:frame_bytes])
                session.audio_buffer = session.audio_buffer[frame_bytes:]
                
                if len(frame) != frame_bytes:
                    continue
                
                if WAKE_WORD_ENABLED and not session.is_wakeword_activated:
                    if session.check_wakeword(frame):
                        await session.activate_wakeword()
                        
                        await websocket.send_json({
                            "type": "wakeword_detected",
                            "status": "activated",
                            "message": "🎯 Hey Jarvis detected! I'm listening...",
                            "timestamp": time.time()
                        })
                    continue
                
                if WAKE_WORD_ENABLED and session.check_timeout():
                    await session.deactivate_wakeword()
                    
                    await websocket.send_json({
                        "type": "wakeword_timeout",
                        "status": "idle",
                        "message": "😴 Going to sleep... Say 'Hey Jarvis' to wake me",
                        "timestamp": time.time()
                    })
                    continue
                
                if session.is_wakeword_activated:
                    is_speech = session.is_voice_with_noise_gate(frame)
                    
                    if is_speech and not session.is_speaking:
                        session.is_speaking = True
                        session.speech_frames = [frame]
                        session.silence_frames = 0
                        print(f"🎤 [{session_id}] Speech started")
                        
                        await websocket.send_json({
                            "type": "speech_started",
                            "status": "speaking",
                            "message": "🎙️ I hear you...",
                            "timestamp": time.time()
                        })
                        
                    elif is_speech and session.is_speaking:
                        session.speech_frames.append(frame)
                        session.silence_frames = 0
                        session.last_activation_time = time.time()
                        
                    elif not is_speech and session.is_speaking:
                        session.speech_frames.append(frame)
                        session.silence_frames += 1
                        
                        if session.silence_frames >= session.silence_timeout_frames:
                            session.is_speaking = False
                            
                            speech_frames_count = len(session.speech_frames) - session.silence_frames
                            if speech_frames_count > 0:
                                speech_audio = b''.join(session.speech_frames[:speech_frames_count])
                                
                                if len(session.speech_frames) >= session.min_speech_frames:
                                    print(f"📊 [{session_id}] Processing: {len(session.speech_frames)} frames")
                                    await session.processing_queue.put(speech_audio)
                                    
                                    await websocket.send_json({
                                        "type": "processing_started",
                                        "status": "processing",
                                        "message": f"🤔 Thinking... (via {'OpenClaw' if session.use_openclaw else 'LM Studio'})",
                                        "timestamp": time.time()
                                    })
                                else:
                                    print(f"⚠️ [{session_id}] Too short: {len(session.speech_frames)} frames")
                            
                            session.speech_frames = []
                            session.silence_frames = 0
                    
    except WebSocketDisconnect:
        print(f"🔇 Disconnected: {session_id}")
    finally:
        status_task.cancel()
        await session.stop()
        if session.worker_task:
            session.worker_task.cancel()
        if session_id in active_sessions:
            del active_sessions[session_id]
        try:
            await status_task
        except:
            pass
        
# ============================================
# HTTP ENDPOINTS
# ============================================
@app.get("/voices")
async def list_voices():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{POCKET_TTS_URL}/v1/audio/voices", timeout=5)
            if response.status_code == 200:
                voices_data = response.json()
                voices = []
                
                if 'voices' in voices_data:
                    for voice in voices_data['voices']:
                        voice_id = voice.get('voice_id') or voice.get('id')
                        voice_name = voice.get('name') or voice_id
                        voices.append({"id": voice_id, "name": voice_name})
                elif isinstance(voices_data, list):
                    for voice in voices_data:
                        voice_id = voice.get('voice_id') or voice.get('id')
                        voice_name = voice.get('name') or voice_id
                        voices.append({"id": voice_id, "name": voice_name})
                
                built_in = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]
                existing_ids = [v['id'] for v in voices]
                for voice_id in built_in:
                    if voice_id not in existing_ids:
                        voices.insert(0, {"id": voice_id, "name": voice_id.capitalize()})
                
                voices.sort(key=lambda x: 0 if x['id'] == 'jonnydepp' else 1)
                
                return {"voices": voices}
    except Exception as e:
        print(f"Error fetching voices: {e}")
    
    return {"voices": [
        {"id": "jonnydepp", "name": "Jonny Depp"},
        {"id": "alba", "name": "Alba"},
        {"id": "marius", "name": "Marius"},
        {"id": "javert", "name": "Javert"},
        {"id": "jean", "name": "Jean"},
        {"id": "fantine", "name": "Fantine"},
        {"id": "cosette", "name": "Cosette"},
        {"id": "eponine", "name": "Eponine"},
        {"id": "azelma", "name": "Azelma"}
    ]}

@app.get("/wakeword/status")
async def wakeword_status():
    return {
        "enabled": WAKE_WORD_ENABLED,
        "model": WAKE_WORD_MODEL,
        "threshold": WAKE_WORD_THRESHOLD,
        "loaded": wakeword_model is not None,
        "library": "livekit-wakeword" if LIVEKIT_WAKEWORD_AVAILABLE else "none"
    }

@app.get("/llm/status")
async def llm_status():
    """Get current LLM configuration status"""
    return {
        "lm_studio": {
            "url": LM_STUDIO_URL,
            "model": LM_STUDIO_MODEL
        },
        "openclaw": {
            "url": OPENCLAW_URL,
            "model": OPENCLAW_MODEL
        }
    }

@app.get("/gpu/status")
async def gpu_status():
    """Get GPU worker status"""
    return {
        "sequential_mode": GPU_SEQUENTIAL_MODE,
        "worker_active": gpu_worker_active,
        "queue_size": gpu_task_queue.qsize() if gpu_task_queue else 0,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/chat")
async def chat(
    audio: UploadFile = File(None),
    text: str = Form(None),
    voice: str = Form(POCKET_TTS_VOICE),
    system_prompt: str = Form(None),
    use_openclaw: bool = Form(False),
    skip_avatar: bool = Form(False),
    transcribe_only: bool = Form(False)      # <-- NEW
):
    session_id = str(uuid.uuid4())[:8]
    perf = PerformanceLogger(session_id)
    user_input = ""
    
    print(f"\n{'='*60}")
    print(f"🎬 NEW REQUEST [{session_id}] - LLM: {'OpenClaw' if use_openclaw else 'LM Studio'} | Skip Avatar: {skip_avatar} | Transcribe Only: {transcribe_only}")
    print(f"{'='*60}")
    
    perf.mark("🚀 Request received")
    
    if audio:
        print(f"\n🎤 Received audio: {audio.filename}")
        
        temp_audio = os.path.join(UPLOAD_DIR, f"audio_{session_id}.webm")
        with open(temp_audio, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        
        perf.mark("📁 Audio file saved")
        
        wav_path = temp_audio.replace('.webm', '.wav')
        subprocess.run([
            'ffmpeg', '-i', temp_audio, '-ar', '16000', '-ac', '1', '-y', wav_path
        ], capture_output=True)
        
        perf.mark("🔄 FFmpeg conversion complete")
        
        user_input = await transcribe_audio(wav_path, perf)
        
        if not user_input:
            return {"success": False, "error": "Failed to transcribe audio"}
        
        print(f"   📝 Transcript ({len(user_input)} chars): {user_input[:100]}...")
        
        os.remove(temp_audio)
        os.remove(wav_path)
        
        perf.mark("🗑️  Temp files cleaned")
    
    elif text and text.strip():
        user_input = text.strip()
        print(f"\n📝 User text ({len(user_input)} chars): {user_input[:100]}...")
        perf.mark("📝 Text input received")
    
    else:
        return {"success": False, "error": "No audio or text input provided"}
    
    # --- NEW: TRANSCRIBE ONLY MODE (PTT voice typing) ---
    if transcribe_only:
        perf.mark("✅ Transcription complete (transcribe_only)")
        return {
            "success": True,
            "transcript": user_input,
            "transcribe_only": True
        }
    
    # Queue LLM query through GPU worker
    llm_response = await queue_gpu_task(
        f"LLM_query_{session_id}",
        query_llm,
        user_input,
        use_openclaw=use_openclaw,
        perf=perf
    )
    
    response_text = llm_response["tts_text"]
    display_text = llm_response["display_text"]
    
    print(f"   🤖 TTS Response ({len(response_text)} chars): {response_text[:100]}...")
    
    # --- SKIP AVATAR MODE: Return audio URL directly ---
    if skip_avatar:
        tts_filename = f"tts_{session_id}.wav"
        tts_path = os.path.join(CHUNK_DIR, tts_filename)
        success = await generate_speech_pockettts(response_text, tts_path, voice, perf)
        if success:
            perf.mark("✅ TTS audio generated")
            return {
                "success": True,
                "skip_avatar": True,
                "audio_url": f"/chunks/{tts_filename}",
                "response_text": response_text,
                "display_text": display_text,
                "voice": voice,
                "llm_used": "OpenClaw" if use_openclaw else "LM Studio"
            }
        else:
            return {"success": False, "error": "TTS generation failed"}
    
    # --- AVATAR MODE: Generate video chunks ---
    chunks = split_text_into_chunks(response_text, target_duration_seconds=3.0)
    
    perf.mark(f"📝 Response split into {len(chunks)} chunks")
    
    chunk_status[session_id] = {
        "total_chunks": len(chunks),
        "ready": [False] * len(chunks),
        "paths": [None] * len(chunks),
        "text_chunks": chunks,
        "response_text": response_text,
        "display_text": display_text,
        "generation_start": time.time()
    }
    
    async def generate_progressively():
        for i, chunk_text in enumerate(chunks):
            start_time = time.time()
            
            chunk_path = await queue_gpu_task(
                f"Video_chunk_{i}_{session_id}",
                generate_chunk_video,
                chunk_text,
                i,
                session_id,
                voice,
                perf
            )
            
            if chunk_path:
                chunk_status[session_id]["paths"][i] = chunk_path
                chunk_status[session_id]["ready"][i] = True
                gen_time = time.time() - start_time
                
                estimated_duration = len(chunk_text) / 20
                print(f"   🚀 Chunk {i+1}/{len(chunks)} ready in {gen_time:.2f}s (speaks for ~{estimated_duration:.1f}s)")
            else:
                print(f"   ❌ Chunk {i+1}/{len(chunks)} failed")
    
    asyncio.create_task(generate_progressively())
    
    perf.mark("✨ Response ready, starting video generation")
    
    return {
        "success": True,
        "session_id": session_id,
        "total_chunks": len(chunks),
        "transcript": user_input if audio else None,
        "response_text": response_text,
        "display_text": display_text,
        "voice": voice,
        "llm_used": "OpenClaw" if use_openclaw else "LM Studio",
        "skip_avatar": False,
        "message": f"Generated {len(chunks)} video chunks"
    }
    
@app.get("/chunk_status/{session_id}")
async def get_chunk_status(session_id: str):
    if session_id not in chunk_status:
        return {"error": "Session not found"}
    
    return {
        "total_chunks": chunk_status[session_id]["total_chunks"],
        "ready": chunk_status[session_id]["ready"],
        "response_text": chunk_status[session_id].get("response_text", ""),
        "display_text": chunk_status[session_id].get("display_text", "")
    }

@app.get("/wait_for_chunk/{session_id}/{chunk_index}")
async def wait_for_chunk(session_id: str, chunk_index: int):
    if session_id not in chunk_status:
        return {"error": "Session not found"}
    
    start_time = time.time()
    timeout = 120
    
    while time.time() - start_time < timeout:
        if chunk_index < len(chunk_status[session_id]["ready"]) and chunk_status[session_id]["ready"][chunk_index]:
            return {
                "ready": True,
                "url": f"/chunks/stream_{session_id}_{chunk_index}.mp4"
            }
        await asyncio.sleep(0.5)
    
    return {"ready": False, "error": "Timeout"}

@app.get("/")
async def index(request: Request):
    avatar_bg = ""
    if os.path.exists(DEFAULT_AVATAR):
        with open(DEFAULT_AVATAR, "rb") as f:
            avatar_bg = base64.b64encode(f.read()).decode()
    
    return templates.TemplateResponse("index.html", {"request": request, "AVATAR_BASE64": avatar_bg})


# ============================================
# LIFESPAN
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, avatar_image, whisper_model, wakeword_model, warmup_complete
    
    print("\n" + "="*60)
    print("🚀 AVATAR STUDIO - GPU SEQUENTIAL + SKIP AVATAR TOGGLE")
    print("="*60)
    
    # Initialize ImTalker
    agent = InferenceAgent()
    
    if os.path.exists(DEFAULT_AVATAR):
        original = Image.open(DEFAULT_AVATAR).convert('RGB')
        avatar_image = resize_avatar(original, AVATAR_SIZE)
        print(f"✅ Avatar loaded and resized to {AVATAR_SIZE[0]}x{AVATAR_SIZE[1]}")
    
    try:
        if agent.try_safe_compile():
            print("✅ GPU Active")
    except: 
        pass
    
    # Initialize Whisper
    if WHISPER_AVAILABLE:
        print(f"\n🎤 Loading Whisper {WHISPER_MODEL_SIZE} model...")
        try:
            load_start = time.time()
            whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device="cuda",
                compute_type=WHISPER_COMPUTE_TYPE,
                download_root="./whisper_models"
            )
            load_time = time.time() - load_start
            print(f"✅ Whisper {WHISPER_MODEL_SIZE} loaded on GPU in {load_time:.2f}s")
        except Exception as e:
            print(f"⚠️ Failed to load faster-whisper: {e}")
            whisper_model = None
    
    # Check LM Studio
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:1234/v1/models", timeout=5)
            if response.status_code == 200:
                print(f"✅ LM Studio connected")
    except:
        print(f"⚠️ LM Studio not responding")
    
    # Check OpenClaw
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://127.0.0.1:18789/v1/models", timeout=5)
            if response.status_code == 200:
                print(f"✅ OpenClaw connected")
    except:
        print(f"⚠️ OpenClaw not responding (will use LM Studio as fallback)")
    
    # Check PocketTTS
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{POCKET_TTS_URL}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ PocketTTS connected on port {POCKET_TTS_URL}")
    except:
        print(f"⚠️ PocketTTS not responding")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(CHUNK_DIR, exist_ok=True)
    
    # Pre-warm voice
    await warmup_pockettts_voice(POCKET_TTS_VOICE)
    
    # Pre-warm wake word
    await warmup_wakeword()
    
    # Start GPU worker
    await start_gpu_worker()
    
    print("\n" + "="*60)
    print("✅ Server Ready!")
    print("📍 http://localhost:8002")
    print("📍 WebSocket: ws://localhost:8002/ws/realtime/{voice}")
    print(f"🎤 VAD: Aggressiveness={VAD_AGGRESSIVENESS}, Silence=900ms")
    print(f"🎯 Wake Word: {'Enabled (Hey Jarvis)' if WAKE_WORD_ENABLED else 'Disabled'}")
    print(f"📚 Wake Word Library: {'livekit-wakeword' if LIVEKIT_WAKEWORD_AVAILABLE else 'none'}")
    print(f"🤖 LLM: LM Studio (Default) + OpenClaw (Toggle)")
    print(f"🎮 GPU Worker: {'Sequential' if GPU_SEQUENTIAL_MODE else 'Parallel'} mode")
    print(f"🎬 Skip Avatar Toggle: Available (Voice/Text can bypass video)")
    print("="*60 + "\n")
    
    yield
    
    # Cleanup
    await stop_gpu_worker()
    for session in active_sessions.values():
        await session.stop()
    active_sessions.clear()

app.router.lifespan_context = lifespan


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🎬 AVATAR STUDIO - GPU SEQUENTIAL + SKIP AVATAR TOGGLE")
    print("="*60)
    print("\n✅ Whisper Small")
    print("✅ LM Studio (Gemma) - Conversational")
    print("✅ OpenClaw - Task-Oriented (Toggle)")
    print("✅ PocketTTS for voice")
    print("✅ WebRTC VAD (Voice Activity Detection)")
    print("✅ livekit-wakeword (Hey Jarvis)")
    print("✅ Real-time WebSocket streaming")
    print("✅ WebSocket LLM toggle support")
    print("✅ GPU Sequential Worker (prevents race conditions)")
    print("✅ Skip Avatar Toggle (Voice Only mode)")
    print("\n🌐 Open: http://localhost:8002")
    print("🔌 WebSocket: ws://localhost:8002/ws/realtime/jonnydepp")
    print("\n📊 Features:")
    print("   - Toggle between LM Studio and OpenClaw")
    print("   - Toggle between Avatar Video and Voice Only (TTS)")
    print("   - Avatar ALWAYS speaks only the final answer")
    print("   - GPU tasks queued sequentially (no more 100% GPU peg!)")
    print("   - Text input always available (no listening required)")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8002)