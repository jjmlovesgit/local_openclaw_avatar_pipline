"""
End-to-End Wake Word Setup for Avatar System
Uses openwakeword's built-in model downloader
"""

import os
import sys
import wave
import json
import asyncio
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime

# ============================================
# STEP 1: DOWNLOAD THE MODEL USING OPENWAKEWORD
# ============================================

def download_jarvis_model():
    """Download the hey_jarvis wake word model using openwakeword"""
    print("\n" + "="*60)
    print("📥 STEP 1: Downloading Wake Word Model")
    print("="*60)
    
    try:
        import openwakeword
        
        # Download all pre-trained models
        print("Downloading pre-trained models from openwakeword...")
        openwakeword.utils.download_models()
        
        # Find the downloaded models
        package_dir = os.path.dirname(openwakeword.__file__)
        models_dir = os.path.join(package_dir, "resources", "models")
        
        print(f"\nModels directory: {models_dir}")
        
        # Look for hey_jarvis model
        jarvis_model = None
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if 'jarvis' in file.lower() and file.endswith('.onnx'):
                    jarvis_model = os.path.join(models_dir, file)
                    print(f"✅ Found: {file}")
                    break
        
        if jarvis_model:
            # Copy to current directory for easy access
            import shutil
            local_path = "hey_jarvis_v0.1.onnx"
            shutil.copy2(jarvis_model, local_path)
            print(f"✅ Copied to: {local_path}")
            return local_path
        else:
            print("❌ Could not find hey_jarvis model")
            print("\nAvailable models:")
            for file in os.listdir(models_dir):
                if file.endswith('.onnx'):
                    print(f"  - {file}")
            return None
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

# ============================================
# STEP 2: TEST WITH SAMPLE AUDIO
# ============================================

def create_test_audio():
    """Create a test audio file"""
    print("\n" + "="*60)
    print("🎤 STEP 2: Creating Test Audio")
    print("="*60)
    
    test_file = "test_audio.wav"
    
    # Create a simple 3-second audio for testing
    sample_rate = 16000
    duration = 3.0
    
    # Generate some test audio (silence with a tone burst)
    audio = np.zeros(int(sample_rate * duration), dtype=np.int16)
    
    with wave.open(test_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    
    print(f"✅ Created test audio: {test_file}")
    return test_file

# ============================================
# STEP 3: WAKE WORD DETECTOR CLASS
# ============================================

class JarvisWakeWordDetector:
    """Complete wake word detector for 'Hey Jarvis'"""
    
    def __init__(self, model_path="hey_jarvis_v0.1.onnx", threshold=0.5):
        print("\n" + "="*60)
        print("🤖 STEP 3: Initializing Wake Word Detector")
        print("="*60)
        
        self.threshold = threshold
        self.model_path = model_path
        
        # Use openwakeword since we know it's installed
        try:
            import openwakeword
            self.model = openwakeword.Model(wakeword_models=[model_path])
            self.use_openwakeword = True
            print(f"✅ Using openwakeword with model: {model_path}")
        except Exception as e:
            raise ImportError(f"Failed to load openwakeword: {e}")
        
        # Audio buffer (2 seconds at 16kHz)
        self.buffer_duration = 2.0
        self.buffer_size = int(16000 * self.buffer_duration)
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # Detection state
        self.is_listening = True
        self.last_detection_time = 0
        self.debounce_seconds = 2.0
        self.detection_count = 0
        
        # For streaming prediction
        self.frame_counter = 0
        
        print(f"✅ Threshold: {threshold}")
        print(f"✅ Buffer: {self.buffer_duration}s ({self.buffer_size} samples)")
        print(f"✅ Debounce: {self.debounce_seconds}s")
        
    def process_audio_chunk(self, audio_chunk):
        """Process a chunk of audio and check for wake word"""
        # Convert to numpy if needed
        if isinstance(audio_chunk, bytes):
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
        else:
            audio_np = audio_chunk
            
        # Add to buffer
        self.audio_buffer.extend(audio_np)
        
        # Only predict every 10 frames to save CPU
        self.frame_counter += 1
        if self.frame_counter % 10 != 0:
            return None
        
        # Check if we have enough audio
        if len(self.audio_buffer) < self.buffer_size:
            return None
        
        # Get the last 2 seconds
        audio_window = np.array(list(self.audio_buffer))[-self.buffer_size:]
        
        # Get prediction from openwakeword
        predictions = self.model.predict(audio_window)
        
        # Extract score for hey_jarvis
        score = 0.0
        for key in predictions.keys():
            if 'jarvis' in key.lower():
                score = predictions[key]
                break
        
        # Check threshold and debounce
        current_time = datetime.now().timestamp()
        if score > self.threshold:
            if current_time - self.last_detection_time > self.debounce_seconds:
                self.last_detection_time = current_time
                self.detection_count += 1
                return {
                    "detected": True,
                    "word": "hey_jarvis",
                    "confidence": float(score),
                    "timestamp": current_time,
                    "count": self.detection_count
                }
        
        return {
            "detected": False,
            "confidence": float(score)
        }
    
    def process_wav_file(self, wav_path):
        """Process an entire WAV file for testing"""
        print(f"\n📁 Processing: {wav_path}")
        
        with wave.open(wav_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())
            
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Duration: {len(audio_data)/2/sample_rate:.2f}s")
        
        # Process in chunks
        chunk_size = 1600  # 100ms at 16kHz
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        detections = []
        for i in range(0, len(audio_np) - chunk_size, chunk_size):
            chunk = audio_np[i:i+chunk_size]
            result = self.process_audio_chunk(chunk)
            
            if result and result.get("detected"):
                detections.append(result)
                print(f"   🎯 Detection #{result['count']}: {result['confidence']:.3f} at {i/sample_rate:.2f}s")
        
        return detections

# ============================================
# STEP 4: ASYNC LISTENER FOR WEBSOCKET
# ============================================

class AsyncJarvisListener:
    """Async listener for WebSocket integration"""
    
    def __init__(self, websocket=None, threshold=0.5):
        self.detector = JarvisWakeWordDetector(threshold=threshold)
        self.websocket = websocket
        self.is_activated = False
        self.activation_timeout = 10.0  # Auto-deactivate after 10s
        self.last_activation = 0
        
    async def process_frame(self, audio_bytes):
        """Process a single audio frame from WebSocket"""
        result = self.detector.process_audio_chunk(audio_bytes)
        
        if result and result.get("detected"):
            self.is_activated = True
            self.last_activation = datetime.now().timestamp()
            
            if self.websocket:
                await self.websocket.send_json({
                    "type": "wake_word_detected",
                    "word": "hey_jarvis",
                    "confidence": result["confidence"],
                    "activated": True
                })
            
            return {"status": "activated", "confidence": result["confidence"]}
        
        # Check for auto-deactivation
        if self.is_activated:
            if datetime.now().timestamp() - self.last_activation > self.activation_timeout:
                self.is_activated = False
                if self.websocket:
                    await self.websocket.send_json({
                        "type": "deactivated",
                        "reason": "timeout"
                    })
                return {"status": "deactivated"}
            else:
                return {"status": "active"}
        
        return {"status": "listening", "confidence": result["confidence"] if result else 0}

# ============================================
# STEP 5: MAIN SETUP AND TEST
# ============================================

def list_available_models():
    """List all available pre-trained models"""
    try:
        import openwakeword
        package_dir = os.path.dirname(openwakeword.__file__)
        models_dir = os.path.join(package_dir, "resources", "models")
        
        if os.path.exists(models_dir):
            print("\n📋 Available pre-trained models:")
            for file in sorted(os.listdir(models_dir)):
                if file.endswith('.onnx'):
                    size = os.path.getsize(os.path.join(models_dir, file))
                    print(f"   - {file} ({size:,} bytes)")
        else:
            print("Models directory not found. Run download_jarvis_model() first.")
    except Exception as e:
        print(f"Error listing models: {e}")

def main():
    """Main setup and test function"""
    print("\n" + "="*60)
    print("🚀 WAKE WORD SETUP COMPLETE")
    print("="*60)
    
    # Download model
    model_path = download_jarvis_model()
    if not model_path:
        print("\n❌ Failed to download model")
        print("\nTrying alternative: list available models...")
        list_available_models()
        return None
    
    # Initialize detector
    try:
        detector = JarvisWakeWordDetector(model_path, threshold=0.5)
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return None
    
    # Test with sample audio
    test_file = create_test_audio()
    detector.process_wav_file(test_file)
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE - Ready to use!")
    print("="*60)
    print("\n📝 Integration Example for your FastAPI app:")
    print("""
# In your main FastAPI app:
from setup_wakeword import AsyncJarvisListener

@app.websocket("/ws/realtime/{voice}")
async def websocket_realtime(websocket: WebSocket, voice: str):
    await websocket.accept()
    listener = AsyncJarvisListener(websocket=websocket, threshold=0.5)
    
    session_id = str(uuid.uuid4())[:8]
    session = RealtimeSession(session_id, voice, websocket)
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            # Check wake word first
            result = await listener.process_frame(data)
            
            if result["status"] == "activated":
                # Wake word detected! Now process with VAD
                await process_avatar_response(data, session)
            elif result["status"] == "active":
                # Already activated, keep processing
                await process_avatar_response(data, session)
    
    except WebSocketDisconnect:
        print(f"Disconnected: {session_id}")
    """)
    
    return detector

# ============================================
# RUN SETUP
# ============================================

if __name__ == "__main__":
    # Run setup
    detector = main()
    
    if detector:
        print("\n" + "="*60)
        print("🎉 READY TO INTEGRATE WITH YOUR AVATAR SYSTEM!")
        print("="*60)
        print("\nQuick test - say 'Hey Jarvis' to activate!")
        print("Then speak your command normally.")
        
        # Optional: Simple microphone test
        try:
            import pyaudio
            print("\n🎤 Starting microphone listener...")
            print("Speak now... (Press Ctrl+C to stop)\n")
            
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=480
            )
            
            while True:
                data = stream.read(480, exception_on_overflow=False)
                result = detector.process_audio_chunk(data)
                
                if result and result.get("detected"):
                    print(f"\n🎯 WAKE WORD DETECTED! (Confidence: {result['confidence']:.3f})")
                    print("   Avatar activated - listening for command...\n")
                    
        except ImportError:
            print("\n⚠️ pyaudio not installed - skipping microphone test")
            print("   Install with: pip install pyaudio")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            if 'p' in locals():
                p.terminate()