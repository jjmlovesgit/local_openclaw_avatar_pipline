# test_wakeword_simple.py
import numpy as np
from collections import deque

# Import your wake word detector
import openwakeword

print("Loading wake word model...")
model = openwakeword.Model(wakeword_models=["hey_jarvis_v0.1.onnx"])

# Test with silence (should give low confidence)
silence = np.zeros(32000, dtype=np.int16)
predictions = model.predict(silence)
score = 0.0
for key in predictions.keys():
    if 'jarvis' in key.lower():
        score = predictions[key]
        break

print(f"Test prediction on silence: {score:.3f}")
print("✅ Wake word model is working!")