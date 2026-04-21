# 🎬 IMTalker Avatar Studio – Setup Guide (RTX 30/40 Series)

![License](https://img.shields.io/badge/license-MIT-blue)
![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-green)
![GPU](https://img.shields.io/badge/GPU-RTX%2030%2F40-orange)

---

## 🚀 Overview

Welcome! This guide will help you install and run the complete Avatar Studio system on your Windows computer.

You will create an AI-powered avatar that can:
- Speak
- Listen for the wake word **"Hey Jarvis"**
- Run in an always‑on‑top desktop window

**⏱️ Estimated Time:** 30–45 minutes  

---

## 📸 Preview 
<img width="617" height="653" alt="image" src="https://github.com/user-attachments/assets/c0080be2-49b6-4f21-b9b6-ef572a4d4e9e" />

```md



```

---

## 💻 Requirements

- Windows 10 or 11  
- NVIDIA RTX 30‑ or 40‑series GPU (**8 GB VRAM+**)  
- 20 GB free disk space  
- Microphone  

> ⚠️ RTX 50-series users → see `README_BLACKWELL.md`

---

## 📦 System Components

1. 🧠 Python Backend (IMTalker)
2. 🔊 PocketTTS Voice Server
3. 🪟 Electron Frontend

---

## 🧰 Prerequisites

<details>
<summary>Click to expand</summary>

| Tool | Link |
|------|------|
| Miniconda | https://docs.conda.io/en/latest/miniconda.html |
| Git | https://git-scm.com/download/win |
| Node.js | https://nodejs.org/ |
| LM Studio | https://lmstudio.ai/ |

👉 Restart your PC after installing

</details>

---

# 🧠 Part 1: Backend Setup

<details>
<summary>Full Instructions</summary>

### Open Terminal
Press `Win + R` → type `powershell`

### Create Environment
```powershell
conda create -n avatar python=3.11 -y
conda activate avatar
```

### Install PyTorch
```powershell
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify GPU
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

### Clone Project
```powershell
git clone https://github.com/jjmlovesgit/local_openclaw_avatar_pipline
cd imtalker-avatar-studio
pip install -r requirements.txt
```

### Models
```
./checkpoints/
├── renderer.ckpt
├── generator.ckpt
└── wav2vec2-base-960h/
```

### Wake Word
```powershell
python -c "import openwakeword; openwakeword.utils.download_models()"
```

### Run Server
```powershell
python python_avatar_server_electron.py
```

➡ Open: http://localhost:8002

</details>

---

# 🔊 Part 2: Voice

<details>
<summary>Setup PocketTTS</summary>

```powershell
conda activate avatar
pip install pocket-tts
pocket-tts serve
```

➡ Test: http://localhost:8000

</details>

---

# 🪟 Part 3: Electron UI

<details>
<summary>Run App</summary>

```powershell
cd electron-app
npm install
npm start
```

</details>

---

## 🎤 Features

- Wake word: **Hey Jarvis**
- Voice + text input
- Floating desktop avatar
- Real-time AI responses

---

## ⚙️ Troubleshooting

<details>
<summary>Common Issues</summary>

| Problem | Fix |
|--------|-----|
| CUDA false | Update drivers |
| Port in use | Change port |
| No audio | Check speakers |
| Wake word fails | Lower threshold |

</details>

---

## 📚 Citation

```bibtex
@article{imtalker2025,
  title={IMTalker: Efficient Audio-driven Talking Face Generation},
  author={Bo, Chen et al.},
  year={2025}
}
```

---

## 📁 Project Structure

```
imtalker-avatar-studio/
├── python_avatar_server_electron.py
├── checkpoints/
├── electron-app/
└── README.md
```

---

## ⭐ Support

Star the repo if useful ⭐
