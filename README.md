---
# 🎬 IMTalker Avatar Studio – Setup Guide (RTX 30/40/50 Series)

![License](https://img.shields.io/badge/license-MIT-blue)
![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-green)
![GPU](https://img.shields.io/badge/GPU-RTX%2030%2F40%2F50-orange)

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
<img width="1791" height="1083" alt="image" src="https://github.com/user-attachments/assets/b7c661e5-b768-4257-a01d-c004c3283724" />
Link:  
https://youtu.be/CIhQwnUNoAk

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
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
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

Download each file individually and place them in the correct location:

Step 1: Create the checkpoints directory

powershell
mkdir checkpoints
cd checkpoints
Step 2: Download the generator and renderer

Download these two files from the Hugging Face page:

generator.ckpt (621 MB)

renderer.ckpt (2.12 GB)

Click each link, then click the "Download" button. Save both files directly into:

text
B:\avatar_pipeline\local_openclaw_avatar_pipline\checkpoints\
Step 3: Download the wav2vec2 model

Create the wav2vec2 folder and download its files:

powershell
mkdir wav2vec2-base-960h
cd wav2vec2-base-960h
Download these three files from the wav2vec2-base-960h folder:

config.json

pytorch_model.bin

preprocessor_config.json (if present)

Save them into:

text
B:\avatar_pipeline\local_openclaw_avatar_pipline\checkpoints\wav2vec2-base-960h\

Since you're in a Python environment, this is even more reliable:

cmd
python -c "import urllib.request; print('Downloading generator.ckpt...'); urllib.request.urlretrieve('https://huggingface.co/cbsjtu01/IMTalker/resolve/main/generator.ckpt', 'checkpoints/generator.ckpt')"
Then do the same for the other files:

cmd
python -c "import urllib.request; print('Downloading renderer.ckpt (2.12 GB)...'); urllib.request.urlretrieve('https://huggingface.co/cbsjtu01/IMTalker/resolve/main/renderer.ckpt', 'checkpoints/renderer.ckpt')"
cmd
python -c "import urllib.request, os; os.makedirs('checkpoints/wav2vec2-base-960h', exist_ok=True); print('Downloading config.json...'); urllib.request.urlretrieve('https://huggingface.co/cbsjtu01/IMTalker/resolve/main/wav2vec2-base-960h/config.json', 'checkpoints/wav2vec2-base-960h/config.json')"
cmd
python -c "import urllib.request; print('Downloading pytorch_model.bin...'); urllib.request.urlretrieve('https://huggingface.co/cbsjtu01/IMTalker/resolve/main/wav2vec2-base-960h/pytorch_model.bin', 'checkpoints/wav2vec2-base-960h/pytorch_model.bin')"

python -c "import urllib.request; print('Downloading preprocessor_config.json...'); urllib.request.urlretrieve('https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/preprocessor_config.json', 'checkpoints/wav2vec2-base-960h/preprocessor_config.json')"

(avatar) B:\avatar_pipeline\local_openclaw_avatar_pipline>dir checkpoints
 Volume in drive B is Secondary
 Volume Serial Number is 0E23-5AF2

 Directory of B:\avatar_pipeline\local_openclaw_avatar_pipline\checkpoints

04/22/2026  09:00 AM    <DIR>          .
04/22/2026  08:52 AM    <DIR>          ..
04/22/2026  08:59 AM       621,318,134 generator.ckpt
04/22/2026  09:00 AM     2,121,068,003 renderer.ckpt
04/22/2026  09:00 AM    <DIR>          wav2vec2-base-960h
               2 File(s)  2,742,386,137 bytes
               3 Dir(s)  513,914,503,168 bytes free

(avatar) B:\avatar_pipeline\local_openclaw_avatar_pipline>dir checkpoints\wav2vec2-base-960h
 Volume in drive B is Secondary
 Volume Serial Number is 0E23-5AF2

 Directory of B:\avatar_pipeline\local_openclaw_avatar_pipline\checkpoints\wav2vec2-base-960h

04/22/2026  09:00 AM    <DIR>          .
04/22/2026  09:00 AM    <DIR>          ..
04/22/2026  09:00 AM             1,596 config.json
04/22/2026  09:00 AM       377,667,514 pytorch_model.bin
               2 File(s)    377,669,110 bytes
               2 Dir(s)  513,914,503,168 bytes free

(avatar) B:\avatar_pipeline\local_openclaw_avatar_pipline>


checkpoints\
├── generator.ckpt          (621 MB)
├── renderer.ckpt           (2.12 GB)
└── wav2vec2-base-960h\
    ├── config.json         (1.6 KB)
    └── pytorch_model.bin   (377 MB)


```
mkdir generated_clips
mkdir uploads
mkdir chunks

### Run Server
```powershell
python python_avatar_server_electron.py
```

➡ Open: http://localhost:8002

</details>

---

# 🔊 Part 2: Voice   unzip the pocket-tts-openai-streaming-server_release_v1.zip

<details>
<summary>Setup PocketTTS</summary>

```powershell
conda activate avatar (Set up your python 311 with requirements .txt from the zip)
pip install pocket-tts
pocket-tts serve
```

➡ Test: http://localhost:8000

</details>

---

# 🪟 Part 3: Electron UI - unzip the electronJon_release_v1.zip 

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
