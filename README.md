<div align="center">
<p align="center">
  <h1>IMTalker: Efficient Audio-driven Talking Face Generation with Implicit Motion Transfer</h1>


## 📖 Overview

IMTalker accepts diverse portrait styles and achieves 40 FPS for video-driven and 42 FPS for audio-driven talking-face generation when tested on an NVIDIA RTX 4090 GPU at 512 × 512 resolution. It also enables diverse controllability by allowing precise head-pose and eye-gaze inputs alongside audio

<div align="center">
  <img src="assets/teaser.png" alt="" width="1000">
</div>

## 📢 News

- **[2025.11]** 🚀 The inference code and pretrained weights are released!

## 🛠️ Installation

### 1. Environment Setup

```bash
conda create -n IMTalker python=3.10
conda activate IMTalker
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

**2. Install with pip:**

```bash
git clone https://github.com/cbsjtu01/IMTalker.git
cd IMTalker
pip install -r requirement.txt
```

## ⚡ Quick Start

You can simply run the Gradio demo to get started. The script will **automatically download** the required pretrained models to the `./checkpoints` directory if they are missing.

```bash
python app.py
```

## 📦 Model

Please download the pretrained models and place them in the `./checkpoints` directory.

| Component               | Checkpoint             | Description             |                                   Download                                   |
| :---------------------- | :--------------------- | :---------------------- | :---------------------------------------------------------------------------: |
| **Audio Encoder** | `wav2vec2-base-960h` | Wav2Vec2 Base model     | [🤗 Link](https://huggingface.co/cbsjtu01/IMTalker/tree/main/wav2vec2-base-960h) |
| **Generator**     | `generator.ckpt`     | Flow Matching Generator |   [🤗 Link](https://huggingface.co/cbsjtu01/IMTalker/blob/main/generator.ckpt)   |
| **Renderer**      | `renderer.ckpt`      | IMT Renderer            |   [🤗 Link](https://huggingface.co/cbsjtu01/IMTalker/blob/main/renderer.ckpt)   |

### 📂 Directory Structure

Ensure your file structure looks like this after downloading:

```text
./checkpoints
├── renderer.ckpt                     # The main renderer
├── generator.ckpt                    # The main generator
└── wav2vec2-base-960h/               # Audio encoder folder
    ├── config.json
    ├── pytorch_model.bin
    └── ...
```

# 🎭 Avatar Studio: High-Fidelity Real-Time AI Avatar

**Avatar Studio** is a low-latency, stability-first pipeline for generating interactive digital humans. By combining **Flow Matching Transformers (FMT)** with a high-resolution **IMTRenderer**, this project achieves synchronized lip-sync and natural macro-motions (head nods, blinks) in real-time.

## ✨ Key Features

* **Flow Matching Transformer (FMT):** Predicts natural facial trajectories with high temporal consistency.
* **Low-Latency WebRTC Engine:** Optimized for sub-1.5s Time-to-First-Frame (TTFF) using chunked progressive rendering.
* **Hybrid Motion Driver:** Blend audio-driven lip-sync with video-driven head movements for realistic presence.
* **Production-Ready Backend:** Built-in support for **FastAPI**, **WebRTC VAD**, and **LiveKit Wake Word** ("Hey Jarvis").
* **Hardware Optimized:** Native support for `torch.compile` and **NVIDIA Blackwell/Grace** architectures, with fallbacks for consumer RTX cards.

---

## 🚀 Quick Start

### 1. Prerequisites

* **Hardware:** NVIDIA GPU (24GB+ VRAM recommended for local inference).
* **Environment:** Python 3.11+.
* **FFmpeg:** Required for high-speed video encoding.

### 2. Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/avatar-studio.git](https://github.com/your-username/avatar-studio.git)
cd avatar-studio

# Create and activate environment
conda create -n avatar-env python=3.11
conda activate avatar-env

# Install dependencies
pip install -r requirements.txt
```

📜 ***Citation***

If you find our work useful for your research, please consider citing:

```bibtex
@article{imtalker2025,
  title={IMTalker: Efficient Audio-driven Talking Face Generation with Implicit Motion Transfer},
  author={Bo, Chen and Tao, Liu and Qi, Chen and  Xie, Chen and  Zilong Zheng}, 
  journal={arXiv preprint arXiv:2511.22167},
  year={2025}
}
```

## 🙏 Acknowledgement

We express our sincerest gratitude to the excellent previous works that inspired this project:

- **[IMF](https://github.com/ueoo/IMF)**: We adapted the framework and training pipeline from IMF and its reproduction code [IMF](https://github.com/johndpope/IMF).
- **[FLOAT](https://github.com/deepbrainai-research/float)**: We referenced the model architecture and implementation of Float for our generator.
- **[Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h)**: We utilized Wav2Vec as our audio encoder.
- **[Face-Alignment](https://github.com/1adrianb/face-alignment)**: We used FaceAlignment for cropping images and videos.
