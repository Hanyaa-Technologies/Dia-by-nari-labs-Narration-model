# 🎙️ Dia Narration Model
### *Advanced Text-to-Speech with Multi-Speaker Dialogue*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-Interactive-orange.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 Overview

**Dia Narration Model** is a cutting-edge text-to-speech synthesis tool built on Nari Labs' powerful **Dia TTS model** — a 1.6B parameter open-source neural network that generates incredibly realistic multi-speaker dialogue. Transform your scripts into natural-sounding audio with advanced voice conditioning and seamless speaker transitions.

### ✨ What Makes It Special?

- 🎭 **Multi-Speaker Support** — Natural dialogue with distinct voices
- 🎨 **Voice Cloning** — Condition on reference audio for custom voices  
- 🧠 **Smart Chunking** — Intelligent text processing for long narrations
- 🔗 **Seamless Audio Chaining** — Continuous output without breaks
- 🎛️ **Fine-Tuned Control** — Adjustable parameters for perfect results

---

## 🚀 Key Features

### 🔤 **Advanced Tokenization**
- Intelligent sentence splitting using NLTK
- GPT-2 tokenizer for precise chunk sizing
- Optimized processing for any text length

### 📊 **Dynamic Chunking System**
- Adaptive ~64 token chunks for optimal processing
- Smart handling of short texts (≤80 tokens)
- Prevents over-splitting while maintaining quality

### 🎵 **Seamless Audio Chaining**
- Smooth transitions between audio segments
- 2.5-second overlap for natural continuity
- No awkward pauses or breaks

### 🎤 **Voice Conditioning & Cloning**
- Upload reference audio for voice matching
- Emotion and tone preservation
- Advanced conditioning capabilities

### 🖥️ **Interactive Gradio Interface**
- User-friendly web interface
- Real-time parameter adjustment
- Professional audio controls

---

## 📋 Prerequisites

Before getting started, ensure you have:

- **Python 3.10+** installed
- **Git** for repository cloning
- **Audio drivers** for playback
- At least **4GB RAM** recommended

---

## ⚡ Quick Start

### 1️⃣ **Clone Repository**
```bash
git clone https://github.com/Hanyaa-Technologies/Dia-by-nari-labs-Narration-model.git
cd Dia-by-nari-labs-Narration-model
```

### 2️⃣ **Environment Setup**

Choose your preferred method:

#### 🐍 **Option A: Using venv + pip**
```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate        # 🐧 Linux/macOS
# OR
venv\Scripts\activate           # 🪟 Windows

# Install dependencies
pip install -r requirements.txt
```

#### 🐍 **Option B: Using conda**
```bash
# Create conda environment
conda create -n dia-tts python=3.10 -y

# Activate environment
conda activate dia-tts

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ **Launch Application**
```bash
python experiment4.py
```
Video link - https://1drv.ms/v/c/0fd555f519abe747/ET5IMBLdwqREjsFnXe-PBnQBc9IqTXnaH2qsZxcpzvSjNg?e=SQyRQN

🎉 **That's it!** Your Gradio interface will open automatically in your default browser.

---

## 📁 Project Structure

```
📦 Dia-Narration-Model/
├── 🎙️ dia/                    # Core TTS modules
├── 🐳 docker/                 # Docker configuration
├── 📝 example/                # Sample files & demos
├── ⚙️ cli.py                  # Command line interface
├── 🎵 example_prompt.mp3      # Sample audio prompt
├── 🧪 experiment4.py          # Main Gradio application
├── 📊 test.py                 # Testing utilities
├── 🖼️ horizontal.png          # Project banner
├── 📋 requirements.txt        # Python dependencies
├── 📄 README.md              # This file
├── ⚖️ LICENSE                # License information
└── 🔒 pyproject.toml         # Project configuration
```

---

## 🎯 Usage Guide

### 🖥️ **Web Interface**

1. **Launch the app**: Run `python experiment4.py`
2. **Enter your script**: Use `[S1]` and `[S2]` tags for different speakers
3. **Upload reference audio** (optional): For voice cloning
4. **Adjust parameters**: Fine-tune CFG scale, temperature, etc.
5. **Generate**: Click the magic button and listen!

### 📝 **Script Format Example**

```
[S1] Welcome to our podcast! I'm excited to discuss today's topic.
[S2] Thanks for having me! This is going to be a great conversation.
[S1] Let's dive right in. What are your thoughts on...
```

### 🎚️ **Parameter Controls**

| Parameter | Description | Recommended Range |
|-----------|-------------|-------------------|
| **CFG Scale** | Controls adherence to prompt | 1.0 - 3.0 |
| **Temperature** | Output randomness | 0.7 - 1.2 |
| **Top-p** | Nucleus sampling threshold | 0.8 - 0.95 |
| **Seed** | Reproducibility control | Any integer |

---

## 🛠️ Advanced Configuration

### 🎵 **Audio Settings**
- **Sample Rate**: 24kHz (default)
- **Bit Depth**: 16-bit
- **Format**: WAV/MP3 compatible
- **Chunk Size**: Automatically optimized

### 💾 **Memory Optimization**
- Automatic GPU/CPU detection
- Efficient batch processing  
- Memory-mapped audio loading
- Smart caching system

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 🚨 **"ModuleNotFoundError: No module named 'xxx'"**
```bash
pip install -r requirements.txt --upgrade
```

#### 🚨 **NLTK Data Missing**
```python
import nltk
nltk.download('punkt')
```

#### 🚨 **CUDA Out of Memory**
- Reduce chunk size in settings
- Close other GPU-intensive applications
- Try CPU-only mode

#### 🚨 **Audio Quality Issues**
- Check sample rate compatibility
- Verify reference audio quality
- Adjust CFG scale parameter

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. 🍴 **Fork** the repository
2. 🌟 **Create** a feature branch
3. 💻 **Commit** your changes
4. 📤 **Push** to your branch
5. 🔄 **Create** a Pull Request

### 📋 **Contribution Guidelines**
- Follow PEP 8 coding standards
- Add tests for new features
- Update documentation
- Be respectful and collaborative

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Nari Labs** for the incredible Dia TTS model
- **Hugging Face** for the Transformers library
- **Gradio** team for the amazing interface framework
- **PyTorch** community for the deep learning foundation

<div align="center">

### 🌟 **Star this project if you found it helpful!** ⭐

**Made with ❤️ by [Hanyaa Technologies](https://github.com/Hanyaa-Technologies)**

</div>
