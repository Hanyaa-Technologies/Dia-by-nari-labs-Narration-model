# Dia-by-nari-labs-Narration-model  
# Narration Synthesis (Dia-based TTS)

## Project Overview

This tool builds on Nari Labs' Dia text-to-speech model, a 1.6B-parameter open-source TTS model. Dia generates highly realistic multi-speaker dialogue from transcripts and can be conditioned on example audio to control voice and tone. The Gradio interface here lets users enter narration scripts (using `[S1]` / `[S2]` tags for speakers) and optional reference audio prompts. The input text is automatically tokenized and split into manageable chunks, and the resulting audio segments are chained together for smooth, continuous output.

## Key Features

* **Tokenization**: Splits input into sentences (using NLTK) and counts tokens with a GPT-2 tokenizer to guide chunk sizes.
* **Dynamic Chunking**: Groups sentences into chunks of ~64 tokens (adaptive to text length). Very short texts (≤80 tokens) are processed as a single chunk to avoid over-splitting.
* **Audio Chaining**: Ensures continuity by feeding the last ~2.5 seconds of each audio chunk as a prompt into the next chunk's generation.
* **Audio Prompt / Voice Cloning**: Supports an optional audio file (plus its transcript) to guide voice style and emotion. This leverages Dia's ability to condition on example audio [^2].
* **Interactive UI**: A custom Gradio app with fields for text input, audio prompt, random seed, and sliders (CFG scale, temperature, top-p, etc.) for fine-tuning. Users click **Generate Audio** to synthesize speech.

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/Hanyaa-Technologies/Dia-by-nari-labs-Narration-model.git
    cd Text-To-Speech
    ```

2. **Set Up a Virtual Environment**

    Choose one of the following methods:

    #### 👉 Using `venv` + `pip`

    ```bash
    python -m venv venv
    source venv/bin/activate        # Linux/macOS
    venv\Scripts\activate           # Windows
    
    pip install -r requirements.txt
    ```

    #### 👉 Using `conda**

    ```bash
    conda create -n env_name python=3.10 -y
    conda activate env_name
    
    pip install -r requirements.txt
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    This installs PyTorch, Transformers, Gradio, SoundFile, scipy, NLTK, and other required libraries.

4. **NLTK data**: Ensure NLTK's Punkt tokenizer is available. The script will auto-download it if missing.

## Directory Structure

The project follows this directory structure:
  ```
      .
      ├── dia/
      ├── docker/
      ├── example/
      ├── .gitignore
      ├── .python-version
      ├── LICENSE
      ├── README.md
      ├── cli.py
      ├── example_prompt.mp3
      ├── experiment.py
      ├── experiment2.py
      ├── experiment3.py
      ├── experiment4.py
      ├── horizontal.png
      ├── pyproject.toml
      ├── test.py
      └── uv.lock
  ```
  ---
- `experiment4.py`: The main script to run the Gradio app  
- `requirements.txt`: Lists all dependencies required for the project  

## Usage

Run the Gradio app with:

```bash
python experiment4.py

## Usage

  Run the Gradio app with:
  ```bash
  python experiment4.py
```
