import argparse
import tempfile
import time
import random
import io
import contextlib
from pathlib import Path
from typing import Optional, Tuple, List
from scipy.signal import resample
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import math
from dia.model import Dia
from transformers import GPT2TokenizerFast
import nltk
from tqdm.auto import tqdm # Keep this import for Gradio to track

# --- Global Setup ---
parser = argparse.ArgumentParser(description="Gradio interface for Nari TTS")
parser.add_argument("--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')")
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
args = parser.parse_args()

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')
    print("Download complete.")

# Determine device
if args.device:
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# --- Load Model and a Compatible Tokenizer for Counting ---
print("Loading Nari model and a compatible tokenizer...")
try:
    # Load the Dia model
    MODEL_NAME = "nari-labs/Dia-1.6B"
    dtype_map = {"cpu": "float32", "mps": "float32", "cuda": "float16"}
    dtype = dtype_map.get(device.type, "float16")
    print(f"Using device: {device}, attempting to load model with {dtype}")
    model = Dia.from_pretrained(MODEL_NAME, compute_dtype=dtype, device=device)

    # Load a standard, reliable tokenizer for estimating token counts.
    TOKENIZER_NAME = "gpt2"
    print(f"Loading '{TOKENIZER_NAME}' tokenizer for chunking estimation...")
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_NAME)

except Exception as e:
    print(f"Error loading Nari model or tokenizer: {e}")
    raise


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Reinstated Robust Token-Based Chunking ---
def split_and_tokenize_sentences(text: str) -> List[Tuple[str, int]]:
    """
    Splits text into sentences using NLTK and calculates the token count for each
    using our compatible (GPT-2) tokenizer.
    """
    sentences = nltk.sent_tokenize(text)
    sentence_tuples = []
    for sent in sentences:
        if not sent.strip():
            continue
        # Use the gpt2 tokenizer to get an estimated token count
        token_ids = tokenizer.encode(sent)
        sentence_tuples.append((sent, len(token_ids)))
    return sentence_tuples


def chunk_text(text: str) -> tuple[list[str], list[int]]:
    """
    Splits text into sentences and dynamically groups them into uniform chunks.
    
    NEW: If the total text is below a safe token threshold, it's treated as a
    single chunk to avoid unnecessary splitting of short paragraphs.
    
    For longer texts, it uses a logarithmic approach to determine an adaptive
    chunk size, creating smaller chunks for short texts and slightly larger
    (but still safe) chunks for very long texts.
    """
    sentence_tuples = split_and_tokenize_sentences(text)
    if not sentence_tuples:
        return [], []

    total_tokens = sum(tc for _, tc in sentence_tuples)

    # --- ADDED: Do-Not-Chunk Threshold ---
    # If the total text is short enough to be processed safely in one go,
    # don't chunk it at all. 150 is a conservative and safe value.
    DO_NOT_CHUNK_THRESHOLD = 80
    if total_tokens <= DO_NOT_CHUNK_THRESHOLD:
        print(f"(Text is short enough ({total_tokens} tokens), processing as a single chunk.)")
        # Return the entire text as a single chunk
        return [" ".join(s for s, _ in sentence_tuples)], [total_tokens]
    # --- End of Threshold Logic ---


    IDEAL_CHUNK_TOKENS = 64
 
    num_chunks = max(1, math.ceil(total_tokens / IDEAL_CHUNK_TOKENS))
    dynamic_target_size = math.ceil(total_tokens / num_chunks)

    chunks, counts = [], []
    current_chunk_sentences, current_tokens = [], 0

    for sent, tc in sentence_tuples:
        if tc > dynamic_target_size:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                counts.append(current_tokens)
                current_chunk_sentences, current_tokens = [], 0
            chunks.append(sent)
            counts.append(tc)
            continue
        if current_chunk_sentences and current_tokens + tc > dynamic_target_size:
            chunks.append(" ".join(current_chunk_sentences))
            counts.append(current_tokens)
            current_chunk_sentences, current_tokens = [sent], tc
        else:
            current_chunk_sentences.append(sent)
            current_tokens += tc

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
        counts.append(current_tokens)

    return chunks, counts


def run_inference(
    text_input: str,
    audio_prompt_text_input: str,
    audio_prompt_input: Optional[Tuple[int, np.ndarray]],
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    speed_factor: float,
    seed: Optional[int] = None,
    # MODIFIED: Add the progress argument for Gradio UI progress bar
    progress=gr.Progress(track_tqdm=True),
):
    global model, device, tokenizer
    console_output_buffer = io.StringIO()
    
    if not text_input or text_input.isspace():
        raise gr.Error("Text input cannot be empty.")

    with contextlib.redirect_stdout(console_output_buffer):
        if audio_prompt_input and (not audio_prompt_text_input or audio_prompt_text_input.isspace()):
            raise gr.Error("Audio Prompt Text input cannot be empty when an audio prompt is provided.")

        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)
            print(f"\nNo seed provided, generated random seed: {seed}\n")
        else:
            print(f"\nUsing user-selected seed: {seed}\n")
        set_seed(seed)

        initial_audio_prompt_path = None
        try:
            if audio_prompt_input is not None:
                sr, audio_data = audio_prompt_input
                if audio_data is None or audio_data.size == 0 or audio_data.max() == 0:
                    gr.Warning("Audio prompt seems empty or silent, ignoring prompt.")
                else:
                    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f_audio:
                        initial_audio_prompt_path = f_audio.name
                        if np.issubdtype(audio_data.dtype, np.integer):
                            max_val = np.iinfo(audio_data.dtype).max
                            audio_data = audio_data.astype(np.float32) / max_val
                        if audio_data.ndim > 1:
                            audio_data = np.mean(audio_data, axis=-1)
                        sf.write(initial_audio_prompt_path, audio_data, sr, subtype="FLOAT")
                        print(f"Created temporary audio prompt file: {initial_audio_prompt_path}")

            print("Step 1: Dynamically grouping text into uniform, token-aware chunks.")
            chunks, token_counts = chunk_text(text_input.strip())
            print(f"Successfully created {len(chunks)} chunks.")

            audio_segments = []
            previous_chunk_audio = None
            start_time = time.time()
            
            chunk_iterator = tqdm(zip(chunks, token_counts), total=len(chunks), desc="Generating Audio Chunks")

            for idx, (chunk, T_chunk) in enumerate(chunk_iterator):
                max_tokens_for_chunk = model.config.data.audio_length
                temp_chain_prompt_path = None
                
                try:
                    prompt_for_this_chunk = None
                    text_for_this_chunk = chunk.strip()
                    
                    print(f"Generating chunk {idx+1}/{len(chunks)} ({T_chunk} estimated tokens)...")

                    if idx == 0:
                        prompt_for_this_chunk = initial_audio_prompt_path
                        if prompt_for_this_chunk:
                            text_for_this_chunk = (audio_prompt_text_input + "\n" + chunk).strip()
                    else:
                        if previous_chunk_audio is not None and len(previous_chunk_audio) > 0:
                            CHAIN_PROMPT_DURATION_S = 2.5
                            sample_rate = 44100
                            prompt_samples = int(CHAIN_PROMPT_DURATION_S * sample_rate)
                            audio_prompt_slice = previous_chunk_audio[-prompt_samples:]
                            with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f_chain:
                                temp_chain_prompt_path = f_chain.name
                                sf.write(temp_chain_prompt_path, audio_prompt_slice, sample_rate, subtype="FLOAT")
                                prompt_for_this_chunk = temp_chain_prompt_path
                    
                    with torch.inference_mode():
                        generated_chunk_audio = model.generate(
                            text_for_this_chunk,
                            max_tokens=max_tokens_for_chunk, cfg_scale=cfg_scale, temperature=temperature,
                            top_p=top_p, cfg_filter_top_k=cfg_filter_top_k, use_torch_compile=False,
                            audio_prompt=prompt_for_this_chunk,
                        )
                    
                    if generated_chunk_audio is not None and len(generated_chunk_audio) > 0:
                        audio_segments.append(generated_chunk_audio)
                        previous_chunk_audio = generated_chunk_audio
                    else:
                        previous_chunk_audio = None
                finally:
                    if temp_chain_prompt_path and Path(temp_chain_prompt_path).exists():
                        Path(temp_chain_prompt_path).unlink()

            if not audio_segments:
                output_audio = (44100, np.zeros(1, dtype=np.int16))
                gr.Warning("Generation produced no valid audio output.")
            else:
                print("Step 2: Merging generated audio chunks.")
                output_audio_np = np.concatenate(audio_segments)
                output_sr = 44100
                
                original_len = len(output_audio_np)
                speed_factor = max(0.1, min(speed_factor, 5.0))
                target_len = int(original_len / speed_factor)
                if target_len != original_len and target_len > 0:
                    resampled_audio_np = resample(output_audio_np, target_len)
                else:
                    resampled_audio_np = output_audio_np
                
                final_audio_int16 = (np.clip(resampled_audio_np, -1.0, 1.0) * 32767).astype(np.int16)
                output_audio = (output_sr, final_audio_int16)

            end_time = time.time()
            print(f"\nGeneration finished in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            raise gr.Error(f"Inference failed: {e}")
        finally:
            if initial_audio_prompt_path and Path(initial_audio_prompt_path).exists():
                try: Path(initial_audio_prompt_path).unlink()
                except OSError as e: print(f"Warning: Error deleting temp file {initial_audio_prompt_path}: {e}")

        console_output = console_output_buffer.getvalue()

    return output_audio, seed, console_output


# --- Create Gradio Interface ---
css = """
gradio-app {
    background-color: #1e747c !important; /* Main UI background */
    padding: 1px 0; /* Ensures background coverage */
}

#app-content-blob {
    max-width: 1000px; /* Max width of the white blob */
    margin-top: 50px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    margin-bottom: 140px !important;
    background-color: rgba(255, 255, 255, 0.80) !important;
    padding: 20px 30px 30px 30px !important; /* Padding inside the blob */
    border-radius: 20px !important; /* Rounded corners for the blob */
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25) !important; /* Shadow for the blob */
}

#main-title h1 {
    color: #1e747c !important; /* Main title color */
    text-align: center !important;
    margin-top: 10px !important;
    margin-bottom: 25px !important; /* Space below title */
    font-size: 2.2em !important; /* Larger title */
    font-weight: 600 !important;
}

/* ---- BUTTON STYLING ---- */
.gr-button-primary,
#app-content-blob button.primary {
    background: #1e747c !important; /* Primary button background */
    color: white !important;
    border: 1px solid #155e63 !important; /* Darker shade for border */
    border-radius: 8px !important; /* Rounded button corners */
    font-weight: bold !important; /* Ensured text is bold */
    padding: 10px 15px !important; /* Button padding */
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    transition: background-color 0.2s ease, border-color 0.2s ease !important;
}
.gr-button-primary:hover,
#app-content-blob button.primary:hover {
    background: #155e63 !important; /* Darker shade on hover */
    border-color: #0e484d !important; /* Even darker for hover border */
}

/* ---- ACCORDION STYLING ---- */
.gr-accordion > .gr-block > .label-wrap { /* Accordion header */
    background-color: #1e747c !important; /* Accordion header background */
    border-radius: 10px 10px 0 0 !important;
    border: 1px solid #155e63 !important;
    border-bottom: none !important;
    padding: 12px 15px !important;
    margin-bottom: 0 !important;
}
.gr-accordion > .gr-block > .label-wrap span { /* Accordion title text */
    color: white !important; /* Accordion header text to white */
    font-weight: 600 !important; /* Kept at 600, can be 'bold' if desired */
    font-size: 1.05em !important;
}
.gr-accordion > .gr-block.gr-box { /* Accordion content area */
    border: 1px solid #d0e0e0 !important;
    border-top: none !important; /* Header defines top separation */
    border-radius: 0 0 10px 10px !important;
    padding: 15px !important;
    background-color: #fdfdfd !important; /* Slightly off-white */
}

/* ---- COMPONENT LABEL "PILLS" ---- */
#app-content-blob span[data-testid="block-info"] {
    background-color: #1e747c !important;
    color: white !important;
    padding: 0.3rem 0.7rem !important;
    border-radius: 0.375rem !important;
    margin-bottom: 8px !important;
    display: inline-block !important;
    line-height: 1.5 !important;
    font-weight: bold !important;
    border: none !important;
}

#app-content-blob label[data-testid="block-label"] {
    background-color: #1e747c !important;
    color: white !important;
    padding: 0.3rem 0.7rem !important;
    border-radius: 0.375rem !important;
    margin-bottom: 8px !important;
    display: inline-flex !important;
    align-items: center !important;
    line-height: 1.5 !important;
    font-weight: bold !important;
    border: none !important;
}

#app-content-blob label[data-testid="block-label"] > span > svg {
    stroke: white !important;
    margin-right: 0.3rem;
    width: 1em !important;
    height: 1em !important;
}

#app-content-blob label[data-testid="block-label"]:has(span[data-testid="block-info"]) {
    background-color: transparent !important;
    color: inherit !important;
    padding: 0 !important;
    border-radius: 0 !important;
    margin-bottom: 0 !important;
    display: block !important;
    font-weight: inherit !important;
    border: none !important;
}

.gr-info {
    color: #555 !important;
    font-size: 0.85em !important;
}

/* ---- EXAMPLES SECTION STYLING ---- */
.gr-examples {
    margin-top: 25px !important;
}
.gr-examples table thead th {
    background-color: #d1e9eb !important;
    color: #155e63 !important;
    font-weight: bold !important;
    border-bottom: 2px solid #1e747c !important;
}
.gr-examples table tbody tr.selected {
  background-color: #b0dde1 !important;
}
.gr-examples > .label-wrap > span {
    font-size: 1.2em !important;
    color: #1e747c !important;
    font-weight: 600 !important;
}

/* ---- LOGO STYLING ---- */
#app-logo-main {
    max-width: 250px !important;
    height: 50px !important;
    display: block !important;
    margin-top: 10px !important;
    margin-bottom: 10px !important;
    margin-left: 0px !important;
    margin-right: auto !important;
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
    position: relative; 
}
#app-logo-main img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
    background-color: transparent !important;
}
#app-logo-main .icon-button-wrapper.top-panel {
    display: none !important;
}
"""

default_text = "[S1] Dia is an open weights text to dialogue model. \n[S2] You get full control over scripts and voices. \n[S1] Wow. Amazing. (laughs) \n[S2] Try it now on Git hub or Hugging Face."
example_txt_path = Path("./example.txt")
if example_txt_path.exists():
    try:
        default_text = example_txt_path.read_text(encoding="utf-8").strip()
        if not default_text:
            default_text = "Example text file was empty."
    except Exception as e:
        print(f"Warning: Could not read example.txt: {e}")

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue=gr.themes.colors.teal, secondary_hue=gr.themes.colors.cyan)) as demo:
    # FIX: Replaced the '...' placeholder with the correct gr.Image component call.
    gr.Image(
        "horizontal.png",
        elem_id="app-logo-main",
        show_label=False,
        interactive=False,
        show_download_button=False,
        show_fullscreen_button=False,
        height=60
    )
    with gr.Column(elem_id="app-content-blob"):
        gr.Markdown("# Narration Synthesis", elem_id="main-title")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                with gr.Accordion("Audio Reference Prompt (Optional)", open=False):
                    audio_prompt_input = gr.Audio(
                        label="Audio Prompt File",
                        sources=["upload", "microphone"],
                        type="numpy",
                    )
                    audio_prompt_text_input = gr.Textbox(
                        label="Transcript of Audio Prompt (Required if using Audio Prompt)",
                        placeholder="Enter the exact text spoken in the audio prompt...",
                        lines=2,
                    )
                text_input = gr.Textbox(
                    label="Text To Generate",
                    placeholder="Enter text here...",
                    value=default_text,
                    lines=5,
                )
                with gr.Accordion("Generation Parameters", open=False):
                    cfg_scale = gr.Slider(label="CFG Scale (Guidance Strength)", minimum=1.0, maximum=5.0, value=3.0, step=0.1)
                    temperature = gr.Slider(label="Temperature (Randomness)", minimum=1.0, maximum=1.5, value=1.3, step=0.05)
                    top_p = gr.Slider(label="Top P (Nucleus Sampling)", minimum=0.80, maximum=1.0, value=0.95, step=0.01)
                    cfg_filter_top_k = gr.Slider(label="CFG Filter Top K", minimum=15, maximum=50, value=30, step=1)
                    speed_factor_slider = gr.Slider(label="Speed Factor", minimum=0.8, maximum=1.2, value=0.94, step=0.02)
                    seed_input = gr.Number(label="Generation Seed (Optional)", value=-1, precision=0, step=1, interactive=True)

                run_button = gr.Button("Generate Audio", variant="primary")

            with gr.Column(scale=2):
                audio_output = gr.Audio(label="Generated Audio", type="numpy", autoplay=False)
                seed_output = gr.Textbox(label="Generation Seed Used", interactive=False)
                console_output = gr.Textbox(label="Console Output Log", lines=10, interactive=False)

    run_button.click(
        fn=run_inference,
        inputs=[
            text_input,
            audio_prompt_text_input,
            audio_prompt_input,
            cfg_scale,
            temperature,
            top_p,
            cfg_filter_top_k,
            speed_factor_slider,
            seed_input,
        ],
        outputs=[
            audio_output,
            seed_output,
            console_output,
        ],
        api_name="generate_audio",
    )

    # Replace the entire examples_list in your file with this corrected version

    example_prompt_path = "./example_prompt.mp3"
    examples_list = [
        [
            # CORRECTED: This list now has 9 items, matching the 9 inputs.
            "[S1] Oh fire! Oh my goodness! What's the procedure? What to we do people? The smoke could be coming through an air duct! \n[S2] Oh my god! Okay.. it's happening. Everybody stay calm! \n[S1] What's the procedure... \n[S2] Everybody stay fucking calm!!!... Everybody fucking calm down!!!!! \n[S1] No! No! If you touch the handle, if its hot there might be a fire down the hallway!",
            "",   # audio_prompt_text_input
            None, # audio_prompt_input
            3.0,  # cfg_scale
            1.3,  # temperature
            0.95, # top_p
            35,   # cfg_filter_top_k
            0.94, # speed_factor
            -1,   # seed
        ],
        [
            # This list was already correct with 9 items.
            "[S1] Open weights text to dialogue model. \n[S2] You get full control over scripts and voices. \n[S1] I'm biased, but I think we clearly won. \n[S2] Hard to disagree. (laughs) \n[S1] Thanks for listening to this demo. \n[S2] Try it now on Git hub and Hugging Face. \n[S1] If you liked our model, please give us a star and share to your friends. \n[S2] This was Nari Labs.",
            "Open Wait's Text-to-Dialogue model. You get full control over scripts and voices.",
            example_prompt_path if Path(example_prompt_path).exists() else None,
            3.0,
            1.3,
            0.95,
            35,
            0.94,
            -1,
        ],
    ]


    if Path(example_prompt_path).exists():
        gr.Examples(
            examples=examples_list,
            inputs=[
                text_input, audio_prompt_text_input, audio_prompt_input, cfg_scale, temperature,
                top_p, cfg_filter_top_k, speed_factor_slider, seed_input,
            ],
            outputs=[audio_output, seed_output, console_output],
            fn=run_inference, cache_examples=True, label="Examples (Click to Run)",
        )
    else:
        print("Warning: example_prompt.mp3 not found. Examples section will be simplified.")
        gr.Examples(
            examples=[examples_list[0]],
            inputs=[
                text_input, audio_prompt_text_input, audio_prompt_input, cfg_scale, temperature,
                top_p, cfg_filter_top_k, speed_factor_slider, seed_input,
            ],
            outputs=[audio_output, seed_output, console_output],
            fn=run_inference, cache_examples=True, label="Examples (Click to Run)",
        )

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch(share=args.share)