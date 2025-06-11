import argparse
import tempfile
import time
import random
import io
import contextlib
from pathlib import Path
from typing import Optional, Tuple
from scipy.signal import resample
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import re
import math
from dia.model import Dia


# --- Global Setup ---
parser = argparse.ArgumentParser(description="Gradio interface for Nari TTS")
parser.add_argument("--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')")
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")

args = parser.parse_args()


# Determine device
if args.device:
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
# Simplified MPS check for broader compatibility
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # Basic check is usually sufficient, detailed check can be problematic
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load Nari model and config
print("Loading Nari model...")
try:
    dtype_map = {
        "cpu": "float32",
        "mps": "float32",  # Apple M series – better with float32
        "cuda": "float16",  # NVIDIA – better with float16
    }

    dtype = dtype_map.get(device.type, "float16")
    print(f"Using device: {device}, attempting to load model with {dtype}")
    model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype=dtype, device=device)
except Exception as e:
    print(f"Error loading Nari model: {e}")
    raise

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into sentences using punctuation as boundaries.
    Keeps the delimiters so we don’t lose the end-of-sentence markers.
    """
    # This regex will split after ., ?, or ! (including when followed by quotes or parentheses)
    sentence_enders = re.compile(r'([.!?]["\')\]]*\s+)')
    parts = sentence_enders.split(text)
    sentences = []
    # Combine the text and its delimiter
    for i in range(0, len(parts) - 1, 2):
        sentence = (parts[i] + parts[i+1]).strip()
        if sentence:
            sentences.append(sentence)
    # If there's a tail without punctuation, include it
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    return [s for s in sentences if s] # Filter out any empty strings

def chunk_sentences(text: str) -> tuple[list[str], list[int]]:
    """
    Splits text into sentences and then dynamically groups them into uniform chunks.
    It calculates the optimal number of chunks to keep the token count per chunk
    within a desirable range (ideally around 125 tokens).

    Args:
        text: The full input text to be chunked.

    Returns:
        A tuple containing:
        - A list of text chunks.
        - A list of the token counts for each corresponding chunk.
    """
    # First, split the entire text into sentences
    sentences = split_into_sentences(text)
    if not sentences:
        return [], []

    # Define the ideal token count for a chunk
    IDEAL_CHUNK_TOKENS = 64
    
    # Calculate token count for each sentence
    sentence_token_counts = [len(s.split()) for s in sentences]
    total_tokens = sum(sentence_token_counts)

    # Determine the optimal number of chunks based on the ideal size
    # Use max(1, ...) to avoid division by zero and ensure at least one chunk
    num_chunks = max(1, math.ceil(total_tokens / IDEAL_CHUNK_TOKENS))

    # Calculate the dynamic chunk size to aim for
    # This creates a uniform target size for each chunk
    dynamic_target_size = math.ceil(total_tokens / num_chunks)

    # --- Use the robust sentence-packing logic with the new dynamic target size ---
    chunks = []
    counts = []
    current_chunk_sentences = []
    current_tokens = 0

    for sent, tc in zip(sentences, sentence_token_counts):
        # Handle a single sentence that is larger than the target size
        if tc > dynamic_target_size:
            # If there's a pending chunk, finalize it first
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                counts.append(current_tokens)
                current_chunk_sentences = []
                current_tokens = 0
            # Add the oversized sentence as its own chunk
            chunks.append(sent)
            counts.append(tc)
            continue

        # If adding the next sentence would exceed the target, finalize the current chunk
        if current_chunk_sentences and current_tokens + tc > dynamic_target_size:
            chunks.append(" ".join(current_chunk_sentences))
            counts.append(current_tokens)
            # Start a new chunk with the current sentence
            current_chunk_sentences = [sent]
            current_tokens = tc
        else:
            # Otherwise, add the sentence to the current chunk
            current_chunk_sentences.append(sent)
            current_tokens += tc

    # Add the last remaining chunk if it exists
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
    # MODIFIED: chunk_size is removed from inputs
    seed: Optional[int] = None,
):
    """
    Runs Nari inference by splitting text into sentences, dynamically grouping them into
    token-limited chunks, generating audio for each chunk, and merging the results.
    """
    global model, device
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

        # Preprocess audio prompt if provided
        temp_audio_prompt_path = None
        prompt_path_for_generate = None
        try:
            if audio_prompt_input is not None:
                sr, audio_data = audio_prompt_input
                if audio_data is None or audio_data.size == 0 or audio_data.max() == 0:
                    gr.Warning("Audio prompt seems empty or silent, ignoring prompt.")
                else:
                    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f_audio:
                        temp_audio_prompt_path = f_audio.name
                        if np.issubdtype(audio_data.dtype, np.integer):
                            max_val = np.iinfo(audio_data.dtype).max
                            audio_data = audio_data.astype(np.float32) / max_val
                        if audio_data.ndim > 1:
                            audio_data = np.mean(audio_data, axis=-1)
                        sf.write(temp_audio_prompt_path, audio_data, sr, subtype="FLOAT")
                        prompt_path_for_generate = temp_audio_prompt_path
                        print(f"Created temporary audio prompt file: {temp_audio_prompt_path}")

            # --- Text Processing: Split by Sentence, then Chunk by Token Limit ---
            print("Step 1: Dynamically grouping text into uniform chunks.")
            # MODIFIED: chunk_sentences now takes the raw text and determines chunking automatically.
            chunks, token_counts = chunk_sentences(text_input.strip())
            print(f"Successfully created {len(chunks)} chunks.")

            # --- Generation Loop for each chunk ---
            audio_segments = []
            start_time = time.time()
            
            for idx, (chunk, T_chunk) in enumerate(zip(chunks, token_counts)):
                max_tokens_for_chunk = model.config.data.audio_length
                
                # Use the audio prompt only for the very first chunk
                if idx == 0 and prompt_path_for_generate:
                    chunk_input = (audio_prompt_text_input + "\n" + chunk).strip()
                    audio_prompt = prompt_path_for_generate
                else:
                    chunk_input = chunk.strip()
                    audio_prompt = None
                
                print(f"Generating chunk {idx+1}/{len(chunks)} ({T_chunk} input tokens)...")
           
                with torch.inference_mode():
                    generated_chunk_audio = model.generate(
                        chunk_input,
                        max_tokens=max_tokens_for_chunk,
                        cfg_scale=cfg_scale,
                        temperature=temperature,
                        top_p=top_p,
                        cfg_filter_top_k=cfg_filter_top_k,
                        use_torch_compile=False,
                        audio_prompt=audio_prompt,
                    )

                if generated_chunk_audio is not None and len(generated_chunk_audio) > 0:
                    audio_segments.append(generated_chunk_audio)
                    # Add a small silence between chunks for better pacing
                    if idx < len(chunks) - 1:
                        silence_duration_sec = 0.2
                        silence_samples = int(44100 * silence_duration_sec)
                        silence = np.zeros(silence_samples, dtype=np.float32)
                        audio_segments.append(silence)

            if not audio_segments:
                output_audio = (44100, np.zeros(1, dtype=np.int16))
                gr.Warning("Generation produced no valid audio output.")
            else:
                # --- Final Merging and Postprocessing ---
                print("Step 2: Merging generated audio chunks.")
                output_audio_np = np.concatenate(audio_segments)
                output_sr = 44100
                
                # Adjust speed if necessary
                original_len = len(output_audio_np)
                speed_factor = max(0.1, min(speed_factor, 5.0))
                target_len = int(original_len / speed_factor)

                if target_len != original_len and target_len > 0:
                    resampled_audio_np = resample(output_audio_np, target_len)
                    print(f"Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed.")
                else:
                    resampled_audio_np = output_audio_np
                
                # Convert to int16 for Gradio compatibility
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
            if temp_audio_prompt_path and Path(temp_audio_prompt_path).exists():
                try:
                    Path(temp_audio_prompt_path).unlink()
                    print(f"Deleted temporary audio prompt file: {temp_audio_prompt_path}")
                except OSError as e:
                    print(f"Warning: Error deleting temp file {temp_audio_prompt_path}: {e}")

        console_output = console_output_buffer.getvalue()

    return output_audio, seed, console_output


# --- Create Gradio Interface ---
# The CSS remains unchanged.
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


# Attempt to load default text from example.txt
default_text = "[S1] Dia is an open weights text to dialogue model. \n[S2] You get full control over scripts and voices. \n[S1] Wow. Amazing. (laughs) \n[S2] Try it now on Git hub or Hugging Face."
example_txt_path = Path("./example.txt")
if example_txt_path.exists():
    try:
        default_text = example_txt_path.read_text(encoding="utf-8").strip()
        if not default_text:  # Handle empty example file
            default_text = "Example text file was empty."
    except Exception as e:
        print(f"Warning: Could not read example.txt: {e}")

# Build Gradio UI
with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue=gr.themes.colors.teal, secondary_hue=gr.themes.colors.cyan)) as demo:
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
                    # MODIFIED: chunk_size number input has been removed.
                    cfg_scale = gr.Slider(
                        label="CFG Scale (Guidance Strength)",
                        minimum=1.0,
                        maximum=5.0,
                        value=3.0,
                        step=0.1,
                        info="Higher values increase adherence to the text prompt.",
                    )
                    temperature = gr.Slider(
                        label="Temperature (Randomness)",
                        minimum=1.0,
                        maximum=1.5,
                        value=1.3,
                        step=0.05,
                        info="Lower values make the output more deterministic, higher values increase randomness.",
                    )
                    top_p = gr.Slider(
                        label="Top P (Nucleus Sampling)",
                        minimum=0.80,
                        maximum=1.0,
                        value=0.95,
                        step=0.01,
                        info="Filters vocabulary to the most likely tokens cumulatively reaching probability P.",
                    )
                    cfg_filter_top_k = gr.Slider(
                        label="CFG Filter Top K",
                        minimum=15,
                        maximum=50,
                        value=30,
                        step=1,
                        info="Top k filter for CFG guidance.",
                    )
                    speed_factor_slider = gr.Slider(
                        label="Speed Factor",
                        minimum=0.8,
                        maximum=1.2,
                        value=0.94,
                        step=0.02,
                        info="Adjusts the speed of the final audio (1.0 = original speed).",
                    )
                    seed_input = gr.Number(
                        label="Generation Seed (Optional)",
                        value=-1,
                        precision=0,
                        step=1,
                        interactive=True,
                        info="Set a seed for reproducible outputs. Leave empty or -1 for a random seed.",
                    )

                run_button = gr.Button("Generate Audio", variant="primary")

            with gr.Column(scale=2):
                audio_output = gr.Audio(
                    label="Generated Audio",
                    type="numpy",
                    autoplay=False,
                )
                seed_output = gr.Textbox(
                    label="Generation Seed Used",
                    interactive=False
                )
                console_output = gr.Textbox(
                    label="Console Output Log", lines=10, interactive=False
                )

    # MODIFIED: Link button click to function with updated inputs list
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
            # chunk_size is removed
            seed_input,
        ],
        outputs=[
            audio_output,
            seed_output,
            console_output,
        ],
        api_name="generate_audio",
    )

    # Add examples
    example_prompt_path = "./example_prompt.mp3"
    # MODIFIED: The examples list is updated to remove the chunk_size value (was after speed_factor)
    examples_list = [
        [
            "[S1] Oh fire! Oh my goodness! What's the procedure? What to we do people? The smoke could be coming through an air duct! \n[S2] Oh my god! Okay.. it's happening. Everybody stay calm! \n[S1] What's the procedure... \n[S2] Everybody stay fucking calm!!!... Everybody fucking calm down!!!!! \n[S1] No! No! If you touch the handle, if its hot there might be a fire down the hallway!",
            "",   # No audio prompt text
            None, # No audio prompt file
            3.0,  # cfg_scale
            1.3,  # temperature
            0.95, # top_p
            35,   # cfg_filter_top_k
            0.94, # speed_factor
            # chunk_size (256) was here, now removed
            -1,   # seed
        ],
        [
            "[S1] Open weights text to dialogue model. \n[S2] You get full control over scripts and voices. \n[S1] I'm biased, but I think we clearly won. \n[S2] Hard to disagree. (laughs) \n[S1] Thanks for listening to this demo. \n[S2] Try it now on Git hub and Hugging Face. \n[S1] If you liked our model, please give us a star and share to your friends. \n[S2] This was Nari Labs.",
            "Open Wait's Text-to-Dialogue model. You get full control over scripts and voices.", # Transcript for the audio prompt
            example_prompt_path if Path(example_prompt_path).exists() else None,
            3.0,
            1.3,
            0.95,
            35,
            0.94,
            # chunk_size (256) was here, now removed
            -1,
        ],
    ]

    if Path(example_prompt_path).exists():
        gr.Examples(
            examples=examples_list,
            # MODIFIED: The inputs list is updated to remove chunk_size
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
            outputs=[audio_output, seed_output, console_output],
            fn=run_inference,
            cache_examples=True,
            label="Examples (Click to Run)",
        )
    else:
        print("Warning: example_prompt.mp3 not found. Examples section will be simplified.")
        # Provide a fallback example that does not require the audio file
        gr.Examples(
            examples=[examples_list[0]], # Only show the first example
            # MODIFIED: The inputs list is updated to remove chunk_size
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
            outputs=[audio_output, seed_output, console_output],
            fn=run_inference,
            cache_examples=True,
            label="Examples (Click to Run)",
        )


# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch(share=args.share)