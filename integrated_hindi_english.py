import argparse
import tempfile
import time
import random
import io
import contextlib
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from scipy.signal import resample
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import math
from dia.model import Dia
from transformers import GPT2TokenizerFast
import nltk
from tqdm.auto import tqdm

# --- 1. SETUP & CONFIGURATION ---
parser = argparse.ArgumentParser(description="Gradio interface for Dia with language selection")
parser.add_argument("--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')")
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
args = parser.parse_args()

ENGLISH_MODEL_NAME = "nari-labs/Dia-1.6B"
# --- !!! IMPORTANT: CONFIGURE YOUR HINDI MODEL PATHS HERE !!! ---
HINDI_CHECKPOINT_PATH = r"C:\Users\Ajay\Desktop\dia_try2\dia-finetuning\checkpoints\hindi_v1\ckpt_step12430.pth"
HINDI_CONFIG_PATH = r"C:\Users\Ajay\Desktop\dia_try2\dia-finetuning\dia\config.json"

if not Path(HINDI_CHECKPOINT_PATH).is_file():
    raise FileNotFoundError(f"CRITICAL: Hindi checkpoint file not found. Path: {HINDI_CHECKPOINT_PATH}")
if not Path(HINDI_CONFIG_PATH).is_file():
    raise FileNotFoundError(f"CRITICAL: Hindi config file not found. Path: {HINDI_CONFIG_PATH}")

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

if args.device: device = torch.device(args.device)
elif torch.cuda.is_available(): device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cpu")
print(f"Using device: {device}")


# --- 2. STATE MANAGEMENT & DYNAMIC MODEL LOADING ---
loaded_models: Dict[str, Optional[Dia]] = {"English": None, "Hindi": None}
active_model: Optional[Dia] = None
active_model_language: Optional[str] = None # NEW: Track the active language
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def load_model_by_language(language: str, progress: gr.Progress):
    global loaded_models, active_model_language
    if loaded_models.get(language):
        active_model_language = language
        print(f"Using cached '{language}' model.")
        return loaded_models[language]

    progress(0, desc=f"Loading {language} model...")
    try:
        if language == "English":
            print(f"Loading '{language}' model from Hugging Face Hub: {ENGLISH_MODEL_NAME}")
            dtype_map = {"cpu": "float32", "mps": "float32", "cuda": "float16"}
            dtype = dtype_map.get(device.type, "float16")
            model = Dia.from_pretrained(ENGLISH_MODEL_NAME, compute_dtype=dtype, device=device)
        elif language == "Hindi":
            print(f"Loading '{language}' model from local checkpoint: {HINDI_CHECKPOINT_PATH}")
            model = Dia.from_local(config_path=HINDI_CONFIG_PATH, checkpoint_path=HINDI_CHECKPOINT_PATH, device=device)
        else:
             raise ValueError(f"Unknown language selected: {language}")
        progress(1, desc=f"{language} model loaded successfully!")
        loaded_models[language] = model
        active_model_language = language # Set the active language
        return model
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Failed to load the {language} model. Check console for details.")

# --- Core Logic (set_seed, chunking) ---
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def split_and_tokenize_sentences(text: str) -> List[Tuple[str, int]]:
    sentences = nltk.sent_tokenize(text)
    return [(sent, len(tokenizer.encode(sent))) for sent in sentences if sent.strip()]

def chunk_text(text: str) -> tuple[list[str], list[int]]:
    sentence_tuples = split_and_tokenize_sentences(text)
    if not sentence_tuples: return [], []
    total_tokens = sum(tc for _, tc in sentence_tuples)
    DO_NOT_CHUNK_THRESHOLD = 80
    if total_tokens <= DO_NOT_CHUNK_THRESHOLD: return [" ".join(s for s, _ in sentence_tuples)], [total_tokens]
    IDEAL_CHUNK_TOKENS = 64
    num_chunks = max(1, math.ceil(total_tokens / IDEAL_CHUNK_TOKENS))
    dynamic_target_size = math.ceil(total_tokens / num_chunks)
    chunks, counts, current_chunk_sentences, current_tokens = [], [], [], 0
    for sent, tc in sentence_tuples:
        if tc > dynamic_target_size:
            if current_chunk_sentences: chunks.append(" ".join(current_chunk_sentences)); counts.append(current_tokens); current_chunk_sentences, current_tokens = [], 0
            chunks.append(sent); counts.append(tc)
        elif current_chunk_sentences and current_tokens + tc > dynamic_target_size:
            chunks.append(" ".join(current_chunk_sentences)); counts.append(current_tokens); current_chunk_sentences, current_tokens = [sent], tc
        else: current_chunk_sentences.append(sent); current_tokens += tc
    if current_chunk_sentences: chunks.append(" ".join(current_chunk_sentences)); counts.append(current_tokens)
    return chunks, counts

def run_inference(text_input: str, audio_prompt_text_input: str, audio_prompt_input: Optional[Tuple[int, np.ndarray]], cfg_scale: float, temperature: float, top_p: float, cfg_filter_top_k: int, speed_factor: float, seed: Optional[int] = None, progress=gr.Progress(track_tqdm=True)):
    global active_model, active_model_language
    if active_model is None: raise gr.Error("Please select a language first.")
    
    console_output_buffer = io.StringIO()
    if not text_input or text_input.isspace(): raise gr.Error("Text input cannot be empty.")
    
    with contextlib.redirect_stdout(console_output_buffer):
        if audio_prompt_input and (not audio_prompt_text_input or audio_prompt_text_input.isspace()): raise gr.Error("Audio Prompt Text is required.")
        if seed is None or seed < 0: seed = random.randint(0, 2**32 - 1)
        set_seed(seed)
        
        initial_audio_prompt_path = None
        try:
            # --- Prepare initial audio prompt (used by both methods) ---
            if audio_prompt_input is not None:
                sr, audio_data = audio_prompt_input
                if audio_data is not None and audio_data.size > 0 and audio_data.max() > 0:
                    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f_audio:
                        initial_audio_prompt_path = f_audio.name
                        if np.issubdtype(audio_data.dtype, np.integer): audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
                        if audio_data.ndim > 1: audio_data = np.mean(audio_data, axis=-1)
                        sf.write(initial_audio_prompt_path, audio_data, sr, subtype="FLOAT")

            audio_segments = []
            start_time = time.time()
            
            # --- CONDITIONAL GENERATION LOGIC ---
            if active_model_language == "Hindi":
                # --- HINDI: Use the simple, manual chunking method from your notebook ---
                print("Using Hindi generation strategy (manual chunking by newline)...")
                # Each line in the text box is a chunk
                text_chunks = [line for line in text_input.strip().split('\n') if line.strip()]
                
                for idx, chunk in enumerate(tqdm(text_chunks, desc="Generating Hindi Chunks")):
                    prompt_for_this_chunk = None
                    text_for_this_chunk = chunk
                    # Only use the main audio prompt for the very first chunk
                    if idx == 0 and initial_audio_prompt_path:
                        prompt_for_this_chunk = initial_audio_prompt_path
                        text_for_this_chunk = (audio_prompt_text_input + "\n" + chunk).strip()

                    with torch.inference_mode():
                        # Note: No voice chaining is used here.
                        generated_chunk_audio = active_model.generate(
                            text_for_this_chunk, temperature=temperature, cfg_scale=cfg_scale, 
                            top_p=top_p, cfg_filter_top_k=cfg_filter_top_k, audio_prompt=prompt_for_this_chunk
                        )
                    if generated_chunk_audio is not None and len(generated_chunk_audio) > 0:
                        audio_segments.append(generated_chunk_audio)

            else: # English or default logic
                # --- ENGLISH: Use the advanced automatic chunking and voice chaining ---
                print("Using English generation strategy (automatic chunking and voice chaining)...")
                chunks, _ = chunk_text(text_input.strip())
                previous_chunk_audio = None
                for idx, chunk in enumerate(tqdm(chunks, desc="Generating English Chunks")):
                    temp_chain_prompt_path = None
                    try:
                        prompt_for_this_chunk, text_for_this_chunk = None, chunk.strip()
                        if idx == 0:
                            prompt_for_this_chunk = initial_audio_prompt_path
                            if prompt_for_this_chunk: text_for_this_chunk = (audio_prompt_text_input + "\n" + chunk).strip()
                        elif previous_chunk_audio is not None and len(previous_chunk_audio) > 0:
                            audio_prompt_slice = previous_chunk_audio[-int(2.5 * 44100):]
                            with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f_chain:
                                temp_chain_prompt_path = f_chain.name
                                sf.write(temp_chain_prompt_path, audio_prompt_slice, 44100, subtype="FLOAT")
                                prompt_for_this_chunk = temp_chain_prompt_path
                        with torch.inference_mode():
                            generated_chunk_audio = active_model.generate(text_for_this_chunk, max_tokens=active_model.config.data.audio_length, cfg_scale=cfg_scale, temperature=temperature, top_p=top_p, cfg_filter_top_k=cfg_filter_top_k, use_torch_compile=False, audio_prompt=prompt_for_this_chunk)
                        if generated_chunk_audio is not None and len(generated_chunk_audio) > 0:
                             audio_segments.append(generated_chunk_audio); previous_chunk_audio = generated_chunk_audio
                        else: previous_chunk_audio = None
                    finally:
                        if temp_chain_prompt_path and Path(temp_chain_prompt_path).exists(): Path(temp_chain_prompt_path).unlink()

            # --- Post-processing (common for both methods) ---
            if not audio_segments:
                output_audio = (44100, np.zeros(1, dtype=np.int16)); gr.Warning("Generation produced no audio.")
            else:
                output_audio_np = np.concatenate(audio_segments)
                target_len = int(len(output_audio_np) / max(0.1, min(speed_factor, 5.0)))
                resampled_audio = resample(output_audio_np, target_len) if target_len != len(output_audio_np) and target_len > 0 else output_audio_np
                output_audio = (44100, (np.clip(resampled_audio, -1.0, 1.0) * 32767).astype(np.int16))
            
            print(f"Generation finished in {time.time() - start_time:.2f} seconds.")

        finally:
            if initial_audio_prompt_path and Path(initial_audio_prompt_path).exists():
                try: Path(initial_audio_prompt_path).unlink()
                except OSError as e: print(f"Warning: Error deleting temp file: {e}")
        console_output = console_output_buffer.getvalue()
    return output_audio, seed, console_output

# --- 3. UI TEXT & EXAMPLES ---
UI_TEXTS = {
    "English": {"title": "# Narration Synthesis", "audio_prompt_header": "Audio Reference Prompt (Optional)", "audio_prompt_label": "Audio Prompt File", "audio_prompt_text_label": "Transcript of Audio Prompt", "audio_prompt_text_placeholder": "Enter the exact text spoken in the audio prompt...", "text_input_label": "Text To Generate", "text_input_placeholder": "Enter text here...", "generate_button": "Generate Audio", "output_audio_label": "Generated Audio", "examples_label": "Examples (Click to Run)"},
    "Hindi": {"title": "# हिंदी संवाद संश्लेषण (Hindi Dialogue Synthesis)", "audio_prompt_header": "ऑडियो प्रॉम्प्ट (वैकल्पिक / Optional)", "audio_prompt_label": "ऑडियो प्रॉम्प्ट फ़ाइल", "audio_prompt_text_label": "ऑडियो प्रॉम्प्ट का टेक्स्ट", "audio_prompt_text_placeholder": "ऑडियो प्रॉम्प्ट में बोले गए सटीक टेक्स्ट को यहाँ दर्ज करें...", "text_input_label": "संश्लेषण के लिए टेक्स्ट (हर लाइन एक अलग हिस्सा है)", "text_input_placeholder": "यहां हिंदी में टेक्स्ट दर्ज करें...\nहर लाइन को अलग से बनाया जाएगा।", "generate_button": "ऑडियो बनाएं (Generate Audio)", "output_audio_label": "जेनरेट किया गया ऑडियो", "examples_label": "उदाहरण (चलाने के लिए क्लिक करें)"}
}
default_english_text = "[S1] This is the first sentence. It will be automatically chunked.\n[S2] This is the second sentence, and it will be chained to the first one for voice consistency."
english_examples = [[default_english_text, "", None, 3.0, 1.3, 0.95, 30, 0.94, -1]]

# --- UPDATED: Hindi example now uses newlines for manual chunking ---
default_hindi_text = "आधुनिक तकनीक ने हमारे समाज को पूरी तरह से बदल दिया है।\nसंचार के साधनों में क्रांति आ गई है।\nसोशल मीडिया ने लोगों को जोड़ने का काम किया है।\nहमें यह सुनिश्चित करना होगा कि हम तकनीक का उपयोग मानवता की भलाई के लिए करें।"
hindi_examples = [[default_hindi_text, "", None, 3.0, 1.3, 0.95, 30, 0.94, -1]]

# --- 4. GRADIO UI DEFINITION ---
css = """...""" # Keep your original CSS

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue=gr.themes.colors.teal)) as demo:
    # MODIFIED: Removed the download button from the logo image
    gr.Image("horizontal.png", elem_id="app-logo-main", show_label=False, interactive=False, height=60, show_download_button=False)
    
    with gr.Column(elem_id="app-content-blob"):
        language_selector = gr.Radio(["English", "Hindi"], label="1. Select Language", info="Choose a language. The model will be loaded automatically.")

        with gr.Group(visible=False) as generation_ui_group:
            main_title = gr.Markdown(UI_TEXTS["English"]["title"], elem_id="main-title")
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    with gr.Accordion(UI_TEXTS["English"]["audio_prompt_header"], open=False) as audio_prompt_accordion:
                        audio_prompt_input = gr.Audio(label=UI_TEXTS["English"]["audio_prompt_label"], sources=["upload", "microphone"], type="numpy")
                        audio_prompt_text_input = gr.Textbox(label=UI_TEXTS["English"]["audio_prompt_text_label"], placeholder=UI_TEXTS["English"]["audio_prompt_text_placeholder"], lines=2)
                    text_input = gr.Textbox(label=UI_TEXTS["English"]["text_input_label"], placeholder=UI_TEXTS["English"]["text_input_placeholder"], value=default_english_text, lines=5)
                    with gr.Accordion("Generation Parameters", open=False):
                        cfg_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=5.0, value=3.0, step=0.1)
                        temperature = gr.Slider(label="Temperature", minimum=0.3, maximum=1.5, value=0.5, step=0.05)
                        top_p = gr.Slider(label="Top P", minimum=0.80, maximum=1.0, value=0.95, step=0.01)
                        cfg_filter_top_k = gr.Slider(label="CFG Filter Top K", minimum=15, maximum=50, value=30, step=1)
                        speed_factor_slider = gr.Slider(label="Speed Factor", minimum=0.8, maximum=1.2, value=0.94, step=0.02)
                        seed_input = gr.Number(label="Generation Seed (Optional)", value=-1, precision=0, step=1, interactive=True)
                    run_button = gr.Button(UI_TEXTS["English"]["generate_button"], variant="primary")
                with gr.Column(scale=2):
                    # MODIFIED: Added a visible download button to the audio output
                    audio_output = gr.Audio(label=UI_TEXTS["English"]["output_audio_label"], type="numpy", autoplay=False, show_download_button=True)
                    seed_output = gr.Textbox(label="Generation Seed Used", interactive=False)
                    console_output = gr.Textbox(label="Console Output Log", lines=10, interactive=False)
        
        with gr.Column(visible=False) as english_examples_wrapper:
            gr.Examples(examples=english_examples, inputs=[text_input, audio_prompt_text_input, audio_prompt_input, cfg_scale, temperature, top_p, cfg_filter_top_k, speed_factor_slider, seed_input], outputs=[audio_output, seed_output, console_output], fn=run_inference, cache_examples=False, label=UI_TEXTS["English"]["examples_label"])
        
        with gr.Column(visible=False) as hindi_examples_wrapper:
            gr.Examples(examples=hindi_examples, inputs=[text_input, audio_prompt_text_input, audio_prompt_input, cfg_scale, temperature, top_p, cfg_filter_top_k, speed_factor_slider, seed_input], outputs=[audio_output, seed_output, console_output], fn=run_inference, cache_examples=False, label=UI_TEXTS["Hindi"]["examples_label"])

    # --- 5. UI EVENT HANDLING ---
    def select_language(language: str, progress=gr.Progress(track_tqdm=True)):
        global active_model
        if not language: return (gr.update(visible=False),) * 10
        
        model = load_model_by_language(language, progress)
        active_model = model
        
        ui_config = UI_TEXTS[language]
        default_text = default_english_text if language == "English" else default_hindi_text
        
        return (
            gr.update(visible=True), gr.update(value=ui_config["title"]), gr.update(label=ui_config["audio_prompt_header"]), gr.update(label=ui_config["audio_prompt_label"]),
            gr.update(label=ui_config["audio_prompt_text_label"], placeholder=ui_config["audio_prompt_text_placeholder"]),
            gr.update(label=ui_config["text_input_label"], placeholder=ui_config["text_input_placeholder"], value=default_text),
            gr.update(value=ui_config["generate_button"]), gr.update(label=ui_config["output_audio_label"]),
            gr.update(visible=language == "English"), gr.update(visible=language == "Hindi"),
        )

    language_selector.change(
        fn=select_language,
        inputs=[language_selector],
        outputs=[
            generation_ui_group, main_title, audio_prompt_accordion, audio_prompt_input,
            audio_prompt_text_input, text_input, run_button, audio_output,
            english_examples_wrapper, hindi_examples_wrapper
        ]
    )
    
    run_button.click(fn=run_inference, inputs=[text_input, audio_prompt_text_input, audio_prompt_input, cfg_scale, temperature, top_p, cfg_filter_top_k, speed_factor_slider, seed_input], outputs=[audio_output, seed_output, console_output], api_name="generate_audio")

if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch(share=args.share)