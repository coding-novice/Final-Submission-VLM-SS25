# maira2_local.py  ────────────────────────────────────────────────────────────
"""
Run MAIRA-2 locally on CPU (INT8/FP16/FP32) or on CUDA GPU (4-bit/8-bit/FP16/FP32).
"""

# ───────── 0) GLOBAL CONFIG ────────────────────────────────────────────────
USE_CUDA   = True       # only usable with NVIDIA GPU

# use "fp32" for inference with full weights (recommended)
# "8bit" and "4bit" only work with bitsandbytes -> requires cuda
QUANT_MODE = "fp32"      # "fp32" , "fp16", "int8", "8bit", "4bit", or "no loading"  







# ───────── 0.1) AUTO PATH RESOLUTION (inputs & outputs) ────────────────────
from pathlib import Path
import os

# Folder that *contains* this script, e.g. '/vol/miltank/users/vac/VLM'
SCRIPT_DIR = Path(__file__).resolve().parent

# Path for all HuggingFace caches

# ───────── 0.2) AUTO CACHE DIR RESOLUTION ───────────────────────────────
# Automatically point cache_dir to '<script-folder>/model_download'
CACHE_DIR_PATH = SCRIPT_DIR / "model_download"
CACHE_DIR_PATH.mkdir(exist_ok=True)      # create it if missing
cache_dir = str(CACHE_DIR_PATH)         # HF env vars need a string, not a Path


# Default roots — can still be overwritten further down
IN_ROOT  = SCRIPT_DIR / "in"
OUT_ROOT = SCRIPT_DIR / "out"
OUT_ROOT.mkdir(exist_ok=True)          # create folder './out' if it does not exist

# -------- Input folders (defaults) --------
x_ray_images_unprocessed_filepath               = IN_ROOT / "dataset"           / "chest_xrays"      / "images"
x_ray_images_processed_with_black_bars_filepath = IN_ROOT / "dataset_processed" / "processed_chest_with_bars"
x_ray_images_processed_no_bars_filepath         = IN_ROOT / "dataset_processed" / "processed_chest_no_bars"
brain_images_filepath                           = IN_ROOT / "dataset"           / "nova_brain"       / "images"

# -------- Helper for ALL output files --------
def out_file(name: str) -> str:
    """
    Returns '<SCRIPT_DIR>/out/<name>' (absolute) and makes sure the parent
    folder exists so later CSV/JSON writes won’t crash.
    """
    path = OUT_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)          # keep downstream code unchanged (expects str)




type_of_run = "brain MRI used data"
# Possible values:
#   "chest X-ray used data"
#   "brain MRI used data"
#   "chest X-ray extended data for unprocessed images"
#   "chest X-ray extended data for processed images with black bars"
#   "chest X-ray extended data for processed images no bars"
#   "brain MRI extended data for simple prompt"
#   "brain MRI extended data for brain-specific prompt"

#run 1-3 (three total runs, change all paths twice: 1) unprocessed input (& respective output names) 2) processed with bars 3) processed without bars )

# ───────── 0.1) ARGUMENT PARSER (choose run on CLI) ────────────────────────
import argparse, sys

# Map each CLI flag to its canonical run-type string
_RUN_FLAGS = {
    "chest_used"            : "chest X-ray used data",
    "brain_used"            : "brain MRI used data",
    "chest_ext_unproc"      : "chest X-ray extended data for unprocessed images",
    "chest_ext_proc_bars"   : "chest X-ray extended data for processed images with black bars",
    "chest_ext_proc_no_bars": "chest X-ray extended data for processed images no bars",
    "brain_ext_simple"      : "brain MRI extended data for simple prompt",
    "brain_ext_specific"    : "brain MRI extended data for brain-specific prompt",
}

parser = argparse.ArgumentParser(
    description="Select which MAIRA-2 run configuration to execute"
)

# Only ONE flag may be given at a time
group = parser.add_mutually_exclusive_group()
for flag, run_name in _RUN_FLAGS.items():
    group.add_argument(f"--{flag}", action="store_true", help=f"Use '{run_name}'")

# Parse *known* args so any extra MAIRA-specific flags you add later won’t crash
args, _unknown = parser.parse_known_args()

# Overwrite the default if a flag was provided
for flag, run_name in _RUN_FLAGS.items():
    if getattr(args, flag):
        type_of_run = run_name
        break


if type_of_run in [
"chest X-ray extended data for unprocessed images",
"chest X-ray extended data for processed images with black bars",
"chest X-ray extended data for processed images no bars"
]:
    

    run_non_grounded_report_all = True 
    start_print_text = "Starting 6.1.2) non-grounded report [X-Ray] // loop all 50 images"
    indication_prompt_param_ng_all = None
    technique_prompt_param_ng_all = "PA chest X-Ray"

    run_grounded_report_all = True  
    print_text_grounded_rep_all = "Starting grounded report // loop all 50 images (X-Ray) [not used for Task, just as backup]"
    indication_grounded = None
    technique_grounded = "PA chest X-Ray"

    run_phrase_grounded_reports_all = True 
    ADD_PERIOD_TO_PHRASES = False # Optional setting for debugging. Recommendation: do not change.


    run_phrase_grounded_reports_single_phrase_92_brain = False # Brain T2 (option C) (should only run during first brain run for type_of_run == "brain MRI extended data for simple prompt")

elif type_of_run in [
"brain MRI extended data for simple prompt",
"brain MRI extended data for brain-specific prompt"
]:
    # specify filepath for brain images
    x_ray_or_brain_images_filepath = brain_images_filepath
  

if type_of_run == "chest X-ray extended data for unprocessed images":
    # RUN 1 (X Ray unprocessed)
    print("---------- Report Generation for X-Rays: unprocessed images ---------- ")
    x_ray_or_brain_images_filepath = x_ray_images_unprocessed_filepath



    non_grounded_reports_out_filepath     = out_file("non_grounded_reports_X_Ray_unproc.csv")
    grounded_reports_out_filepath         = out_file("grounded_reports_X_Ray_unproc.csv")
    phrase_grounded_reports_out_filepath  = out_file("phrase_grounded_reports_X_Ray_unproc.json")


elif type_of_run == "chest X-ray extended data for processed images with black bars":
    # RUN 2 (X Ray processed with black bars)
    print("---------- Report Generation for X-Rays: processed images with black bars ---------- ")
    x_ray_or_brain_images_filepath = x_ray_images_processed_with_black_bars_filepath



    non_grounded_reports_out_filepath     = out_file("non_grounded_reports_X_Ray_proc_w_bars.csv")
    grounded_reports_out_filepath         = out_file("grounded_reports_X_Ray_proc_w_bars.csv")
    phrase_grounded_reports_out_filepath  = out_file("phrase_grounded_reports_X_Ray_proc_w_bars.json")


elif type_of_run == "chest X-ray extended data for processed images no bars":   
    # RUN 3 (X Ray processed no bars)
    print("---------- Report Generation for X-Rays: processed images with NO black bars ---------- ")
    x_ray_or_brain_images_filepath =  x_ray_images_processed_no_bars_filepath



    non_grounded_reports_out_filepath     = out_file("non_grounded_reports_X_Ray_proc_no_bars.csv")
    grounded_reports_out_filepath         = out_file("grounded_reports_X_Ray_proc_no_bars.csv")
    phrase_grounded_reports_out_filepath  = out_file("phrase_grounded_reports_X_Ray_proc_no_bars.json")

#run 4 and 5 (brain)

elif type_of_run == "brain MRI extended data for simple prompt":
    # First brain run
    print("---------- Report Generation for Brain MRIs (Run 4): simple prompt for both ungrounded and grounded report, phrase grounded reports  ---------- ")
    run_non_grounded_report_all = True # Brain T1 (option A)

    non_grounded_reports_out_filepath =  out_file("non_grounded_reports_brain(prompt_simple).csv")

    start_print_text = "Starting 6.1.2) non-grounded report [brain] // loop all 92 images (simple prompt)"
    indication_prompt_param_ng_all = None
    technique_prompt_param_ng_all = "Brain MRI"

    run_grounded_report_all = True # Brain T2 (option A)
    grounded_reports_out_filepath         = out_file("grounded_reports_brain(prompt_simple).csv")
    print_text_grounded_rep_all = "Starting grounded report // loop all 92 images (brain) (simple prompt)"
    indication_grounded = None
    technique_grounded = "Brain MRI"

    run_phrase_grounded_reports_single_phrase_92_brain = True # Brain T2 (option C)
    phrase_grounded_reports_single_phrase_92_out_filepath = out_file("phrase_grounded_reports_brain.json")
    phrase_grounded_reports_single_phrase_92_brain_print_text = "Starting phrase_grounded_reports_single_phrase (92 images) (brain)"
    phrase_grounded_single_phrase_92_report_phrase = "localize possible pathologies in the brain MRI image"




    run_phrase_grounded_reports_all = False  

elif type_of_run == "brain MRI extended data for brain-specific prompt":
    # second brain run
    print("---------- Report Generation for Brain MRIs (Run 5): brain-specific prompt for both ungrounded and grounded report (no phrase grounded reports)  ---------- ")
    run_non_grounded_report_all = True # Brain T1 (option B)
    non_grounded_reports_out_filepath = out_file("non_grounded_reports_brain(prompt_brain_specific).csv")
    start_print_text = "Starting 6.1.2) non-grounded report [brain] // loop all 92 images (more specific prompt to trigger brain knowledge)"
    indication_prompt_param_ng_all = "Describe the brain in the picture. Point out pathologies, if present." # [this exact indication occurs twice in this code]
    technique_prompt_param_ng_all = "Brain MRI"

    run_grounded_report_all = True 
    grounded_reports_out_filepath         =  out_file("grounded_reports_brain(prompt_brain_specific).csv")
    print_text_grounded_rep_all = "Starting grounded report // loop all 92 images (more specific prompt to trigger brain knowledge) "
    indication_grounded = "Describe the brain in the picture. Point out pathologies, if present." # [this exact indication occurs twice in this code]
    technique_grounded = "Brain MRI"

    run_phrase_grounded_reports_single_phrase_92_brain = False

    run_phrase_grounded_reports_all = False 

elif type_of_run == "chest X-ray used data":

    print("---------- Report Generation for Chest X-Ray : data used in evaluation // ungrounded reports (processed images), phrase grounded reports (unprocessed images)  ---------- ")

    x_ray_or_brain_images_filepath = x_ray_images_processed_with_black_bars_filepath


    non_grounded_reports_out_filepath     = out_file("non_grounded_reports_X_Ray_proc_w_bars.csv") 
    phrase_grounded_reports_out_filepath  = out_file("phrase_grounded_reports_X_Ray_unproc.json") 

    run_non_grounded_report_all = True 
    start_print_text = "Starting 6.1.2) non-grounded report [X-Ray] // loop all 50 images"
    indication_prompt_param_ng_all = None
    technique_prompt_param_ng_all = "PA chest X-Ray"

    run_grounded_report_all = False  


    run_phrase_grounded_reports_all = True 
    ADD_PERIOD_TO_PHRASES = False # Optional setting for debugging. Recommendation: do not change.


    run_phrase_grounded_reports_single_phrase_92_brain = False # Brain T2 (option C) (should only run during first brain run for type_of_run == "brain MRI extended data for simple prompt" or type_of_run == "brain MRI used data")

    

elif type_of_run == "brain MRI used data":

    x_ray_or_brain_images_filepath = brain_images_filepath

    print("---------- Report Generation for Brain MRIs : data used in evaluation // simple prompt for ungrounded report, phrase grounded reports  ---------- ")
    run_non_grounded_report_all = True # Brain T1 (option A)
    non_grounded_reports_out_filepath = out_file("non_grounded_reports_brain(prompt_simple).csv")
    start_print_text = "Starting 6.1.2) non-grounded report [brain] // loop all 92 images (simple prompt)"
    indication_prompt_param_ng_all = None
    technique_prompt_param_ng_all = "Brain MRI"

    run_grounded_report_all = True # Brain T2 (option A)
    grounded_reports_out_filepath         =  out_file("grounded_reports_brain(prompt_simple).csv")
    print_text_grounded_rep_all = "Starting grounded report // loop all 92 images (brain) (simple prompt)"
    indication_grounded = None
    technique_grounded = "Brain MRI"


    run_phrase_grounded_reports_all = False


    run_phrase_grounded_reports_single_phrase_92_brain = True # Brain T2 (option C)
    phrase_grounded_reports_single_phrase_92_out_filepath =  out_file("phrase_grounded_reports_brain.json")
    phrase_grounded_reports_single_phrase_92_brain_print_text = "Starting phrase_grounded_reports_single_phrase (92 images) (brain)"
    phrase_grounded_single_phrase_92_report_phrase = "localize possible pathologies in the brain MRI image"




else:
    raise ValueError(f"Unknown type_of_run: {type_of_run}")

# ───────── 1) ENV-VARS & AUTH ──────────────────────────────────────────────
import os, warnings, time #part of standard library
os.environ["HF_HOME"]             = cache_dir
os.environ["TRANSFORMERS_CACHE"]  = cache_dir   # still needed for old commit
os.environ["HF_DATASETS_CACHE"]   = cache_dir
os.environ["HF_METRICS_CACHE"]    = cache_dir
# Optionally silence symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import login
login(new_session=False)   # paste HF token once; it’s cached afterwards

# ───────── 2) IMPORTS ──────────────────────────────────────────────────────
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image


# Optional: INT8 quant (always CPU) & BnB config (GPU)
from torch.quantization import quantize_dynamic
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None  # not available on the pinned transformers commit

#runtime printout
import time
from datetime import datetime
script_start_time = time.time()
system_start_time = datetime.now()
print(f"[INFO] Script started at: {system_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Imports for helpers "parse_bboxes_from_string" and "visualize_bounding_boxes"
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# for json output
import json

# ───────── 3) DEVICE SELECTION ─────────────────────────────────────────────
device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
print(f"[INFO] Using device: {device}")

# ───────── 4) MODEL LOADING ────────────────────────────────────────────────
model_name  = "microsoft/maira-2"
model_kwargs = {"trust_remote_code": True}

# ---- 4.1  Quantization / precision rules ----
if QUANT_MODE == "fp32":
    # Default: weights loaded as float32
    model_kwargs["torch_dtype"] = torch.float32
    model_kwargs.pop("quantization_config", None)
    model_kwargs.pop("device_map", None)

elif QUANT_MODE == "fp16":
    # For CPU or CUDA; works on both, best on CUDA
    model_kwargs["torch_dtype"] = torch.float16
    model_kwargs["low_cpu_mem_usage"] = True
    model_kwargs.pop("quantization_config", None)
    model_kwargs.pop("device_map", None)

elif QUANT_MODE == "int8":
    # Do NOT set torch_dtype; model must load as float32 for quantize_dynamic
    model_kwargs["low_cpu_mem_usage"] = True
    model_kwargs.pop("torch_dtype", None)
    model_kwargs.pop("quantization_config", None)
    model_kwargs.pop("device_map", None)

elif QUANT_MODE in ("8bit", "4bit"):
    if BitsAndBytesConfig is None:
        raise RuntimeError(
            "BitsAndBytesConfig not available with this transformers commit. Use fp16/int8/fp32 for now."
        )
    load_in_8 = QUANT_MODE == "8bit"
    quant_cfg = BitsAndBytesConfig(
        load_in_8bit = load_in_8,
        load_in_4bit = not load_in_8,
        bnb_4bit_quant_type      = "nf4",
        bnb_4bit_compute_dtype   = torch.float16,
        bnb_4bit_use_double_quant= True,
    )
    model_kwargs["quantization_config"] = quant_cfg
    model_kwargs["device_map"] = "auto"
    # Remove torch_dtype to avoid conflict
    model_kwargs.pop("torch_dtype", None)

elif QUANT_MODE == "no loading":
        print ("no model loaded")

else:
    raise ValueError("QUANT_MODE must be fp32, fp16, int8, 8bit, 4bit, or no loading")

print(f"[INFO] Loading model with QUANT_MODE={QUANT_MODE} …")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir = cache_dir,
    **model_kwargs
)
load_secs = time.time() - t0
print(f"[INFO] Model loaded in {load_secs:.1f}s")

# ---- 4.2 (Optional) dynamic INT8 quantisation on CPU ----
if QUANT_MODE == "int8":
    print("[INFO] Applying dynamic INT8 quantisation …")
    model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    print("[INFO] Quantisation done.")

# ---- 4.3 final device placement ----
if device.type == "cuda":
    model.to("cuda")
model.eval()

processor = AutoProcessor.from_pretrained(
    model_name, cache_dir=cache_dir, trust_remote_code=True
)

# ───────── 5) LOAD TEST IMAGE ──────────────────────────────────────────────
# image_test_path = "/vol/miltank/users/vac/VLM/in/dataset/chest_xrays/images/8004676ecf95af8cee446cbcd139a938.png"
# image_test = Image.open(image_test_path)
#image_test.show() # testing purposes only // opens a window



# -------------- Helper Functions --------

def parse_bboxes_from_string(result_string):
    """
    Robustly parses:
      - Bullet format: '• Sentence. [boxes]'
      - Flat format: 'Sentence. [boxes]; ...'
    Returns: list of (sentence, [boxes]) pairs.
    """
    import ast, re

    results = []

    # Try phrase_grounding style
    try:
        if result_string.strip().startswith("[") or result_string.strip().startswith("Phrase grounding result:"):
            m = re.search(r"(\[\(.*\)\])", result_string, re.DOTALL)
            if m:
                results = ast.literal_eval(m.group(1))
                return results
            results = ast.literal_eval(result_string.split(":", 1)[-1].strip())
            return results
    except Exception:
        pass

    # Bullet-point format
    if "•" in result_string:
        for line in result_string.splitlines():
            line = line.strip()
            if not line.startswith("•"):
                continue
            m = re.match(r"• (.*?)(?:\.|:)(.*)", line)
            if not m:
                continue
            sentence = m.group(1).strip()
            after_dot = m.group(2).strip()
            if "None" in after_dot:
                boxes = []
            else:
                try:
                    boxes = ast.literal_eval(after_dot)
                    if isinstance(boxes, tuple) and len(boxes) == 4:
                        boxes = [boxes]
                except Exception:
                    boxes = []
            results.append((sentence, boxes))
        return results

    # Semicolon/flat format - look for "Sentence. [boxes]" or "Sentence. None"
    # Regex: sentence ending with a dot, followed by (None or [box])
    segments = [seg.strip() for seg in result_string.split(";") if seg.strip()]
    for seg in segments:
        # If there’s a box list “[ … ]” at all
        if "[" in seg and "]" in seg:
            # sentence = everything up to the last period before the “[”
            idx_box = seg.find("[")
            dotpos = seg.rfind(".", 0, idx_box)
            sentence = seg[:dotpos+1].strip()
            # extract the bracketed text
            box_str = seg[seg.find("[", idx_box): seg.find("]", idx_box)+1]
            try:
                boxes = ast.literal_eval(box_str)
                if isinstance(boxes, tuple):
                    boxes = [boxes]
            except:
                boxes = []
        else:
            # no box: sentence up to last period, no boxes
            if "." in seg:
                sentence = seg[:seg.rfind(".")+1].strip()
            else:
                sentence = seg
            boxes = []

        results.append((sentence, boxes))
    return results



def visualize_bounding_boxes(results, img, box_label="coords"):
    """
    Visualize bounding boxes from MAIRA2 outputs on an image.

    Args:
        results: list of (sentence, boxes), or string of lines to parse
        img: PIL.Image (e.g., 'frontal')
        box_label: 'coords' (default, normalized coords) or 'sentence'
    """
    # If input is string, parse it
    if isinstance(results, str):
        results = parse_bboxes_from_string(results)
    
    img_arr = np.array(img.convert("RGB"))
    fig, ax = plt.subplots(1)
    ax.imshow(img_arr, cmap='gray')

    colors = [
        "lime", "red", "yellow", "cyan", "magenta", "orange", "blue", "lawngreen",
        "springgreen", "violet"
    ]

    box_count = 0
    for idx, item in enumerate(results):
        if isinstance(item, tuple) and len(item) == 2:
            sentence, boxes = item
        else:
            continue  # skip bad format

        if boxes is None or (isinstance(boxes, list) and len(boxes) == 0):
            continue
        if isinstance(boxes[0], (float, int)):
            boxes = [boxes]

        for box in boxes:
            color = colors[box_count % len(colors)]
            width, height = img.size
            x_min = box[0] * width
            y_min = box[1] * height
            x_max = box[2] * width
            y_max = box[3] * height
            # Draw the bounding box
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2.5,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            # Label: coords or sentence
            if box_label == "coords":
                label = f"[{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]"
            elif box_label == "sentence":
                label = sentence
            else:
                label = ""
            ax.text(
                x_min,
                max(y_min - 8, 5),
                label,
                color=color,
                fontsize=10,
                weight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.3')
            )
            box_count += 1

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def warn_if_no_eos_token(out_tensor, processor, section=""):
    """
    Checks if the EOS token is present in the generated output.
    Prints a warning if not found.
    This helps to notice that the max number of output tokens is too small.
    """
    # Get eos token id (may be called tokenizer or similar)
    eos_token_id = getattr(processor, "eos_token_id", None)
    if eos_token_id is None and hasattr(processor, "tokenizer"):
        eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        print(f"[WARN] Could not determine eos_token_id for warning check.")
        return

    # out_tensor could be shape [batch, seq] or [seq]
    tokens = out_tensor[0] if out_tensor.ndim > 1 else out_tensor

    no_eos_token = (eos_token_id not in tokens.tolist())

    if no_eos_token:
        warning_message = f"""
    =====================[ MAIRA-2 OUTPUT WARNING ]=====================
    !!! [SECTION: {section}] !!!
    --------------------------------------------------------------------
    NO END-OF-SEQUENCE (EOS) TOKEN FOUND IN MODEL OUTPUT!
    The generated report may be INCOMPLETE or TRUNCATED.
    Consider increasing max_new_tokens, or check the model/prompt.
    --------------------------------------------------------------------
    ====================[ MAIRA-2 OUTPUT WARNING ]===================
    """
        warnings.warn(warning_message)
    # else: print(f"[{section}] EOS token found.")

    #print(f"testprint: no_eos_token value: {no_eos_token}")

    return no_eos_token



# ───────── 6) INFERENCE ─────────────────────────────────────
# ───────── 6.1.1) non-grounded //single report  ─────────────────────────────────────

run_non_grounded_report_single = False # change input filepath of image below (frontal_path_GS) accordingly, if you turn this on

if run_non_grounded_report_single:
    frontal_path_GS = "/vol/miltank/users/vac/VLM/in/dataset_processed/processed_chest_with_bars/fd03dd80e6e2940d7527e52670f3c21a.png"

     
    frontal_GS = Image.open(frontal_path_GS)

    print (f"Now running non grounded report for a single image with image path: {frontal_path_GS}")
    
    # -------- optional metadata ---------
    indication_NG = None
    #indication_NG = "Routine chest X-ray for health check."
    technique_NG = "PA chest X-Ray" # vorher: 1) "Frontal Chest X-Ray" 2) "PA chest X-Ray" 3) for re-rung of image 49 in ungrounded report based on processed image with bars: "PA chest X-Ray."
    assistant_text_NG = None # should not be used (reason see below)
    #assistant_text_NG = "Classify the patient as either healthy or unhealthy. Also include the reasoning for the classification."
    print (f"technique string:---{technique_NG}---")

    inputs = processor.format_and_preprocess_reporting_input( # details on what the inputs mean: see paper, description of Fig. 1
        current_frontal = frontal_GS, # only required input, all others are optional (can be None)
        current_lateral = None, # Type: image
        prior_frontal   = None, # Type: image
        indication      = indication_NG, # Type: str   // provides clinical context on the patient and influences interpretation and reporting
        technique       = technique_NG, # Type: str    // describes acquired views and sometimes patient positioning (e.g. supine, lateral)
        comparison      = None, # Type: str   // prior report // indicates whether the radiologist consulted prior studies. e.g.: there is no prior study, so the model receives no prior frontal image or prior report
        prior_report    = None, # Type: str
        assistant_text = assistant_text_NG, # should not be used (no mentioning in MAIRA-2 Paper or huggingface documentation -> probably a leftover underlying LLava framework)
        return_tensors  = "pt", # makes sure the output is formatted as PyTorch tensors rather than, for example, NumPy arrays or plain Python lists/dicts
        get_grounding   = False
    ).to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=450, use_cache=True)

    warn_if_no_eos_token(out, processor, section="6.1 non-grounded (single)")
    
    #warn_if_no_eos_token_returnValue = warn_if_no_eos_token(out, processor, section="6.1 non-grounded (single)")
    #print(f"testprint in section 6.1.1: warn_if_no_eos_token_returnValue: {warn_if_no_eos_token_returnValue}")




    prompt_len = inputs["input_ids"].shape[-1]
    raw_text   = processor.decode(out[0][prompt_len:], skip_special_tokens=True).lstrip()
    report     = processor.convert_output_to_plaintext_or_grounded_sequence(raw_text)
    print("\n=== Non-grounded findings ===\n", report)

        


#  -------------- 6.1.2) non-grounded report // loop all 50 images ------ #


if run_non_grounded_report_all:
    
    print(start_print_text)
    import csv
    import glob
    

    images_folder = x_ray_or_brain_images_filepath
    output_csv_path = non_grounded_reports_out_filepath


    # find and list all PNG files
    image_files = sorted(glob.glob(os.path.join(images_folder, '*.png')))

    # keep updating csv
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_name', 'report'])


    for idx, image_file in enumerate(image_files):
        image_name = os.path.basename(image_file)
        image_path = image_file

        try:
            frontal = Image.open(image_path)
        except Exception as e:
            print(f"[ERROR] Could not open {image_file}: {e}")
            continue

        # --- Optional metadata ---

        indication_NG = indication_prompt_param_ng_all
        technique_NG = technique_prompt_param_ng_all
        assistant_text_NG = None

        # --- Preprocess input ---
        inputs = processor.format_and_preprocess_reporting_input(
            current_frontal = frontal,
            current_lateral = None,
            prior_frontal   = None,
            indication      = indication_NG,
            technique       = technique_NG,
            comparison      = None,
            prior_report    = None,
            assistant_text  = assistant_text_NG,
            return_tensors  = "pt",
            get_grounding   = False
        ).to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=450, use_cache=True) # no eos token warning for chest xray data with max_new_tokens = 300

        warn_if_no_eos_token_returnValue = warn_if_no_eos_token(out, processor, section=f"non-grounded (Warning 1) // all images // image: ({image_name})")

        if (not warn_if_no_eos_token_returnValue):
            prompt_len = inputs["input_ids"].shape[-1]
            raw_text   = processor.decode(out[0][prompt_len:], skip_special_tokens=True).lstrip()
            report     = processor.convert_output_to_plaintext_or_grounded_sequence(raw_text) 
        else:
            print("Re-trying report generation with different technique string (added whitespace at the end)")
            # --- Preprocess input ---
            inputs = processor.format_and_preprocess_reporting_input(
                current_frontal = frontal,
                current_lateral = None,
                prior_frontal   = None,
                indication      = indication_NG,
                technique       = (technique_NG + " "),
                comparison      = None,
                prior_report    = None,
                assistant_text  = assistant_text_NG,
                return_tensors  = "pt",
                get_grounding   = False
            ).to(device)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=450, use_cache=True) # no eos token warning for chest xray data with max_new_tokens = 300

            warn_if_no_eos_token_returnValue2 = warn_if_no_eos_token(out, processor, section=f"non-grounded (Warning 2)// all images // image: ({image_name})")
            if (not warn_if_no_eos_token_returnValue2):
                    print("Re-try successful. Now saving the report for the technique prompt with the additional whitespace. (Instead of the report without an EOS-token)")
            
            prompt_len = inputs["input_ids"].shape[-1]
            raw_text   = processor.decode(out[0][prompt_len:], skip_special_tokens=True).lstrip()
            report     = processor.convert_output_to_plaintext_or_grounded_sequence(raw_text) 

        # Save result to CSV immediately
        with open(output_csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([image_name, report])

        # Debug statement per your request
        print(f"{image_name} // report successfully generated and saved to CSV ({idx+1}/{len(image_files)})")


# ───────── 6.2.1) grounded (bounding boxes) // single report  ─────────────────────────────────────

run_grounded_report_single = False # change input filepath of image below (frontal_path) accordingly, if you turn this on

if run_grounded_report_single:
    frontal_path = "/vol/miltank/users/vac/VLM/in/dataset/chest_xrays/images/5562ea946b0ed8574dd20d05a001d6c4.png"
    frontal = Image.open(frontal_path)

    # -------- optional metadata ---------
    indication_G = None
    technique_G = "PA chest X-Ray" # vorher: "PA chest X-Ray"
    assistant_text_G = None # should not be used (reason see grounded report annotation)




    inputs_g = processor.format_and_preprocess_reporting_input(
        current_frontal=frontal,
        current_lateral=None,
        prior_frontal=None,
        indication=indication_G,
        technique=technique_G,
        comparison=None,
        prior_report=None,
        return_tensors="pt",
        assistant_text = assistant_text_G,
        get_grounding=True
    ).to(device)

    with torch.no_grad():
        out_g = model.generate(**inputs_g, max_new_tokens=600, use_cache=True)

    warn_if_no_eos_token(out_g, processor, section="6.2 grounded")


    prompt_len = inputs_g["input_ids"].shape[-1]
    raw_g = processor.decode(out_g[0][prompt_len:], skip_special_tokens=True).lstrip()
    #grounded = processor.convert_output_to_plaintext_or_grounded_sequence(raw_g)
    try:
        grounded = processor.convert_output_to_plaintext_or_grounded_sequence(raw_g)
    except AssertionError:
        print(f"[ERROR] Could not parse grounded sequence: missing end token. Raw output:\n{raw_g}\nresponsible image:{frontal_path}")
        grounded = [("UNPARSEABLE OUTPUT", raw_g)]


    print("\nGrounded findings (+ boxes):")
    for sentence, boxes in grounded:
        print(" •", sentence, boxes)

    # visualizing bounding boxes
    #visualize_bounding_boxes(grounded, frontal, box_label="coords")
    #print("\n[✓] successfully visualized bounding boxes")


#  -------------- 6.2.2) grounded report // loop all 50 images ------ #


if run_grounded_report_all:
    
    print(print_text_grounded_rep_all)
    import csv
    import glob
    import os

    images_folder   = x_ray_or_brain_images_filepath
    output_csv_path = grounded_reports_out_filepath

    


    image_files = sorted(glob.glob(os.path.join(images_folder, "*.png")))
    total = len(image_files)

    # Initialize CSV with header if not present so you can open during processing
    if not os.path.isfile(output_csv_path):
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_name', 'grounded_report'])

    for idx, img_path in enumerate(image_files, start=1):
        img_name = os.path.basename(img_path)
        try:
            frontal = Image.open(img_path)
        except Exception as e:
            print(f"[ERROR] Could not open {img_name}: {e}")
            continue


        indication_G     = indication_grounded
        technique_G      = technique_grounded
        assistant_text_G = None

        # Prepare input for grounded inference
        inputs_g = processor.format_and_preprocess_reporting_input(
            current_frontal = frontal,
            current_lateral = None,
            prior_frontal   = None,
            indication      = indication_G,
            technique       = technique_G,
            comparison      = None,
            prior_report    = None,
            assistant_text  = assistant_text_G,
            return_tensors  = "pt",
            get_grounding   = True
        ).to(device)

        with torch.no_grad():
            out_g = model.generate(**inputs_g, max_new_tokens=600, use_cache=True)

        warn_if_no_eos_token(out_g, processor, section=f"grounded Try 1 // image: ({img_name})") 

        # Decode and parse
        prompt_len = inputs_g["input_ids"].shape[-1]
        raw_g = processor.decode(out_g[0][prompt_len:], skip_special_tokens=True).lstrip()
        
        
        # ---------- Handling Assertion Error (due to missing EOS token) and re-trying with slightly mofied prompt (additional whitespace in "technique") ----------
        try:
            grounded = processor.convert_output_to_plaintext_or_grounded_sequence(raw_g)
        except AssertionError:
            print(f"[ERROR] Could not parse grounded sequence: missing end token. Raw output:\n{raw_g}\nresponsible image:{img_name}\n Now re-trying inference with an added whitespace at the end of the 'technique' prompt")
            # Fallback 1: re-trying inference with an added whitespace at the end of the 'technique' prompt

            raw_g_Cache = raw_g 

            inputs_g = processor.format_and_preprocess_reporting_input(
                current_frontal = frontal,
                current_lateral = None,
                prior_frontal   = None,
                indication      = indication_G,
                technique       = (technique_grounded + " "),
                comparison      = None,
                prior_report    = None,
                assistant_text  = assistant_text_G,
                return_tensors  = "pt",
                get_grounding   = True
            ).to(device)
            
            with torch.no_grad():
                out_g = model.generate(**inputs_g, max_new_tokens=600, use_cache=True)

            warn_if_no_eos_token(out_g, processor, section=f"grounded Try 2 // image: ({img_name})")

            # Decode and parse
            prompt_len = inputs_g["input_ids"].shape[-1]
            raw_g = processor.decode(out_g[0][prompt_len:], skip_special_tokens=True).lstrip()

            try:
                grounded = processor.convert_output_to_plaintext_or_grounded_sequence(raw_g)
                print("Re-try succesful. Saved grounded report for the adapted prompt. (Technique string with an added whitespace at the end)")
            except AssertionError:
                print(f"[ERROR] Re-try failed. Could not parse grounded sequence again: missing end token, also for the adaptet prompt. Raw output:\n{raw_g}\nresponsible image:{img_name}\n Fallback: saving")

                #Fallback 2: re-try failed. treat the whole string (with original prompt) after start token as one phrase
                grounded = [("UNPARSEABLE OUTPUT", raw_g_Cache)]
        # ---------- Handling Assertion Error (due to missing EOS token) and re-trying with slightly mofied prompt (additional whitespace in "technique") ----------



        report_str = "; ".join(f"{sent} {boxes}" for sent, boxes in grounded)

        # Save result to CSV immediately
        with open(output_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([img_name, report_str])

        print(f"{img_name} // {idx}/{total} reports successfully generated")

# ───────── 6.3) phrase grounding  ─────────────────────────────────────
# -------- 6.3.1) single output --------

run_phrase_grounding = False # change input filepath of image below (frontal_path_PG) accordingly, if you turn this on


if run_phrase_grounding:
    frontal_path_PG = "/vol/miltank/users/vac/VLM/in/dataset/chest_xrays/images/f5eb3e7e9ee9c4d08377de30251a94e2.png"
    frontal_PG = Image.open(frontal_path_PG)
    phrase_PG = "Pulmonary fibrosis."  #"Pleural effusion."  # or any phrase you want to find # "Pulmonary fibrosis" or "Pulmonary fibrosis " cause AssertionError for f5eb3e7e9ee9c4d08377de30251a94e2 -> "Pulmonary fibrosis." works

    inputs_p = processor.format_and_preprocess_phrase_grounding_input(
        frontal_image=frontal_PG,
        phrase=phrase_PG,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out_p = model.generate(**inputs_p, max_new_tokens=150, use_cache=True)

    warn_if_no_eos_token(out_p, processor, section="6.3 phrase grounding")


    prompt_len = inputs_p["input_ids"].shape[-1]
    raw_p = processor.decode(out_p[0][prompt_len:], skip_special_tokens=True).lstrip()
    phrase_grounding = processor.convert_output_to_plaintext_or_grounded_sequence(raw_p) #this line sometimes leads to an AssertionError for certain phrases
    print(f"Phrase grounding for the following phrase: '{phrase_PG}'")
    print("\nPhrase grounding result:", phrase_grounding)


# -------- 6.3.2) all 49 outputs (12 images, 49 classes) --------

# Directory of input images
images_dir = x_ray_or_brain_images_filepath


if run_phrase_grounded_reports_all:
    if type_of_run == "chest X-ray used data":
        x_ray_or_brain_images_filepath = x_ray_images_unprocessed_filepath

    phrase_grounded_print = "Starting phrase-grounded report // loop all 12 images (49 classes) (X-Ray)"
    print(phrase_grounded_print)
    # 1) Define which images + phrases to run
    if ADD_PERIOD_TO_PHRASES:
        image_phrases = {
            "07c12d0f562f17579aabc18c11e2ad54": [
                "Aortic enlargement.", "ILD.",
                "Infiltration.", "Cardiomegaly.", "Pleural thickening."
            ],
            "23b0639cd035140def992b0ee7fc34f2": [
                "Cardiomegaly.", "Aortic enlargement."
            ],
            "277b457e1e341a9194249937b68cd2c2": [
                "Lung Opacity.", "Pleural effusion."
            ],
            "4a24da485b9550c8df8b19caff945cdc": [
                "Cardiomegaly.", "Other lesion.",
                "Aortic enlargement.", "Calcification."
            ],
            "5562ea946b0ed8574dd20d05a001d6c4": [
                "Lung Opacity.", "Consolidation.", "Mass.",
                "Other lesion.", "Calcification."
            ],
            "8004676ecf95af8cee446cbcd139a938": [
                "Aortic enlargement.", "Cardiomegaly.",
                "Pulmonary fibrosis.", "Mass."
            ],
            "8de556d9cd8d026b8eba03870cc6acba": [
                "Pleural effusion.", "Pulmonary fibrosis.",
                "Lung Opacity."
            ],
            "985be77c13eb905ee8e19a45e46ab785": [
                "Pleural effusion.", "Cardiomegaly.", "Other lesion."
            ],
            "a537060564b5e08c80f46362deb565e8": [
                "Emphysema.", "Pleural effusion.", "Pleural thickening.",
                "Pulmonary fibrosis.", "Pneumothorax.", "Other lesion.",
                "Lung cyst.", "Rib fracture.", "ILD."
            ],
            "af4c1f381399cfac17a6e0b983261a4e": [
                "Cardiomegaly.", "Aortic enlargement."
            ],
            "e4e32ce0e061d700c0afda13faa45b1d": [
                "Pleural effusion.", "Rib fracture.",
                "Other lesion.", "Infiltration."
            ],
            "f5eb3e7e9ee9c4d08377de30251a94e2": [
                "Pulmonary fibrosis.", "Pleural effusion.",
                "Pleural thickening.", "Aortic enlargement.",
                "Other lesion.", "Lung Opacity."
            ],
        }
    else:
    
        image_phrases = {
        "07c12d0f562f17579aabc18c11e2ad54": [
            "Aortic enlargement", "ILD",
            "Infiltration", "Cardiomegaly", "Pleural thickening"
        ],
        "23b0639cd035140def992b0ee7fc34f2": [
            "Cardiomegaly", "Aortic enlargement"
        ],
        "277b457e1e341a9194249937b68cd2c2": [
            "Lung Opacity", "Pleural effusion"
        ],
        "4a24da485b9550c8df8b19caff945cdc": [
            "Cardiomegaly", "Other lesion",
            "Aortic enlargement", "Calcification"
        ],
        "5562ea946b0ed8574dd20d05a001d6c4": [
            "Lung Opacity", "Consolidation", "Mass",
            "Other lesion", "Calcification"
        ],
        "8004676ecf95af8cee446cbcd139a938": [
            "Aortic enlargement", "Cardiomegaly",
            "Pulmonary fibrosis", "Mass"
        ],
        "8de556d9cd8d026b8eba03870cc6acba": [
            "Pleural effusion", "Pulmonary fibrosis",
            "Lung Opacity"
        ],
        "985be77c13eb905ee8e19a45e46ab785": [
            "Pleural effusion", "Cardiomegaly", "Other lesion"
        ],
        "a537060564b5e08c80f46362deb565e8": [
            "Emphysema", "Pleural effusion", "Pleural thickening",
            "Pulmonary fibrosis", "Pneumothorax", "Other lesion",
            "Lung cyst", "Rib fracture", "ILD"
        ],
        "af4c1f381399cfac17a6e0b983261a4e": [
            "Cardiomegaly", "Aortic enlargement"
        ],
        "e4e32ce0e061d700c0afda13faa45b1d": [
            "Pleural effusion", "Rib fracture",
            "Other lesion", "Infiltration"
        ],
        "f5eb3e7e9ee9c4d08377de30251a94e2": [
            "Pulmonary fibrosis", "Pleural effusion",
            "Pleural thickening", "Aortic enlargement",
            "Other lesion", "Lung Opacity"
        ],
    }

    # 2) Prepare JSON file for streaming results
    json_path = phrase_grounded_reports_out_filepath
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    if not os.path.exists(json_path):
        with open(json_path, 'w') as f:
            json.dump({}, f)  # start with empty object

    # 3) Load any existing results so we can append in‐memory
    with open(json_path, 'r') as f:
        pg_results = json.load(f)

    

    # Count total phrase‐runs for progress tracking
    total_runs = sum(len(v) for v in image_phrases.values())
    run_idx = 0

    for img_id, phrases in image_phrases.items():
        img_path = os.path.join(images_dir, f"{img_id}.png")
        pg_results.setdefault(img_id, {})  # ensure key exists

        for phrase in phrases:
            run_idx += 1
            try:
                img = Image.open(img_path)
                inputs_p = processor.format_and_preprocess_phrase_grounding_input(
                    frontal_image=img,
                    phrase=phrase,
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    out_p = model.generate(**inputs_p, max_new_tokens=450, use_cache=True)

                warn_if_no_eos_token(out_p, processor, section=img_id)

                prompt_len = inputs_p["input_ids"].shape[-1]
                raw_p = processor.decode(out_p[0][prompt_len:], skip_special_tokens=True).lstrip()
                phrase_grounding = processor.convert_output_to_plaintext_or_grounded_sequence(raw_p)

                # Build array-of-arrays: [x0,y0,x1,y1,sentence] or [None,sentence]
                entries = []
                for sentence, boxes in phrase_grounding:
                    if not boxes:
                        entries.append([None, sentence])
                    else:
                        for (x0, y0, x1, y1) in boxes:
                            entries.append([x0, y0, x1, y1, sentence])

                pg_results[img_id][phrase] = entries

            except AssertionError as ae:
                # pg_results[img_id][phrase] = {"error": f"AssertionError: {ae}"}
                print(f"AssertionError caught when trying to predict Bboxes for image // phrase: \n {img_id} // {phrase}")

                extended_phrase = phrase + "."

                print("Starting second try to predict the Bbox with 2 added whitespaces after the phrase. This time, the same phrase is extended by a period '.' " \
                f"\n Concretely, the second try usesthis phrase: '{extended_phrase}'")

                try:
                    img = Image.open(img_path)
                    inputs_p = processor.format_and_preprocess_phrase_grounding_input(
                        frontal_image=img,
                        phrase=extended_phrase,
                        return_tensors="pt"
                    ).to(device)

                    with torch.no_grad():
                        out_p = model.generate(**inputs_p, max_new_tokens=450, use_cache=True)

                    warn_if_no_eos_token(out_p, processor, section=img_id)

                    prompt_len = inputs_p["input_ids"].shape[-1]
                    raw_p = processor.decode(out_p[0][prompt_len:], skip_special_tokens=True).lstrip()
                    phrase_grounding = processor.convert_output_to_plaintext_or_grounded_sequence(raw_p)

                    # Build array-of-arrays: [x0,y0,x1,y1,sentence] or [None,sentence]
                    entries = []
                    for sentence, boxes in phrase_grounding:
                        if not boxes:
                            entries.append([None, sentence])
                        else:
                            for (x0, y0, x1, y1) in boxes:
                                entries.append([x0, y0, x1, y1, sentence])

                    pg_results[img_id][extended_phrase] = entries
                    print("Second try was successful!")

                except AssertionError as ae:
                    print("Re-try lead to a second AssertionError. Now writing 'AssertionError : <Error_description_if_available>' to the .json result file 'phrase_grounded_reports_X_Ray_unproc.json ")
                    pg_results[img_id][phrase] = {"error": f"Phrases '{phrase}' and '{extended_phrase}' both lead to an. The latter lead to this AssertionError: {ae}"}
                
                except Exception as e:
                    print("Re-try lead to another Error. Now writing 'Error : <Error_description_if_available>' to the .json result file 'phrase_grounded_reports_X_Ray_unproc.json ")
                    pg_results[img_id][phrase] = {"error": str(e)}

            except Exception as e:
                print(f"Error caught when trying to predict Bboxes for image // phrase: \n {img_id} // {phrase}\n" \
                    "Now writing 'Error : <Error_description_if_available>' to the .json result file 'phrase_grounded_reports_X_Ray_unproc.json ")
                pg_results[img_id][phrase] = {"error": str(e)}
                


            # Write updated JSON after each phrase‐run
            with open(json_path, 'w') as f:
                json.dump(pg_results, f, indent=2)

            # Debug print
            print(f"{img_id} // {run_idx}/{total_runs} reports successfully generated")




# -------- 6.3.3) phrase grounded // all 92 outputs (same phrase for all 92) --------

# Directory of input images
images_dir = x_ray_or_brain_images_filepath

if run_phrase_grounded_reports_single_phrase_92_brain:
    print(phrase_grounded_reports_single_phrase_92_brain_print_text)
    # 1) Define a single, easily-modifiable phrase and collect all image IDs
    
    image_ids = [
        os.path.splitext(fname)[0]
        for fname in os.listdir(images_dir)
        if fname.lower().endswith(".png")
    ]

    # 2) Debug print
    phrase_grounded_print = f"Starting phrase-grounded report // loop all {len(image_ids)} images (1 phrase)"
    print(phrase_grounded_print)

    # 3) Prepare JSON file for streaming results

    json_path = phrase_grounded_reports_single_phrase_92_out_filepath
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    if not os.path.exists(json_path):
        with open(json_path, 'w') as f:
            json.dump({}, f)

    # 4) Load any existing results so we can append in memory
    with open(json_path, 'r') as f:
        pg_results = json.load(f)

    # 5) Count total runs and initialize counter
    total_runs = len(image_ids)
    run_idx = 0

    # 6) Loop over every image, using the same phrase each time
    for img_id in image_ids:
        img_path = os.path.join(images_dir, f"{img_id}.png")
        pg_results.setdefault(img_id, {})  # ensure key exists

        run_idx += 1
        try:
            img = Image.open(img_path)
            inputs_p = processor.format_and_preprocess_phrase_grounding_input(
                frontal_image=img,
                phrase=phrase_grounded_single_phrase_92_report_phrase,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                out_p = model.generate(**inputs_p, max_new_tokens=300, use_cache=True)

            warn_if_no_eos_token(out_p, processor, section=img_id)

            prompt_len = inputs_p["input_ids"].shape[-1]
            raw_p = processor.decode(out_p[0][prompt_len:], skip_special_tokens=True).lstrip()
            phrase_grounding = processor.convert_output_to_plaintext_or_grounded_sequence(raw_p)

            # Build array-of-arrays: [x0, y0, x1, y1, sentence] or [None, sentence]
            entries = []
            for sentence, boxes in phrase_grounding:
                if not boxes:
                    entries.append([None, sentence])
                else:
                    for (x0, y0, x1, y1) in boxes:
                        entries.append([x0, y0, x1, y1, sentence])

            # Store under the single phrase key
            pg_results[img_id][phrase_grounded_single_phrase_92_report_phrase] = entries

        except AssertionError as ae:
            pg_results[img_id][phrase_grounded_single_phrase_92_report_phrase] = {"error": f"AssertionError: {ae}"}
        except Exception as e:
            pg_results[img_id][phrase_grounded_single_phrase_92_report_phrase] = {"error": str(e)}

        # 7) Write updated JSON after each image
        with open(json_path, 'w') as f:
            json.dump(pg_results, f, indent=2)

        # 8) Progress print
        print(f"{img_id} // {run_idx}/{total_runs} report successfully generated")


# ───────── 8.1) DONE ─────────────────────────────────────────────────────────
print("\n[✓] Finished run.")

#runtime printout
total_runtime = time.time() - script_start_time
print(f"[INFO] Total runtime: {total_runtime:.2f} seconds")


# ───────── 8.2) Post-MAIRA ─────────────────────────────────────────────────────────

# visualizing bounding boxes -> see after grounded report
#visualizing without report generation:

visualize = False

if visualize:
    frontal_path_visualize = "C:\\Users\\franc\\OneDrive - TUM\\1 - VLM - AI for Vision Lang Mod in Med Seminar\\VLM_dataset\\chest_xrays\\images\\8004676ecf95af8cee446cbcd139a938.png"
    frontal_visualize = Image.open(frontal_path_visualize)

    text_report = """Grounded findings (+ boxes):
    • The lungs are adequately inflated. None
    • No focal airspace opacity. None
    • No pleural effusion. None
    • No pneumothorax. None
    • Peribronchial cuffing is suggested. [(0.295, 0.245, 0.675, 0.545)]
    • Normal cardiomediastinal silhouette. None
    • Normal imaged portion of the upper abdomen. None
    • Degenerative changes are present at the spine. [(0.405, 0.185, 0.585, 0.975)]"""
    # visualize_bounding_boxes(text_report, frontal_visualize, box_label="coords")


    #visualization 2

    frontal_path_visualize2 = "C:\\Users\\franc\\OneDrive - TUM\\1 - VLM - AI for Vision Lang Mod in Med Seminar\\VLM_dataset\\chest_xrays\\images\\5562ea946b0ed8574dd20d05a001d6c4.png"
    frontal_visualize2 = Image.open(frontal_path_visualize2)
    text_report2 = "The cardiac silhouette is within normal limits. None; 3.8 cm masslike opacity in the right lower lung. [(0.185, 0.505, 0.405, 0.705)]; 1.5 cm nodular opacity in the left upper lung. [(0.705, 0.265, 0.765, 0.345)]; No pleural effusion. None; No pneumothorax. None; No acute osseous abnormality. None"
    # visualize_bounding_boxes(text_report2, frontal_visualize2, box_label="sentence")






# Plays a short melody when code has finished running
playSound = False
if playSound:

    import winsound
    import time
    # Defining a melody as (frequency in Hz, duration in ms)
    melody = [
        (659, 200),  # E5
        (784, 200),  # G5
        #(880, 200),  # A5
        #(784, 200),  # G5
        #(659, 400),  # E5
    ]

    for freq, dur in melody:
        winsound.Beep(freq, dur)
        time.sleep(0.025)  # brief pause between notes