# run_inference_fully.py

import os
import yaml
import h5py
import torch
from types import SimpleNamespace

# Model imports
from timebase.models.models import Classifier
from timebase.data.static import EMOTIBIT_CHANNELS, EMOTIBIT_CHANNELS_FREQ

# --- Config ---
SUPERVISED_RUN_DIR = "runs/sl_ann_test"
TARGET_H5_FILE = "data/preprocessed/sl512_ss128/data/raw_data/barcelona/1-1/61.h5"
SEGMENT_LENGTH = 512

# --- Step 1: Load config and input shapes ---
def load_args(run_dir, h5_file, used_channels):
    args_path = os.path.join(run_dir, 'args.yaml')
    with open(args_path, "r") as f:
        args_dict = yaml.safe_load(f)
    args = SimpleNamespace(**args_dict)
    args.ds_info = {'channel_freq': EMOTIBIT_CHANNELS_FREQ, 'segment_length': SEGMENT_LENGTH}
    with h5py.File(h5_file, "r") as f:
        args.input_shapes = {ch: f[ch].shape for ch in f if ch in used_channels}
    return args

# --- Step 2: Load model ---
def load_model(args, run_dir):
    model = Classifier(args)
    ckpt_path = os.path.join(run_dir, 'ckpt_classifier/model_state.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")['model'])
    model.eval()
    return model

# --- Step 3: Load data ---
def load_data(h5_file, used_channels):
    input_data = {}
    with h5py.File(h5_file, "r") as f:
        for ch in used_channels:
            if ch in f:
                input_data[ch] = torch.tensor(f[ch][()], dtype=torch.float32).unsqueeze(0)
    return input_data

# --- Step 4: Inference ---
def run_inference(model, inputs):
    with torch.no_grad():
        logits, _ = model(inputs)
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).int().item()
    return logits.item(), prob.item(), pred

# --- Step 5: Print result ---
def print_result(logit, prob, pred):
    print("\n--- Inference Result ---")
    print(f"Logits: {logit:.4f}")
    print(f"Probability (emotional episode): {prob:.2%}")
    print(f"Predicted label: {pred} (0 = non-episode, 1 = emotional episode)")

# --- Main ---
if __name__ == "__main__":
    try:
        print("--- Script started ---")
        args = load_args(SUPERVISED_RUN_DIR, TARGET_H5_FILE, EMOTIBIT_CHANNELS)
        model = load_model(args, SUPERVISED_RUN_DIR)
        print("✅ Model loaded successfully")

        input_data = load_data(TARGET_H5_FILE, EMOTIBIT_CHANNELS)
        print("✅ Data loaded successfully")

        logit, prob, pred = run_inference(model, input_data)
        print_result(logit, prob, pred)

    except Exception as e:
        print("\n❌ Script encountered an error:")
        import traceback
        traceback.print_exc()
