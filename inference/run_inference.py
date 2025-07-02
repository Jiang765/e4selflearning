# run_inference_v2.py

import os
import yaml
import h5py
import torch
from types import SimpleNamespace

# Direct model and function imports
from timebase.models.models import Classifier
from timebase.data.static import CHANNELS_FREQ
from train_ann import load_pre_trained_parameters, load

# --- Configurable paths ---
TARGET_H5_FILE = "data/preprocessed/sl512_ss128_unlabelled/wesad/S2/S2_E4_Data/0.h5"
FINETUNE_CKPT_PATH = "runs/masked_prediction_fine_tuning_test/ckpt_classifier/model_state.pt"
FINETUNE_ARGS_PATH = "runs/masked_prediction_fine_tuning_test/args.yaml"
USED_CHANNELS = ['ACC_x', 'ACC_y', 'ACC_z', 'BVP', 'EDA', 'TEMP']

# --- Step 1: Load args from YAML and merge with pretraining config ---
def load_args(finetune_args_path, used_channels):
    with open(finetune_args_path, "r") as f:
        args_dict = yaml.safe_load(f)

    # Merge with pretraining config if exists
    if 'path2pretraining_res' in args_dict:
        pretrain_args_path = os.path.join(args_dict['path2pretraining_res'], 'args.yaml')
        if os.path.exists(pretrain_args_path):
            with open(pretrain_args_path, 'r') as pf:
                pretrain_args = yaml.safe_load(pf)
                pretrain_args.update(args_dict)
                args_dict = pretrain_args

    args = SimpleNamespace(**args_dict)
    args.ds_info = {'channel_freq': CHANNELS_FREQ, 'segment_length': 512}
    args.input_shapes = {ch: (L,) for ch, L in CHANNELS_FREQ.items() if ch in used_channels}
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args

# --- Step 2: Load model and weights ---
def load_model_from_ckpt(args, ckpt_path):
    model = Classifier(args)
    load_pre_trained_parameters(args, classifier=model, path2pretraining_res=args.path2pretraining_res)
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model

# --- Step 3: Load data from HDF5 file ---
def load_data_from_h5(h5_file_path, used_channels):
    input_data_dict = {}
    with h5py.File(h5_file_path, "r") as f:
        for ch in used_channels:
            if ch in f:
                input_data_dict[ch] = torch.tensor(f[ch][()], dtype=torch.float32).unsqueeze(0)
    return input_data_dict

# --- Step 4: Run inference ---
def run_inference(model, input_data, device):
    with torch.no_grad():
        inputs_on_device = load(input_data, device=device)
        logits, _ = model(inputs_on_device)
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).int()
    return logits.item(), prob.item(), pred.item()

# --- Step 5: Print result ---
def print_inference_result(h5_file, logit, prob, pred):
    print("\n--- Inference Complete ---")
    print(f"File: {h5_file}")
    print(f"Logits: {logit:.4f}")
    print(f"Probability (class 1): {prob:.2%}")
    print(f"Predicted Label: {pred} (0 = non-episode, 1 = emotional episode)")

# --- Main ---
if __name__ == "__main__":
    try:
        print("--- Script started ---")
        args = load_args(FINETUNE_ARGS_PATH, USED_CHANNELS)
        print("✅ Args loaded.")
        # Cell 2: Load model
        model = load_model_from_ckpt(args, FINETUNE_CKPT_PATH)
        print("✅ Model loaded.")
        # Cell 3: Load data
        input_data = load_data_from_h5(TARGET_H5_FILE, USED_CHANNELS)
        print("✅ Data loaded.")
        # Cell 4: Inference
        logit, prob, pred = run_inference(model, input_data, args.device)
        print_inference_result(TARGET_H5_FILE, logit, prob, pred)

    except Exception as e:
        print("\n❌ Script encountered an error:")
        import traceback
        traceback.print_exc()