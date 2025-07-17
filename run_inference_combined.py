import os
import yaml
import h5py
import torch
from types import SimpleNamespace

from timebase.models.models import Classifier
from timebase.data.static import CHANNELS_FREQ
from train_ann import load_pre_trained_parameters, load  # used only for finetuned

# --- Shared Constants ---
USED_CHANNELS = ['ACC_x', 'ACC_y', 'ACC_z', 'BVP', 'EDA', 'TEMP']

# --- Common Step 1: Load args.yaml ---
def load_args(args_path, used_channels, allow_pretrain_merge=True):
    with open(args_path, "r") as f:
        args_config = yaml.safe_load(f)

    if allow_pretrain_merge and 'path2pretraining_res' in args_config:
        pretrain_args_path = os.path.join(args_config['path2pretraining_res'], 'args.yaml')
        if os.path.exists(pretrain_args_path):
            with open(pretrain_args_path, 'r') as pf:
                pretrain_args = yaml.safe_load(pf)
                pretrain_args.update(args_config)
                args_config = pretrain_args

    args = SimpleNamespace(**args_config)
    args.ds_info = {'channel_freq': CHANNELS_FREQ, 'segment_length': 512}
    args.input_shapes = {ch: (L,) for ch, L in CHANNELS_FREQ.items() if ch in used_channels}
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args

# --- Common Step 2: Load H5 data ---
def load_data_from_h5(h5_path, used_channels):
    input_data = {}
    with h5py.File(h5_path, "r") as f:
        for ch in used_channels:
            if ch in f:
                input_data[ch] = torch.tensor(f[ch][()], dtype=torch.float32).unsqueeze(0)
    return input_data

# --- Common Step 3: Print results ---
def print_inference_result(h5_file, logit, prob, pred):
    print("\n--- Inference Complete ---")
    print(f"File: {h5_file}")
    print(f"Logits: {logit:.4f}")
    print(f"Probability (class 1): {prob:.2%}")
    print(f"Predicted Label: {pred} (0 = non-episode, 1 = emotional episode)")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# === Inference mode 1: Fully supervised ===
def run_inference_supervised(h5_path, ckpt_path, args_path):
    print("🔵 Running fully supervised inference...")

    args = load_args(args_path, USED_CHANNELS, allow_pretrain_merge=False)
    model = Classifier(args)

    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(model)
    print(count_parameters(model))

    input_data = load_data_from_h5(h5_path, USED_CHANNELS)

    with torch.no_grad():
        logits, _ = model(input_data)
        prob_class_1 = torch.sigmoid(logits)
        pred_label = (prob_class_1 > 0.5).int()

    print_inference_result(h5_path, logits.item(), prob_class_1.item(), pred_label.item())

# === Inference mode 2: Fine-tuned with pretrained backbone ===
def run_inference_finetuned(h5_path, ckpt_path, args_path):
    print("🟢 Running fine-tuned inference (with pretraining)...")

    args = load_args(args_path, USED_CHANNELS, allow_pretrain_merge=True)
    model = Classifier(args)

    load_pre_trained_parameters(args, classifier=model, path2pretraining_res=args.path2pretraining_res)

    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(model)
    print(count_parameters(model))

    input_data = load_data_from_h5(h5_path, USED_CHANNELS)

    with torch.no_grad():
        inputs_on_device = load(input_data, device=args.device)
        logits, _ = model(inputs_on_device)
        prob_class_1 = torch.sigmoid(logits)
        pred_label = (prob_class_1 > 0.5).int()

    print_inference_result(h5_path, logits.item(), prob_class_1.item(), pred_label.item())

if __name__ == "__main__":
    print("🧪 Running both inference modes from __main__...")

    # === Fully supervised ===
    run_inference_supervised(
        h5_path="data/preprocessed/sl512_ss128_unlabelled/wesad/S2/S2_E4_Data/0.h5",
        ckpt_path="runs/sl_ann_test/ckpt_classifier/model_state.pt",
        args_path="runs/sl_ann_test/args.yaml"
    )

    print("\n" + "="*60 + "\n")

    # === Fine-tuned with pretrained backbone ===
    run_inference_finetuned(
        h5_path="data/preprocessed/sl512_ss128_unlabelled/wesad/S2/S2_E4_Data/0.h5",
        ckpt_path="runs/masked_prediction_fine_tuning_test/ckpt_classifier/model_state.pt",
        args_path="runs/masked_prediction_fine_tuning_test/args.yaml"
    )
