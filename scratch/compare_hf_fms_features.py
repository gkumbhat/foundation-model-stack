"""
Quick and dirty script for comparing features dumped out from somewhere in a transformer,
e.g., during prefill.
"""
import os
import torch

OUT_DIR = os.path.join(os.path.dirname(__file__), "encoder_outputs")
HF_OUT_DIR = os.path.join(OUT_DIR, "hf")
FMS_OUT_DIR = os.path.join(OUT_DIR, "fms")
MATCH_STR = "✅"
MISMATCH_STR = "❌"

# Put these in order that they are encountered in prefill
# so that we can see the first place at which the visual inputs
# diverge.
TENSOR_FILE_ORDER = [
    # "pixel_values.pt",
    # "patch_embeds.pt",
    # "normed_flat_patch_embeds.pt",
    # "position_ids.pt", # < maybe need to unsqueeze here, not sure
    # ### Dying somewhere between this
    "block_1_input.pt",
    "block_1_attn_norm.pt",
    "q_proj.pt",
    # "k_proj.pt",
    # "v_proj.pt", 
    # # ^^^ This is all ok; prior to transposition, kqv projections DO match correctly.
    # # In transformers, we pass Rope: [1, 16, 1064, 64]
    # # But in FMS it's: [1, 1064, 16, 64] - note, dim is the head dimension
    # # When we actually apply it. Maybe that is causing problems?
    # "inv_freq.pt",
    # "q_proj_pos_encoded.pt",
    # "k_proj_pos_encoded.pt",
    
    # ### Outputs diverge here! So definitely a problem in attn
    "block_1_post_attn_no_residual.pt",
    "block_1_ff_norm.pt",
    "block_1_hidden_states.pt",
    # "image_features.pt",
    # "inputs_embeds.pt",    
]

fmt = lambda matches: MATCH_STR if matches else MISMATCH_STR

def show_prop_name_diff(attr_name, hf_out, fms_out):
    hf_attr = getattr(hf_out, attr_name)
    fms_attr = getattr(fms_out, attr_name)
    is_same = hf_attr == fms_attr
    print(f"    -> Same {attr_name}? {fmt(is_same)}")
    if not is_same:
        print(f"       -> FMS {attr_name} {fms_attr}")
        print(f"       -> HF {attr_name} {hf_attr}")
    else:
        print(f"       -> {attr_name} {fms_attr}")

def compare_and_print(file_name, hf_out, fms_out):
    same_dtype = fms_out.dtype == hf_out.dtype
    same_shape = fms_out.shape == hf_out.shape
    print("-------------{file_name}------------")
    show_prop_name_diff("dtype", hf_out, fms_out)
    show_prop_name_diff("shape", hf_out, fms_out)
    # Check the actual data to see closeness
    max_diff = torch.max(hf_out - fms_out)
    match = torch.allclose(fms_out, hf_out)
    print(f"    -> Tensors all close? {fmt(match)}")
    if not match:
        print(f"        -> Tensors diverge! Max diff {max_diff:.2f}")
    print("\n\n")


for file in TENSOR_FILE_ORDER:
    print("Comparing files {}".format(file))
    fms_out = torch.load(os.path.join(FMS_OUT_DIR, file))
    hf_out = torch.load(os.path.join(HF_OUT_DIR, file))
    compare_and_print(file, hf_out, fms_out)
