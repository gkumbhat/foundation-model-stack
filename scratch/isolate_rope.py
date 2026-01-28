"""
An isolated test for checking the correctness of Pixtral Rope in FMS against
HF Transformers; we need to be careful about this due to Pixtral Rope being 2D,
then flattening to 1D patches, because the channel order in Transformers/HF
rotary embeddings is different.
"""
import torch
from transformers import AutoConfig
from transformers.models.pixtral.modeling_pixtral import PixtralRotaryEmbedding as TransformersPixtralEmb
from transformers.models.pixtral.modeling_pixtral import apply_rotary_pos_emb, position_ids_in_meshgrid
from fms.modules.positions import PixtralRotaryEmbedding as FMSPixtralEmb
from fms.models.hf.config_utils.param_builders import build_pixtral_params
from fms.models.pixtral_vision import PixtralVisionConfig

MODEL_PATH = "/home/senuser/projects/vllm-spyre/examples/offline_inference/Mistral-Small-3.2-24B-Instruct-2506"
# Example inputs from the pokemon example dumped after the input transform,
# i.e., the kq that we are applying the positional encoding to
Q_PROJ = "/home/senuser/projects/foundation-model-stack/scratch/encoder_outputs/hf/q_proj.pt"
K_PROJ = "/home/senuser/projects/foundation-model-stack/scratch/encoder_outputs/hf/k_proj.pt"


def load_kq():
    """Load example key / query dumped from running Alex's script in Transformers

    This is only for debugging (NOTE: alex dumped q_proj BEFORE the transpose, which is actually the FMS input;
    so the transpose here gives the format for transformers, which is the same as the zerp shape.
    """
    q = torch.load(Q_PROJ).transpose(1, 2)
    k = torch.load(K_PROJ).transpose(1, 2)
    return q, k

def get_sample_inputs(model_path, is_zero):
    # Number of zero convolutional patches
    patch_h, patch_w = 1, 2 # FIXME - why does it pass with zerod inputs?
    num_patches = patch_h * patch_w # 1064    

    hf_config = AutoConfig.from_pretrained(model_path).vision_config
    fms_config = PixtralVisionConfig(**build_pixtral_params(hf_config))

    # Create a sample input to be positionally encoded; we create a dtype half
    # of shape [1, 16, 1064, 64], which is [bsz, num_heads, num_patches, head_dim];
    # this is the shape of a query projection to this model in HF.
    head_dim = fms_config.hidden_size // fms_config.nheads
    # TODO - once it works, we can remove the is_zero flag and use this as a test
    if is_zero:
        query_proj = torch.ones((1, fms_config.nheads, num_patches, head_dim), dtype=torch.float16)
        key_proj = torch.ones((1, fms_config.nheads, num_patches, head_dim), dtype=torch.float16)
    else:
        query_proj, key_proj = load_kq()
    position_ids = position_ids_in_meshgrid( # position_ids = tensor([0, 1])
        patch_embeds_list=[torch.rand((fms_config.hidden_size, patch_h, patch_w))],
        max_width=hf_config.image_size // hf_config.patch_size,
    )
    # kwargs that can be directly expanded to both the transformers / fms wrappers
    return {
        "hf_config": hf_config,
        "fms_config": fms_config,
        "head_dim": head_dim,
        "position_ids": position_ids,
        "query_proj": query_proj,
        "key_proj": key_proj,
    }


def get_hf_pixtral_rope_adjusted_kq(*, hf_config, position_ids, query_proj, key_proj, **kwargs):
    transformers_emb = TransformersPixtralEmb(hf_config)
    cos, sin = transformers_emb(query_proj, position_ids)
    query_hf, key_hf = apply_rotary_pos_emb(query_proj, key_proj, cos, sin, unsqueeze_dim=0)
    return query_hf, key_hf

def get_fms_pixtral_rope_adjusted_kq(*, fms_config, head_dim, position_ids, query_proj, key_proj, **kwargs):
    fms_emb = FMSPixtralEmb(
        dim=head_dim,
        ratio=fms_config.rope_theta,
        image_size=fms_config.image_size,
        patch_size=fms_config.patch_size,
    )

    # In FMS, the query and key are viewed as [1, 1064, 16, 64], i.e,.
    # [bsz, num_patches, num_heads, head_dim] prior to invoking the rotational
    # embeddings, so we need to permute the inputs.
    query_proj_fms = query_proj.transpose(1, 2)
    key_proj_fms = key_proj.transpose(1, 2)
    query_fms, key_fms = fms_emb.adjusted_qk(
        query_proj_fms,
        key_proj_fms,
        position_ids.unsqueeze(0), # FMS expects [bsz, seq_len]
        past_kv_state=None,
        use_cache=False,
    )
    query_fms = query_fms.transpose(1, 2)
    key_fms = key_fms.transpose(1, 2)
    return query_fms, key_fms

def test_hf_fms_pixtral_rope_correctness(model_path, is_zero: bool=True):
    inps = get_sample_inputs(model_path, is_zero=is_zero)
    q_hf, k_hf = get_hf_pixtral_rope_adjusted_kq(**inps)
    q_fms, k_fms = get_fms_pixtral_rope_adjusted_kq(**inps)
    # Ensure k/q adjusted have the same dtype and shape
    assert q_hf.dtype == q_fms.dtype
    assert k_hf.dtype == k_fms.dtype
    assert q_hf.shape == q_fms.shape
    assert k_hf.shape == k_fms.shape

    q_matches = torch.allclose(q_hf, q_fms)
    k_matches = torch.allclose(k_hf, k_fms)
    
    if not q_matches:
        print(f"Query doesn't match! Max absolute diff: {torch.max(q_hf - q_fms)}") # 5.62
    if not k_matches:
        print(f"Key doesn't match! Max absolute diff: {torch.max(k_hf - k_fms)}") # 6.34

    if q_matches and k_matches:
        print("Query and key match!")
    

if __name__ == "__main__":
    # NOTE: We can use zero=False to fix the positional encoding issues,
    # and then we can use zero=True for parity in a unit test within FMS
    # for a more comprehensive test.
    test_hf_fms_pixtral_rope_correctness(MODEL_PATH, is_zero=True)
