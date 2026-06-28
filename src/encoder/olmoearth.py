"""
OLMoEarth encoder wrapper.

Supports two modes:
- local: loads from local weights.pth state_dict
- cloud: downloads weights.pth from HuggingFace and loads

Both modes use the same minimal encoder architecture (no rslearn needed).
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class PatchEmbed(nn.Module):
    """Per-modality patch embedding: pixel_proj + linear proj."""
    def __init__(self, in_channels, embed_dim=768, out_dim=128):
        super().__init__()
        self.pixel_proj = nn.Linear(in_channels, in_channels)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.squeeze(-1).squeeze(-1)
        x = self.pixel_proj(x)
        if x.shape[-1] < 768:
            pad = torch.zeros(*x.shape[:-1], 768 - x.shape[-1], device=x.device)
            x = torch.cat([x, pad], dim=-1)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""
    def __init__(self, dim=128, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class OlmoEarthEncoder(nn.Module):
    """
    Minimal OLMoEarth encoder built from state_dict keys.
    Only the encoder path is used for embedding extraction.
    """
    def __init__(self, dim=128, n_blocks=4, modality_configs=None):
        super().__init__()
        self.dim = dim

        if modality_configs is None:
            modality_configs = {
                "sentinel2_l2a": 12,
                "sentinel1": 2,
                "landsat": 11,
                "worldcover": 1,
                "srtm": 1,
                "openstreetmap_raster": 30,
                "wri_canopy_height_map": 1,
                "cdl": 1,
                "worldcereal": 8,
            }

        self.patch_embeddings = nn.ModuleDict()
        for name, in_ch in modality_configs.items():
            self.patch_embeddings[name] = PatchEmbed(in_ch, 768, dim)

        self.composite_encodings = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim) for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(dim)
        self.project_and_aggregate = nn.Sequential(
            nn.Linear(dim, dim),
        )

    def forward(self, x, modality="sentinel2_l2a"):
        if modality not in self.patch_embeddings:
            modality = list(self.patch_embeddings.keys())[0]

        if x.shape[-1] == 13:
            x = torch.cat([x[:, :, :8], x[:, :, 9:]], dim=-1)

        h = self.patch_embeddings[modality](x)
        h = h + self.composite_encodings

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)
        h = h.mean(dim=1)
        h = self.project_and_aggregate(h)
        return h

    @staticmethod
    def from_state_dict(weights_path: str, device: str = "cpu"):
        state_dict = torch.load(weights_path, map_location=device)

        enc_keys = {k: v for k, v in state_dict.items()
                    if k.startswith("encoder.")}

        model = OlmoEarthEncoder()

        mapped = {}
        for k, v in enc_keys.items():
            new_k = k.replace("encoder.", "", 1)
            mapped[new_k] = v

        model.load_state_dict(mapped, strict=False)
        model.eval()
        model.to(device)
        return model


class OLMoEarthEncoder:
    """Wrapper supporting both local and cloud modes."""
    def __init__(self, mode="local", local_weights_path=None,
                 cloud_model_id=None, device="cpu"):
        if device == "cuda" and not torch.cuda.is_available():
            print("[OLMoEarth] CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = device
        self.mode = mode

        if mode == "local":
            self.model = OlmoEarthEncoder.from_state_dict(
                local_weights_path, device
            )
        else:
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(cloud_model_id, "weights.pth")
            self.model = OlmoEarthEncoder.from_state_dict(
                weights_path, device
            )

    @torch.inference_mode()
    def encode(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        use_amp = self.device == "cuda"

        for i in tqdm(range(0, len(X), batch_size), desc="Encoding"):
            batch = X[i:i + batch_size]
            tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(tensor)
            else:
                output = self.model(tensor)
            all_embeddings.append(output.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)
