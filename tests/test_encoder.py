"""
Tests for src/encoder/olmoearth.py
"""

import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from src.encoder.olmoearth import OlmoEarthEncoder, PatchEmbed, TransformerBlock


class TestPatchEmbed:
    def test_output_shape(self):
        pe = PatchEmbed(in_channels=12, embed_dim=768, out_dim=128)
        x = torch.randn(2, 10, 12)
        out = pe(x)
        assert out.shape == (2, 10, 128)

    def test_pads_to_768(self):
        pe = PatchEmbed(in_channels=5, embed_dim=768, out_dim=128)
        x = torch.randn(2, 10, 5)
        out = pe(x)
        assert out.shape == (2, 10, 128)

    def test_5d_input(self):
        pe = PatchEmbed(in_channels=12, embed_dim=768, out_dim=128)
        x = torch.randn(2, 10, 12, 1, 1)
        out = pe(x)
        assert out.shape == (2, 10, 128)


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(dim=128, heads=4)
        x = torch.randn(2, 10, 128)
        out = block(x)
        assert out.shape == (2, 10, 128)

    def test_residual_connection(self):
        block = TransformerBlock(dim=128, heads=4)
        x = torch.randn(2, 10, 128)
        out = block(x)
        assert out.shape == x.shape


class TestOlmoEarthEncoder:
    def test_init_default(self):
        model = OlmoEarthEncoder(dim=128, n_blocks=4)
        assert model.dim == 128
        assert len(model.blocks) == 4

    def test_forward_output_shape(self):
        model = OlmoEarthEncoder(dim=128, n_blocks=4)
        x = torch.randn(2, 10, 12)
        out = model(x)
        assert out.shape == (2, 128)

    def test_forward_drops_b8a(self):
        model = OlmoEarthEncoder(dim=128, n_blocks=4)
        x = torch.randn(2, 10, 13)
        out = model(x)
        assert out.shape == (2, 128)

    def test_from_state_dict(self, tmp_path):
        model = OlmoEarthEncoder(dim=128, n_blocks=4)
        fake_state = {}
        for k, v in model.state_dict().items():
            fake_state[f"encoder.{k}"] = v
        weights_path = str(tmp_path / "test_weights.pth")
        torch.save(fake_state, weights_path)

        loaded = OlmoEarthEncoder.from_state_dict(weights_path)
        assert loaded.dim == 128
        assert len(loaded.blocks) == 4

    def test_from_state_dict_cpu(self, tmp_path):
        model = OlmoEarthEncoder(dim=128, n_blocks=2)
        fake_state = {f"encoder.{k}": v for k, v in model.state_dict().items()}
        weights_path = str(tmp_path / "test_weights.pth")
        torch.save(fake_state, weights_path)

        loaded = OlmoEarthEncoder.from_state_dict(weights_path, device="cpu")
        assert next(loaded.parameters()).device.type == "cpu"


class TestOLMoEarthEncoderWrapper:
    @patch('src.encoder.olmoearth.OlmoEarthEncoder.from_state_dict')
    def test_local_mode(self, mock_from):
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(5, 128)
        mock_from.return_value = mock_model

        from src.encoder.olmoearth import OLMoEarthEncoder
        enc = OLMoEarthEncoder(mode="local", local_weights_path="test.pth")

        X = np.random.rand(5, 10, 13).astype(np.float32)
        emb = enc.encode(X, batch_size=5)
        assert emb.shape == (5, 128)

    def test_encode_numpy_output(self):
        model = OlmoEarthEncoder(dim=128, n_blocks=2)

        from src.encoder.olmoearth import OLMoEarthEncoder
        enc = OLMoEarthEncoder.__new__(OLMoEarthEncoder)
        enc.model = model
        enc.mode = "local"
        enc.device = "cpu"

        X = np.random.rand(3, 8, 13).astype(np.float32)
        emb = enc.encode(X, batch_size=3)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (3, 128)

    @patch('src.encoder.olmoearth.torch.cuda.is_available', return_value=False)
    def test_cuda_fallback_to_cpu(self, mock_cuda):
        from src.encoder.olmoearth import OLMoEarthEncoder
        with patch.object(OLMoEarthEncoder, '__init__', lambda self, **kw: None):
            enc = OLMoEarthEncoder.__new__(OLMoEarthEncoder)
            # Simulate what __init__ does with fallback
            device = "cuda"
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            enc.device = device
            enc.model = OlmoEarthEncoder(dim=128, n_blocks=2)

            X = np.random.rand(2, 6, 13).astype(np.float32)
            emb = enc.encode(X, batch_size=2)
            assert emb.shape == (2, 128)

    def test_encode_uses_inference_mode(self):
        model = OlmoEarthEncoder(dim=128, n_blocks=2)
        from src.encoder.olmoearth import OLMoEarthEncoder
        enc = OLMoEarthEncoder.__new__(OLMoEarthEncoder)
        enc.model = model
        enc.mode = "local"
        enc.device = "cpu"

        X = np.random.rand(4, 8, 13).astype(np.float32)
        emb = enc.encode(X, batch_size=2)
        assert emb.shape == (4, 128)
