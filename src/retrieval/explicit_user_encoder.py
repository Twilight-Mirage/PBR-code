import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def l2_normalize(vec):
    arr = np.asarray(vec, dtype=np.float32)
    return arr / (np.linalg.norm(arr) + 1e-8)


class ExplicitUserProjector(nn.Module):
    """
    Lightweight projector for explicit user representations.
    Input/Output dim stays the same as retriever embedding dim.
    """

    def __init__(self, in_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x):
        return self.net(x)


class ExplicitUserEncoderAdapter:
    """
    Runtime adapter:
    - use retriever model to get base embedding
    - optionally project it via trained explicit-user projector
    """

    def __init__(self, projector=None, meta=None, device="cpu"):
        self.projector = projector
        self.meta = meta or {}
        self.device = device
        if self.projector is not None:
            self.projector.to(self.device)
            self.projector.eval()

    @property
    def available(self):
        return self.projector is not None

    def project_embeddings(self, embeddings):
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if not self.available:
            return np.asarray([l2_normalize(x) for x in arr], dtype=np.float32)

        with torch.no_grad():
            x = torch.from_numpy(arr).to(self.device)
            y = self.projector(x).detach().cpu().numpy()
        return np.asarray([l2_normalize(v) for v in y], dtype=np.float32)

    def encode_texts(self, texts, retriever_model):
        emb = retriever_model.encode(texts, convert_to_numpy=True)
        return self.project_embeddings(emb)

    @classmethod
    def from_checkpoint(cls, ckpt_path, map_location="cpu"):
        if ckpt_path is None or str(ckpt_path).strip() == "":
            return cls(projector=None, meta={"reason": "empty_path"}, device=map_location)
        p = Path(ckpt_path)
        if not p.exists():
            raise FileNotFoundError(f"Explicit-user encoder checkpoint not found: {ckpt_path}")

        ckpt = torch.load(p, map_location=map_location)
        in_dim = int(ckpt["in_dim"])
        hidden_dim = int(ckpt.get("hidden_dim", 512))
        dropout = float(ckpt.get("dropout", 0.1))
        projector = ExplicitUserProjector(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        projector.load_state_dict(ckpt["state_dict"])
        meta = ckpt.get("meta", {})
        meta["checkpoint"] = str(p)
        return cls(projector=projector, meta=meta, device=map_location)


def save_explicit_user_encoder_checkpoint(
    output_path,
    projector,
    in_dim,
    hidden_dim,
    dropout,
    meta=None,
):
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": projector.state_dict(),
        "in_dim": int(in_dim),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "meta": meta or {},
    }
    torch.save(payload, p)


def save_training_summary(path, summary_dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary_dict, indent=2, ensure_ascii=False), encoding="utf-8")
