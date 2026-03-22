import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.retrieval.explicit_user_encoder import (
    ExplicitUserProjector,
    save_explicit_user_encoder_checkpoint,
    save_training_summary,
)


def l2_normalize_tensor(x):
    return F.normalize(x, p=2, dim=-1, eps=1e-8)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dedup_encode_texts(model, texts, batch_size=128):
    uniq = list(dict.fromkeys(texts))
    emb = model.encode(uniq, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    table = {t: emb[i] for i, t in enumerate(uniq)}
    return table


class TripletEmbeddingDataset(Dataset):
    def __init__(self, anchors, positives, negatives):
        self.a = np.asarray(anchors, dtype=np.float32)
        self.p = np.asarray(positives, dtype=np.float32)
        self.n = np.asarray(negatives, dtype=np.float32)

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        return self.a[idx], self.p[idx], self.n[idx]


def compute_loss(projector, a, p, n, temperature=0.07, inbatch_weight=0.5):
    z = l2_normalize_tensor(projector(a))
    p = l2_normalize_tensor(p)
    n = l2_normalize_tensor(n)

    pos_logits = torch.sum(z * p, dim=-1) / temperature
    neg_logits = torch.sum(z * n, dim=-1) / temperature
    pair_logits = torch.stack([pos_logits, neg_logits], dim=1)
    pair_labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
    pair_loss = F.cross_entropy(pair_logits, pair_labels)

    # In-batch InfoNCE: current anchor should align with its paired positive.
    sim_mat = torch.matmul(z, p.T) / temperature
    inbatch_labels = torch.arange(z.size(0), dtype=torch.long, device=z.device)
    inbatch_loss = F.cross_entropy(sim_mat, inbatch_labels)

    return pair_loss + inbatch_weight * inbatch_loss, pair_loss.detach(), inbatch_loss.detach()


def evaluate(projector, loader, device, temperature=0.07, inbatch_weight=0.5):
    projector.eval()
    losses = []
    with torch.no_grad():
        for a, p, n in loader:
            a = a.to(device)
            p = p.to(device)
            n = n.to(device)
            loss, _, _ = compute_loss(
                projector=projector,
                a=a,
                p=p,
                n=n,
                temperature=temperature,
                inbatch_weight=inbatch_weight,
            )
            losses.append(float(loss.item()))
    if not losses:
        return float("inf")
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser(description="Train explicit-user projector with contrastive triplets.")
    parser.add_argument("--triplets_jsonl", type=str, required=True, help="Path to triplets jsonl.")
    parser.add_argument("--output_ckpt", type=str, required=True, help="Output checkpoint path.")
    parser.add_argument("--output_summary", type=str, default="", help="Optional summary json path.")
    parser.add_argument(
        "--retrieval_model_name",
        type=str,
        default="multi-qa-MiniLM-L6-cos-v1",
        help="Base embedding model.",
    )
    parser.add_argument("--max_triplets", type=int, default=0, help="Optional cap on triplets.")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Projector hidden dim.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Projector dropout.")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature.")
    parser.add_argument("--inbatch_weight", type=float, default=0.5, help="Weight of in-batch loss term.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.triplets_jsonl)
    if args.max_triplets > 0:
        rows = rows[: args.max_triplets]
    if not rows:
        raise ValueError("No triplets loaded.")

    anchors_text = [str(x["anchor_profile_text"]) for x in rows]
    positives_text = [str(x["positive_text"]) for x in rows]
    negatives_text = [str(x["negative_text"]) for x in rows]

    base_model = SentenceTransformer(args.retrieval_model_name, trust_remote_code=True)
    text_table = dedup_encode_texts(base_model, anchors_text + positives_text + negatives_text)

    anchors = np.asarray([text_table[t] for t in anchors_text], dtype=np.float32)
    positives = np.asarray([text_table[t] for t in positives_text], dtype=np.float32)
    negatives = np.asarray([text_table[t] for t in negatives_text], dtype=np.float32)

    n = len(anchors)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = int(max(1, round(n * args.val_ratio))) if n > 10 else 1
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) == 0:
        train_idx = idx[:-1]
        val_idx = idx[-1:]

    train_ds = TripletEmbeddingDataset(anchors[train_idx], positives[train_idx], negatives[train_idx])
    val_ds = TripletEmbeddingDataset(anchors[val_idx], positives[val_idx], negatives[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    in_dim = int(anchors.shape[1])
    projector = ExplicitUserProjector(in_dim=in_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    train_curve = []
    val_curve = []

    for epoch in range(1, args.epochs + 1):
        projector.train()
        epoch_losses = []
        for a, p, n in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            a = a.to(args.device)
            p = p.to(args.device)
            n = n.to(args.device)

            optimizer.zero_grad(set_to_none=True)
            loss, pair_loss, inbatch_loss = compute_loss(
                projector=projector,
                a=a,
                p=p,
                n=n,
                temperature=args.temperature,
                inbatch_weight=args.inbatch_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        val_loss = evaluate(
            projector=projector,
            loader=val_loader,
            device=args.device,
            temperature=args.temperature,
            inbatch_weight=args.inbatch_weight,
        )
        train_curve.append(train_loss)
        val_curve.append(val_loss)
        print(f"[E{epoch}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            save_explicit_user_encoder_checkpoint(
                output_path=args.output_ckpt,
                projector=projector,
                in_dim=in_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                meta={
                    "retrieval_model_name": args.retrieval_model_name,
                    "triplets_jsonl": str(args.triplets_jsonl),
                    "num_triplets": int(n),
                    "num_train": int(len(train_idx)),
                    "num_val": int(len(val_idx)),
                    "best_epoch": int(best_epoch),
                    "best_val_loss": float(best_val),
                    "temperature": float(args.temperature),
                    "inbatch_weight": float(args.inbatch_weight),
                },
            )

    summary = {
        "retrieval_model_name": args.retrieval_model_name,
        "triplets_jsonl": str(args.triplets_jsonl),
        "output_ckpt": str(args.output_ckpt),
        "num_triplets": int(n),
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "train_curve": train_curve,
        "val_curve": val_curve,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    summary_path = args.output_summary.strip() if args.output_summary.strip() else str(Path(args.output_ckpt).with_suffix(".summary.json"))
    save_training_summary(summary_path, summary)


if __name__ == "__main__":
    main()
