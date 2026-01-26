import os
import argparse
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import torch  # noqa: F401

def build_dictionary(texts: List[str], vectorizer: TfidfVectorizer, top_k: int = 10000) -> List[str]:
    # Use average TF-IDF across corpus to rank tokens
    X = vectorizer.transform(texts)
    avg_scores = np.asarray(X.mean(axis=0)).ravel()
    vocab = np.array(vectorizer.get_feature_names_out())
    idx = np.argsort(-avg_scores)[:top_k]
    return vocab[idx].tolist()


def train_autoencoder(texts: List[str], vectorizer: TfidfVectorizer, latent_dim: int = 128, epochs: int = 5, lr: float = 1e-3):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    X = vectorizer.transform(texts)
    X_dense = X.astype(np.float32).toarray()
    input_dim = X_dense.shape[1]

    class AE(nn.Module):
        def __init__(self, inp, latent):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(inp, 1024), nn.ReLU(),
                nn.Linear(1024, 256), nn.ReLU(),
                nn.Linear(256, latent)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent, 256), nn.ReLU(),
                nn.Linear(256, 1024), nn.ReLU(),
                nn.Linear(1024, inp), nn.Sigmoid()
            )
        def forward(self, x):
            z = self.encoder(x)
            out = self.decoder(z)
            return out, z

    model = AE(input_dim, latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(X_dense))
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model.train()
    for e in range(epochs):
        total = 0.0
        for (xb,) in dl:
            opt.zero_grad()
            recon, _ = model(xb)
            loss = crit(recon, xb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f"[AE] Epoch {e+1}/{epochs} loss={total/len(ds):.6f}")

    # Extract latent vectors
    model.eval()
    with torch.no_grad():
        latents = []
        for (xb,) in dl:
            _, z = model(xb)
            latents.append(z.numpy())
    latents = np.vstack(latents)
    return model, latents


def main():
    parser = argparse.ArgumentParser(description="Autoencoder over TF-IDF and dictionary rebuild")
    parser.add_argument("--input", default=os.path.join("data", "combined_text_only.csv"), help="Input CSV with no,text")
    parser.add_argument("--text-col", default="text", help="Text column name")
    parser.add_argument("--dict-out", default=os.path.join("data", "dictionary_rebuilt.txt"), help="Output dictionary file path")
    parser.add_argument("--model-out", default=os.path.join("models", "autoencoder.pt"), help="Path to save AE model (Torch)")
    parser.add_argument("--latents-out", default=os.path.join("models", "autoencoder_latents.npy"), help="Path to save latents (NPY)")
    parser.add_argument("--top-k", type=int, default=10000, help="Top-K tokens for rebuilt dictionary")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension for AE")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--skip-ae", action="store_true", help="Skip AE training; only rebuild dictionary")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column '{args.text_col}' in {args.input}")
    texts = df[args.text_col].astype(str).tolist()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, lowercase=True)
    vectorizer.fit(texts)

    # Dictionary rebuild
    tokens = build_dictionary(texts, vectorizer, top_k=args.top_k)
    os.makedirs(os.path.dirname(args.dict_out), exist_ok=True)
    with open(args.dict_out, "w", encoding="utf-8") as f:
        for t in tokens:
            f.write(t + "\n")
    print(f"[OK] Wrote rebuilt dictionary: {args.dict_out} ({len(tokens)} tokens)")

    # Autoencoder training (optional)
    if args.skip_ae:
        return
    if not ensure_torch():
        print("[WARN] PyTorch not installed. Install torch to enable AE training.")
        return

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.latents_out), exist_ok=True)

    model, latents = train_autoencoder(texts, vectorizer, latent_dim=args.latent_dim, epochs=args.epochs, lr=args.lr)

    # Save model
    import torch
    torch.save(model.state_dict(), args.model_out)
    print(f"[OK] Saved AE model: {args.model_out}")

    # Save latents
    np.save(args.latents_out, latents)
    print(f"[OK] Saved latents: {args.latents_out} shape={latents.shape}")


if __name__ == "__main__":
    main()
