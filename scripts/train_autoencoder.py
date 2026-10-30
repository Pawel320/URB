"""
Trenuje autoencoder na danych PO przeskalowaniu (X_train.npy) i zapisuje:
  - trained_encoder.pt  (TYLKO encoder - dokładnie to wczytuje observations.py)
  - ae_report.txt

Użycie:
  python train_autoencoder.py              (na serwerze, dane i wyniki w RESULTS_DIR)
  python train_autoencoder.py --dir . --epochs 3   # lokalny test
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

# Ścieżka zahardkowana zgodnie z routerl/environment/observations.py
RESULTS_DIR = "/home/z1201158/URB_Dim_Reduction/URB/results"
LATENT_DIM = 35    # musi być zgodne z latent_dim=35 w observations.py
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
PATIENCE = 10      # epok bez poprawy R2 i kończymy trening


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder MUSI mieć dokładnie tę samą budowę co w observations.py,
        # żeby pasowały klucze state_dict (0/2/4.weight) przy wczytaniu z dysku.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=RESULTS_DIR, help="katalog z X_train.npy / X_val.npy (i na wyjście)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    X_train = np.load(os.path.join(args.dir, "X_train.npy"))
    X_val = np.load(os.path.join(args.dir, "X_val.npy"))
    print(f"Dane: train {X_train.shape}, val {X_val.shape}")

    np.random.seed(42)
    torch.manual_seed(42)

    model = Autoencoder(X_train.shape[1], LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()  # loss do uczenia; R2 jest tylko metryką raportowaną

    best_r2 = -np.inf
    best_epoch = 0
    best_encoder_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(X_train))
        for start in range(0, len(X_train), BATCH_SIZE):
            batch = torch.from_numpy(X_train[order[start:start + BATCH_SIZE]])
            loss = loss_fn(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            X_val_reconstructed = model(torch.from_numpy(X_val)).numpy()
        val_r2 = float(r2_score(X_val, X_val_reconstructed))
        print(f"epoka {epoch:3d}/{args.epochs} | R2 (val) = {val_r2:.4f}")

        if val_r2 > best_r2:
            best_r2 = val_r2
            best_epoch = epoch
            epochs_without_improvement = 0
            # zapisujemy TYLKO encoder, bo tylko on trafia do runtime
            best_encoder_state = {k: v.clone() for k, v in model.encoder.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping (brak poprawy przez {PATIENCE} epok)")
                break

    torch.save(best_encoder_state, os.path.join(args.dir, "trained_encoder.pt"))

    report = (
        "Autoencoder - raport\n"
        "==============================\n"
        f"wektor wejsciowy:      {X_train.shape[1]} cech\n"
        f"latent (wymiar):       {LATENT_DIM}\n"
        f"najlepsza epoka:       {best_epoch}\n"
        f"R2 rekonstrukcji (val): {best_r2:.4f}\n"
    )
    print(report)
    with open(os.path.join(args.dir, "ae_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Zapisano w {args.dir}: trained_encoder.pt, ae_report.txt")


if __name__ == "__main__":
    main()
