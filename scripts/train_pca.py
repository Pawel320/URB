"""
Trenuje PCA na danych PO przeskalowaniu (X_train.npy) i zapisuje:
  - trained_pca_model.pkl
  - pca_report.txt

PCA liczy się jednym razem (SVD) - nie ma tu epok treningu.

Użycie:
  python train_pca.py            (na serwerze, dane i wyniki w RESULTS_DIR)
  python train_pca.py --dir .     # lokalny test
"""

import argparse
import os

import joblib
import numpy as np
from sklearn.decomposition import PCA

# Ścieżka zahardkowana zgodnie z routerl/environment/observations.py
RESULTS_DIR = "/home/z1201158/URB_Dim_Reduction/URB/results"
K = 35           # tyle komponentów używa runtime (n_components=35)
VARIANCE = 0.95  # raport: ile komponentów wystarczy do 95% wariancji


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=RESULTS_DIR, help="katalog z X_train.npy / X_val.npy (i na wyjście)")
    args = parser.parse_args()

    X_train = np.load(os.path.join(args.dir, "X_train.npy"))
    X_val = np.load(os.path.join(args.dir, "X_val.npy"))
    print(f"Dane: train {X_train.shape}, val {X_val.shape}")

    pca = PCA().fit(X_train)

    # explained_variance_ratio_[i] = jaka część CAŁEJ wariancji przypada na i-ty komponent
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    variance_at_k = float(cumulative[min(K, len(cumulative)) - 1])
    k_for_target = int(np.searchsorted(cumulative, VARIANCE) + 1)

    # R2 rekonstrukcji przez PIERWSZE K komponentów (dokładnie tak, jak robi to
    # runtime) - dzięki temu porównywalne z R2 autoencodera o latent_dim=35.
    # Ręcznie, bo inverse_transform wymaga wszystkich komponentów:
    # X = reduced @ components_ + mean_
    reduced_val = pca.transform(X_val)[:, :K]
    X_val_reconstructed = reduced_val @ pca.components_[:K] + pca.mean_
    # R2 GLOBALNE (ważone wariancją) zamiast sklearn r2_score, który uśrednia
    # osobne R2 per cecha - przy martwych cechach (wariancja ~0 na val)
    # taka średnia wybija się do ~-1e9. Globalne: martwe cechy nie ważą.
    val_r2 = 1.0 - ((X_val - X_val_reconstructed) ** 2).sum() / ((X_val - X_val.mean(0)) ** 2).sum()

    report = (
        "PCA - raport\n"
        "==============================\n"
        f"wektor wejsciowy:      {X_train.shape[1]} cech\n"
        f"komponentow w modelu:  {pca.n_components_}\n"
        f"wariancja @{K} kom.:   {100 * variance_at_k:.2f}%\n"
        f"komponentow do {VARIANCE:.0%} wariancji: {k_for_target}\n"
        f"R2 global (val, przez {K} kom.): {val_r2:.4f}\n"
    )
    print(report)

    joblib.dump(pca, os.path.join(args.dir, "trained_pca_model.pkl"))
    with open(os.path.join(args.dir, "pca_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Zapisano w {args.dir}: trained_pca_model.pkl, pca_report.txt")


if __name__ == "__main__":
    main()
