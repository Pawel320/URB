"""
Przygotowanie danych pod PCA i Autoencoder.

Wczytuje surowe wektory stanu krawędzi zebrane przez ippo_clusters.py,
miesza, dzieli na train/val, uczy scaler i zapisuje:
  - scaler.pkl    (używany w runtime przez klasy PCA/AE)
  - X_train.npy   (dane treningowe PO przeskalowaniu)
  - X_val.npy     (dane walidacyjne PO przeskalowaniu)

Użycie:
  python prepare_compression_data.py --data ../results/<exp_id>/surowe_dane_sumo.npy
  (na serwerze --dir można pominąć, domyślnie to RESULTS_DIR)
"""

import argparse
import json
import os

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Ścieżka zahardkowana zgodnie z routerl/environment/observations.py
RESULTS_DIR = "/home/z1201158/URB_Dim_Reduction/URB/results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="plik surowe_dane_sumo.npy")
    parser.add_argument("--dir", default=RESULTS_DIR, help="gdzie zapisać scaler.pkl i X_*.npy")
    parser.add_argument("--val-frac", type=float, default=0.1, help="udział danych walidacyjnych")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1) Wczytanie surowych danych
    X = np.load(args.data)
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    print(f"Wczytano {X.shape[0]} wektorów o długości {X.shape[1]}")
    if X.ndim != 2:
        raise ValueError(f"Oczekiwano macierzy 2D, dostano kształt {X.shape}")

    # 2) Kontrola metadanych (żeby nie trenować na danych z innej mapy/klasy)
    meta_path = os.path.join(os.path.dirname(args.data), "surowe_dane_sumo_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("edge_vec_len") != X.shape[1]:
            raise ValueError(
                f"Metadata mówią edge_vec_len={meta.get('edge_vec_len')}, "
                f"ale dane mają {X.shape[1]} cech."
            )
        if meta.get("observation_type") not in (None, "trip_info_eta_sumo"):
            raise ValueError(
                f"Dane zbierano dla observation_type={meta.get('observation_type')}, "
                "a potrzebne są surowe dane (trip_info_eta_sumo)."
            )

    # 3) Wymieszanie + podział train/val
    rng = np.random.default_rng(args.seed)
    X = X[rng.permutation(X.shape[0])]
    n_val = max(1, int(len(X) * args.val_frac))
    X_val, X_train = X[:n_val], X[n_val:]
    print(f"Podział: train {X_train.shape[0]}, val {X_val.shape[0]}")

    # 4) Scaler liczony TYLKO na danych treningowych
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    # 5) Zapis
    os.makedirs(args.dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(args.dir, "scaler.pkl"))
    np.save(os.path.join(args.dir, "X_train.npy"), X_train)
    np.save(os.path.join(args.dir, "X_val.npy"), X_val)
    print(f"Zapisano w {args.dir}: scaler.pkl, X_train.npy, X_val.npy")


if __name__ == "__main__":
    main()
