import numpy as np
import pandas as pd
import os


BINARY_CSV_PATH = "influence/binary_influence_matrix.csv"
OUTPUT_CLUSTERS_CSV_PATH = "config/groupings/agent_clusters.csv"

ALPHA = 0.5   # Waga nagrody za interakcje WEWNĄTRZ zbioru T
BETA = 0.5    # Waga kary za interakcje ZEWNĘTRZNE (między T a resztą świata)
MAX_K = 50    # Maksymalny rozmiar zbioru T (wpisz 1035 lub None, aby wyłączyć limit)


def calculate_objective_score(binary_matrix: np.ndarray, in_cluster_mask: np.ndarray, alpha: float, beta: float):
    """Oblicza dokładną wartość funkcji celu J(T) = alpha * Int_wew - beta * Int_zew."""
    outside_cluster_mask = ~in_cluster_mask

    # Suma jedynek wewnątrz klastra T (i należy do T oraz j należy do T)
    internal_interactions = binary_matrix[np.ix_(in_cluster_mask, in_cluster_mask)].sum()

    # Suma jedynek wychodzących poza klaster T (i należy do T, k NIE należy do T)
    external_interactions = binary_matrix[np.ix_(in_cluster_mask, outside_cluster_mask)].sum()

    score = (alpha * internal_interactions) - (beta * external_interactions)
    return score, int(internal_interactions), int(external_interactions)


def find_optimal_subgraph_T(binary_matrix: np.ndarray, alpha: float, beta: float, max_k: int = None, num_starts: int = 50):
    """Buduje gęsty podgraf T, testując wielokrotny start z najbardziej konfliktowych węzłów."""
    num_agents = binary_matrix.shape[0]
    if max_k is None or max_k >= num_agents:
        max_k = num_agents

    # Liczymy, ile wzajemnych konfliktów (wartość 2) ma każdy agent
    mutual_influence = (binary_matrix + binary_matrix.T) == 2
    mutual_degree_per_agent = mutual_influence.sum(axis=1)

    # Wybieramy N najbardziej konfliktowych agentów jako kandydatów na ziarno startowe
    candidate_seeds = np.argsort(mutual_degree_per_agent)[::-1][:num_starts]

    total_outgoing_per_agent = binary_matrix.sum(axis=1)
    global_best_mask = None
    global_best_score = -np.inf

    for seed_agent in candidate_seeds:
        in_cluster_mask = np.zeros(num_agents, dtype=bool)
        in_cluster_mask[seed_agent] = True

        edges_from_T_to_candidate = binary_matrix[seed_agent, :].copy()
        edges_from_candidate_to_T = binary_matrix[:, seed_agent].copy()

        current_score, _, _ = calculate_objective_score(binary_matrix, in_cluster_mask, alpha, beta)
        local_best_mask = in_cluster_mask.copy()
        local_best_score = current_score

        for step in range(2, max_k + 1):
            delta_internal = edges_from_T_to_candidate + edges_from_candidate_to_T
            delta_external = -edges_from_T_to_candidate + (total_outgoing_per_agent - edges_from_candidate_to_T)
            delta_score = (alpha * delta_internal) - (beta * delta_external)
            delta_score[in_cluster_mask] = -np.inf

            best_candidate = int(np.argmax(delta_score))

            if max_k == num_agents and delta_score[best_candidate] <= 0:
                break

            in_cluster_mask[best_candidate] = True
            current_score += delta_score[best_candidate]
            edges_from_T_to_candidate += binary_matrix[best_candidate, :]
            edges_from_candidate_to_T += binary_matrix[:, best_candidate]

            if current_score > local_best_score:
                local_best_score = current_score
                local_best_mask = in_cluster_mask.copy()

        # Sprawdzamy, czy start z tego ziarna dał najlepszy wynik ze wszystkich prób
        if local_best_score > global_best_score:
            global_best_score = local_best_score
            global_best_mask = local_best_mask.copy()

    return global_best_mask


if __name__ == "__main__":

    # 1. Wczytanie macierzy binarnej
    df_binary = pd.read_csv(BINARY_CSV_PATH, index_col=0)
    agent_ids = df_binary.index.astype(int).values
    B_matrix = df_binary.values

    # 2. Uruchomienie algorytmu szukającego zbioru T

    GRID_PARAMS = [
    # (alpha, beta, max_k)

    # ==========================================
    # CZĘŚĆ 1: Zamrożona ALPHA = 1.0, zmienna BETA
    # ==========================================
    (1.0, 0.05, 50),
    (1.0, 0.20, 50),
    (1.0, 0.50, 50),
    (1.0, 0.70, 50),
    (1.0, 1.0, 50),

    (1.0, 0.05, 100),
    (1.0, 0.20, 100),
    (1.0, 0.50, 100),
    (1.0, 0.70, 100),
    (1.0, 1.0, 100),

    (1.0, 0.05, 400),
    (1.0, 0.20, 400),
    (1.0, 0.50, 400),
    (1.0, 0.70, 400),
    (1.0, 1.0, 400),

    (1.0, 0.05, 1035),
    (1.0, 0.20, 1035),
    (1.0, 0.50, 1035),
    (1.0, 0.70, 1035),
    (1.0, 1.0, 1035),

    # ==========================================
    # CZĘŚĆ 2: Zamrożona BETA = 1.0, zmienna ALPHA
    # (pominięto 1.0, 1.0, aby nie dublować plików z Części 1)
    # ==========================================
    (0.05, 1.0, 50),
    (0.20, 1.0, 50),
    (0.50, 1.0, 50),
    (0.70, 1.0, 50),

    (0.05, 1.0, 100),
    (0.20, 1.0, 100),
    (0.50, 1.0, 100),
    (0.70, 1.0, 100),

    (0.05, 1.0, 400),
    (0.20, 1.0, 400),
    (0.50, 1.0, 400),
    (0.70, 1.0, 400),

    (0.05, 1.0, 1035),
    (0.20, 1.0, 1035),
    (0.50, 1.0, 1035),
    (0.70, 1.0, 1035),
]

    os.makedirs("config/groupings/grid", exist_ok=True)
    for alpha, beta, max_k in GRID_PARAMS:
        mask = find_optimal_subgraph_T(B_matrix, alpha=alpha, beta=beta, max_k=max_k)
        score, w, z = calculate_objective_score(B_matrix, mask, alpha, beta)
        
        fname = f"ingolstadt_custom2_a{alpha}_b{beta}_k{max_k}.csv"
        pd.DataFrame({"agent_id": agent_ids, "cluster": mask.astype(int)}).to_csv(
            f"config/groupings/grid/{fname}", index=False
        )
        print(f"{fname} -> |T|={mask.sum()}, Wew={w}, Zew={z}, J(T)={score:.2f}")