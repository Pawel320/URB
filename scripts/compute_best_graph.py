import numpy as np
import pandas as pd


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
    optimal_T_mask = find_optimal_subgraph_T(B_matrix, alpha=ALPHA, beta=BETA, max_k=MAX_K)

    # 3. Wyliczenie finalnych statystyk dla znalezionego zbioru
    final_score, int_wew, int_zew = calculate_objective_score(B_matrix, optimal_T_mask, ALPHA, BETA)
    selected_agents_count = optimal_T_mask.sum()

    print("=== WYNIK OPTYMALIZACJI ZBIORU T ===")
    print(f"Parametry:                     ALPHA={ALPHA}, BETA={BETA}, MAX_K={MAX_K}")
    print(f"Liczba aut w zbiorze T:        {selected_agents_count} / {len(agent_ids)}")
    print(f"Interakcje wewnętrzne (Wew):   {int_wew}")
    print(f"Interakcje zewnętrzne (Zew):   {int_zew}")
    print(f"Wartość funkcji celu J(T):     {final_score:.2f}")
    print("====================================")

    # 4. Eksport do formatu zgodnego z SubAV (agent_id, cluster)
    clusters_df = pd.DataFrame({
        "agent_id": agent_ids,
        "cluster": optimal_T_mask.astype(int)  # 1 = zbiór T (cooperative), 0 = reszta (selfish)
    })
    clusters_df.to_csv(OUTPUT_CLUSTERS_CSV_PATH, index=False)
    print(f"Zapisano podział klastrów do:  {OUTPUT_CLUSTERS_CSV_PATH}")