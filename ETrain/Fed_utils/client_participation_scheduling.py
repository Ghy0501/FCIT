import numpy as np


def client_selection(num_clients, client_selection_frac, client_selection_strategy, previous_set, other_info=None):
    np.random.seed(other_info)
    all_clients_set = set(np.arange(num_clients))
    if client_selection_strategy == "random":
        num_selected = max(int(client_selection_frac * num_clients), 1)
        selected_clients_set = set(np.random.choice(np.arange(num_clients), num_selected, replace=False))
    if client_selection_strategy == "non-repeat":
        remaining_clients_set = all_clients_set - previously_selected_clients_set
        num_selected = max(int(client_selection_frac * num_clients), 1)
        num_selected = min(num_selected, len(remaining_clients_set))
        selected_clients_set = set(np.random.choice(list(remaining_clients_set), num_selected, replace=False))

    return selected_clients_set