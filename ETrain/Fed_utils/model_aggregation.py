from CoIN.peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
import torch
import os
from torch.nn.functional import normalize
import torch.nn.functional as F

def Avg_mm_projector(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch):
    device = next(model.parameters()).device
    weights_array = normalize(
        torch.tensor(local_dataset_len_dict,
                     dtype=torch.float32, device=device),
        p=1, dim=0)

    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, "llava_lora_finetune_epoch_{}".format(str(epoch)), "local_output_{}".format(client_id),
                                         "non_lora_trainables.bin")
        non_lora_trainables = torch.load(single_output_dir, map_location=device)     # base_model.model.model.mm_projector.0.weight
        if k == 0:
            weighted_single_weights = {key: non_lora_trainables[key] * (weights_array[k]) for key in
                                       non_lora_trainables.keys()}
        else:
            weighted_single_weights = {key: weighted_single_weights[key] + non_lora_trainables[key] * (weights_array[k])
                                       for key in
                                       non_lora_trainables.keys()}

    model.load_state_dict(weighted_single_weights, strict=False)
            
    return model

def process_text_feature(model, local_dataset_len_dict, cluster_text_feature, cluster_image_feature, cluster_text_size, clusters, init_clusters, threshold=0.9):
    averaged_text_features = {
        k: torch.mean(torch.stack(v), dim=0) 
        for k, v in model.text_features_dict.items() if len(v) > 0
    }

    averaged_image_features = {
        k: torch.mean(torch.stack(v), dim=0) 
        for k, v in model.image_features_dict.items() if len(v) > 0
    }

    client_ids = list(averaged_text_features.keys())
    num_clients = len(client_ids)

    weights_array = normalize(
        torch.tensor(local_dataset_len_dict,
                     dtype=torch.float32, device='cpu'),
        p=1, dim=0)
    
    if clusters is None and init_clusters is None:
        cluster_text_feature = []
        cluster_image_feature = []
        cluster_text_size = []

        similarity_matrix = torch.eye(num_clients)  
        
        for i, client_a in enumerate(client_ids):
            for j, client_b in enumerate(client_ids):
                if i < j:
                    sim = F.cosine_similarity(
                        averaged_text_features[client_a].unsqueeze(0),
                        averaged_text_features[client_b].unsqueeze(0)
                    )
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        # similarity_matrix_ = torch.eye(num_clients)  
        
        # for i, client_a in enumerate(client_ids):
        #     for j, client_b in enumerate(client_ids):
        #         if i < j:
        #             sim = F.cosine_similarity(
        #                 averaged_image_features[client_a].unsqueeze(0),
        #                 averaged_image_features[client_b].unsqueeze(0)
        #             )
        #             similarity_matrix_[i, j] = sim
        #             similarity_matrix_[j, i] = sim
        # similarity_matrix = (similarity_matrix + similarity_matrix_) / 2
        clusters = []
        visited = set()

        for i in range(num_clients):
            if i not in visited:
                cluster = [client_ids[i]]
                for j in range(num_clients):
                    if i != j and similarity_matrix[i, j] >= threshold:
                        cluster.append(client_ids[j])
                        visited.add(j)
                visited.add(i)
                clusters.append(cluster)

        valid_clusters = []
        for cluster in clusters:
            if len(cluster) == 1:  
                client_idx = client_ids.index(cluster[0]) 
                client_sample_ratio = weights_array[client_idx].item()  
                if client_sample_ratio >= 1e-3: 
                    valid_clusters.append(cluster)
                else:
                    max_sim = -1
                    best_cluster_idx = -1

                    for idx, valid_cluster in enumerate(valid_clusters):
                        valid_client_idx = client_ids.index(valid_cluster[0])
                        sim = similarity_matrix[client_idx, valid_client_idx].item()

                        if sim > max_sim:
                            max_sim = sim
                            best_cluster_idx = idx

                    if best_cluster_idx != -1:
                        valid_clusters[best_cluster_idx].append(client_ids[client_idx])
            else:
                valid_clusters.append(cluster) 

        for cluster in valid_clusters:
            all_text_features = []
            all_image_features = []
            total_weight = 0  
            for client in cluster:
                client_text_features = torch.mean(torch.stack(model.text_features_dict[client]), dim=0)
                client_image_features = torch.mean(torch.stack(model.image_features_dict[client]), dim=0)
                client_weight = len(model.text_features_dict[client]) 
                all_text_features.append(client_text_features * client_weight) 
                all_image_features.append(client_image_features * client_weight) 
                total_weight += client_weight
            cluster_text_feature.append(torch.sum(torch.stack(all_text_features), dim=0) / total_weight)
            cluster_image_feature.append(torch.sum(torch.stack(all_image_features), dim=0) / total_weight)
            cluster_text_size.append(total_weight)
        return cluster_text_feature, cluster_image_feature, cluster_text_size, valid_clusters
    elif clusters is None and init_clusters is not None:
        similarity_matrix = torch.eye(num_clients)  
        
        for i, client_a in enumerate(client_ids):
            for j, client_b in enumerate(client_ids):
                if i < j:
                    sim = F.cosine_similarity(
                        averaged_text_features[client_a].unsqueeze(0),
                        averaged_text_features[client_b].unsqueeze(0)
                    )
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        # similarity_matrix_ = torch.eye(num_clients)  
        
        # for i, client_a in enumerate(client_ids):
        #     for j, client_b in enumerate(client_ids):
        #         if i < j:
        #             sim = F.cosine_similarity(
        #                 averaged_image_features[client_a].unsqueeze(0),
        #                 averaged_image_features[client_b].unsqueeze(0)
        #             )
        #             similarity_matrix_[i, j] = sim
        #             similarity_matrix_[j, i] = sim
        # similarity_matrix = (similarity_matrix + similarity_matrix_) / 2
        clusters = []
        visited = set()

        for i in range(num_clients):
            if i not in visited:
                cluster = [client_ids[i]]
                for j in range(num_clients):
                    if i != j and similarity_matrix[i, j] >= threshold:
                        cluster.append(client_ids[j])
                        visited.add(j)
                visited.add(i)
                clusters.append(cluster)

        valid_clusters = []
        for cluster in clusters:
            if len(cluster) == 1:  
                client_idx = client_ids.index(cluster[0]) 
                client_sample_ratio = weights_array[client_idx].item()  
                if client_sample_ratio >= 1e-3: 
                    valid_clusters.append(cluster)
                else:
                    max_sim = -1
                    best_cluster_idx = -1

                    for idx, valid_cluster in enumerate(valid_clusters):
                        valid_client_idx = client_ids.index(valid_cluster[0])
                        sim = similarity_matrix[client_idx, valid_client_idx].item()

                        if sim > max_sim:
                            max_sim = sim
                            best_cluster_idx = idx

                    if best_cluster_idx != -1:
                        valid_clusters[best_cluster_idx].append(client_ids[client_idx])
            else:
                valid_clusters.append(cluster) 
        sorted_valid_clusters = []
        for cluster in init_clusters:
            if cluster == [-1]:
                sorted_valid_clusters.append([-1])
            else:
                for v_cluster in valid_clusters:
                    if set(cluster).intersection(set(v_cluster)) and v_cluster not in sorted_valid_clusters:
                        sorted_valid_clusters.append(v_cluster)
        new_clusters = sorted_valid_clusters.copy()
        for i, c in enumerate(init_clusters):
            if c == [-1] and new_clusters[i] != [-1]:
                new_clusters.insert(i, [-1])
        for idx, cluster in enumerate(new_clusters):
            all_text_features = []
            all_image_features = []
            total_weight = 0  
            if cluster != [-1]:
                for client in cluster:
                    client_text_features = torch.mean(torch.stack(model.text_features_dict[client]), dim=0)
                    client_image_features = torch.mean(torch.stack(model.image_features_dict[client]), dim=0)
                    client_weight = len(model.text_features_dict[client]) 
                    all_text_features.append(client_text_features * client_weight) 
                    all_image_features.append(client_image_features * client_weight) 
                    total_weight += client_weight
                cluster_text_feature[idx] = (cluster_text_feature[idx] * cluster_text_size[idx] + torch.sum(torch.stack(all_text_features), dim=0) / (total_weight + cluster_text_size[idx]))
                cluster_image_feature[idx] = (cluster_image_feature[idx] * cluster_text_size[idx] + torch.sum(torch.stack(all_image_features), dim=0) / (total_weight + cluster_text_size[idx]))
                cluster_text_size[idx] += total_weight
        return cluster_text_feature, cluster_image_feature, cluster_text_size, new_clusters
    else:
        for client_id in client_ids:
            current_text_feature = averaged_text_features[client_id]
            current_image_feature = averaged_image_features[client_id]
            max_sim = -1
            best_cluster_idx = -1
            
            for idx, cluster_center in enumerate(cluster_text_feature):
                sim = F.cosine_similarity(
                    current_text_feature.unsqueeze(0), 
                    cluster_center.unsqueeze(0)
                )
                if sim > max_sim:
                    max_sim = sim
                    best_cluster_idx = idx
    
            # if max_sim >= threshold:
            total_weight = cluster_text_size[best_cluster_idx]
            new_weight = len(model.text_features_dict[client_id])
                
            cluster_text_feature[best_cluster_idx] = (
                    (cluster_text_feature[best_cluster_idx] * total_weight + current_text_feature * new_weight) 
                    / (total_weight + new_weight)
                )
            cluster_image_feature[best_cluster_idx] = (
                    (cluster_image_feature[best_cluster_idx] * total_weight + current_image_feature * new_weight) 
                    / (total_weight + new_weight)
                )
            cluster_text_size[best_cluster_idx] += new_weight
            # else:
            #     client_idx = client_ids.index(client_id)
            #     client_sample_ratio = weights_array[client_idx].item()
                
            #     if client_sample_ratio >= 1e-3:
            #         cluster_text_feature.append(current_text_feature)
            #         cluster_image_feature.append(current_image_feature)
            #         cluster_text_size.append(len(model.text_features_dict[client_id]))
            #     else:
            #         total_weight = cluster_text_size[best_cluster_idx]
            #         new_weight = len(model.text_features_dict[client_id])
                    
            #         cluster_text_feature[best_cluster_idx] = (
            #             (cluster_text_feature[best_cluster_idx] * total_weight + current_text_feature * new_weight) 
            #             / (total_weight + new_weight)
            #         )

            #         cluster_image_feature[best_cluster_idx] = (
            #             (cluster_image_feature[best_cluster_idx] * total_weight + current_image_feature * new_weight) 
            #             / (total_weight + new_weight)
            #         )
            #         cluster_text_size[best_cluster_idx] += new_weight
        return cluster_text_feature, cluster_image_feature, cluster_text_size, clusters



def FedAvg(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch, clusters, lora_weight_id):
    device = next(model.parameters()).device

    client_weights = torch.tensor(local_dataset_len_dict, dtype=torch.float32, device=device)

    aggregated_weights = {}
    weighted_single_weights = {}

    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, "llava_lora_finetune_epoch_{}".format(str(epoch)), "local_output_{}".format(client_id),
                                         "pytorch_model.bin")
        single_weights = torch.load(single_output_dir, map_location=device)

        lora_weight_id = lora_weight_id if lora_weight_id is not None else clusters
        lora_id = next((idx for idx, cluster in enumerate(lora_weight_id) if k in cluster), -1)
        for key in single_weights.keys():
            if 'loraA.{}'.format(lora_id) in key or 'loraB.{}'.format(lora_id) in key:

                cluster_idx = next(
                (idx for idx, cluster in enumerate(clusters) if k in cluster), -1)

                if cluster_idx != -1:
                    new_key = key.split(".")
                    new_key[-2] = str(cluster_idx)  
                    new_key = ".".join(new_key)

                    weight = client_weights[k]

                    cluster_key = (new_key, cluster_idx)
                    if cluster_key not in aggregated_weights:
                        aggregated_weights[cluster_key] = single_weights[key] * weight
                    else:
                        aggregated_weights[cluster_key] += single_weights[key] * weight
            else:
                weighted_single_weights[key] = single_weights[key]
    
    for (key, cluster_idx), weight_sum in aggregated_weights.items():
        total_weight = sum(client_weights[k] for k in clusters[cluster_idx])
        weighted_single_weights[key] = weight_sum / total_weight 
                                       
    model = Avg_mm_projector(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch)

    set_peft_model_state_dict(model, weighted_single_weights, "default")

    return model