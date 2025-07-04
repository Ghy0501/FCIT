import torch
import copy 

def get_proxy_dict(args, global_dict):
    opt_proxy_dict = None
    proxy_dict = None
    if args.fed_alg in ['fedadagrad', 'fedyogi', 'fedadam']:
        proxy_dict, opt_proxy_dict = {}, {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key], device = args.device)
            opt_proxy_dict[key] = torch.ones_like(global_dict[key], device = args.device) * args.fedopt_tau**2
    elif args.fed_alg == 'fedavgm':
        proxy_dict = {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key], device = args.device)
    return proxy_dict, opt_proxy_dict

def get_auxiliary_dict(args, global_dict):

    if args.fed_alg in ['scaffold']:
        global_auxiliary = {}             
        for key in global_dict.keys():
            global_auxiliary[key] = torch.zeros_like(global_dict[key], device = args.device)
        auxiliary_model_list = [copy.deepcopy(global_auxiliary) for _ in range(args.num_clients)]  
        auxiliary_delta_dict = [copy.deepcopy(global_auxiliary) for _ in range(args.num_clients)]  

    else:
        global_auxiliary = None
        auxiliary_model_list = [None]*args.num_clients
        auxiliary_delta_dict = [None]*args.num_clients

    return global_auxiliary, auxiliary_model_list, auxiliary_delta_dict