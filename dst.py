import torch
import torch.cuda
from torch.nn import functional as F
import argparse
import numpy as np
import os
import time
from copy import deepcopy
from tqdm import tqdm


from datasets import get_dataset
# import models
from client import Client
from models import all_models, needs_mask, initialize_mask


def device_list(x):
    if x == 'cpu' or x == 'mps':
        return [x]
    return [int(y) for y in x.split(',')]


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
parser.add_argument('-C', '--clients', type=int, help='number of clients per round', default=20)
parser.add_argument('-R', '--rounds', type=int, help='number of global rounds', default=400)
parser.add_argument('-E','--epochs', type=int, help='number of local epochs', default=10)
parser.add_argument('--dataset', type=str, choices=('mnist', 'emnist', 'cifar10', 'cifar100','pacs', 'office', 'officehome'),
                    default='mnist', help='Dataset to use')
parser.add_argument("--source", nargs='+', help='specified when using DG datasets')
parser.add_argument("--target", nargs='+', help="Target")
parser.add_argument('--distribution', type=str, choices=('dirichlet', 'lotteryfl', 'iid', 'noniid'), default='dirichlet',
                    help='how should the dataset be distributed?')
parser.add_argument('--beta', type=float, default=0.1, help='Beta parameter (unbalance rate) for Dirichlet distribution')
parser.add_argument('-K', '--total-clients', type=int, help='split the dataset between this many clients. Ignored for EMNIST.', default=400)
parser.add_argument('--min-samples', type=int, default=0, help='minimum number of samples required to allow a client to participate')
parser.add_argument('--samples-per-client', type=int, default=0, help='samples to allocate to each client (per class, for lotteryfl, or per client, for iid)')
parser.add_argument('--prox', type=float, default=0, help='coefficient to proximal term (i.e. in FedProx)')

# Pruning and regrowth options
parser.add_argument('-S', '--sparsity', type=float, default=0.1, help='sparsity from 0 to 1')
parser.add_argument('-d', '--direction-ratio', type=float, default=0.0, help='from 0 to 1')
parser.add_argument('--rate-decay-method', default='cosine', choices=('constant', 'cosine'), help='annealing for readjustment ratio')
parser.add_argument('--rate-decay-end', default=None, type=int, help='round to end annealing')
parser.add_argument('--readjustment-ratio', type=float, default=0.5, help='readjust this many of the weights each time')
parser.add_argument('--pruning-begin', type=int, default=9, help='first epoch number when we should readjust')
parser.add_argument('--pruning-interval', type=int, default=10, help='epochs between readjustments')
parser.add_argument('--rounds-between-readjustments', type=int, default=10, help='rounds between readjustments')
parser.add_argument('--remember-old', default=False, action='store_true', help="remember client's old weights when aggregating missing ones")
parser.add_argument('--drill', default=False, action='store_true', help="drill run for quick testing")
parser.add_argument('--sparsity-distribution', default='erk', choices=('uniform', 'er', 'erk'))
parser.add_argument('--final-sparsity', type=float, default=None, help='final sparsity to grow to, from 0 to 1. default is the same as --sparsity')

parser.add_argument('-B', '--batch-size', type=int, default=32,
                    help='local client batch size')
parser.add_argument('--l2', default=0.001, type=float, help='L2 regularization strength')
parser.add_argument('--momentum', default=0.9, type=float, help='Local client SGD momentum parameter')
parser.add_argument('--cache-test-set', default=False, action='store_true', help='Load test sets into memory')
parser.add_argument('--cache-test-set-gpu', default=False, action='store_true', help='Load test sets into GPU memory')
parser.add_argument('--test-batches', default=0, type=int, help='Number of minibatches to test on, or 0 for all of them')
parser.add_argument('--eval-every', default=50, type=int, help='Evaluate on test set every N rounds')
parser.add_argument('--device', default='0', type=device_list, help='Device to use for compute. Use "cpu" to force CPU. Otherwise, separate with commas to allow multi-GPU.')
parser.add_argument('--min-votes', default=0, type=int, help='Minimum votes required to keep a weight')
parser.add_argument('--no-eval', default=True, action='store_false', dest='eval')
parser.add_argument('--grasp', default=False, action='store_true')
parser.add_argument('--fp16', default=False, action='store_true', help='upload as fp16')
parser.add_argument('-o', '--outfile', default='output', type=str)


args = parser.parse_args()

args.seed = 0

seed_val = args.seed
torch.manual_seed(seed_val)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed_val)

rng = np.random.default_rng()

flog = open(args.outfile+'.log', 'a')
args.log_file = flog
fcsv = open(args.outfile+'.csv', 'a')

devices = [torch.device(x) for x in args.device] if torch.cuda.is_available() else [torch.device('cpu')]
args.pid = os.getpid()

if args.rate_decay_end is None:
    args.rate_decay_end = args.rounds // 2
if args.final_sparsity is None:
    args.final_sparsity = args.sparsity
if args.sparsity <= 0 :
    args.readjustment_ratio = 0

args.use_DG_dataset = args.dataset in ['pacs', 'officehome', 'vlcs']


def print_to_log(*arg, **kwargs):
    print(*arg, **kwargs, file=flog)
    print(*arg, **kwargs)

def print_to_csv(*arg, **kwargs):
    print(*arg, **kwargs, file=fcsv)
    # print(*arg, **kwargs)

def print_csv_line(**kwargs):
    print_to_csv(','.join(str(x) for x in kwargs.values()))

def nan_to_num(x, nan=0, posinf=0, neginf=0):
    x = x.clone()
    x[x != x] = nan
    x[x == -float('inf')] = neginf
    x[x == float('inf')] = posinf
    return x.clone()

def evaluate_global_model(global_model, loaders):
    target_loaders = [(dm, loaders[dm]) for dm in args.target] if args.use_DG_dataset else loaders.items()
    global_test_data = []
    for _, (_, cl_loaders) in enumerate(target_loaders):
        _, _, test_data = cl_loaders
        global_test_data += test_data
    return evaluate_model(global_model, global_test_data)


def evaluate_model(model, test_data):
    with torch.no_grad():
        correct = 0.
        total = 0.
        _model = model.to(devices[0])
        _model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_data):
                if not args.cache_test_set_gpu:
                    inputs = inputs.to(devices[0])
                    labels = labels.to(devices[0])
                outputs = _model(inputs)
                outputs = torch.argmax(outputs, dim=-1)
                correct += sum(labels == outputs)
                total += len(labels)
        global_model_acc = correct / total
        return global_model_acc


def evaluate_global_clients(client_ids, global_model, progress=False, n_batches=0):
    with torch.no_grad():
        accuracies = {}
        sparsities = {}

        if progress:
            enumerator = tqdm(client_ids)
        else:
            enumerator = client_ids

        for client_id in enumerator:
            _, _, test_data = client_loaders[client_id]
            accuracies[client_id] = evaluate_model(global_model, test_data)
            client = get_client_instance(client_id)
            sparsities[client_id] = client.sparsity()

    return accuracies, sparsities


def evaluate_local(clients, global_model, progress=False, n_batches=0):

    # we need to perform an update to client's weights.
    with torch.no_grad():
        accuracies = {}
        sparsities = {}

        if progress:
            enumerator = tqdm(clients.items())
        else:
            enumerator = clients.items()

        for client_id, client in enumerator:
            client.reset_weights(global_state=global_model.state_dict(), use_global_mask=True)
            accuracies[client_id] = client.test().item()
            sparsities[client_id] = client.sparsity()

    return accuracies, sparsities



# Fetch and cache the dataset
if args.drill:
    print('Enter drill mode ...')

print('Fetching dataset...')
print_to_log(args)

'''
if os.path.isfile(args.dataset + '.pickle'):
    with open(args.dataset + '.pickle', 'rb') as f:
        loaders = pickle.load(f)
else:
    loaders = get_dataset(args.dataset, clients=args.total_clients,
                          batch_size=args.batch_size, devices=cache_devices,
                          min_samples=args.min_samples)
    with open(args.dataset + '.pickle', 'wb') as f:
        pickle.dump(loaders, f)
'''

loaders = get_dataset(args.dataset, clients=args.total_clients, mode=args.distribution,
                      beta=args.beta, batch_size=args.batch_size, devices=devices,
                      min_samples=args.min_samples, samples=args.samples_per_client)

# initialize clients
print('Initializing clients...')
client_ids = []
dataset_classes = {
    'mnist': 10,
    'cifar10': 10,
    'cifar100': 100,
    'pacs': 7,
    'officehome': 65,
    'vlcs': 5,
}

clients_loaders_map = [(dm, loaders[dm]) for dm in args.source] if args.use_DG_dataset else list(loaders.items())
client_epochs = {}
client_loaders = {}
# for i, (client_id, client_loaders) in tqdm(enumerate(loaders.items())):
for i, (client_id, cl_loaders) in enumerate(clients_loaders_map):
    # cl = Client(client_id, *cl_loaders, net=all_models[args.dataset],
    #             learning_rate=args.eta, local_epochs=args.epochs,
    #             target_sparsity=args.sparsity, classes=dataset_classes[args.dataset], global_args=args)
    device, train_data, test_data = cl_loaders
    print(f"Client-{client_id}: {len(train_data)} iters/epoch")
    # clients[client_id] = cl
    client_ids.append(client_id)
    client_epochs[client_id] = 0
    client_loaders[client_id] = cl_loaders
    torch.cuda.empty_cache()

# initialize global model
global_model = all_models[args.dataset](device=devices[0], classes=dataset_classes[args.dataset], global_args=args)
initialize_mask(global_model)

global_model.layer_prune(sparsity=args.sparsity, sparsity_distribution=args.sparsity_distribution)

initial_global_state = deepcopy(global_model.state_dict())

# we need to accumulate compute/DL/UL costs regardless of round number, resetting only
# when we actually report these numbers
compute_times = np.zeros(len(client_ids)) # time in seconds taken on client-side for round
download_cost = np.zeros(len(client_ids))
upload_cost = np.zeros(len(client_ids))

print(f'Total clients: {client_ids}')
# for each round t = 1, 2, ... do
# for server_round in tqdm(range(args.rounds)): # 默认 400
cumulative_ul, cumulative_dl = 0, 0
global_params_direction = {}

dataset_classes = {
    'mnist': 10,
    'cifar10': 10,
    'cifar100': 100,
    'pacs': 7,
    'officehome': 65,
    'vlcs': 5,
}

client_temp_models = {}
for device in devices:
    device_idx = devices[0] if device.type == devices[0] else str(device.index)
    client_temp_model = all_models[args.dataset](device=device, classes=dataset_classes[args.dataset], global_args=args).to(device)
    client_temp_models[device_idx] = client_temp_model

def get_client_model(device):
    device_idx = devices[0] if device.type == devices[0] else str(device.index)
    return client_temp_models[device_idx]

def get_client_instance(client_id):
    cl_device, train_data, test_data = client_loaders[client_id]
    client = Client(i, cl_device, train_data, test_data, net=get_client_model(cl_device),
                    learning_rate=args.lr, local_epochs=args.epochs,
                    target_sparsity=args.sparsity, classes=dataset_classes[args.dataset], global_args=args, curr_epoch=client_epochs[client_id])
    return client

for server_round in range(args.rounds): # 默认 400
    # sample clients
    client_indices = rng.choice(list(client_ids), size=args.clients, replace=False)
    print_to_log(f"Selected clients: {client_indices} at round {server_round+1}")

    global_state = global_model.state_dict()
    aggregated_weight_params = {}
    aggregated_weight_params_needmsk = {} # This is the final aggregated params for global model.
    aggregated_mask_params = {}
    
    # set server parameters to 0 in preparation for aggregation,
    for key, val in global_state.items():
        if key.endswith('_mask'):
            continue
        aggregated_weight_params[key] = torch.zeros_like(val, dtype=torch.float, device=devices[0])
        aggregated_weight_params_needmsk[key] = torch.zeros_like(val, dtype=torch.float, device=devices[0])
        if needs_mask(key): # weight 都需要 mask， bias 不需要。
            aggregated_mask_params[key] = torch.zeros_like(val, device=devices[0])

    # for each client k \in S_t in parallel do    
    if args.rate_decay_method == 'cosine':
        readjustment_ratio = args.readjustment_ratio * global_model._decay(server_round, alpha=args.readjustment_ratio, t_end=args.rate_decay_end)
    else:
        readjustment_ratio = args.readjustment_ratio
    readjust = (server_round - 1) % args.rounds_between_readjustments == 0 and readjustment_ratio > 0. # bool 值
    # determine sparsity desired at the end of this round
    # ...via linear interpolation
    if server_round <= args.rate_decay_end:
        round_sparsity = args.sparsity * (args.rate_decay_end - server_round) / args.rate_decay_end + args.final_sparsity * server_round / args.rate_decay_end
    else:
        round_sparsity = args.final_sparsity
    if readjust:
        print_to_log(f'R-{server_round}: S.S:{round_sparsity}; ReAdjust:{readjustment_ratio}')
    
    for client_id in client_indices:
        i = client_ids.index(client_id)
        client = get_client_instance(client_id)
        # Local client training.
        t0 = time.process_time()

        # actually perform training
        train_result = client.train(global_params=global_state, initial_global_params=initial_global_state,
                                    readjustment_ratio=readjustment_ratio,
                                    readjust=readjust, sparsity=round_sparsity, global_params_direction=global_params_direction)
        client_epochs[client_id] = client_epochs[client_id] + 1
        client_state = train_result['state'] # with mask_name

        download_cost[i] = train_result['dl_cost']
        upload_cost[i] = train_result['ul_cost']
        cumulative_ul = cumulative_ul + upload_cost[i] # int((upload_cost[i])/8.0/1e6)
        cumulative_dl = cumulative_dl + download_cost[i] # int((download_cost[i])/8.0/1e6)

        t1 = time.process_time()
        compute_times[i] = t1 - t0
        client.net.clear_gradients() # to save memory

        # add this client's params to the aggregate
        cl_weight_all_params = {} # key 是所有的 param_name， value 是 weight
        cl_mask_params_needmsk = {} # key 是需要mask的 param_name （不是 mask_name）， value 是 mask

        # first deduce masks for the received weights
        for key, cl_val in client_state.items():
            if key.endswith('_orig'):
                name = key[:-5]
            elif key.endswith('_mask'):
                name = key[:-5]
                cl_mask_params_needmsk[name] = cl_val.to(device=devices[0], copy=True)
                continue

            cl_weight_all_params[key] = cl_val.to(device=devices[0], copy=True)
            if args.fp16:
                cl_weight_all_params[key] = cl_weight_all_params[key].to(torch.bfloat16).to(torch.float)

        # This client ended up with current round training.
        # at this point, we have weights and masks (possibly all-ones)
        # for this client. we will proceed by applying the mask and adding
        # the masked received weights to the aggregate, and adding the mask
        # to the aggregate as well.

        for name, cl_weight in cl_weight_all_params.items():
            if name in cl_mask_params_needmsk:
                # things like weights have masks
                cl_mask = cl_mask_params_needmsk[name]
                sv_mask = global_state[name+'_mask'].to(devices[0], copy=True)

                # calculate Hamming distance of masks for debugging
                # if readjust:
                #     print(f'{client.id} {name} d_h=', torch.sum(cl_mask ^ sv_mask).item())
                aggregated_weight_params[name].add_(client.train_size() * cl_weight * cl_mask)
                aggregated_weight_params_needmsk[name].add_(client.train_size() * cl_weight * cl_mask)
                aggregated_mask_params[name].add_(client.train_size() * cl_mask) # 这里就体现出了 votes，train_size 越大，对应 mask votes越大。

                if args.remember_old: # By default, this is False.
                    sv_mask[cl_mask] = 0
                    sv_param = global_state[name].to(devices[0], copy=True)
                    # remember_old 为 True 的时候，aggregated_params_for_mask 才会与 aggregated_weight_params 在 weight 上的值不同。
                    # 加上 server 的 weight 值，但是最后 global_model inference 时并没有用到，只是用于重新计算 server mask
                    aggregated_weight_params_needmsk[name].add_(client.train_size() * sv_param) #* sv_mask)
                    aggregated_mask_params[name].add_(client.train_size() * sv_mask)
            else:
                # things like biases don't have masks
                aggregated_weight_params[name].add_(client.train_size() * cl_weight)

    # at this point, ALL selected clients ended up with current round training.
    # we have the sum of client parameters
    # in aggregated_params, and the sum of masks in aggregated_masks. We
    # can take the average now by simply dividing...
    for name, param in aggregated_weight_params.items():
        # if this parameter has no associated mask, simply take the average.
        if name not in aggregated_mask_params:
            
            train_size = sum(len(x) for x in client_loaders[client_id][1])
            aggregated_weight_params[name] /= sum(train_size for i in client_indices)
            continue
        
        # 接下里的 name 都是需要 mask 的。
        # drop parameters with not enough votes
        aggregated_mask_params[name] = F.threshold_(aggregated_mask_params[name], args.min_votes, 0)
        # otherwise, we are taking the weighted average w.r.t. the number of 
        # samples present on each of the clients.
        aggregated_weight_params[name] /= aggregated_mask_params[name]
        aggregated_weight_params_needmsk[name] /= aggregated_mask_params[name]
        aggregated_mask_params[name] /= aggregated_mask_params[name]
        # it's possible that some weights were pruned by all clients. In this
        # case, we will have divided by zero. Those values have already been
        # pruned out, so the values here are only placeholders.
        aggregated_weight_params[name] = torch.nan_to_num(aggregated_weight_params[name],
                                                   nan=0.0, posinf=0.0, neginf=0.0)
        aggregated_weight_params_needmsk[name] = torch.nan_to_num(aggregated_weight_params[name],
                                                   nan=0.0, posinf=0.0, neginf=0.0)
        aggregated_mask_params[name] = torch.nan_to_num(aggregated_mask_params[name],
                                                  nan=0.0, posinf=0.0, neginf=0.0)

    # masks are parameters too!
    for name, mask in aggregated_mask_params.items():
        mask_name = name + '_mask'
        aggregated_weight_params_needmsk[mask_name] = mask

    global_state_pre_rd = deepcopy(global_model.state_dict())
 
    # reset global params to aggregated values
    # 这里只 load 了带 mask的参数，没有 load bias什么的。
    # 因为只是要计算一下新的 mask，所以不需要 load 全部。
    global_model.load_state_dict(aggregated_weight_params_needmsk)
    if global_model.sparsity() < round_sparsity: #
        # we now have denser networks than we started with at the beginning of
        # the round. reprune on the server to get back to the desired sparsity.
        # we use layer-wise magnitude pruning as before.
        global_model.layer_prune(sparsity=round_sparsity, sparsity_distribution=args.sparsity_distribution)

    # discard old weights and apply new mask
    global_state = global_model.state_dict()
    aggregated_state = aggregated_weight_params # 这里的 aggregated_state 没有 mask，只有 weight

    for name, mask in aggregated_mask_params.items():
        delta_weight = global_state[name] - global_state_pre_rd[name]
        global_params_direction[name] = torch.sign(delta_weight).to(devices[0])

        new_mask = global_state[name+'_mask']
        aggregated_state[name+'_mask'] = new_mask
        aggregated_state[name][~new_mask] = 0

    global_model.load_state_dict(aggregated_state) # 这次所有的 bias 啥的都 load进去了

    # evaluate performance
    torch.cuda.empty_cache()
    if server_round % args.eval_every == 0 and args.eval:
        accuracies, sparsities = evaluate_global_clients(client_ids, global_model, progress=False,
                                                 n_batches=args.test_batches)

    global_acc = evaluate_global_model(global_model, loaders)
    for client_id in client_ids:
        i = client_ids.index(client_id)
        if server_round == 0:
            client = get_client_instance(client_id)
            client.initial_global_params = initial_global_state
        if (server_round % args.eval_every == 0) and args.eval:
            print_csv_line(
                # pid=args.pid,
                           round=server_round,
                           target_sparsity=round_sparsity,
                        #    initial_pruning_threshold='',
                        #    final_pruning_threshold='',
                        #    pruning_threshold_growth_method='',
                        #    pruning_method='',
                        #    lth=False,
                           client_id=client_id,
                           sparsity=sparsities[client_id],
                           compute_time=compute_times[i],
                           download_cost=download_cost[i],
                           upload_cost=upload_cost[i],
                           accuracy=accuracies[client_id].item(),
                           global_acc=global_acc.item())

    global_acc = evaluate_global_model(global_model, loaders)
    if server_round == args.rounds-1:
        print_to_log(f'FINAL ROUND! {server_round}')
        print_to_log('Final Global Acc: ', global_acc)
    else:
        print_to_log(f'Global Acc @{server_round+1}: {global_acc}, CUL:{int(cumulative_ul/8.0/1e6)} MiB, CDL:{int(cumulative_dl/8.0/1e6)} MiB')

        # if we didn't send initial global params to any clients in the first round, send them now.
        # (in the real world, this could be implemented as the transmission of
        # a random seed, so the time and place for this is not a concern to us)

    if server_round % args.eval_every == 0 and args.eval:
        # clear compute, UL, DL costs
        compute_times[:] = 0
        download_cost[:] = 0
        upload_cost[:] = 0

flog.close()
fcsv.close()

#print2('OVERALL SUMMARY')
#print2()
#print2(f'{args.total_clients} clients, {args.clients} chosen each round')
#print2(f'E={args.epochs} local epochs per round, B={args.batch_size} mini-batch size')
#print2(f'{args.rounds} rounds of federated learning')
#print2(f'Target sparsity r_target={args.target_sparsity}, pruning rate (per round) r_p={args.pruning_rate}')
#print2(f'Accuracy threshold starts at {args.pruning_threshold} and ends at {args.final_pruning_threshold}')
#print2(f'Accuracy threshold growth method "{args.pruning_threshold_growth_method}"')
#print2(f'Pruning method: {args.pruning_method}, resetting weights: {args.reset_weights}')
#print2()
#print2(f'ACCURACY: mean={np.mean(accuracies)}, std={np.std(accuracies)}, min={np.min(accuracies)}, max={np.max(accuracies)}')
#print2(f'SPARSITY: mean={np.mean(sparsities)}, std={np.std(sparsities)}, min={np.min(sparsities)}, max={np.max(sparsities)}')
#print2()
#print2()

