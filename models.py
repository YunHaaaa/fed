import sys
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils import prune
import torchvision
import prune as torch_prune
import warnings
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, ResNet18_Weights, Bottleneck
from collections import OrderedDict
import math

def print_to_log(*arg, log_file=None, **kwargs):
    print(*arg, file=log_file, **kwargs)
    print(*arg, **kwargs)

# Utility functions
def needs_mask(name):
    return (name.endswith('weight')) and ('bn' not in name)


def initialize_mask(model, dtype=torch.bool):
    layers_to_prune = ((layername, layer) for layername, layer in model.leaf_modules())
    for layername, layer in layers_to_prune:
        for name, param in layer.named_parameters():
            if not needs_mask(layername + '.' +name):
                continue
            if hasattr(layer, layername + '.' +name + '_mask'):
                warnings.warn(
                    'Parameter has a pruning mask already. '
                    'Reinitialize to an all-one mask.'
                )
            bname = name + '_mask'
            # register_buffer 是用来在模块中添加一个不是模型参数的缓冲区的方法。
            # 这通常用于注册一些不需要通过梯度更新的缓冲区，但是又是模型状态的一部分，比如 BatchNorm 的 running_mean。
            # 注册缓冲区的好处是，当你保存或者移动模型时，缓冲区也会跟着保存或者移动。
            # 注册缓冲区的方法是在模块上调用 register_buffer 方法，传入一个名字和一个初始值。
            # 比如说：pytorch 中 BN 层的 running_mean 和 running_var 是注册在模块中的缓冲区，而不是模型参数。
            layer.register_buffer(bname, torch.ones_like(param, dtype=dtype))

class PrunableNet(nn.Module):
    '''Common functionality for all networks in this experiment.'''

    def leaf_modules(self):
        moudles = []
        for name, md in self.named_modules():
            l =  (list(md.modules()))
            if len(l) == 1:
                moudles.append((name, md))
        return moudles


    def __init__(self, device='cpu', global_args=None):
        super(PrunableNet, self).__init__()
        self.device = device
        self.global_args = global_args
        self.communication_sparsity = 0


    def init_param_sizes(self):
        # bits required to transmit mask and parameters?
        self.mask_size = 0
        self.param_size = 0
        for _, layer in self.leaf_modules():
            for name, param in layer.named_parameters():
                param_size = np.prod(param.size())
                self.param_size += param_size * 32 # FIXME: param.dtype.size?
                if needs_mask(name):
                    self.mask_size += param_size
        #print(f'Masks require {self.mask_size} bits.')
        #print(f'Weights require {self.param_size} bits.')
        #print(f'Unmasked weights require {self.param_size - self.mask_size*32} bits.')


    def clear_gradients(self):
        for _, param in self.named_parameters():
            del param.grad
        torch.cuda.empty_cache()


    def infer_mask(self, masking):
        for name, param in self.state_dict().items():
            if needs_mask(name) and name in masking.masks:
                mask_name = name + '_mask'
                mask = self.state_dict()[mask_name]
                mask.copy_(masking.masks[name])


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    def _decay(self, t, alpha=0.3, t_end=400):
        if t >= t_end:
            return 0
        return alpha/2 * (1 + np.cos(t*np.pi / t_end))


    def _weights_by_layer(self, sparsity=0.1, sparsity_distribution='erk'):
        with torch.no_grad():
            layer_names = []
            sparsities = np.empty(len(list(self.leaf_modules()))) # 3层layers => sparsities: [S1, S2, S3] 
            # sparsities = []
            n_weights = np.zeros_like(sparsities, dtype=int)

            for i, (name, layer) in enumerate(self.leaf_modules()):
                layer_names.append(name)
                kernel_size = None

                if isinstance(layer, nn.modules.conv._ConvNd):
                    neur_out = layer.out_channels
                    neur_in = layer.in_channels
                    kernel_size = layer.kernel_size
                elif isinstance(layer, nn.Linear):
                    neur_out = layer.out_features
                    neur_in = layer.in_features
                elif isinstance(layer, nn.modules.container.Sequential):
                    neur_in, neur_out = self.layer_channels[name]
                else:
                    continue

                for pname, param in layer.named_parameters():
                    n_weights[i] += param.numel()

                if sparsity_distribution == 'uniform':
                    sparsities[i] = sparsity
                    continue
                if sparsity_distribution == 'er':
                    sparsities[i] = 1 - (neur_in + neur_out) / (neur_in * neur_out)
                elif sparsity_distribution == 'erk':
                    if isinstance(layer, nn.modules.conv._ConvNd):
                        sparsities[i] = 1 - (neur_in + neur_out + np.sum(kernel_size)) / (neur_in * neur_out * np.prod(kernel_size))
                    else:
                        sparsities[i] = 1 - (neur_in + neur_out) / (neur_in * neur_out)
                else:
                    raise ValueError('Unsupported sparsity distribution ' + sparsity_distribution)
                
            # Now we need to renormalize sparsities.
            # We need global sparsity S = sum(s * n) / sum(n) equal to desired
            # sparsity, and s[i] = C n[i]
            sparsities *= sparsity * np.sum(n_weights) / np.sum(sparsities * n_weights)
            n_weights = np.floor((1-sparsities) * n_weights)
            ret =  {layer_names[i]: n_weights[i] for i in range(len(layer_names))}
            return ret


    def layer_prune(self, sparsity=0.1, sparsity_distribution='erk', dire_ratio = 0.1, global_directions={}):
        '''
        Prune the network to the desired sparsity, following the specified
        sparsity distribution. The weight magnitude is used for pruning.

        uniform: layer sparsity = global sparsity
        er: Erdos-Renyi
        erk: Erdos-Renyi Kernel
        '''

        #print('desired sparsity', sparsity)
        with torch.no_grad():
            weights_by_layer = self._weights_by_layer(sparsity=sparsity, sparsity_distribution=sparsity_distribution)
            for name, layer in self.leaf_modules():
                # We need to figure out how many to prune
                n_total = 0
                for bname, buf in layer.named_buffers():
                    n_total += buf.numel()
                # DEBUG: sometimes the n_prune_tmp is NaN
                n_prune_tmp = n_total - weights_by_layer[name]
                if math.isnan(n_prune_tmp):
                    print_to_log(f'Weird Error: {name},{n_total},{weights_by_layer[name]},{n_prune_tmp}', log_file=self.global_args.log_file)
                    n_prune = 0
                n_prune = int(n_prune_tmp)
                if n_prune >= n_total or n_prune < 0:
                    continue
                print_to_log(f'Prune (total): {name}, {n_prune}', log_file=self.global_args.log_file)
                for pname, param in layer.named_parameters():
                    if not needs_mask(pname):
                        continue

                    # Determine smallest indices
                    if dire_ratio <= 0.0:
                        _, prune_indices = torch.topk(torch.abs(param.data.flatten()), n_prune, largest=False)
                        param.data.view(param.data.numel())[prune_indices] = 0
                        for bname, buf in layer.named_buffers():
                            if bname == pname + '_mask':
                                buf.view(buf.numel())[prune_indices] = 0
                        continue

                    same_directions = []
                    if param.grad is not None and len(global_directions) > 0:
                        grad_direction = torch.sign(param.grad.flatten())
                        x = global_directions[name+'.'+pname].flatten().to(self.device)
                        same_directions = (grad_direction == x)

                    n_prune_weight, n_prune_dir = n_prune, 0
                    prune_indices_dir = None
                    if len(same_directions) > 0:
                        diff_directions_count = torch.sum(~same_directions)
                        n_prune_dir = int(dire_ratio * n_prune)
                        n_prune_dir = min(n_prune_dir, diff_directions_count)
                        n_prune_weight = n_prune - n_prune_dir
                        if n_prune_dir > 0:
                            paradata_tmp = torch.abs(param.data.flatten())
                            paradata_tmp[same_directions] = 999999
                            _, prune_indices_dir = torch.topk(paradata_tmp, n_prune_dir, largest=False)

                    prune_indices_weight = None
                    if prune_indices_dir is not None:
                        paradata_tmp = torch.abs(param.data.flatten())
                        paradata_tmp[prune_indices_dir] == 999999
                        _, prune_indices_weight = torch.topk(paradata_tmp, n_prune_weight, largest=False)

                    if prune_indices_weight is not None:
                        print_to_log('Prune (WithDir, weight): ', len(prune_indices_dir), len(prune_indices_weight))
                        prune_indices = torch.cat((prune_indices_dir, prune_indices_weight))
                    else:
                        _, prune_indices = torch.topk(torch.abs(param.data.flatten()), n_prune, largest=False)

                    param.data.view(param.data.numel())[prune_indices] = 0
                    for bname, buf in layer.named_buffers():
                        if bname == pname + '_mask':
                            buf.view(buf.numel())[prune_indices] = 0
            #print('pruned sparsity', self.sparsity())


    def layer_grow(self, sparsity=0.1, sparsity_distribution='erk', dire_ratio = 0.1, global_directions={}):
        '''
        Grow the network to the desired sparsity, following the specified
        sparsity distribution.
        The gradient magnitude is used for growing weights.

        uniform: layer sparsity = global sparsity
        er: Erdos-Renyi
        erk: Erdos-Renyi Kernel
        '''

        #print('desired sparsity', sparsity)
        with torch.no_grad():
            weights_by_layer = self._weights_by_layer(sparsity=sparsity, sparsity_distribution=sparsity_distribution)
            for name, layer in self.leaf_modules():

                # We need to figure out how many to grow
                n_nonzero = 0
                for bname, buf in layer.named_buffers():
                    n_nonzero += buf.count_nonzero().item()
                n_grow = int(weights_by_layer[name] - n_nonzero) # 期望的非0参数的个数
                if n_grow < 0:
                    continue
                print('Grow (total)', name, n_grow)

                for pname, param in layer.named_parameters():
                    if not needs_mask(pname):
                        continue

                    # Determine smallest indices
                    if dire_ratio <= 0.0:
                        _, grow_indices = torch.topk(torch.abs(param.grad.flatten()), n_grow, largest=True)
                        param.data.view(param.data.numel())[grow_indices] = 0
                        for bname, buf in layer.named_buffers():
                            if bname == pname + '_mask':
                                buf.view(buf.numel())[grow_indices] = 1
                        continue

                    same_directions = []
                    if param.grad is not None and len(global_directions) > 0:
                        grad_direction = torch.sign(param.grad.flatten())
                        x = global_directions[name+'.'+pname].flatten().to(self.device)
                        same_directions = (grad_direction == x)

                    n_grow_grad, n_grow_dir, grow_indices_dir = n_grow, 0, None
                    if len(same_directions) > 0:
                        same_directions_count = torch.sum(same_directions)
                        n_grow_dir = min(int(dire_ratio * n_grow), same_directions_count)
                        n_grow_grad = n_grow - n_grow_dir
                        if n_grow_dir > 0:
                            para_grad_tmp = torch.abs(param.grad.flatten())
                            para_grad_tmp[~same_directions] = -1.0
                            _, grow_indices_dir = torch.topk(para_grad_tmp, n_grow_dir, largest=True)

                    grow_indices_grad = None
                    if grow_indices_dir is not None:
                        para_grad_tmp = torch.abs(param.grad.flatten())
                        para_grad_tmp[grow_indices_dir] == -1.0
                        _, grow_indices_grad = torch.topk(para_grad_tmp, n_grow_grad, largest=True)

                    if grow_indices_grad is not None:
                        print('Grow (WithDir, grad):', len(grow_indices_dir), len(grow_indices_grad))
                        grow_indices = torch.cat((grow_indices_dir, grow_indices_grad))
                    else:
                        _, grow_indices = torch.topk(torch.abs(param.grad.flatten()), n_grow, largest=True)

                    # Write and apply mask
                    param.data.view(param.data.numel())[grow_indices] = 0
                    for bname, buf in layer.named_buffers():
                        if bname == pname + '_mask':
                            buf.view(buf.numel())[grow_indices] = 1
            #print('grown sparsity', self.sparsity())


    def prunefl_readjust(self, aggregate_gradients, layer_times, prunable_params=0.3):
        with torch.no_grad():
            importances = []
            for i, g in enumerate(aggregate_gradients):
                g.square_()
                g = g.div(layer_times[i])
                importances.append(g)

            t = 0.2
            delta = 0
            cat_grad = torch.cat([torch.flatten(g) for g in aggregate_gradients])
            cat_imp = torch.cat([torch.flatten(g) for g in importances])
            indices = torch.argsort(cat_grad, descending=True)
            n_required = (1 - prunable_params) * cat_grad.numel()
            n_grown = 0

            masks = []
            for i, g in enumerate(aggregate_gradients):
                masks.append(torch.zeros_like(g, dtype=torch.bool))

            for j, i in enumerate(indices):
                if cat_imp[i] >= delta/t or n_grown <= n_required:
                    index_within_layer = i.item()
                    for layer in range(len(layer_times)):
                        numel = aggregate_gradients[layer].numel()
                        if index_within_layer >= numel:
                            index_within_layer -= numel
                        else:
                            break

                    delta += cat_grad[i]
                    t += layer_times[layer]

                    shape = tuple(masks[layer].shape)
                    masks[layer][np.unravel_index(index_within_layer, shape)] = 1
                    n_grown += 1
                else:
                    break

            print('readj density', n_grown / cat_imp.numel())

            # set masks
            state = self.state_dict()
            i = 0
            n_differences = 0
            for name, param in state.items():
                if name.endswith('_mask'):
                    continue
                if not needs_mask(name):
                    continue

                n_differences += torch.count_nonzero(state[name + '_mask'].to('cpu') ^ masks[i].to('cpu'))
                state[name + '_mask'] = masks[i]
                i += 1

            print('mask changed percent', n_differences / cat_imp.numel())
                    
            self.load_state_dict(state)
            return n_differences/cat_imp.numel()


    def prune(self, pruning_rate=0.2):
        with torch.no_grad():
            # prune (self.pruning_rate) of the remaining weights
            parameters_to_prune = []
            layers_to_prune = (layer for _, layer in self.leaf_modules())
            for layer in layers_to_prune:
                for name, param in layer.named_parameters():
                    if needs_mask(name):
                        parameters_to_prune.append((layer, name))

            # (actually perform pruning)
            torch_prune.global_unstructured(
                parameters_to_prune,
                pruning_method=torch_prune.L1Unstructured,
                amount=pruning_rate
            )

    # Seems not get called at all.
    def grow(self, indices):
        with torch.no_grad():
            state = self.state_dict()
            keys = list(state.keys())
            for grow_index in indices:
                mask_name = keys[grow_index[0]] + "_mask"
                state[mask_name].flatten()[grow_index[1]] = 1
            self.load_state_dict(state)

    def reset_weights(self, global_state=None, use_global_mask=False,
                      keep_local_masked_weights=False,
                      global_communication_mask=False):
        '''Reset weights to the given global state and apply the mask.
        - If global_state is None, then only apply the mask in the current state.
        - use_global_mask will reset the local mask to the global mask.
        - keep_local_masked_weights will use the global weights where masked 1, and
          use the local weights otherwise.
        '''

        with torch.no_grad():
            mask_changed = False
            local_state = self.state_dict()

            # If no global parameters were specified, that just means we should
            # apply the local mask, so the local state should be used as the
            # parameter source.
            if global_state is None:
                param_source = local_state
            else:
                param_source = global_state

            # We may wish to apply the global parameters but use the local mask.
            # In these cases, we will use the local state as the mask source.
            if use_global_mask:
                apply_mask_source = global_state
            else:
                apply_mask_source = local_state

            # We may wish to apply the global mask to the global parameters,
            # but not overwrite the local mask with it.
            if global_communication_mask:
                copy_mask_source = local_state
            else:
                copy_mask_source = apply_mask_source

            self.communication_sparsity = self.sparsity(apply_mask_source.items())

            # Empty new state to start with.
            new_state = {}

            # Copy over the params, masking them off if needed.
            for name, param in param_source.items():
                if name.endswith('_mask'):
                    # skip masks, since we will copy them with their corresponding
                    # layers, from the mask source.
                    continue
                new_state[name] = local_state[name]
                if needs_mask(name) and (name + '_mask') in apply_mask_source:
                    mask_name =  name + '_mask'
                    mask_to_apply = apply_mask_source[mask_name].to(device=self.device, dtype=torch.bool)
                    mask_to_copy = copy_mask_source[mask_name].to(device=self.device, dtype=torch.bool)
                    gpu_param = param[mask_to_apply].to(self.device)

                    # copy weights provided by the weight source, where the mask
                    # permits them to be copied
                    new_state[name][mask_to_apply] = gpu_param

                    # Don't bother allocating a *new* mask if not needed
                    if mask_name in local_state:
                        new_state[mask_name] = local_state[mask_name] 

                    new_state[mask_name].copy_(mask_to_copy) # copy mask from mask_source into this model's mask

                    # what do we do with shadowed weights?
                    if not keep_local_masked_weights: # by default, keep_local_masked_weights is False, this if-clause will enter-in
                        new_state[name][~mask_to_apply] = 0

                    if mask_name not in local_state or not torch.equal(local_state[mask_name], mask_to_copy):
                        mask_changed = True
                else:
                    # biases and other unmasked things
                    gpu_param = param.to(self.device)
                    new_state[name].copy_(gpu_param)

                # clean up copies made to gpu
                if gpu_param.data_ptr() != param.data_ptr():
                    del gpu_param
            self.load_state_dict(new_state)
        return mask_changed


    def proximal_loss(self, last_state):

        loss = torch.tensor(0.).to(self.device)

        state = self.state_dict()
        for i, (name, param) in enumerate(state.items()):
            if name.endswith('_mask'):
                continue
            gpu_param = last_state[name].to(self.device)
            loss += torch.sum(torch.square(param - gpu_param))
            if gpu_param.data_ptr != last_state[name].data_ptr:
                del gpu_param

        return loss


    def topk_changes(self, last_state, count=5, mask_behavior='invert'):
        '''Find the top `count` changed indices and their values
        since the given last_state.
        - mask_behavior determines how the layer mask is used:
          'normal' means to take the top-k which are masked 1 (masked in)
          'invert' means to take the top-k which are masked 0 (masked out)
          'all' means to ignore the mask

        returns (values, final_indices) tuple. Where values has zeroes,
        we were only able to find top k0 < k.
        '''

        with torch.no_grad():
            state = self.state_dict()
            topk_values = torch.zeros(len(state), count)
            topk_indices = torch.zeros_like(topk_values)

            for i, (name, param) in enumerate(state.items()):
                if name.endswith('_mask') or not needs_mask(name):
                    continue
                mask_name = name + '_mask'
                haystack = param - last_state[name]
                if mask_name in state and mask_behavior != 'all':
                    mask = state[mask_name]
                    if mask_behavior == 'invert':
                        mask = 1 - mask
                    haystack *= mask

                haystack = haystack.flatten()
                layer_count = min((count, haystack.numel()))
                vals, idxs = torch.topk(torch.abs(haystack), k=layer_count, largest=True, sorted=False)
                topk_indices[i, :layer_count] = idxs
                topk_values[i, :layer_count] = haystack[idxs]

            # Get the top-k collected
            vals, idxs = torch.topk(torch.abs(topk_values).flatten(), k=count, largest=True, sorted=False)
            vals = topk_values.flatten()[idxs]
            final_indices = torch.zeros(count, 2)
            final_indices[:, 0] = idxs // count # which parameters do they belong to?
            final_indices[:, 1] = topk_indices.flatten()[idxs] # indices within

        return vals, final_indices


    def sparsity(self, buffers=None):

        if buffers is None:
            buffers = self.named_buffers()

        n_ones = 0
        mask_size = 0
        for name, buf in buffers:
            if name.endswith('mask'):
                # print(name, torch.sum(buf), buf.nelement())
                n_ones += torch.sum(buf)
                mask_size += buf.nelement()

        return 1 - (n_ones / mask_size).item()


#############################
# Subclasses of PrunableNet #
#############################

class MNISTNet(PrunableNet):

    def __init__(self, classes=10, *args, **kwargs):
        super(MNISTNet, self).__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(1, 10, 5) # "Conv 1-10"
        self.conv2 = nn.Conv2d(10, 20, 5) # "Conv 10-20"

        self.fc1 = nn.Linear(20 * 16 * 16, 50)
        self.fc2 = nn.Linear(50, 10)

        self.init_param_sizes()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        x = x.view(-1, self.num_flat_features(x)) # flatten
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class CIFAR10Net(PrunableNet):

    def __init__(self, classes=10, *args, **kwargs):
        super(CIFAR10Net, self).__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 20 * 20, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.init_param_sizes()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
class CIFAR100Net(PrunableNet):

    def __init__(self, classes=100, *args, **kwargs):
        super(CIFAR100Net, self).__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 20 * 20, 120)
        self.fc2 = nn.Linear(120, 100)

        self.init_param_sizes()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class EMNISTNet(PrunableNet):

    def __init__(self, *args, **kwargs):
        super(EMNISTNet, self).__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(1, 10, 5) # "Conv 1-10"
        self.conv2 = nn.Conv2d(10, 20, 5) # "Conv 10-20"

        self.fc1 = nn.Linear(20 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 62)

        self.init_param_sizes()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        x = x.view(-1, self.num_flat_features(x)) # flatten
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class Conv2(PrunableNet):
    '''The EMNIST model from LEAF:
    https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py
    '''

    def __init__(self, *args, **kwargs):
        super(Conv2, self).__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, 62)

        self.init_param_sizes()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, stride=2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class ResNet(PrunableNet):
    def __init__(self, block, layers, *args, classes=7, num_channels=3, **kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__(*args,  **kwargs)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer_channels = {'layer1': (64, 64),
                               'layer2': (64, 128),
                               'layer3': (128, 256),
                               'layer4': (256, 512)}
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.class_classifier = nn.Linear(512 * block.expansion, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.init_param_sizes()

    # stride 参数决定了 该 Residual Layer 的第一个 block 的 stride，进而决定了这个layer 会不会缩减图片尺寸
    # res18 里只有第一个 Layer 不需要缩减，其他 Layer 都需要减半尺寸
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # stride ！=1 意味着，第一个 block 需要缩减 size（第一个 block 的 X 和 Y 都需要进行 size 调整）， downsample 调整 X
        # block.expansion 是针对 Bottleneck Block，该类型 Block 会 4 倍地 expand input dimensions 
        # 所以其实 downsample 这里做了两件事 1. 调整 size  2. 调整 dimensions
        # 每一个 Layer 都有且只有一个 downsample
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample)) # 第一个 Block 用了 stride 参数，改变 size 且使用 downsample
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes)) # 后面就默认为1，不改变 size

        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        # layers_output_dict['max_pool']=x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # layers_output_dict['avg_pool']=x
        x = x.view(x.size(0), -1)
        out = self.class_classifier(x)
        # layers_output_dict['fc']=out
        return out

def resnet18(pretrained=False, *args, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2],*args, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(ResNet18_Weights.IMAGENET1K_V1.url), strict=False)
    return model




class VggNet(PrunableNet):
    def __init__(self, *args, classes=7, vgg_no='vgg11', **kwargs):
        super(VggNet, self).__init__(*args,  **kwargs)
        self.vgg_no = vgg_no
        model = torchvision.models.vgg16(num_classes=classes)
        
        self.feature_layer_names = {
            'vgg11': ['conv1_1', 'relu1_1',
                      'conv1_2', 'relu1_2',
                      'pool1',
                      'conv2_1', 'relu2_1',
                      'conv2_2', 'relu2_2',
                      'pool2',
                      'conv3_1', 'relu3_1',
                      'conv3_2', 'relu3_2',
                      'conv3_3', 'relu3_3',
                      'pool3',
                      'conv4_1', 'relu4_1',
                      'conv4_2', 'relu4_2',
                      ],
            'vgg16': ['conv1_1', 'relu1_1',
                   'conv1_2', 'relu1_2',
                   'pool1',
                   'conv2_1', 'relu2_1',
                   'conv2_2', 'relu2_2',
                   'pool2',
                   'conv3_1', 'relu3_1',
                   'conv3_2', 'relu3_2',
                   'conv3_3', 'relu3_3',
                   'pool3',
                   'conv4_1', 'relu4_1',
                   'conv4_2', 'relu4_2',
                   'conv4_3', 'relu4_3',
                   'pool4',
                   'conv5_1', 'relu5_1',
                   'conv5_2', 'relu5_2',
                   'conv5_3', 'relu5_3',
                   'pool5']
             }
        
        for idx in range(len(self.feature_layer_names[vgg_no])):
            setattr(self, self.feature_layer_names[vgg_no][idx], model.features[idx])


        # layers = collections.OrderedDict(zip(self.layer_names, model.features))
        # self.features = torch.nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier_layer_names = [
            'fc6', 'relu6',
            'drop6',
            'fc7', 'relu7',
            'drop7',
            'fc8a']
        for idx in range(len(self.classifier_layer_names)):
            setattr(self, self.classifier_layer_names[idx], model.classifier[idx])
        # self.classifier = torch.nn.Sequential(OrderedDict(zip(self.classifier_layer_names, model.classifier)))
        self.init_param_sizes()

    def forward(self, x, **kwargs):
        for layername in self.feature_layer_names[self.vgg_no]:
            x = getattr(self, layername)(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for layername in self.classifier_layer_names:
            x = getattr(self, layername)(x)
        # x = self.classifier(x)
        return  x

def vgg11(pretrained=False, *args, **kwargs):
    model = VggNet(*args, **kwargs, vgg_no='vgg11')
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']), strict=False)
    return model

def vgg16(pretrained=False, *args, **kwargs):
    model = VggNet(*args, **kwargs, vgg_no='vgg16')
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
    return model

all_models = {
        'mnist': MNISTNet,
        'emnist': Conv2,
        'cifar10': vgg11,
        'cifar100': vgg16,
        'pacs': resnet18,
        'office': resnet18,
        'office-home': resnet18,
}
