import torch
import torch.nn as nn

import models
from models import all_models, needs_mask, initialize_mask

def print_to_log(*arg, log_file=None, **kwargs):
    print(*arg, file=log_file, **kwargs)
    print(*arg, **kwargs)

class Client:

    def __init__(self, id, device, train_data, test_data, net=models.MNISTNet,
                 local_epochs=10, learning_rate=0.01, target_sparsity=0.1, classes=10, global_args=None, curr_epoch=0):
        '''Construct a new client.

        Parameters:
        id : object
            a unique identifier for this client. For EMNIST, this should be
            the actual client ID.
        train_data : iterable of tuples of (x, y)
            a DataLoader or other iterable giving us training samples.
        test_data : iterable of tuples of (x, y)
            a DataLoader or other iterable giving us test samples.
            (we will use this as the validation set.)
        local_epochs : int
            the number of local epochs to train for each round

        Returns: a new client.
        '''
        self.id = id
        self.global_args = global_args

        self.train_data, self.test_data = train_data, test_data

        self.device = device
        # self.net = net(device=self.device, classes=classes).to(self.device)
        self.net = net
        initialize_mask(self.net)
        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.reset_optimizer()

        self.local_epochs = local_epochs
        self.curr_epoch = curr_epoch

        # save the initial global params given to us by the server
        # for LTH pruning later.
        self.initial_global_params = None


    def reset_optimizer(self):
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=self.global_args.momentum, weight_decay=self.global_args.l2)


    def reset_weights(self, *args, **kwargs):
        return self.net.reset_weights(*args, **kwargs)


    def sparsity(self, *args, **kwargs):
        return self.net.sparsity(*args, **kwargs)


    def train_size(self):
        return sum(len(x) for x in self.train_data)


    # 这里的这个 initial_global_params 似乎没啥用，据说是给 PruneFL 的实现使用的
    def train(self, global_params=None, initial_global_params=None,
              readjustment_ratio=0.5, readjust=False, sparsity=0.0, global_params_direction={}):
        '''Train the client network for a single round.'''

        ul_cost = 0
        dl_cost = 0
        if global_params:
            # this is a FedAvg-like algorithm, where we need to reset
            # the client's weights every round
            mask_changed = self.reset_weights(global_state=global_params, use_global_mask=True)

            # Try to reset the optimizer state.
            self.reset_optimizer()

            if mask_changed:
                dl_cost += self.net.mask_size # need to receive mask

            if not self.initial_global_params:
                self.initial_global_params = initial_global_params
                # no DL cost here: we assume that these are transmitted as a random seed
            else:
                # otherwise, there is a DL cost: we need to receive all parameters masked '1' and
                # all parameters that don't have a mask (e.g. biases in this case)
                dl_cost += (1-self.net.sparsity()) * self.net.mask_size * 32 + (self.net.param_size - self.net.mask_size * 32)

        #pre_training_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        for epoch in range(self.local_epochs):

            self.net.train()

            running_loss = 0.

            for i, (inputs, labels) in enumerate(self.train_data):
                if self.global_args.drill and i >= 3:
                    print_to_log(f'drill training: Client-{self.id}', log_file=self.global_args.log_file)
                    break
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                if self.global_args.prox > 0:
                    loss += self.global_args.prox / 2. * self.net.proximal_loss(global_params)
                loss.backward()
                self.optimizer.step()

                self.reset_weights() # applies the mask, without changing weights.
                # if i % 10 == 0:
                #     print(f"iteration: {i}, loss: {loss.item()}")
                running_loss += loss.item()
            if readjust:
                print('client-'+str(self.id), self.curr_epoch)
            if self.curr_epoch >= self.global_args.pruning_begin and (self.curr_epoch - self.global_args.pruning_begin) % self.global_args.pruning_interval == 0 and readjust:
                prune_sparsity = sparsity + (1 - sparsity) * readjustment_ratio
                # recompute gradient if we used FedProx penalty
                print_to_log(f"Client-{self.id} start-pruning, sparsity: {prune_sparsity}, epoch: {self.curr_epoch}", 
                             log_file=self.global_args.log_file)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                self.criterion(outputs, labels).backward()

                self.net.layer_prune(sparsity=prune_sparsity, sparsity_distribution=self.global_args.sparsity_distribution, 
                                     dire_ratio = self.global_args.direction_ratio, global_directions=global_params_direction)
                self.net.layer_grow(sparsity=sparsity, sparsity_distribution=self.global_args.sparsity_distribution,
                                    dire_ratio = self.global_args.direction_ratio, global_directions=global_params_direction)
                ul_cost += (1-self.net.sparsity()) * self.net.mask_size # need to transmit mask, unit: bit
            self.curr_epoch += 1

        # we only need to transmit the masked weights and all biases
        if self.global_args.fp16:
            ul_cost += (1-self.net.sparsity()) * self.net.mask_size * 16 + (self.net.param_size - self.net.mask_size * 16)
        else:
            ul_cost += (1-self.net.sparsity()) * self.net.mask_size * 32 + (self.net.param_size - self.net.mask_size * 32)
        ret = dict(state=self.net.state_dict(), dl_cost=dl_cost, ul_cost=ul_cost)

        #dprint(global_params['conv1.weight_mask'][0, 0, 0], '->', self.net.state_dict()['conv1.weight_mask'][0, 0, 0])
        #dprint(global_params['conv1.weight'][0, 0, 0], '->', self.net.state_dict()['conv1.weight'][0, 0, 0])
        return ret

    def test(self, model=None, n_batches=0):
        '''Evaluate the local model on the local test set.

        model - model to evaluate, or this client's model if None
        n_batches - number of minibatches to test on, or 0 for all of them
        '''
        correct = 0.
        total = 0.

        if model is None:
            model = self.net
            _model = self.net
        else:
            _model = model.to(self.device)

        _model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.test_data):
                if i > n_batches and n_batches > 0:
                    break
                if not self.global_args.cache_test_set_gpu:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                outputs = _model(inputs)
                outputs = torch.argmax(outputs, dim=-1)
                correct += sum(labels == outputs)
                total += len(labels)

        # remove copies if needed
        if model is not _model:
            del _model

        return correct / total
