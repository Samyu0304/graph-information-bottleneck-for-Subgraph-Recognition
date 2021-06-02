import torch
import random
from tqdm import trange
from layers import Subgraph, Discriminator
from utils import  GraphDatasetGenerator
import itertools
import json
from tqdm import tqdm
import numpy as np
import os

class Subgraph_Learning(object):

    def __init__(self, args):
        super(Subgraph_Learning, self).__init__()
        self.args = args
        self.dataset_generator = GraphDatasetGenerator(self.args.data)
        self.batch_size = self.args.batch_size
        self.train_percent = self.args.train_percent
        self.valiate_percent = self.args.validate_percent
        self.D_criterion = torch.nn.BCEWithLogitsLoss()
        self.inner_loop = self.args.inner_loop

    def _dataset_spilt(self):
        Data_Length = len(self.dataset_generator.graphs)
        Training_Length = int(self.train_percent * Data_Length)
        Validate_Length = int(self.valiate_percent * Data_Length)
        Testing_Length = Data_Length - Training_Length - Validate_Length
        test_ind = [i for i in range(0, Testing_Length)]
        all_ind = [j for j in range(0, Data_Length)]
        train_val_ind = list(set(all_ind)-set(test_ind))
        train_ind = train_val_ind[0:Training_Length]
        validate_ind = train_val_ind[Training_Length:]
        self.training_data = [self.dataset_generator.graphs[i] for i in train_ind]
        self.valiate_data = [self.dataset_generator.graphs[i] for i in validate_ind]
        self.testing_data = [self.dataset_generator.graphs[i] for i in test_ind]

    def _setup_model(self):
        self.model = Subgraph(self.args, self.dataset_generator.number_of_features)
        self.discriminator = Discriminator(self.args)

        if torch.cuda.is_available():
            self.discriminator = Discriminator(self.args).cuda()
            self.model = Subgraph(self.args, self.dataset_generator.number_of_features).cuda()


    def set_requires_grad(self, net, requires_grad=False):

        if net is not None:

            for param in net.parameters():
                param.requires_grad = requires_grad

    def fit_a_single_model(self):
        self._dataset_spilt()
        self._setup_model()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        Data_Length = len(self.training_data)
        Num_split = int(Data_Length / self.batch_size)

        for _ in tqdm(range(self.args.epochs)):

            for i in range(0, Num_split):
                data = self.training_data[int(i*self.batch_size): min(int((i+1)*self.batch_size),Data_Length)]
                embeddings, positive, negative,  cls_loss, positive_penalty = self.model(data)

                for j in range(0, self.inner_loop):

                    optimizer_local = torch.optim.Adam(self.discriminator.parameters(),
                                                       lr=self.args.learning_rate,
                                                       weight_decay=self.args.weight_decay)
                    optimizer_local.zero_grad()
                    local_loss = - self.MI_Est(self.discriminator, embeddings, positive)
                    local_loss.backward(retain_graph = True)
                    optimizer_local.step()

                mi_loss = self.MI_Est(self.discriminator, embeddings, positive)
                optimizer.zero_grad()
                loss = cls_loss + positive_penalty + self.args.mi_weight * mi_loss
                loss.backward()
                optimizer.step()
                print("Loss:%.2f"%(loss))

    def MI_Est(self, discriminator, embeddings, positive):
        shuffle_embeddings = embeddings[torch.randperm(self.batch_size)]
        joint = discriminator(embeddings,positive)
        margin = discriminator(shuffle_embeddings,positive)
        mi_est = torch.mean(joint) - torch.log(torch.mean(torch.exp(margin)))

        return mi_est

    def return_index(self,data):
        self.model.eval()
        ind = self.model.assemble(data)

        return ind

    def validate(self):
        ind = self.return_index(self.valiate_data)
        count = 0

        for data in ind:
            save_path = os.path.join(self.args.save_validate, str(count) + '.json')
            dump_data = json.dumps(data)
            F = open(save_path, 'w')
            F.write(dump_data)
            F.close()
            count += 1

    def test(self):
        ind = self.return_index(self.testing_data)
        count = 0

        for data in ind:
            save_path = os.path.join(self.args.save_test, str(count) + '.json')
            dump_data = json.dumps(data)
            F = open(save_path, 'w')
            F.write(dump_data)
            F.close()
            count += 1

    def fit(self):
        print("\nTraining started.\n")
        self.fit_a_single_model()