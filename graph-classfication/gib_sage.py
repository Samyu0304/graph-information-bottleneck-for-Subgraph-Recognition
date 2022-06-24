import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool, JumpingKnowledge, SAGEConv
from torch_geometric.utils import to_dense_adj

class GIBSAGE(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GIBSAGE, self).__init__()
        self.conv1 = SAGEConv(dataset.num_features, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        #attention = torch.nn.functional.softmax(self.fully_connected_2(abstract_features_1), dim=1)
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

        self.cluster1 = Linear(hidden, hidden)
        self.cluster2 = Linear(hidden, 2)
        self.mse_loss = nn.MSELoss()


    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.cluster1.reset_parameters()
        self.cluster2.reset_parameters()

    def assignment(self,x):

        return self.cluster2(torch.tanh(self.cluster1(x)))

    #for i in batch:
    def aggregate(self, assignment, x, batch, edge_index):

        max_id = torch.max(batch)
        if torch.cuda.is_available():
            EYE = torch.ones(2).cuda()
        else:
            EYE = torch.ones(2)

        all_adj = to_dense_adj(edge_index)[0]

        all_pos_penalty = 0
        all_graph_embedding = []
        all_pos_embedding = []

        st = 0
        end = 0

        for i in range(int(max_id + 1)):

            #print(i)

            j = 0

            while batch[st + j] == i and st + j <= len(batch) - 2:
                j += 1

            end = st + j

            if end == len(batch) - 1:
                end += 1

            one_batch_x = x[st:end]
            one_batch_assignment = assignment[st:end]

            #sum_assignment = torch.sum(x, dim=0, keepdim=False)[0]

            group_features = torch.mm(torch.t(one_batch_assignment), one_batch_x)

            pos_embedding = group_features[0].unsqueeze(dim=0)

            Adj = all_adj[st:end,st:end]
            new_adj = torch.mm(torch.t(one_batch_assignment), Adj)
            new_adj = torch.mm(new_adj, one_batch_assignment)


            normalize_new_adj = F.normalize(new_adj, p=1, dim=1, eps = 0.00001)


            norm_diag = torch.diag(normalize_new_adj)
            pos_penalty = self.mse_loss(norm_diag, EYE)

            graph_embedding = torch.mean(x, dim=0, keepdim=True)

            #print(pos_embedding)
            #print(graph_embedding)

            all_pos_embedding.append(pos_embedding)
            all_graph_embedding.append(graph_embedding)

            all_pos_penalty = all_pos_penalty + pos_penalty

            st = end



        all_pos_embedding = torch.cat(tuple(all_pos_embedding), dim=0)
        all_graph_embedding = torch.cat(tuple(all_graph_embedding), dim=0)
        all_pos_penalty = all_pos_penalty / (max_id + 1)

        return all_pos_embedding,all_graph_embedding, all_pos_penalty




    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))

        x = F.relu(self.conv2(x, edge_index))

        #print(x)

        assignment = torch.nn.functional.softmax(self.assignment(x), dim=1)

        all_pos_embedding, all_graph_embedding,all_pos_penalty = self.aggregate(assignment, x, batch, edge_index)


        x = F.relu(self.lin1(all_pos_embedding))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), all_pos_embedding, all_graph_embedding, all_pos_penalty


    def __repr__(self):
        return self.__class__.__name__


class Discriminator(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()

        self.input_size = 2 * hidden_size
        self.hidden_size = hidden_size
        self.lin1 = torch.nn.Linear(self.input_size,self.hidden_size)
        self.lin2 = torch.nn.Linear(self.hidden_size, 1)
        self.relu = torch.nn.ReLU()


    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, embeddings,positive):

        cat_embeddings = torch.cat((embeddings, positive),dim = -1)

        pre = self.relu(self.lin1(cat_embeddings))
        pre = self.relu(self.lin2(pre))

        return pre




class GIB0WithJK(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, mode='cat'):
        super(GIB0WithJK, self).__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(dataset.num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=False))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__




if __name__ == '__main__':
    import argparse
    from itertools import product
    from datasets import get_dataset
    from diff_pool import DiffPool
    from train_eval import cross_validation_with_val_set

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)  # default = 100
    parser.add_argument('--batch_size', type=int, default=4)  # default = 128
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=50)
    parser.add_argument('--dataset', type=str, default='MUTAG')
    args = parser.parse_args()


    layers = [2]
    hiddens = [16, 32, 64, 128]
    # datasets = ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'DD', 'COLLAB']  # , 'COLLAB']DD
    datasets = [args.dataset]
    nets = [GIB0]

    results = []
    for dataset_name, Net in product(datasets, nets):
        best_result = (float('inf'), 0, 0)  # (loss, acc, std)
        print('-----\n{} - {}'.format(dataset_name, Net.__name__))
        for num_layers, hidden in product(layers, hiddens):
            dataset = get_dataset(dataset_name, sparse=Net != DiffPool)
            model = Net(dataset, num_layers, hidden)
            loss, acc, std = cross_validation_with_val_set(
                dataset,
                model,
                folds=10,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                lr_decay_factor=args.lr_decay_factor,
                lr_decay_step_size=args.lr_decay_step_size,
                weight_decay=0,
                logger=None,
            )
            if loss < best_result[0]:
                best_result = (loss, acc, std)

        desc = '{:.3f} ± {:.3f}'.format(best_result[1], best_result[2])
        print('Best result - {}'.format(desc))
        results += ['{} - {}: {}'.format(dataset_name, model, desc)]
    print('-----\n{}'.format('\n'.join(results)))