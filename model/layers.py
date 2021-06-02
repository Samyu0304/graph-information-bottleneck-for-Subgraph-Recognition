import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GIB(torch.nn.Module):

    def __init__(self, args, number_of_features):

        super(GIB, self).__init__()
        self.args = args
        self.number_of_features = number_of_features
        self._setup()
        self.mseloss = torch.nn.MSELoss()
        self.relu = torch.nn.ReLU()
        self.subgraph_const = self.args.subgraph_const

    def _setup(self):

        self.graph_convolution_1 = GCNConv(self.number_of_features,
                                           self.args.first_gcn_dimensions)

        self.graph_convolution_2 = GCNConv(self.args.first_gcn_dimensions,
                                           self.args.second_gcn_dimensions)

        self.fully_connected_1 = torch.nn.Linear(self.args.second_gcn_dimensions,
                                                 self.args.first_dense_neurons)

        self.fully_connected_2 = torch.nn.Linear(self.args.first_dense_neurons,
                                                 self.args.second_dense_neurons)

    def forward(self, data):

        edges = data["edges"]
        features = data["features"]
        node_features_1 = torch.nn.functional.relu(self.graph_convolution_1(features, edges))
        node_features_2 = self.graph_convolution_2(node_features_1, edges)
        abstract_features_1 = torch.tanh(self.fully_connected_1(node_features_2))
        assignment = torch.nn.functional.softmax(self.fully_connected_2(abstract_features_1), dim=1)
        group_features = torch.mm(torch.t(assignment),node_features_2)

        if torch.cuda.is_available():
            EYE = torch.ones(2).cuda()
        else:
            EYE = torch.ones(2)

        Adj = to_dense_adj(edges)[0]
        Adj.requires_grad = False
        new_adj = torch.mm(torch.t(assignment),Adj)
        new_adj = torch.mm(new_adj,assignment)
        positive = torch.clamp(group_features[0].unsqueeze(dim = 0),-100,100)
        negative = torch.clamp(group_features[1].unsqueeze(dim = 0),-100,100)
        normalize_new_adj = F.normalize(new_adj, p=1, dim=1)
        norm_diag = torch.diag(normalize_new_adj)
        pos_penalty = self.mseloss(norm_diag, EYE)
        graph_embedding = torch.mm(torch.t(assignment), node_features_2)
        graph_embedding = torch.mean(graph_embedding,dim = 0,keepdim= True)

        return graph_embedding, positive, negative,  pos_penalty

    def return_att(self,data):
        edges = data["edges"]
        features = data["features"]
        node_features_1 = torch.nn.functional.relu(self.graph_convolution_1(features, edges))
        node_features_2 = self.graph_convolution_2(node_features_1, edges)
        abstract_features_1 = torch.tanh(self.fully_connected_1(node_features_2))
        attention = torch.nn.functional.softmax(self.fully_connected_2(abstract_features_1), dim=1)

        return attention


class Discriminator(torch.nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.args = args
        self.input_size = 2 * self.args.second_gcn_dimensions
        self.hidden_size = self.args.dis_hidden_dimensions
        self.fc1 = torch.nn.Linear(self.input_size,self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.relu = torch.nn.ReLU()

        torch.nn.init.constant(self.fc1.weight, 0.01)
        torch.nn.init.constant(self.fc2.weight, 0.01)

    def forward(self, embeddings,positive):
        cat_embeddings = torch.cat((embeddings, positive),dim = -1)
        pre = self.relu(self.fc1(cat_embeddings))
        pre = self.fc2(pre)

        return pre


class Subgraph(torch.nn.Module):

    def __init__(self, args, number_of_features):
        super(Subgraph, self).__init__()

        self.args = args
        self.number_of_features = number_of_features
        self._setup()
        self.mse_criterion = torch.nn.MSELoss(reduction='mean')
        self.relu = torch.nn.ReLU()

    def _setup(self):

        self.graph_level_model = GIB(self.args, self.number_of_features)
        self.classify = torch.nn.Sequential(torch.nn.Linear(self.args.second_gcn_dimensions, self.args.cls_hidden_dimensions), torch.nn.ReLU(),torch.nn.Linear(self.args.cls_hidden_dimensions, 1), torch.nn.ReLU()) # , torch.nn.ReLU()

    def forward(self, graphs):

        embeddings = []
        positive = []
        negative = []
        labels = []
        positive_penalty = 0

        for graph in graphs:
            embedding, pos, neg, pos_penalty = self.graph_level_model(graph)
            embeddings.append(embedding)
            positive.append(pos)
            negative.append(neg)
            positive_penalty += pos_penalty
            labels.append(graph["label"])

        embeddings = torch.cat(tuple(embeddings),dim = 0)
        positive = torch.cat(tuple(positive),dim = 0)
        negative = torch.cat(tuple(negative), dim=0)
        labels = torch.FloatTensor(labels).view(-1,1)
        positive_penalty = positive_penalty/len(graphs)
        cls_loss = self.supervise_classify_loss(embeddings, positive, labels)

        return embeddings, positive, negative, cls_loss, self.args.con_weight * positive_penalty


    def supervise_classify_loss(self,embeddings,positive,labels):

        data = torch.cat((embeddings, positive), dim=0)
        labels = torch.cat((labels,labels),dim = 0)

        if torch.cuda.is_available():
            labels = labels.cuda()

        pred = self.classify(data)
        loss = self.mse_criterion(pred,labels)

        return loss


    def assemble(self, graphs):

        all_index_pair = []
        all_bias = []

        for graph in graphs:
            smiles = graph['smiles']
            attention = self.graph_level_model.return_att(graph)
            _,ind = torch.max(attention,1)
            ind = ind.tolist()
            pos_ind = [i for i,j in enumerate(ind) if j == 0]
            decomposed_cluster, decomposed_subgraphs = self.decompose_cluster(smiles,pos_ind)
            smiles_ind_pairs = {'smiles':smiles,'ind':decomposed_cluster,'subgraphs':decomposed_subgraphs}
            all_index_pair.append(smiles_ind_pairs)

        return all_index_pair, all_bias


    def get_nei(self, atom_ind, edges):
        nei = []

        for bond in edges:

            if atom_ind in bond:
                nei.extend(bond)

        nei.remove(atom_ind)
        nei = list(set(nei))

        return nei

    def decompose_cluster(self,smiles, ind):

        mol = Chem.MolFromSmiles(smiles)
        all_cluster = []
        all_subgraphs = []
        edge = []

        for bond in mol.GetBonds():
            edge.append([bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()])

        for i in ind:
            cluster = [i]
            ind.remove(i)
            nei = self.get_nei(i,edge)
            valid_nei = list(set(nei).intersection(set(ind)))

            while valid_nei != [] :
                cluster.extend(valid_nei)
                new_nei = []

                for j in valid_nei:
                    ind.remove(j)
                    new_nei.extend(self.get_nei(j,edge))
                valid_nei = list(set(new_nei).intersection(set(ind)))

            subgraph = self.get_subgraph_with_idx(smiles,cluster)
            all_subgraphs.append(subgraph)
            all_cluster.append(cluster)

        return all_cluster, all_subgraphs

    def get_subgraph_with_idx(self,smiles,ind):
        mol = Chem.MolFromSmiles(smiles)
        ri = mol.GetRingInfo()

        for ring in ri.AtomRings():
            ring = list(ring)

            if set(ind) >= set(ring):
                pass
            else:
                for atom_ind in ring:
                    broke_atom = mol.GetAtomWithIdx(atom_ind)
                    broke_atom.SetIsAromatic(False)

        edit_mol = Chem.EditableMol(mol)
        del_ind = sorted(list(set(range(mol.GetNumAtoms())) - set(ind)))[::-1]

        for idx in del_ind:
            edit_mol.RemoveAtom(idx)

        new_mol = edit_mol.GetMol()
        subgraph = Chem.MolToSmiles(new_mol)

        if subgraph:
            return subgraph
        else:
            return None

if __name__ == '__main__':
    from rdkit import Chem
    import torch

    from torch_geometric.utils import to_dense_adj

    edge = [[0,1,1,1,2,3,0,4],[1,0,2,3,1,1,4,0]]
    edge = torch.LongTensor(edge)
    batch_id = torch.LongTensor([0,0,1,1,1])
    all_edge = to_dense_adj(edge)[0]
    print(all_edge)
    st = 0
    end = 2
    edge = all_edge[st:end,st:end]
    print(edge)



















