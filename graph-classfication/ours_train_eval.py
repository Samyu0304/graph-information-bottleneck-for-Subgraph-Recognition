import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cross_validation_with_val_set(dataset, model, discriminator, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, inner_loop, mi_weight, pp_weight, logger=None):

    val_losses, accs, durations = [], [], []
    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        discriminator.to(device).reset_parameters()
        optimizer_local = Adam(discriminator.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_loss = train(model, discriminator, optimizer, optimizer_local, train_loader, mi_weight, pp_weight, inner_loop)

            if train_loss != train_loss:
                print('NaN')
                continue

            val_losses.append(eval_loss(model, val_loader))
            accs.append(eval_acc(model, test_loader))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_losses[-1],
                'test_acc': accs[-1],
            }

            print(eval_info)



            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(folds, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    print(acc)
    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(loss_mean, acc_mean, acc_std, duration_mean))

    return loss_mean, acc_mean, acc_std


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, discriminator, optimizer, local_optimizer, loader, mi_weight, pp_weight, inner_loop):
    model.train()

    total_loss = 0
    for data in loader:

        data = data.to(device)
        out, all_pos_embedding, all_graph_embedding, all_pos_penalty = model(data)




        for j in range(0, inner_loop):
            local_optimizer.zero_grad()
            local_loss = - MI_Est(discriminator, all_graph_embedding.detach(), all_pos_embedding.detach())
            local_loss.backward()
            local_optimizer.step()


        optimizer.zero_grad()
        loss = F.nll_loss(out, data.y.view(-1))

        mi_loss = MI_Est(discriminator, all_graph_embedding, all_pos_embedding)

        loss = (1-pp_weight) * (loss + mi_weight*mi_loss) + pp_weight * all_pos_penalty #(1-mi_weight)*

        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)



def MI_Est(discriminator, embeddings, positive):

    batch_size = embeddings.shape[0]

    shuffle_embeddings = embeddings[torch.randperm(batch_size)]

    joint = discriminator(embeddings,positive)

    margin = discriminator(shuffle_embeddings,positive)

    #Donsker
    #mi_est = torch.mean(joint) - torch.clamp(torch.log(torch.mean(torch.exp(margin))),-100000,100000)
    #JSD
    #mi_est = -torch.mean(F.softplus(-joint)) - torch.mean(F.softplus(-margin)+margin)
    #
    mi_est = torch.mean(joint**2) - 0.5* torch.mean((torch.sqrt(margin**2)+1.0)**2)
    return mi_est


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred,_,_,_ = model(data)
            pred = pred.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out,_,_,_ = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)
