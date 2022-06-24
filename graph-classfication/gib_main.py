from itertools import product

import argparse
from datasets import get_dataset
from ours_train_eval import cross_validation_with_val_set

from gib_gin import GIBGIN, Discriminator
from gib_gat import GIBGAT
from gib_sage import GIBSAGE
from gib_gcn  import GIBGCN

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)#default = 100
parser.add_argument('--batch_size', type=int, default=128)#default = 128
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--net', type=int, default=0)
parser.add_argument('--inner_loop', type=int, default=50)
parser.add_argument('--mi_weight', type=float, default=0.1)
parser.add_argument('--pp_weight', type=float, default=0.3)
args = parser.parse_args()

layers = [2]
hiddens = [16,32,64]
datasets = ['MUTAG', 'PROTEINS', 'DD', 'COLLAB']
datasets = [args.dataset]
nets = [GIBGCN, GIBSAGE, GIBGAT, GIBGIN]


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
        fold, epoch, val_loss, test_acc))

results = []
for dataset_name, Net in product(datasets, nets):

    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    print('-----\n{} - {}'.format(dataset_name, Net.__name__))
    for num_layers, hidden in product(layers, hiddens):
        dataset = get_dataset(dataset_name, sparse=True)
        model = Net(dataset, num_layers, hidden)
        discriminator = Discriminator(hidden)
        loss, acc, std = cross_validation_with_val_set(
            dataset,
            model,
            discriminator,
            folds=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            inner_loop = args.inner_loop,
            mi_weight = args.mi_weight,
            pp_weight=args.pp_weight,
            logger= None
        )
        if loss < best_result[0]:
            best_result = (loss, acc, std)

    desc = '{:.3f} , {:.3f}'.format(best_result[1], best_result[2])
    print('Best result - {}'.format(desc))
    results += ['{} - {}: {}'.format(dataset_name, model, desc)]
print('-----\n{}'.format('\n'.join(results)))
