import torch
from utils import tab_printer
from subgraph import Subgraph_Learning
from param_parser import parameter_parser

import torch._utils

def main():

    try:
        torch._utils._rebuild_tensor_v2
    except AttributeError:
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
            tensor.requires_grad = requires_grad
            tensor._backward_hooks = backward_hooks
            return tensor

        torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

    args = parameter_parser()
    tab_printer(args)
    trainer = Subgraph_Learning(args)
    trainer.fit()
    trainer.test()

if __name__ == '__main__':
    main()
