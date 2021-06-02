import argparse
def parameter_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", nargs="?", default="../input/qed/", help="Folder with training graph jsons.")

    parser.add_argument("--train_percent", type = float, default= 0.85, help="trainin_percent.")

    parser.add_argument("--validate_percent", type = float, default= 0.05, help="validate_percent.")

    parser.add_argument("--first-gcn-dimensions", type=int, default=16, help="Filters (neurons) in 1st convolution. Default is 16.")

    parser.add_argument("--second-gcn-dimensions", type=int, default=8, help="Filters (neurons) in 2nd convolution. Default is 8.")

    parser.add_argument("--first-dense-neurons", type=int, default=16, help="Neurons in SAGE aggregator layer. Default is 16.")

    parser.add_argument("--second-dense-neurons", type=int, default=2, help="assignment. Default is 2.")

    parser.add_argument("--cls_hidden_dimensions", type=int, default= 4, help="classifier hidden dims")

    parser.add_argument("--dis_hidden_dimensions", type=int, default= 4, help="discriminator hidden dims")

    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs. Default is 3.")

    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate. Default is 0.010.")

    parser.add_argument("--weight-decay", type=float, default=5*10**-5, help="Adam weight decay. Default is 5*10^-5.")

    parser.add_argument("--gamma", type=float, default=10**-5, help="Attention regularization coefficient. Default is 10^-5.")

    parser.add_argument("--save_test", type=str, default='../test_results/qed/', help="save results .")

    parser.add_argument("--save_validate", type=str, default='../validate_results/qed/', help="save results .")

    parser.add_argument("--batch_size", type=int, default= 128, help="batch_size")

    parser.add_argument("--mi_weight", type=float, default= 0.2, help="classifier hidden dims")

    parser.add_argument("--con_weight", type=float, default= 5, help="classifier hidden dims")

    parser.add_argument("--inner_loop", type=int, default= 150, help="classifier hidden dims")

    return parser.parse_args()
