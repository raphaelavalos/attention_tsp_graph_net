import argparse
import pprint
import os
import numpy as np
from collections import namedtuple
import time
import json

def get_options_and_config():
    parser = argparse.ArgumentParser(description="Attention TSP Train Module")
    parser.add_argument('--nodes', default=20, type=int, help="The number of nodes per graph ex: 20, 50, 100.",
                        required=False)
    parser.add_argument('--cuda', default=True, type=bool, help="Use cuda (default True)", required=False)
    parser.add_argument('--gpu', type=int, default=0,
                        help="The id of the GPU, -1 for CPU (default: 0).",
                        required=False)
    parser.add_argument('--save_dir', default="saved_models/experiments", help="The directory for saving the models.",
                        required=False)
    parser.add_argument('--tensorboard', type=bool, default=True, help="Activate tensorboard (default: True).",
                        required=False)
    parser.add_argument('--epoch', type=int, default=100, help="Number of epochs (default: 100).", required=False)
    parser.add_argument('--rollout_steps', type=int, default=2,
                        help="The number of batch that will be use in the rolling phase", required=False)
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate (default: 0.0001).",
                        required=False)
    parser.add_argument('--save_freq', type=int, default=10, help="Save model every _ epochs (default: 10).",
                        required=False)
    parser.add_argument('--step_per_epoch', type=int, default=2500, help="Number of step per epoch (default: 2500).",
                        required=False)
    # parser.add_argument('--freq_update', type=int, default=500, help="Get update every _ steps (default: 500).",
    #                     required=False)
    parser.add_argument('--batch', type=int, default=512, help="Batch size (default: 512).", required=False)
    parser.add_argument('--encoder_layer', type=int, default=3, help="Number of encoder layers (default: 3).",
                        required=False)
    parser.add_argument('--v', type=bool, default=False, help="Verbose (default: False).", required=False)
    parser.add_argument('--debug', type=bool, default=False, help="Debug (default: False).", required=False)
    parser.add_argument('--val_dataset', type=bool, default=None,
                        help="Validation dataset path, if None the validation dataset will be created and saved.",
                        required=False)

    args = parser.parse_args()

    experiment_name = "TSP_%i_%s" % (args.nodes, time.strftime("%Y%m%d_%H%M%S"))

    # Make save dir compatible
    if args.save_dir[-1] != '/':
        args.save_dir += '/'
    args.save_dir += experiment_name + '/'
    os.makedirs(args.save_dir, exist_ok=True)

    pprint.pprint(vars(args))

    # Build config
    ENCODER_NBR_LAYERS = 3
    EMBEDDING_DIM = 128
    HEAD_NBR = 8
    KEY_DIM = 16
    VALUE_DIM = 16
    QUERY_DIM = 16
    INIT_DIM = 2
    INIT_MAX = 1 / np.sqrt(INIT_DIM)
    FF_HIDDEN_SIZE = 512
    N_NODE = 20
    C = 10

    Config = namedtuple('ConfigTSP', ['batch',
                                      'learning_rate',
                                      'head_nbr',
                                      'key_dim',
                                      'embedding_dim',
                                      'value_dim',
                                      'query_dim',
                                      'init_dim',
                                      'init_max',
                                      'ff_hidden_size',
                                      'n_node',
                                      'c',
                                      'encoder_nbr_layers'])

    conf = Config(batch=args.batch, learning_rate=args.learning_rate, head_nbr=HEAD_NBR, key_dim=KEY_DIM,
                  embedding_dim=EMBEDDING_DIM, value_dim=VALUE_DIM, query_dim=QUERY_DIM, init_dim=INIT_DIM,
                  init_max=INIT_MAX, ff_hidden_size=FF_HIDDEN_SIZE, n_node=args.nodes, c=C,
                  encoder_nbr_layers=ENCODER_NBR_LAYERS)

    with open(args.save_dir + 'args.json', 'w') as f:
        json.dump(vars(args), f)

    with open(args.save_dir + 'conf.json', 'w') as f:
        json.dump(conf._asdict(), f)

    return args, conf, experiment_name
