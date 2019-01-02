from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from graph_nets import utils_tf
from graph_nets import utils_np
from graph_nets import graphs
import argparse
import os
import pprint
import tsp_dataset
from collections import namedtuple
from tensorflow.python import debug as tf_debug
from models import AttentionTspModel
import pickle
from tqdm import trange

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Attention TSP Train Module")
    parser.add_argument('--nodes', default=20, type=int, help="The number of nodes per graph ex: 20, 50, 100.", required=False)
    parser.add_argument('--cuda', default=True, type=bool, help="Use cuda (default True)", required=False)
    parser.add_argument('--gpu', type=int, default=0,
                        help="The id of the GPU, -1 for CPU (default: 0).",
                        required=False)
    parser.add_argument('--save_dir', default="saved_models/experiments", help="The directory for saving the models.",
                        required=False)
    parser.add_argument('--tensorboard', type=bool, default=True, help="Activate tensorboard (default: True).", required=False)
    parser.add_argument('--epoch', type=int, default=1000, help="Number of epochs (default: 1000).", required=False)
    parser.add_argument('--learning_rate', type=float, default=.0001, help="Learning rate (default: 0.0001).", required=False)
    parser.add_argument('--save_freq', type=int, default=10, help="Save model every _ epochs (default: 10).", required=False)
    parser.add_argument('--step_per_epoch', type=int, default=2500, help="Number of step per epoch (default: 2500).",
                        required=False)
    parser.add_argument('--freq_update', type=int, default=500, help="Get update every _ steps (default: 500).", required=False)
    parser.add_argument('--batch', type=int, default=512, help="Batch size (default: 512).", required=False)
    parser.add_argument('--encoder_layer', type=int, default=3, help="Number of encoder layers (default: 3).", required=False)
    parser.add_argument('--v', type=bool, default=False, help="Verbose (default: False).", required=False)

    args = parser.parse_args()
    print(pprint.pprint(args))

    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_dir[-1] != '/':
        args.save_dir += '/'

    val_step = 10000 // args.batch

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
    BATCH = 8
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
                  init_max=INIT_MAX, ff_hidden_size=FF_HIDDEN_SIZE, n_node=N_NODE, c=C,
                  encoder_nbr_layers=ENCODER_NBR_LAYERS)

    pprint.pprint(conf)

    # Device selection
    if not args.cuda or args.gpu == -1:
        device0 = "/cpu:0"
    else:
        device0 = "/device:GPU:%i" % args.gpu

    # Creating Graphs

    graph0 = tf.Graph()

    # Model creation

    sample = tsp_dataset.generate_networkx_batch(args.batch, args.nodes)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=False

    with graph0.device(device0):
        sess0 = tf.Session(config=config)
        # sess0 = tf_debug.TensorBoardDebugWrapperSession(sess0, "Artik:6064")
        model = AttentionTspModel(conf)
        baseline = AttentionTspModel(conf)
        baseline.modify_state(False, True)

        # Create the random nodes
        nodes = tf.random_uniform([conf.batch * conf.n_node, 2], minval=-1., maxval=1.)
        # graph creation operation
        graph_input = graphs.GraphsTuple(
            nodes=nodes,
            edges=None,
            receivers=None,
            senders=None,
            globals=tf.zeros((conf.batch,), dtype=tf.float32),
            n_node=tf.convert_to_tensor(np.full((conf.batch,), conf.n_node, dtype=np.int64)),
            n_edge=tf.zeros((conf.batch,), dtype=tf.int64)
        )

        fully_connected_graph = utils_tf.fully_connect_graph_static(graph_input).replace(
            edges=tf.zeros((2*conf.batch*conf.n_node,), dtype=tf.int64))

        _, _, _, baseline_cost = baseline(fully_connected_graph)
        baseline_cost = tf.stop_gradient(baseline_cost)
        _, computed_log_likelihood, result_graph, cost = model(fully_connected_graph)
        loss = tf.reduce_mean(tf.multiply(tf.subtract(cost, baseline_cost), computed_log_likelihood), name="loss")
        loss0 = tf.reduce_mean(tf.multiply(cost, computed_log_likelihood), name="loss0")

        baseline_cost_mean = tf.reduce_mean(baseline_cost)
        model_cost_mean = tf.reduce_mean(cost)

        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 1.0)

        opt_step = optimizer.minimize(loss=loss, var_list=model.get_variables())
        opt_step0 = optimizer.minimize(loss=loss0, var_list=model.get_variables())

        cost_summary = tf.summary.scalar('Mean cost', tf.reduce_mean(cost))
        loss_summary = tf.summary.scalar('Loss', loss)
        loss_summary0 = tf.summary.scalar('Loss', loss0)

        init0 = tf.global_variables_initializer()
        init0_local = tf.local_variables_initializer()

        writer = tf.summary.FileWriter("log/", sess0.graph)


    with graph0.device(device0):
        sess0.run(init0)
        sess0.run(init0_local)

    for epoch in trange(args.epoch, desc="Epoch"):
        data = utils_np.networkxs_to_graphs_tuple(tsp_dataset.generate_networkx_batch(args.batch, args.nodes))
        for step in trange(args.step_per_epoch, desc="Step", leave=False):
            # create input
            with graph0.device(device0):
                sess0.run(init0_local)
                if epoch == 0:
                    _, cost_summary_,  loss_summary_ = sess0.run(fetches=[opt_step0, cost_summary, loss_summary0],)
                 #                                                feed_dict={input_placeholder: data})

                else:
                    _, cost_summary_, loss_summary_ = sess0.run(fetches=[opt_step, cost_summary, loss_summary],)
                #                                                feed_dict={input_placeholder: data})
                if args.tensorboard:
                    writer.add_summary(cost_summary_, step + epoch * args.step_per_epoch)
                    writer.add_summary(loss_summary_, step + epoch * args.step_per_epoch)

        # Test baseline
        cost_r_m = []
        cost_r_b = []
        if epoch != 0:
            for step in trange(val_step, desc="Rollout step"):
                data = utils_np.networkxs_to_graphs_tuple(tsp_dataset.generate_networkx_batch(args.batch, args.nodes))
                with graph0.device(device0):
                    sess0.run(init0_local)
                    model_cost_mean_, baseline_cost_mean_ = sess0.run(
                        fetches=[model_cost_mean, baseline_cost_mean],)
                    #    feed_dict={input_placeholder: data})
                    cost_r_m.append(model_cost_mean_)
                    cost_r_b.append(baseline_cost_mean_)
            bcm = np.array(cost_r_m).mean()
            mcm = np.array(cost_r_b).mean()
            if bcm - mcm >= .05 * bcm:
                print('Baseline updated')
                with graph0.device(device0):
                    theta = model.save(sess0)
                    baseline.load(theta, sess0)
        else:
            with graph0.device(device0):
                theta = model.save(sess0)
                baseline.load(theta, sess0)

        if epoch != 0 and (epoch % args.save_freq == 0) or epoch == args.epoch - 1:
            with graph0.device(device0):
                theta = model.save(sess0)
            with open(args.save_dir + 'epoch_%i' % epoch, 'w') as outfile:
                pickle.dump(outfile, theta)

    writer.close()
