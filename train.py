from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from graph_nets import utils_tf
from graph_nets import utils_np
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
    parser.add_argument('--nodes', default=20, help="The number of nodes per graph ex: 20, 50, 100.", required=False)
    parser.add_argument('--cuda', default=True, help="Use cuda (default True)", required=False)
    parser.add_argument('--main_gpu', default=0,
                        help="The id of the GPU to use for the main computation, -1 for CPU (default: 0).",
                        required=False)
    parser.add_argument('--baseline_gpu', default=-1,
                        help="The id of the GPU to use for baseline computation, "
                             "-1 for CPU (default: same as main_gpu).",
                        required=False)
    parser.add_argument('--save_dir', default="saved_models/experiments", help="The directory for saving the models.",
                        required=False)
    parser.add_argument('--tensorboard', default=True, help="Activate tensorboard (default: True).", required=False)
    parser.add_argument('--epoch', default=1000, help="Number of epochs (default: 1000).", required=False)
    parser.add_argument('--learning_rate', default=.0001, help="Learning rate (default: 0.0001).", required=False)
    parser.add_argument('--save_freq', default=10, help="Save model every _ epochs (default: 10).", required=False)
    parser.add_argument('--step_per_epoch', default=2500, help="Number of step per epoch (default: 2500).",
                        required=False)
    parser.add_argument('--freq_update', default=500, help="Get update every _ steps (default: 500).", required=False)
    parser.add_argument('--batch', default=8, help="Batch size (default: 512).", required=False)
    parser.add_argument('--encoder_layer', default=3, help="Number of encoder layers (default: 3).", required=False)
    parser.add_argument('--v', default=False, help="Verbose (default: False).", required=False)

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
    device_list = tf.test.gpu_device_name()
    if type(device_list) is not list:
        device_list = list(device_list)
    device_list.append("/cpu:0")
    if not args.cuda:
        device0 = "/cpu:0"
        device1 = "/cpu:0"
    else:
        baseline_gpu = args.main_gpu if args.baseline_gpu is None else args.baseline_gpu
        if baseline_gpu == -1:
            device1 = "/cpu:0"
        else:
            device1 = "/device:GPU:%i" % baseline_gpu
        if args.main_gpu == -1:
            device0 = "/cpu:0"
        else:
            device0 = "/device:GPU:%i" % args.main_gpu

    # Creating Graphs

    graph0 = tf.Graph()
    graph1 = tf.Graph()

    # Model creation

    sample = tsp_dataset.generate_networkx_batch(args.batch, args.nodes)

    with graph1.device(device1):
        sess1 = tf.Session()
        baseline = AttentionTspModel(conf)
        baseline.modify_state(False, True)
        input_placeholder1 = utils_tf.placeholders_from_networkxs(sample)
        _, _, _, baseline_cost = baseline(input_placeholder1)  # Get the cost by the baseline model

        cost_summary_b = tf.summary.scalar('Mean cost - baseline', tf.reduce_mean(baseline_cost))
        cost_summary_b_r = tf.summary.scalar('Mean cost - baseline, rollout', tf.reduce_mean(baseline_cost))

        init1 = tf.global_variables_initializer()
        init1_local = tf.local_variables_initializer()

    with graph0.device(device0):
        sess0 = tf.Session()
        # sess0 = tf_debug.TensorBoardDebugWrapperSession(sess0, "Artik:6064")
        model = AttentionTspModel(conf)
        input_placeholder0 = utils_tf.placeholders_from_networkxs(sample)
        baseline_placeholder = tf.placeholder(tf.float32)
        _, computed_log_likelihood, result_graph, cost = model(input_placeholder0)
        loss = tf.reduce_mean(tf.multiply(tf.subtract(cost, baseline_placeholder), computed_log_likelihood))
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        opt_step = optimizer.minimize(loss=loss)

        cost_summary = tf.summary.scalar('Mean cost', tf.reduce_mean(cost))
        loss_summary = tf.summary.scalar('Loss', loss)

        cost_summary_r = tf.summary.scalar('Mean cost - rollout', tf.reduce_mean(cost))

        init0 = tf.global_variables_initializer()
        init0_local = tf.local_variables_initializer()

    with graph0.device(device0):
        writer = tf.summary.FileWriter("log/", sess0.graph)

    baseline_value = np.zeros((conf.batch,))

    with graph0.device(device0):
        sess0.run(init0)
        sess0.run(init0_local)

    with graph1.device(device1):
        sess1.run(init1)
        sess1.run(init1_local)

    for epoch in trange(args.epoch, desc="Epoch"):
        for step in trange(args.step_per_epoch, desc="Step", leave=False):
            # create input
            data = utils_np.networkxs_to_graphs_tuple(tsp_dataset.generate_networkx_batch(args.batch, args.nodes))
            if epoch != 0:
                with graph1.device(device1):
                    sess1.run(init1_local)
                    baseline_value, cost_summary_b_ = sess1.run(fetches=[baseline_cost, cost_summary_b],
                                                                feed_dict={input_placeholder1: data})
            with graph0.device(device0):
                sess0.run(init0_local)
                loss_value, cost_value, _, cost_summary_, loss_summary_ = sess0.run(
                    fetches=[loss, cost, opt_step, cost_summary, loss_summary],
                    feed_dict={input_placeholder0: data, baseline_placeholder: baseline_value})

            # log
            if args.tensorboard:
                writer.add_summary(cost_summary_, step + epoch * args.step_per_epoch)
                writer.add_summary(loss_summary_, step + epoch * args.step_per_epoch)
                if epoch != 0:
                    writer.add_summary(cost_summary_b_, step + epoch * args.step_per_epoch)

        # Test baseline
        if epoch != 0:
            baseline_cost_list = []
            model_cost_list = []
            for step in trange(val_step, desc="Rollout step"):
                data = utils_np.networkxs_to_graphs_tuple(tsp_dataset.generate_networkx_batch(args.batch, args.nodes))
                with graph1.device(device1):
                    sess1.run(init1_local)
                    baseline_value, cost_summary_b_r_ = sess1.run(fetches=[baseline_cost, cost_summary_b_r],
                                                                  feed_dict={input_placeholder1: data})
                with graph0.device(device0):
                    sess0.run(init0_local)
                    model_cost, cost_summary_r_ = sess0.run(
                        fetches=[cost, cost_summary_r],
                        feed_dict={input_placeholder0: data})
                baseline_cost_list.append(baseline_value)
                model_cost_list.append(model_cost)
                if args.tensorboard:
                    writer.add_summary(cost_summary_r_, step + epoch * val_step)
                    writer.add_summary(cost_summary_b_r_, step + epoch * val_step)
            bcm = np.array(baseline_cost_list).mean()
            mcm = np.array(model_cost_list).mean()
            if bcm - mcm >= .05 * bcm:
                print('Baseline updated')
                with graph0.device(device0):
                    theta = model.save(sess0)
                with graph1.device(device1):
                    baseline.load(theta, sess1)
        else:
            with graph0.device(device0):
                theta = model.save(sess0)
            with graph1.device(device1):
                baseline.load(theta, sess1)

        if epoch != 0 and (epoch % args.save_freq == 0) or epoch == args.epoch - 1:
            with graph0.device(device0):
                theta = model.save(sess0)
            with open(args.save_dir + 'epoch_%i' % epoch, 'w') as outfile:
                pickle.dump(outfile, theta)

    writer.close()
