from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from graph_nets import utils_tf
from graph_nets import utils_np
from graph_nets import graphs
from graph_nets import blocks
from graph_nets import modules
import sonnet as snt
import attention_tsp
import argparse
import os
import tsp_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Attention TSP Train Module")
    parser.add_argument('--nodes', default=50, help="The number of nodes per graph ex: 20, 50, 100.", required=False)
    parser.add_argument('--gpu', default=0, help="The id of the GPU to use. "
                                                 "Doesn't support multiple GPUs yet, -1 for CPU only (default: 0).", required=False)
    parser.add_argument('--save_dir', default="saved_models/experiment", help="The directory for saving the models.", required=False)
    parser.add_argument('--tensorboard', default=True, help="Activate tensorboard (default: True).", required=False)
    parser.add_argument('--epoch', default=100, help="Number of epochs (default: 1000).", required=False)
    parser.add_argument('--learning_rate', default=0.0001, help="Learning rate (default: 0.0001).", required=False)
    parser.add_argument('--save_freq', default=10, help="Save model every _ epochs (default: 10).", required=False)
    parser.add_argument('--step_per_epoch', default=2500, help="Number of step per epoch (default: 2500).", required=False)
    parser.add_argument('--freq_update', default=500, help="Get update every _ steps (default: 500).", required=False)
    parser.add_argument('--batch', default=64, help="Batch size (default: 512).", required=False)
    parser.add_argument('--encoder_layer', default=3, help="Number of encoder layers (default: 3).", required=False)
    parser.add_argument('--v', default=False, help="Verbose (default: False).", required=False)

    args = parser.parse_args()
    print(args)

    os.makedirs(args.save_dir, exist_ok=True)

    ## Creating Graphs and Sessions

    graph0 = tf.Graph()
    graph1 = tf.Graph()
    #sess0 = tf.Session(graph=graph0)
    #sess1 = tf.Session(graph=graph1)


    ## Model creation

    sample = tsp_dataset.generate_networkx_batch(args.batch, args.nodes)

    with graph1.as_default():
        model1 = attention_tsp.TSP_Model(args)
        input_placeholder1 = utils_tf.placeholders_from_networkxs(sample)
        baseline = model1.forward(input_placeholder1)[0].globals  # Get the cost by the baseline model
        init1 = tf.initialize_all_variables()
        sess1 = tf.Session()

    with graph0.as_default():
        model0 = attention_tsp.TSP_Model(args)
        input_placeholder0 = utils_tf.placeholders_from_networkxs(sample)
        baseline_placeholder = tf.placeholder(tf.float32)
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        loss, grad_loss = model0.compute_loss(input_placeholder0, baseline_placeholder)
        #opt_step = optimizer.minimize(loss=loss, grad_loss=grad_loss)
        init0 = tf.initialize_all_variables()
        sess0 = tf.Session()



    baseline_value = None

    writer = tf.summary.FileWriter("log/", graph0)
    with graph0.as_default():
        sess0.run(init0)
    with graph1.as_default():
        sess1.run(init1)

    for epoch in range(args.epoch):
        print('Epoch %i' % epoch)
        for step in range(args.step_per_epoch):
            # create input
            data = utils_np.networkxs_to_graphs_tuple(tsp_dataset.generate_networkx_batch(args.batch, args.nodes))
            if epoch != 0:
                baseline_value = sess1.run(fetches=[baseline], feed_dict={input_placeholder1: data})
            loss_value, grad_loss_value = sess0.run(fetches=[loss, grad_loss],
                                                                    feed_dict={input_placeholder0: data,
                                                                               baseline_placeholder: baseline_value})
            print(grad_loss_value)
    writer.close()
