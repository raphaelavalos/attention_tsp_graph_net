from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from graph_nets import utils_tf
from graph_nets import utils_np
import attention_tsp
import argparse
import os
import pprint
import tsp_dataset
from collections import namedtuple
from tensorflow.python import debug as tf_debug

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Attention TSP Train Module")
    parser.add_argument('--nodes', default=20, help="The number of nodes per graph ex: 20, 50, 100.", required=False)
    parser.add_argument('--gpu', default=0, help="The id of the GPU to use. "
                                                 "Doesn't support multiple GPUs yet, -1 for CPU only (default: 0).",
                        required=False)
    parser.add_argument('--save_dir', default="saved_models/experiment", help="The directory for saving the models.",
                        required=False)
    parser.add_argument('--tensorboard', default=True, help="Activate tensorboard (default: True).", required=False)
    parser.add_argument('--epoch', default=1, help="Number of epochs (default: 1000).", required=False)
    parser.add_argument('--learning_rate', default=10000, help="Learning rate (default: 0.0001).", required=False)
    parser.add_argument('--save_freq', default=10, help="Save model every _ epochs (default: 10).", required=False)
    parser.add_argument('--step_per_epoch', default=1, help="Number of step per epoch (default: 2500).",
                        required=False)
    parser.add_argument('--freq_update', default=500, help="Get update every _ steps (default: 500).", required=False)
    parser.add_argument('--batch', default=8, help="Batch size (default: 512).", required=False)
    parser.add_argument('--encoder_layer', default=3, help="Number of encoder layers (default: 3).", required=False)
    parser.add_argument('--v', default=False, help="Verbose (default: False).", required=False)

    args = parser.parse_args()
    print(pprint.pprint(args))

    os.makedirs(args.save_dir, exist_ok=True)

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
    # Creating Graphs

    graph0 = tf.Graph()
    #graph1 = tf.Graph()
    #sess0 = tf.Session(graph=graph0)

    # Model creation

    sample = tsp_dataset.generate_networkx_batch(args.batch, args.nodes)

    # with graph1.as_default():
    #     model1 = attention_tsp.TSP_Model(args, conf)
    #     input_placeholder1 = utils_tf.placeholders_from_networkxs(sample)
    #     baseline = model1.forward(input_placeholder1)[0].globals  # Get the cost by the baseline model
    #     init1 = tf.global_variables_initializer()
    #     init1_local = tf.local_variables_initializer()
    #     sess1 = tf.Session()

    with graph0.device("/device:GPU:0"):
        sess0 = tf.Session()
        #sess0 = tf_debug.TensorBoardDebugWrapperSession(sess0, "Artik:6064")
        trainer = attention_tsp.TSP_Trainer(args, conf)
        input_placeholder0 = utils_tf.placeholders_from_networkxs(sample)
        # baseline_placeholder = tf.placeholder(tf.float32)
        result_graph, cost, computed_log_likelihood = trainer.forward(input_placeholder0)
        loss = trainer.compute_loss(cost, computed_log_likelihood)
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        #grad = optimizer.compute_gradients(loss)
        #opt_step = optimizer.apply_gradients(grad)
        opt_step = optimizer.compute_gradients(loss=loss)
        init0 = tf.global_variables_initializer()
        init0_local = tf.local_variables_initializer()
        writer = tf.summary.FileWriter("log/")

    baseline_value = None

    # writer = tf.summary.FileWriter("log/", graph0)
    with graph0.device("/device:GPU:0"):
        sess0.run(init0)
        #sess0.run(init0_local)
    # with graph1.as_default():
    #     sess1.run(init1)

    with graph0.device("/device:GPU:0"):
        dic = trainer.save_model(sess0)

    loss_list = []
    for epoch in range(args.epoch):
        print('Epoch %i' % epoch)
        loss_epoch = []
        for step in range(args.step_per_epoch):
            #print('step %i' % step)
            # create input
            data = utils_np.networkxs_to_graphs_tuple(tsp_dataset.generate_networkx_batch(args.batch, args.nodes))
            # if epoch != 0:
            #     with graph1.as_default():
            #         sess1.run(init1_local)
            #         baseline_value = sess1.run(fetches=[baseline], feed_dict={input_placeholder1: data})
            with graph0.device("/device:GPU:0"):
                sess0.run(init0_local)
                loss_value, opt_step_value = sess0.run(
                    fetches=[loss, opt_step, ],
                    feed_dict={input_placeholder0: data})

                #print(trainer.model._decoder.get_variables()[7].eval(sess0))
            loss_epoch.append(loss_value)
        loss_list.append(loss_epoch)

    writer.close()

    with graph0.device("/device:GPU:0"):
        dic2 = trainer.save_model(sess0)
