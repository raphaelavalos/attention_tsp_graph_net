from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
import tensorflow as tf
from graph_nets import graphs
from graph_nets import utils_tf
from tensorflow.python import debug as tf_debug
from tqdm import trange
import options
from models import AttentionTspModel

tf.set_random_seed(1234)


def get_validation_np(args, conf):
    if args.val_dataset is None:
        validation_np = np.random.random((20, args.batch * conf.n_node, conf.init_dim)).astype(np.float32)
        np.save(args.save_dir + 'val_dataset', validation_np)
    else:
        validation_np = np.load(args.val_dataset)
    return validation_np.shape[0], validation_np.reshape((-1, 2))


def get_rollout_np(args, conf):
    rollout_np = np.random.random((args.batch * args.rollout_steps * conf.n_node, conf.init_dim)).astype(np.float32)
    return rollout_np


def get_device(args):
    if not args.cuda or args.gpu == -1:
        device0 = "/cpu:0"
    else:
        device0 = "/device:GPU:%i" % args.gpu
    return device0


if __name__ == '__main__':

    args, conf = options.get_options_and_config()

    # Get or create validation dataset
    validation_step, validation_np = get_validation_np(args, conf)
    rollout_np = get_rollout_np(args, conf)

    # Device selection
    device0 = get_device(args)

    # Creating Graph
    graph0 = tf.Graph()

    # Session config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = False

    with graph0.device(device0):
        sess0 = tf.Session(config=config)
        if args.debug:
            sess0 = tf_debug.TensorBoardDebugWrapperSession(sess0, "localhost:6064")

        rolloutplaceholder = tf.placeholder(tf.float32)

        validation_dataset = tf.data.Dataset.from_tensor_slices(validation_np).repeat().batch(args.batch * conf.n_node)
        training_dataset = tf.data.Dataset.from_tensor_slices(
            tf.random_uniform([args.batch * args.step_per_epoch * conf.n_node, conf.init_dim])).batch(
            args.batch * conf.n_node)
        rollout_dataset = tf.data.Dataset.from_tensor_slices(rolloutplaceholder).batch(args.batch * conf.n_node)
        iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                                   training_dataset.output_shapes)

        training_init_op = iterator.make_initializer(training_dataset)  # creates new data
        rollout_init_op = iterator.make_initializer(rollout_dataset)  # requires placeholder rollout_placeholder
        validation_init_op = iterator.make_initializer(validation_dataset)

        nodes = iterator.get_next()
        nodes = tf.reshape(nodes, [args.batch * conf.n_node, 2])
        # Create the random nodes

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
            edges=tf.zeros((2 * conf.batch * conf.n_node,), dtype=tf.int64))

        model = AttentionTspModel(conf)
        baseline = AttentionTspModel(conf)
        baseline.modify_state(False, True)
        beta = .8
        baseline_exp_cost = tf.get_variable('exp_baseline/v', shape=(1,), dtype=tf.float32, trainable=False)

        _, computed_log_likelihood, result_graph, cost = model(fully_connected_graph)
        model_cost_mean = tf.reduce_mean(cost)
        baseline_cost = tf.stop_gradient(baseline(fully_connected_graph)[3])
        baseline_exp_cost = tf.stop_gradient(
            tf.add(tf.multiply(baseline_exp_cost, beta), (1. - beta) * model_cost_mean))
        loss = tf.reduce_mean(tf.multiply(tf.subtract(cost, baseline_cost), computed_log_likelihood), name="loss")
        loss_exp = tf.reduce_mean(tf.multiply(tf.subtract(cost, baseline_exp_cost), computed_log_likelihood),
                                  name="loss_exp")
        baseline_cost_mean = tf.reduce_mean(baseline_cost)

        baseline_rollout_mean = 0

        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 1.0)
        opt_step = optimizer.minimize(loss=loss, var_list=model.get_variables())
        opt_step_exp = optimizer.minimize(loss=loss_exp, var_list=model.get_variables())

        cost_summary = tf.summary.scalar('Mean cost', tf.reduce_mean(cost))
        loss_summary = tf.summary.scalar('Loss', loss)
        validation_placeholder = tf.placeholder(tf.float32)
        validation_summary = tf.summary.scalar('Validation', validation_placeholder)

        init0 = tf.global_variables_initializer()
        init0_local = tf.local_variables_initializer()

        writer = tf.summary.FileWriter("log/", sess0.graph)

    with graph0.device(device0):
        sess0.run([init0, init0_local])

    for epoch in trange(args.epoch, desc="Epoch"):
        with graph0.device(device0):
            sess0.run([training_init_op])
            for step in trange(args.step_per_epoch, desc="Step", leave=False):
                sess0.run(init0_local)
                if epoch == 0:
                    _, cost_summary_, loss_summary_ = sess0.run(fetches=[opt_step_exp, cost_summary, loss_summary])
                else:
                    _, cost_summary_, loss_summary_ = sess0.run(fetches=[opt_step, cost_summary, loss_summary])
                if args.tensorboard:
                    writer.add_summary(cost_summary_, step + epoch * args.step_per_epoch)
                    writer.add_summary(loss_summary_, step + epoch * args.step_per_epoch)

        # Validation
        with graph0.device(device0):
            sess0.run([validation_init_op])
            val = []
            for step in trange(validation_step, desc='Validation step'):
                sess0.run([init0_local])
                val.append(sess0.run([model_cost_mean])[0])
            val = np.array(val, dtype=np.float32)
            print('Validation: %f +/- %f' % (val.mean(), val.std()))
            writer.add_summary(sess0.run([validation_summary], {validation_placeholder: val.mean()})[0], epoch)

        # Test baseline
        cost_r_m = []
        if epoch != 0:
            with graph0.device(device0):
                sess0.run([rollout_init_op], {rolloutplaceholder: rollout_np})
                for step in trange(args.rollout_steps, desc="Rollout step"):
                    sess0.run(init0_local)
                    model_cost_mean_, = sess0.run(
                        fetches=[model_cost_mean])
                    cost_r_m.append(model_cost_mean_)
            cm = np.array(cost_r_m).mean()
            if baseline_rollout_mean - cm >= .05 * baseline_rollout_mean:
                print('Baseline updated')
                rollout_np = get_rollout_np(args, conf)
                with graph0.device(device0):
                    sess0.run([rollout_init_op], {rolloutplaceholder: rollout_np})
                    theta = model.save(sess0)
                    baseline.load(theta, sess0)
                    baseline_results = []
                    for step in trange(args.rollout_steps, desc="Rollout baseline"):
                        sess0.run(init0_local)
                        baseline_results.append(sess0.run([baseline_cost_mean])[0])
                    baseline_rollout_mean = np.mean(baseline_results)
        else:
            print('Quitting warm up')
            rollout_np = get_rollout_np(args, conf)
            print(rollout_np.shape)
            with graph0.device(device0):
                sess0.run([rollout_init_op], {rolloutplaceholder: rollout_np})
                theta = model.save(sess0)
                baseline.load(theta, sess0)
                baseline_results = []
                for step in trange(args.rollout_steps, desc="Rollout baseline"):
                    sess0.run(init0_local)
                    baseline_results.append(sess0.run([baseline_cost_mean])[0])
                baseline_rollout_mean = np.mean(baseline_results)

        if epoch != 0 and (epoch % args.save_freq == 0) or epoch == args.epoch - 1:
            with graph0.device(device0):
                theta = model.save(sess0)
            with open(args.save_dir + 'epoch_%i' % epoch, 'wb') as f:
                pickle.dump(theta, f)

    writer.close()
