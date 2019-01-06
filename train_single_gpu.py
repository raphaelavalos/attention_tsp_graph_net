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
        validation_np = np.random.random((5, args.batch * conf.n_node, conf.init_dim)).astype(np.float32)
        np.save(args.save_dir + 'val_dataset', validation_np)
    else:
        validation_np = np.load(args.val_dataset)
    return validation_np.shape[0], validation_np.reshape((-1, 2))


def get_rollout_np(args, conf):
    rollout_np = np.random.random((args.batch * args.rollout_steps * conf.n_node, conf.init_dim)).astype(np.float32)
    return rollout_np


def get_device(args):
    if not args.cuda or args.gpu == -1:
        device = "/cpu:0"
    else:
        device = "/device:GPU:%i" % args.gpu
    return device


def compute_baseline_rollout_mean(baseline, rollout_np, sess, model):
    sess.run([rollout_init_op], {rolloutplaceholder: rollout_np})
    theta = model.save(sess)
    baseline.load(theta, sess)
    baseline_results = []
    for _ in trange(args.rollout_steps, desc="Rollout baseline"):
        baseline_results.append(sess.run([baseline_cost_mean])[0])
    baseline_rollout_mean = np.mean(baseline_results)
    return baseline, baseline_rollout_mean


if __name__ == '__main__':

    args, conf, experiment_name = options.get_options_and_config()

    rollout_enable = True

    # Get or create validation dataset
    validation_step, validation_np = get_validation_np(args, conf)
    rollout_np = get_rollout_np(args, conf)

    # Device selection
    device = get_device(args)

    # Creating Graph
    graph = tf.Graph()

    # Session config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = False

    with graph.device(device):
        sess = tf.Session(config=config)
        if args.debug:
            sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:6064")

        with tf.name_scope('dataset_iterator'):
            validation_dataset = tf.data.Dataset.from_tensor_slices(validation_np).repeat().batch(
                args.batch * conf.n_node)
            training_dataset = tf.data.Dataset.from_tensor_slices(
                tf.random_uniform([args.batch * args.step_per_epoch * conf.n_node, conf.init_dim])).batch(
                args.batch * conf.n_node)
            iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
            training_init_op = iterator.make_initializer(training_dataset)  # creates new data
            validation_init_op = iterator.make_initializer(validation_dataset)
            if rollout_enable:
                with tf.name_scope('baseline'):
                    rolloutplaceholder = tf.placeholder(tf.float32)
                    rollout_dataset = tf.data.Dataset.from_tensor_slices(rolloutplaceholder).batch(
                        args.batch * conf.n_node)
                    rollout_init_op = iterator.make_initializer(
                        rollout_dataset)  # requires placeholder rollout_placeholder

        with tf.name_scope('graph_creator'):
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
        computed_log_likelihood, cost = model(fully_connected_graph)
        model_cost_mean = tf.reduce_mean(cost)

        if rollout_enable:
            with tf.name_scope('baseline'):
                baseline = AttentionTspModel(conf, name='baseline')
                # baseline.modify_state(False, True)
                beta = .8
                baseline_exp_cost = tf.get_variable('exp_baseline/v', shape=(1,), dtype=tf.float32, trainable=False)
                baseline_exp_cost = tf.stop_gradient(tf.add(beta * baseline_exp_cost, (1. - beta) * model_cost_mean))
                baseline_cost = tf.stop_gradient(baseline(fully_connected_graph)[1])
                baseline_cost_mean = tf.reduce_mean(baseline_cost)

            baseline_rollout_mean = 0
            with tf.name_scope('loss'):
                loss = tf.reduce_mean((cost - baseline_cost) * computed_log_likelihood, name="loss")
                loss_exp = tf.reduce_mean((cost - baseline_exp_cost) * computed_log_likelihood, name="loss_exp")
        else:
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(cost * computed_log_likelihood, name="loss")

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 1.0)
            opt_step = optimizer.minimize(loss=loss, var_list=model.get_variables())
            if rollout_enable:
                opt_step_exp = optimizer.minimize(loss=loss_exp, var_list=model.get_variables())
        with tf.name_scope('summary'):
            if rollout_enable:
                loss_summary_exp = tf.summary.scalar('Loss', loss_exp)
            cost_summary = tf.summary.scalar('Cost', tf.reduce_mean(cost))
            loss_summary = tf.summary.scalar('Loss', loss)

            validation_placeholder = tf.placeholder(tf.float32)
            validation_summary = tf.summary.scalar('Validation', validation_placeholder)

        init0 = tf.global_variables_initializer()

        writer = tf.summary.FileWriter("log/" + experiment_name, sess.graph)

    with graph.device(device):
        sess.run([init0])

    for epoch in trange(args.epoch, desc="Epoch"):
        with graph.device(device):
            sess.run([training_init_op])
            for step in trange(args.step_per_epoch, desc="Step", leave=False):
                # sess.run(init0_local)
                if epoch == 0 and rollout_enable:
                    _, cost_summary_, loss_summary_ = sess.run(fetches=[opt_step_exp, cost_summary, loss_summary_exp])
                else:
                    _, cost_summary_, loss_summary_ = sess.run(fetches=[opt_step, cost_summary, loss_summary])
                if args.tensorboard:
                    writer.add_summary(cost_summary_, step + epoch * args.step_per_epoch)
                    writer.add_summary(loss_summary_, step + epoch * args.step_per_epoch)

        # Validation
        with graph.device(device):
            sess.run([validation_init_op])
            val = []
            for step in trange(validation_step, desc='Validation step'):
                val.append(sess.run([model_cost_mean])[0])
            val = np.array(val, dtype=np.float32)
            print('Validation: %f' % val.mean())
            writer.add_summary(sess.run([validation_summary], {validation_placeholder: val.mean()})[0], epoch)

        # Test baseline
        if rollout_enable:
            cost_r_m = []
            if epoch != 0:
                with graph.device(device):
                    sess.run([rollout_init_op], {rolloutplaceholder: rollout_np})
                    for step in trange(args.rollout_steps, desc="Rollout step"):
                        model_cost_mean_, = sess.run(fetches=[model_cost_mean])
                        cost_r_m.append(model_cost_mean_)
                cm = np.array(cost_r_m).mean()
                if (1 - .05) * baseline_rollout_mean >= cm:
                    print('Baseline updated')
                    rollout_np = get_rollout_np(args, conf)
                    with graph.device(device):
                        baseline, baseline_rollout_mean = compute_baseline_rollout_mean(baseline, rollout_np, sess,
                                                                                        model)
            else:
                print('Quitting warm up')
                rollout_np = get_rollout_np(args, conf)
                with graph.device(device):
                    baseline, baseline_rollout_mean = compute_baseline_rollout_mean(baseline, rollout_np, sess, model)

        if epoch != 0 and (epoch % args.save_freq == 0) or epoch == args.epoch - 1:
            with graph.device(device):
                theta = model.save(sess)
            with open(args.save_dir + 'epoch_%i' % epoch, 'wb') as f:
                pickle.dump(theta, f)

    writer.close()
