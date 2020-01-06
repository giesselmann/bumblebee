# \MODULE\-------------------------------------------------------------------------
#
#  CONTENTS      : OptimusPrime
#
#  DESCRIPTION   : Nanopore Basecalling
#
#  RESTRICTIONS  : none
#
#  REQUIRES      : none
#
# ---------------------------------------------------------------------------------
# Copyright 2019 Pay Giesselmann, Max Planck Institute for Molecular Genetics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Pay Giesselmann
# ---------------------------------------------------------------------------------
import os, sys, argparse
import yaml
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tf_data import tf_data_basecalling
from tf_util import WarmupLRS
from tf_models import OptimusPrime




class Optimus():
    def __init__(self):
        parser = argparse.ArgumentParser(
        description='OptimusPrime basecaller',
        usage='''Optimus.py <command> [<args>]
Available OptimusPrime commands are:
train       Train OptimusPrime model
predict     Predict sequence from raw fast5
    ''')
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command', file=sys.stderr)
            parser.print_help(file=sys.stderr)
            exit(1)
        getattr(self, args.command)(sys.argv[2:])

    def train(self, argv):
        parser = argparse.ArgumentParser(description="BumbleBee basecaller training")
        parser.add_argument("records", help="Training records")
        parser.add_argument("--config", default=None, help="Transformer config file")
        parser.add_argument("--prefix", default="", help="Checkpoint and event prefix")
        parser.add_argument("--input_min_len", type=int, default=500, help="Input signal minimum length")
        parser.add_argument("--input_max_len", type=int, default=1000, help="Input signal maximum length")
        parser.add_argument("--target_min_len", type=int, default=50, help="Target sequence max length")
        parser.add_argument("--target_max_len", type=int, default=100, help="Target sequence max length")
        parser.add_argument("--minibatch_size", type=int, default=32, help="Minibatch size")
        parser.add_argument("--batches_train", type=int, default=10000, help="Training batches")
        parser.add_argument("--batches_val", type=int, default=1000, help="Validation batches")
        parser.add_argument("--gpus", nargs='+', type=int, default=[], help="GPUs to use")
        args = parser.parse_args(argv)

        #tf.config.experimental_run_functions_eagerly(True)

        # Constants
        alphabet = "ACGT"
        d_output = len(alphabet) + 1    # alphabet + blank label

        # tfRecord files
        record_files = [os.path.join(dirpath, f) for dirpath, _, files
                            in os.walk(args.records) for f in files if f.endswith('.tfrec')]
        random.shuffle(record_files)
        val_rate = args.batches_train // args.batches_val
        val_split = int(max(1, args.batches_val / args.batches_train * len(record_files)))
        test_files = record_files[:val_split]
        train_files = record_files[val_split:]
        print("Training files {}".format(len(train_files)))
        print("Test files {}".format(len(test_files)))

        hparams_file = (args.config or os.path.join('./training_configs', args.prefix, 'hparams.yaml'))
        if os.path.exists(hparams_file):
            with open(hparams_file, 'r') as fp:
                hparams = yaml.safe_load(fp)
        else:
            hparams = {
                        'd_model' : 512,
                        'd_output' : d_output,
                        'cnn_kernel' : 32,
                        'cnn_pool_size' : 6,
                        'cnn_pool_stride' : 5,
                        'dff' : 2048,
                        'nff' : 2,
                        'ff_filter' : 32,
                        'ff_pool_size' : 3,
                        'act_type' : 'point_wise',
                        'act_dff' : 32,
                        'ponder_bias_init' : 2.0,
                        'max_iterations' : 6,
                        'time_penalty' : 0.01
                      }
            os.makedirs(os.path.dirname(hparams_file), exist_ok=True)
            with open(hparams_file, 'w') as fp:
                print(yaml.dump(hparams), file=fp)

        tf_data = tf_data_basecalling(alphabet=alphabet, use_sos=False, use_eos=False,
                            input_min_len=args.input_min_len, input_max_len=args.input_max_len,
                            target_min_len=args.target_min_len, target_max_len=args.target_max_len)
        tf_data_train = tf_data.get_ds_train(train_files, minibatch_size=args.minibatch_size)
        tf_data_test = tf_data.get_ds_test(test_files, minibatch_size=args.minibatch_size)

        strategy = tf.distribute.MirroredStrategy(devices=['/gpu:' + str(i) for i in args.gpus] if args.gpus else ['/cpu:0'])

        tf_data_train_dist = strategy.experimental_distribute_dataset(tf_data_train)
        tf_data_test_dist = strategy.experimental_distribute_dataset(tf_data_test)

        checkpoint_dir = os.path.join('./training_checkpoints', args.prefix)
        os.makedirs(checkpoint_dir, exist_ok=True)
        summary_dir = os.path.join('./training_summaries', args.prefix)
        os.makedirs(summary_dir, exist_ok=True)
        summary_writer = tf.summary.create_file_writer(summary_dir)

        with strategy.scope(), summary_writer.as_default():
            lrs = WarmupLRS(hparams.get('dff'), warmup_steps=1)
            #optimizer = tf.keras.optimizers.Adam(lrs, beta_1=0.9, beta_2=0.98, epsilon=1e-9, amsgrad=False)
            #optimizer = tf.keras.optimizers.SGD(lrs, momentum=0.9, clipnorm=2.0, nesterov=True)
            optimizer = tf.keras.optimizers.RMSprop(0.0001, momentum=0.9, clipnorm=2.0)
            optimus = OptimusPrime(hparams=hparams, name='OptimusPrime')
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=optimus)
            ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
            if ckpt_manager.latest_checkpoint:
                checkpoint.restore(ckpt_manager.latest_checkpoint)
                print('Latest checkpoint restored!!')

            edit_train = tf.keras.metrics.Mean(name='edit distance training')
            edit_test = tf.keras.metrics.Mean(name='edit distance test')

            def train_step(inputs):
                input, target = inputs
                input_seq, input_len = input
                target_seq, target_len = target
                with tf.GradientTape() as tape:
                    predictions, pred_len, act_loss = optimus(input, training=True)
                    loss_ = tf.nn.ctc_loss(target_seq, predictions, tf.squeeze(target_len), pred_len,
                                           logits_time_major=False, blank_index=-1)
                loss = loss_ / tf.cast(pred_len, loss_.dtype) + act_loss
                gradients = tape.gradient([loss_, act_loss], optimus.trainable_variables)
                optimizer.apply_gradients(zip(gradients, optimus.trainable_variables))
                #decoded, _ = tf.nn.ctc_greedy_decoder(tf.transpose(predictions, [1,0,2]), pred_len)
                #target = tf.sparse.from_dense((target_seq + 1) * tf.sequence_mask(tf.squeeze(target_len), args.target_max_len, dtype=target_seq.dtype))
                #target = tf.sparse.SparseTensor(target.indices, target.values - 1, target.dense_shape)
                #dist = tf.edit_distance(tf.cast(decoded[0], target_seq.dtype), target, normalize=True)
                #edit_train.update_state(dist)
                return tf.nn.compute_average_loss(loss, global_batch_size=args.minibatch_size)

            def test_step(inputs):
                input, target = inputs
                input_seq, input_len = input
                target_seq, target_len = target
                predictions, pred_len, act_loss = optimus(input, training=False)
                loss_ = tf.nn.ctc_loss(target_seq, predictions, tf.squeeze(target_len), pred_len,
                                       logits_time_major=False, blank_index=-1)
                loss = loss_ / tf.cast(pred_len, loss_.dtype) + act_loss
                #decoded, _ = tf.nn.ctc_greedy_decoder(tf.transpose(predictions, [1,0,2]), pred_len)
                #target = tf.sparse.from_dense((target_seq + 1) * tf.sequence_mask(tf.squeeze(target_len), args.target_max_len, dtype=target_seq.dtype))
                #target = tf.sparse.SparseTensor(target.indices, target.values - 1, target.dense_shape)
                #dist = tf.edit_distance(tf.cast(decoded[0], target_seq.dtype), target, normalize=True)
                #edit_test.update_state(dist)
                return tf.nn.compute_average_loss(loss, global_batch_size=args.minibatch_size)

            # `experimental_run_v2` replicates the provided computation and runs it
            # with the distributed input.
            @tf.function
            def distributed_train_step(dataset_inputs):
                per_replica_losses = strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
                return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

            @tf.function
            def distributed_test_step(dataset_inputs):
                per_replica_losses = strategy.experimental_run_v2(test_step, args=(dataset_inputs,))
                return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

            best_losses = [float("inf")] * 5
            for epoch in range(20):
                total_loss = 0.0
                num_batches = 0
                tf_data_train_dist_iter = iter(tf_data_train_dist)
                tf_data_test_dist_iter = iter(tf_data_test_dist)
                for batch in tqdm(range(args.batches_train), desc='Training', ncols=0):
                    tf.summary.experimental.set_step(optimizer.iterations)
                    batch_input = next(tf_data_train_dist_iter)
                    loss = distributed_train_step(batch_input)
                    if epoch == 0 and batch == 0:
                        optimus.summary()
                    if loss <= max(best_losses):
                        ckpt_manager.save()
                        best_losses[best_losses.index(max(best_losses))] = loss

                    num_batches += 1
                    total_loss += loss
                    train_loss = total_loss / num_batches
                    tf.summary.scalar("loss", loss)
                    tf.summary.scalar("lr", lrs(optimizer.iterations.numpy().astype(np.float32)))
                    tf.summary.scalar("edit distance training", edit_train.result())
                    if batch % val_rate == 0:
                        batch_input = next(tf_data_test_dist_iter)
                        test_loss = distributed_test_step(batch_input)
                        tf.summary.scalar("loss_val", test_loss)
                        tf.summary.scalar("edit distance test", edit_test.result())
                    edit_train.reset_states()
                    edit_test.reset_states()
            print("Epoch {}: train loss: {}".format(epoch, train_loss))




if __name__ == '__main__':
    tf.config.optimizer.set_jit(True)
    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for d in physical_devices:
        tf.config.experimental.set_memory_growth(d, True)
    Optimus()
