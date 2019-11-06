# \MODULE\-------------------------------------------------------------------------
#
#  CONTENTS      : BumbleBee
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
import os, sys, argparse, yaml, time
import edlib, random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from util import pore_model
from tf_transformer import Transformer
from tf_transformer_util import decode_sequence
from tf_transformer_util import TransformerLRS




class BumbleBee():
    def __init__(self):
        parser = argparse.ArgumentParser(
        description='BumbleBee basecaller',
        usage='''BumbleBee.py <command> [<args>]
Available BumbleBee commands are:
train       Train BumbleBee model
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
        parser.add_argument("--input_length", type=int, default=1000, help="Input signal window")
        parser.add_argument("--target_length", type=int, default=100, help="Target sequence length")
        parser.add_argument("--minibatch_size", type=int, default=32, help="Minibatch size")
        parser.add_argument("--batches_train", type=int, default=10000, help="Training batches")
        parser.add_argument("--batches_val", type=int, default=1000, help="Validation batches")
        parser.add_argument("--gpus", nargs='+', type=int, default=[], help="GPUs to use")
        args = parser.parse_args(argv)
        # Constants
        alphabet = "ACGT"
        tf_alphabet = alphabet + '^$'

        # Model
        d_output = len(tf_alphabet)
        input_max_len = args.input_length
        target_max_len = args.target_length

        # tfRecord files
        record_files = [os.path.join(dirpath, f) for dirpath, _, files
                            in os.walk(args.records) for f in files if f.endswith('.tfrec')]
        random.shuffle(record_files)
        val_rate = args.batches_train // args.batches_val
        val_split = int(max(1, args.batches_val / args.batches_train * len(record_files)))
        val_files = record_files[:val_split]
        train_files = record_files[val_split:]

        print("Training files {}".format(len(train_files)))
        print("Test files {}".format(len(val_files)))
        
        def encode_sequence(sequence):
            ids = {char:tf_alphabet.find(char) for char in tf_alphabet}
            ret = [ids[char] if char in ids else ids[random.choice(alphabet)] for char in '^' + sequence.numpy().decode('utf-8') + '$']
            return tf.cast(ret, tf.int32)

        def tf_parse(eg):
            example = tf.io.parse_example(
                eg[tf.newaxis], {
                    'sequence': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
                    'signal': tf.io.FixedLenFeature(shape=(), dtype=tf.string)})
            seq = tf.py_function(encode_sequence, [example['sequence'][0]], tf.int32)
            sig = tf.expand_dims(
                    tf.cast(
                        tf.io.parse_tensor(example['signal'][0], tf.float16),
                        tf.float32),
                    axis=-1)
            seq_len = tf.expand_dims(tf.size(seq), axis=-1) - 1
            sig_len = tf.expand_dims(tf.size(sig), axis=-1)
            return ((sig, seq[:-1], sig_len, seq_len), seq[1:])

        def tf_filter(input, target):
            #input, target = eg
            return (input[2] <= tf.cast(input_max_len, tf.int32) and input[3] <= tf.cast(target_max_len + 2, tf.int32))[0]


        ds_train = tf.data.Dataset.from_tensor_slices(train_files).shuffle(len(train_files))
        ds_train = (ds_train.interleave(lambda x:
                    tf.data.TFRecordDataset(filenames=x).map(tf_parse, num_parallel_calls=1), cycle_length=8, block_length=8))
        ds_train = (ds_train
                    .filter(tf_filter)
                    .prefetch(args.minibatch_size * 64)
                    .shuffle(args.minibatch_size * 1024)
                    .padded_batch(args.minibatch_size,
                        padded_shapes=(([input_max_len, 1], [target_max_len+2,], [1,], [1,]), [target_max_len+2,]),
                        drop_remainder=True)
                    .repeat())

        ds_test = tf.data.Dataset.from_tensor_slices(val_files)
        ds_test = (ds_test.interleave(lambda x:
                    tf.data.TFRecordDataset(filenames=x).map(tf_parse, num_parallel_calls=1), cycle_length=8, block_length=8))
        ds_test = (ds_test
                    .filter(tf_filter)
                    .prefetch(args.minibatch_size * 32)
                    .padded_batch(args.minibatch_size,
                        padded_shapes=(([input_max_len, 1], [target_max_len+2,], [1,], [1,]), [target_max_len+2,]),
                        drop_remainder=True)
                    .repeat())

        transformer_hparams_file = (args.config or os.path.join('./training_configs', args.prefix, 'hparams.yaml'))
        if os.path.exists(transformer_hparams_file):
            with open(transformer_hparams_file, 'r') as fp:
                transformer_hparams = yaml.safe_load(fp)
        else:
            transformer_hparams = {'d_output' : d_output,
                               'd_model' : 384,
                               'cnn_kernel' : 20,
                               'dff' : 1536,
                               #'dff_type' : 'point_wise',
                               'encoder_dff_type' : 'separable_convolution',
                               'decoder_dff_type' : 'point_wise',
                               'encoder_dff_filter_width' : 24,
                               #'decoder_dff_filter_width' : 20,
                               'num_heads' : 8,
                               'encoder_max_iterations' : 14,
                               'decoder_max_iterations' : 14,
                               'encoder_time_scale' : 10000,
                               'decoder_time_scale' : 1000,
                               'random_shift' : False,
                               'ponder_bias_init' : 1.0,
                               'encoder_act_type' : 'dense',
                               'decoder_act_type' : 'dense',
                               'encoder_time_penalty' : 0.0005,
                               'decoder_time_penalty' : 0.005,
                               'input_memory_comp' : 16,
                               'target_memory_comp' : None
                               }
            os.makedirs(os.path.dirname(transformer_hparams_file), exist_ok=True)
            with open(transformer_hparams_file, 'w') as fp:
                print(yaml.dump(transformer_hparams), file=fp)

        strategy = tf.distribute.MirroredStrategy(devices=['/gpu:' + str(i) for i in args.gpus] if args.gpus else ['/cpu:0'])

        ds_train_dist = strategy.experimental_distribute_dataset(ds_train)
        ds_test_dist = strategy.experimental_distribute_dataset(ds_test)

        checkpoint_dir = os.path.join('./training_checkpoints', args.prefix)
        os.makedirs(checkpoint_dir, exist_ok=True)
        summary_dir = os.path.join('./training_summaries', args.prefix)
        os.makedirs(summary_dir, exist_ok=True)
        summary_writer = tf.summary.create_file_writer(summary_dir)

        with strategy.scope(), summary_writer.as_default():
            learning_rate = TransformerLRS(transformer_hparams.get('dff'), warmup_steps=8000)
            optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9, amsgrad=False)
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
            def loss_function(real, pred, target_lengths):
                loss_ = loss_object(real, pred)
                mask = tf.sequence_mask(target_lengths, target_max_len+2, dtype=loss_.dtype)
                loss_ = (loss_ * mask) # / tf.cast(target_lengths, dtype=loss_.dtype)
                nrm_loss = loss_ / tf.cast(target_lengths, dtype=loss_.dtype)
                return (tf.nn.compute_average_loss(loss_, global_batch_size=args.minibatch_size),
                        tf.nn.compute_average_loss(nrm_loss, global_batch_size=args.minibatch_size))
            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
            prediction_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='prediction_accuracy')

            transformer = Transformer(hparams=transformer_hparams)

            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=transformer)
            ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
            if ckpt_manager.latest_checkpoint:
                checkpoint.restore(ckpt_manager.latest_checkpoint)
                print('Latest checkpoint restored!!')

            def train_step(inputs):
                input, target = inputs
                target_lengths = input[3]
                mask = tf.squeeze(tf.sequence_mask(target_lengths, target_max_len+2, dtype=tf.float32))
                with tf.GradientTape() as tape:
                    predictions = transformer(input, training=True)
                    loss, nrm_loss = loss_function(target, predictions, target_lengths)
                gradients = tape.gradient([loss] + transformer.losses, transformer.trainable_variables)
                optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
                train_accuracy.update_state(target, predictions, mask)
                return nrm_loss

            def test_step(inputs):
                input, target = inputs
                target_lengths = input[3]
                mask = tf.squeeze(tf.sequence_mask(target_lengths, target_max_len+2, dtype=tf.float32))
                predictions = transformer(input, training=False)
                t_loss, nrm_loss = loss_function(target, predictions, target_lengths)
                test_accuracy.update_state(target, predictions, mask)
                return nrm_loss

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
                ds_train_dist_iter = iter(ds_train_dist)
                ds_test_dist_iter = iter(ds_test_dist)
                for batch in tqdm(range(args.batches_train), desc='Training', ncols=0):
                    tf.summary.experimental.set_step(optimizer.iterations)
                    batch_input = next(ds_train_dist_iter)
                    batch_loss = distributed_train_step(batch_input)
                    if batch_loss < max(best_losses):
                        ckpt_manager.save()
                        best_losses[best_losses.index(max(best_losses))] = batch_loss
                    if epoch == 0 and batch == 0:
                        transformer.summary()
                    num_batches += 1
                    total_loss += batch_loss
                    train_loss = total_loss / num_batches
                    tf.summary.scalar("loss", batch_loss)
                    tf.summary.scalar("lr", learning_rate(optimizer.iterations.numpy().astype(np.float32)))
                    if batch % val_rate == 0:
                        batch_input = next(ds_test_dist_iter)
                        test_loss = distributed_test_step(batch_input)
                        tf.summary.scalar("val_loss", test_loss)
                        tf.summary.scalar('train_accuracy', train_accuracy.result())
                        tf.summary.scalar("val_accuracy", test_accuracy.result())
                        train_accuracy.reset_states()
                        test_accuracy.reset_states()
                print("Epoch {}: train loss: {}".format(epoch, train_loss))

    def predict(self, argv):
        parser = argparse.ArgumentParser(description="BumbleBee basecaller prediction")
        parser.add_argument("config", help="BumbleBee config")
        parser.add_argument("checkpoint", help="Training checkpoint")
        parser.add_argument("model", help="Pore model")
        parser.add_argument("fast5", help="Raw signal fast5 file")
        parser.add_argument("--max_signal_length", type=int, default=1200, help="Signal window size")
        parser.add_argument("--minibatch_size", type=int, default=16, help="Batch size")
        parser.add_argument("-t", "--threads", type=int, default=16, help="Threads")
        args = parser.parse_args(argv)
        tf.config.threading.set_inter_op_parallelism_threads(args.threads // 4)
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        # test sim_signal
        batch_gen = BatchGeneratorSig(args.model, args.fast5, max_input_len=args.max_signal_length,
                                        min_target_len=50, max_target_len=100,
                                        batches_train=1, batches_val=0,
                                        minibatch_size=args.minibatch_size)
        batch_gen_train = batch_gen.next_train()
        input_data, target_data, input_lengths, target_lengths = next(batch_gen_train)
        # load and init with dummy data
        with open(args.config, 'r') as fp:
            transformer_hparams = yaml.safe_load(fp)
        transformer = Transformer(hparams=transformer_hparams)
        _ = transformer([input_data, target_data[:,:-1], input_lengths, target_lengths])
        # load weights into initialized model
        transformer.load_weights(args.checkpoint)


        predictions = transformer([input_data, target_data[:,:-1], input_lengths, target_lengths], training=False)
        for target, target_length, prediction, prediction_length in zip(target_data, target_lengths, predictions, target_lengths):
            logits = tf.argmax(prediction, axis=-1)
            print('@seq')
            print(decode_sequence(target[:target_length[0]]))
            print(decode_sequence(logits[:prediction_length[0]]))




if __name__ == '__main__':
    tf.config.optimizer.set_jit(True)
    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for d in physical_devices:
        tf.config.experimental.set_memory_growth(d, True)
    BumbleBee()
