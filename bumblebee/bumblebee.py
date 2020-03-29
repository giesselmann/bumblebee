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
import timeit
import edlib, random, re
import numpy as np
import tensorflow as tf
from scipy.signal import medfilt
from tqdm import tqdm
from util import pore_model
from tf_transformer import Transformer
from tf_discriminator import Discriminator
from tf_util import decode_sequence
from tf_util import WarmupLRS




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
        parser.add_argument("records", nargs='+', help="Training records")
        parser.add_argument("--config", default=None, help="Transformer config file")
        parser.add_argument("--prefix", default="", help="Checkpoint and event prefix")
        parser.add_argument("--input_length", type=int, default=1000, help="Input signal window")
        parser.add_argument("--target_min_length", type=int, default=100, help="Target sequence max length")
        parser.add_argument("--target_length", type=int, default=100, help="Target sequence max length")
        parser.add_argument("--minibatch_size", type=int, default=32, help="Minibatch size")
        parser.add_argument("--batches_train", type=int, default=10000, help="Training batches")
        parser.add_argument("--batches_val", type=int, default=1000, help="Validation batches")
        parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
        parser.add_argument("--gpus", nargs='+', type=int, default=[], help="GPUs to use")
        parser.add_argument("--policy", default='float32', choices=['float32', 'float16', 'mixed_float16'], help='Training data type policy')
        args = parser.parse_args(argv)

        #tf.config.experimental_run_functions_eagerly(True)
        policy = tf.keras.mixed_precision.experimental.Policy(args.policy)
        tf.keras.mixed_precision.experimental.set_policy(policy)

        # Constants
        alphabet = "ACGT"
        tf_alphabet = alphabet + '^$'

        # Model
        d_output = len(tf_alphabet)
        input_max_len = args.input_length
        target_max_len = args.target_length
        target_min_len = args.target_min_length

        # tfRecord files
        record_files = [os.path.join(dirpath, f)
            for record in args.records
                for dirpath, _, files in os.walk(record)
                    for f in files if f.endswith('.tfrec')]
        random.shuffle(record_files)
        val_rate = args.batches_train // args.batches_val
        val_split = int(max(1, args.batches_val / args.batches_train * len(record_files)))
        test_files = record_files[:val_split]
        train_files = record_files[val_split:]

        print("Training files {}".format(len(train_files)))
        print("Test files {}".format(len(test_files)))

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
                        tf.cast(tf.io.parse_tensor(example['signal'][0], tf.float16), tf.float16),
                        axis=-1)
            seq_len = tf.cast(tf.expand_dims(tf.size(seq), axis=-1) - 1, tf.int32)
            sig_len = tf.cast(tf.expand_dims(tf.size(sig), axis=-1), tf.int32)
            return ((sig, seq[:-1], sig_len, seq_len), seq[1:])

        def tf_filter(input, target):
            #input, target = eg
            return (input[2] <= tf.cast(input_max_len, tf.int32) and
                    input[2] >= tf.cast(target_min_len, tf.int32) and
                    input[3] <= tf.cast(target_max_len, tf.int32) and
                    input[3] >= tf.cast(target_min_len + 2, tf.int32))[0]

        ds_train = tf.data.Dataset.from_tensor_slices(train_files)
        ds_train = (ds_train.interleave(lambda x:
                    tf.data.TFRecordDataset(filenames=x).map(tf_parse, num_parallel_calls=1), cycle_length=8, block_length=8))
        ds_train = (ds_train
                    .filter(tf_filter)
                    .prefetch(args.minibatch_size * 64)
                    .shuffle(args.minibatch_size * 1024) # 2048
                    .padded_batch(args.minibatch_size,
                        padded_shapes=(([input_max_len, 1], [target_max_len,], [1,], [1,]), [target_max_len,]),
                        drop_remainder=True)
                    .repeat())

        ds_test = tf.data.Dataset.from_tensor_slices(test_files)
        ds_test = (ds_test.interleave(lambda x:
                    tf.data.TFRecordDataset(filenames=x).map(tf_parse, num_parallel_calls=1), cycle_length=8, block_length=8))
        ds_test = (ds_test
                    .filter(tf_filter)
                    .prefetch(args.minibatch_size * 32)
                    .shuffle(args.minibatch_size * 256)
                    .padded_batch(args.minibatch_size,
                        padded_shapes=(([input_max_len, 1], [target_max_len,], [1,], [1,]), [target_max_len,]),
                        drop_remainder=True)
                    .repeat())

        transformer_hparams_file = (args.config or os.path.join('./training_configs', args.prefix, 'hparams.yaml'))
        if os.path.exists(transformer_hparams_file):
            with open(transformer_hparams_file, 'r') as fp:
                transformer_hparams = yaml.safe_load(fp)
        else:
            transformer_hparams = {'d_output' : d_output,
                               'd_model' : 256,
                               'cnn_features' : 128,
                               'cnn_kernel' : 32,
                               'cnn_pool_size' : 32,
                               'cnn_pool_stride' : 8,
                               'dff' : 1024,
                               'nff' : 2,
                               'encoder_nff' : 2,   # overwrites nff
                               'decoder_nff' : 2,   # overwrites nff
                               #'dff_type' : 'point_wise' or 'separable_convolution' or 'inception'
                               'encoder_dff_type' : 'separable_convolution',
                               'decoder_dff_type' : 'point_wise',
                               #'dff_filter_width' 'dff_pool_size'
                               'encoder_dff_filter_width' : 8,
                               'encoder_dff_pool_size' : 8,
                               'num_heads' : 8,
                               #'mha_qk_equal' : True,
                               'encoder_max_iterations' : 8,   # 14
                               'decoder_max_iterations' : 8,
                               'encoder_time_scale' : 10000,
                               'decoder_time_scale' : 10000,
                               'random_shift' : False,
                               'ponder_bias_init' : 0.0,
                               #'act_type' : 'separable_convolution',
                               'encoder_act_type' : 'point_wise',
                               'decoder_act_type' : 'point_wise',
                               'act_dff' : None,
                               #'act_conv_filter' : 5,
                               'encoder_time_penalty' : 0.005,
                               'decoder_time_penalty' : 0.015,
                               'input_memory_comp' : None,
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
            tf_lrs = WarmupLRS(transformer_hparams.get('d_model'), warmup_steps=4000)
            tf_optimizer = tf.keras.optimizers.Adam(tf_lrs, beta_1=0.9, beta_2=0.98, epsilon=1e-6, amsgrad=False)
            tf_optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(tf_optimizer, loss_scale='dynamic')
            cat_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
            # Transformer loss function
            def tf_loss_function(real, pred, target_lengths, mask):
                mask = mask / tf.reduce_sum(mask, axis=-1, keepdims=True)
                # read length weighted loss with scaling on EOS token
                target_length_idx = tf.expand_dims(tf.stack(
                                            [tf.range(tf.size(target_lengths)),
                                            tf.squeeze(target_lengths) - 1],
                                             axis=-1),
                                             axis=0)
                eos_mask = tf.gather_nd(mask, target_length_idx)
                eos_mask *= tf.cast(len(alphabet), eos_mask.dtype)
                mask = tf.tensor_scatter_nd_update(mask, target_length_idx, eos_mask)
                loss_ = cat_cross_entropy(real, pred, sample_weight=mask) # (batch_size, target_seq_len)
                loss_ = tf.reduce_sum(loss_, axis=-1) # (batch_size)
                # EOS token accuracy
                pred_lbl = tf.argmax(tf.nn.softmax(pred, axis=-1), axis=-1)
                eos_lbl = tf.gather_nd(pred_lbl, target_length_idx)
                eos_acc = tf.equal(eos_lbl, tf.constant(d_output - 1, dtype=tf.int64))
                eos_acc = tf.reduce_mean(tf.cast(eos_acc, tf.float32))
                return (loss_, eos_acc)

            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
            eos_accuracy = tf.keras.metrics.Mean(name='eos_accuracy')

            transformer = Transformer(hparams=transformer_hparams, name='Transformer', dtype=policy)

            tf_checkpoint = tf.train.Checkpoint(optimizer=tf_optimizer, transformer=transformer)
            tf_ckpt_manager = tf.train.CheckpointManager(tf_checkpoint, os.path.join(checkpoint_dir, 'transformer'), max_to_keep=5)

            if tf_ckpt_manager.latest_checkpoint:
                tf_checkpoint.restore(tf_ckpt_manager.latest_checkpoint)
                print('Latest transformer checkpoint restored!!')

            def train_step(inputs):
                input, target = inputs
                input_data, target_data, input_lengths, target_lengths = input
                mask = tf.sequence_mask(tf.squeeze(target_lengths), target_max_len, dtype=tf.float32)
                # Transformer gradient
                with tf.GradientTape() as tape:
                    tf_predictions, _, _, _dec_loss, _enc_loss = transformer(input, training=True)
                    loss_, eos_acc = tf_loss_function(target, tf_predictions, target_lengths, mask)
                    tf_loss = loss_  + _dec_loss + _enc_loss
                    scaled_tf_loss = tf_optimizer.get_scaled_loss(tf_loss)
                    scaled_loss = tf.nn.compute_average_loss(scaled_tf_loss, global_batch_size=args.minibatch_size)
                    loss = tf.nn.compute_average_loss(tf_loss, global_batch_size=args.minibatch_size)
                scaled_tf_gradients = tape.gradient([scaled_loss], transformer.trainable_variables)
                tf_gradients = tf_optimizer.get_unscaled_gradients(scaled_tf_gradients)
                #tf_gradients, _ = tf.clip_by_global_norm(tf_gradients, 10.0)
                # Apply gradients
                tf_optimizer.apply_gradients(zip(tf_gradients, transformer.trainable_variables))
                # reset and update accuracies
                train_accuracy.update_state(target, tf_predictions, mask)
                eos_accuracy.update_state(eos_acc)
                return loss

            def test_step(inputs):
                input, target = inputs
                target_lengths = input[3]
                mask = tf.sequence_mask(tf.squeeze(target_lengths), target_max_len, dtype=tf.float32)
                predictions, _, _, _dec_loss, _enc_loss = transformer(input, training=False)
                loss_, _ = tf_loss_function(target, predictions, target_lengths, mask)
                tf_loss = loss_ + _dec_loss + _enc_loss
                loss = tf.nn.compute_average_loss(tf_loss, global_batch_size=args.minibatch_size)
                test_accuracy.update_state(target, predictions, mask)
                return loss

            # `experimental_run_v2` replicates the provided computation and runs it
            # with the distributed input.
            @tf.function
            def distributed_train_step(dataset_inputs):
                per_replica_tf_losses = strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
                return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_tf_losses, axis=None)

            @tf.function
            def distributed_test_step(dataset_inputs):
                per_replica_losses = strategy.experimental_run_v2(test_step, args=(dataset_inputs,))
                return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

            best_losses = [float("inf")] * 5
            min_steps = 250
            steps = 0
            for epoch in range(args.epochs):
                total_loss = 0.0
                num_batches = 0
                ds_train_dist_iter = iter(ds_train_dist)
                ds_test_dist_iter = iter(ds_test_dist)
                for batch in tqdm(range(args.batches_train), desc='Training', ncols=0):
                    tf.summary.experimental.set_step(tf_optimizer.iterations)
                    batch_input = next(ds_train_dist_iter)
                    tf_loss = distributed_train_step(batch_input)
                    steps += 1
                    if epoch == 0 and batch == 0:
                        print("========== Transformer model ==========")
                        transformer.summary()
                    if tf_loss <= max(best_losses) and steps > min_steps:
                        tf_ckpt_manager.save()
                        best_losses[best_losses.index(max(best_losses))] = tf_loss
                        steps = 0
                    num_batches += 1
                    total_loss += tf_loss
                    train_loss = total_loss / num_batches
                    tf.summary.scalar("loss", tf_loss)
                    #tf.summary.scalar("lr", tf_lrs(tf_optimizer.iterations.numpy().astype(np.float32)))
                    tf.summary.scalar("lr", tf_lrs(tf_optimizer.iterations))
                    if batch % val_rate == 0:
                        batch_input = next(ds_test_dist_iter)
                        test_loss = distributed_test_step(batch_input)
                        tf.summary.scalar("loss_val", test_loss)
                        tf.summary.scalar('acc_train', train_accuracy.result())
                        tf.summary.scalar("acc_val", test_accuracy.result())
                        tf.summary.scalar("acc_eos", eos_accuracy.result())
                        train_accuracy.reset_states()
                        test_accuracy.reset_states()
                        eos_accuracy.reset_states()
                print("Epoch {}: train loss: {}".format(epoch, train_loss))

    def weights(self, argv):
        parser = argparse.ArgumentParser(description="BumbleBee basecaller prediction")
        parser.add_argument("config", help="BumbleBee config")
        parser.add_argument("checkpoint", help="Training checkpoint")
        args = parser.parse_args(argv)
        # Load config
        with open(args.config, 'r') as fp:
            transformer_hparams = yaml.safe_load(fp)
        strategy = tf.distribute.MirroredStrategy(devices=['/cpu:0'])
        #tf.config.experimental_run_functions_eagerly(True)
        with strategy.scope():
            transformer = Transformer(hparams=transformer_hparams, name='Transformer')

            input_data = tf.zeros((1, 100, 1), dtype=tf.float32)
            target_data = tf.zeros((1, 10), dtype=tf.int32)
            input_lengths = tf.ones((1,), dtype=tf.int32)
            target_lengths = tf.ones((1,), dtype=tf.int32)
            # init model on dummy data
            _ = transformer((input_data, target_data, input_lengths, target_lengths))

            optimizer = tf.keras.optimizers.Adam(0.0)
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, transformer=transformer)
            ckpt_manager = tf.train.CheckpointManager(checkpoint, args.checkpoint, max_to_keep=5)

            if ckpt_manager.latest_checkpoint:
                checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
                print('Latest checkpoint restored!!')
            else:
                print("Checkpoint not found!!")
                exit()

            transformer.summary()
            transformer.save_weights(os.path.join(os.path.dirname(args.config), 'weights.h5'), save_format='h5')

    def validate(self, argv):
        parser = argparse.ArgumentParser(description="BumbleBee basecaller prediction")
        parser.add_argument("config", help="BumbleBee config")
        parser.add_argument("weights", help="BumbleBee weights")
        parser.add_argument("records", help="Raw signal fast5 file")
        parser.add_argument("--input_min_length", type=int, default=500, help="Signal window size")
        parser.add_argument("--input_max_length", type=int, default=1200, help="Signal window size")
        parser.add_argument("--minibatch_size", type=int, default=16, help="Batch size")
        parser.add_argument("-t", "--threads", type=int, default=16, help="Threads")
        args = parser.parse_args(argv)
        tf.config.threading.set_inter_op_parallelism_threads(args.threads)
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        #tf.config.experimental_run_functions_eagerly(True)

        # Constants
        alphabet = "ACGT"
        tf_alphabet = alphabet + '^$'

        # Model
        d_output = len(tf_alphabet)
        input_min_len = args.input_min_length
        input_max_len = args.input_max_length
        target_max_len = input_max_len // 8

        # Load config
        with open(args.config, 'r') as fp:
            transformer_hparams = yaml.safe_load(fp)

        # tfRecord files
        if os.path.isfile(args.records):
            record_files = [args.records]
        else:
            record_files = [os.path.join(dirpath, f) for dirpath, _, files
                            in os.walk(args.records) for f in files if f.endswith('.tfrec')]

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
            seq_len = tf.cast(tf.expand_dims(tf.size(seq), axis=-1) - 1, tf.int32)
            sig_len = tf.cast(tf.expand_dims(tf.size(sig), axis=-1), tf.int32)
            return ((sig, seq[:-1], sig_len, seq_len), seq[1:])

        def tf_filter(input, target):
            #input, target = eg
            return (input[2] >= tf.cast(input_min_len, tf.int32) and
                    input[2] <= tf.cast(input_max_len, tf.int32) and
                    input[3] <= tf.cast(target_max_len + 2, tf.int32))[0]

        ds_val = tf.data.Dataset.from_tensor_slices(record_files)
        ds_val = (ds_val.interleave(lambda x:
                    tf.data.TFRecordDataset(filenames=x).map(tf_parse, num_parallel_calls=1), cycle_length=8, block_length=8))
        ds_val = (ds_val
                    .filter(tf_filter)
                    .prefetch(args.minibatch_size * 64)
                    .padded_batch(args.minibatch_size,
                        padded_shapes=(([input_max_len, 1], [target_max_len+2,], [1,], [1,]), [target_max_len+2,]),
                        drop_remainder=True)
                    )

        def decode_predictions(input, predictions, predicted_lengths=None):
            ret = []
            input_data, target_data, input_lengths, _target_lengths = input
            predicted_lengths = predicted_lengths if predicted_lengths is not None else _target_lengths
            #print(len(predicted_lengths))
            for target, target_length, prediction, predicted_length in zip(target_data, _target_lengths, predictions, predicted_lengths):
                logits = tf.argmax(prediction, axis=-1)
                target_sequence = decode_sequence(target[1:target_length[0]])
                predicted_sequence = decode_sequence(logits[:predicted_length[0]-1])
                algn = edlib.align(predicted_sequence, target_sequence, mode='NW', task='path')
                ref_iter = iter(target_sequence)
                res_iter = iter(predicted_sequence)
                cigar_ops = [(int(op[:-1]), op[-1]) for op in re.findall('(\d*\D)',algn['cigar'])]
                ref_exp = ''.join([''.join([next(ref_iter) for _ in range(n)]) if op in 'M=XD' else '-' * n for n, op in cigar_ops ])
                res_exp = ''.join([''.join([next(res_iter) for _ in range(n)]) if op in 'M=XI' else '-' * n for n, op in cigar_ops ])
                match_exp = ''.join('|' * n if op in 'M=' else '.' * n if op in 'X' else ' ' * n for n, op in cigar_ops)
                acc = match_exp.count('|') / len(match_exp) if len(match_exp) else 0.0
                yield (acc, ref_exp, match_exp, res_exp)

        @tf.function
        def validate_batch(input):
            # input_data, target_data, input_lengths, target_lengths = input
            predictions, _, _, _ = transformer(input, training=False)
            return predictions

        @tf.function
        def predict_batch_v1(input):
            # input_data, target_data, input_lengths, target_lengths = input
            pass

        @tf.function
        def predict_batch_v2(input):
            # input_data, input_lengths = input
            predictions, target_lengths, _, _ = transformer(input, training=False)
            return predictions, target_lengths

        transformer_hparams['target_max_len'] = target_max_len + 2
        transformer = Transformer(hparams=transformer_hparams)
        input, target = next(iter(ds_val))
        input_data, target_data, input_lengths, target_lengths = input
        print("init")
        predictions = validate_batch(input)
        transformer.load_weights(args.weights)

        t0 = timeit.default_timer()
        predictions = validate_batch(input)
        t1 = timeit.default_timer()
        accs = []
        with open('algn_val.txt', 'w') as fp:
            for acc, ref_exp, match_exp, res_exp in decode_predictions(input, predictions):
                print('@seq {0:.2f}'.format(acc), file=fp)
                print(ref_exp, file=fp)
                print(match_exp, file=fp)
                print(res_exp, file=fp)
                accs.append(acc)
        print("validation in {:.3f} with {:.2f} mean, {:.2f} median accuracy".format(t1-t0, np.mean(accs), np.median(accs)))

        #input, input_lengths, target_max = inputs
        t0 = timeit.default_timer()
        predictions, target_lengths = predict_batch_v2((input_data, input_lengths))
        t1 = timeit.default_timer()
        accs = []
        with open("algn_test.txt", 'w') as fp:
            for acc, ref_exp, match_exp, res_exp in decode_predictions(input, predictions, predicted_lengths=target_lengths):
                print('@seq {0:.2f}'.format(acc), file=fp)
                print(ref_exp, file=fp)
                print(match_exp, file=fp)
                print(res_exp, file=fp)
                accs.append(acc)
        print("prediction in {:.3f} with {:.2f} mean, {:.2f} median accuracy".format(t1-t0, np.mean(accs), np.median(accs)))
        exit(0)




if __name__ == '__main__':
    tf.config.optimizer.set_jit(True)
    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for d in physical_devices:
        tf.config.experimental.set_memory_growth(d, True)
    BumbleBee()
