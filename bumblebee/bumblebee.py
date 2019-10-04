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
import os, argparse, yaml, time
import edlib
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tf_transformer import Transformer
from tf_transformer_util import BatchGeneratorSim, BatchGeneratorSig, TransformerLRS




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BumbleBee basecaller")
    parser.add_argument("model", help="Pore model")
    parser.add_argument("event", help="Event file")
    parser.add_argument("--config", default=None, help="Transformer config file")
    parser.add_argument("--prefix", default="", help="Checkpoint and event prefix")
    parser.add_argument("--input_length", type=int, default=1000, help="Input signal window")
    parser.add_argument("--target_length", type=int, default=100, help="Target sequence length")
    parser.add_argument("--minibatch_size", type=int, default=32, help="Minibatch size")
    parser.add_argument("--batches_train", type=int, default=10000, help="Training batches")
    parser.add_argument("--batches_val", type=int, default=1000, help="Validation batches")
    parser.add_argument("--gpus", nargs='+', type=int, default=[0], help="GPUs to use")
    args = parser.parse_args()
    input_max_len = args.input_length
    target_max_len = args.target_length
    batch_size = args.minibatch_size
    #batch_gen = BatchGeneratorSim(args.model, max_input_len=input_max_len,
    #                              min_target_len=50, max_target_len=target_max_len,
    #                              batches_train=args.batches_train, batches_val=args.batches_val,
    #                              minibatch_size=batch_size)
    batch_gen = BatchGeneratorSig(args.model, args.event, max_input_len=input_max_len,
                                  min_target_len=50, max_target_len=target_max_len,
                                  batches_train=args.batches_train, batches_val=args.batches_val,
                                  minibatch_size=batch_size)
    batches_train = batch_gen.batches_train
    batches_val = batch_gen.batches_val
    val_rate = batches_train // batches_val
    d_output = batch_gen.target_dim
    transformer_hparams_file = (args.config or os.path.join('./training_configs', args.prefix, 'hparams.yaml'))
    if os.path.exists(transformer_hparams_file):
        with open(transformer_hparams_file, 'r') as fp:
            transformer_hparams = yaml.safe_load(fp)
    else:
        transformer_hparams = {'d_output' : d_output,
                           'd_model' : 64,
                           'cnn_kernel' : 16,
                           'dff' : 256,
                           'dff_type' : 'separable_convolution',
                           'encoder_dff_filter_width' : 16,
                           'decoder_dff_filter_width' : 8,
                           'num_heads' : 4,
                           'encoder_max_iterations' : 8,
                           'decoder_max_iterations' : 16,
                           'encoder_time_scale' : 10000,
                           'decoder_time_scale' : 1000,
                           'random_shift' : False,
                           'ponder_bias_init' : 1.0,
                           'encoder_time_penalty' : 0.001,
                           'decoder_time_penalty' : 0.01,
                           'input_memory_comp' : 16,
                           'target_memory_comp' : None,
                           'input_memory_comp_pad' : 'same',
                           'target_memory_comp_pad' : 'causal'}
        os.makedirs(os.path.dirname(transformer_hparams_file), exist_ok=True)
        with open(transformer_hparams_file, 'w') as fp:
            print(yaml.dump(transformer_hparams), file=fp)

    tf.config.set_soft_device_placement(False)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for d in physical_devices:
        tf.config.experimental.set_memory_growth(d, True)

    strategy = tf.distribute.MirroredStrategy(devices=['/gpu:' + str(i) for i in args.gpus] if args.gpus else ['/cpu:0'])
    checkpoint_dir = os.path.join('./training_checkpoints', args.prefix)

    summary_writer = tf.summary.create_file_writer(os.path.join('./training_summaries', args.prefix))

    with strategy.scope(), summary_writer.as_default():
        learning_rate = TransformerLRS(transformer_hparams.get('d_model'), warmup_steps=8000)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9, amsgrad=False)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        def loss_function(real, pred, target_lengths):
            mask = tf.sequence_mask(target_lengths, target_max_len+2, dtype=tf.float32)
            loss_ = loss_object(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            return tf.nn.compute_average_loss(loss_, global_batch_size=batch_size * strategy.num_replicas_in_sync)
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        #with tf.device('/cpu:0'):
        transformer = Transformer(hparams=transformer_hparams)
        #transformer.summary()

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=transformer)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            checkpoint.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        def train_step(inputs):
            replica_context = tf.distribute.get_replica_context()  # for strategy
            id = tf.get_static_value(replica_context.replica_id_in_sync_group)
            input, target = inputs[id]
            target_lengths = input[3]
            with tf.GradientTape() as tape:
                predictions = transformer(input, training=True)
                loss = loss_function(target, predictions, target_lengths)
            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
            train_accuracy.update_state(target, predictions)
            return loss

        def test_step(inputs):
            replica_context = tf.distribute.get_replica_context()  # for strategy
            id = tf.get_static_value(replica_context.replica_id_in_sync_group)
            input, target = inputs[id]
            target_lengths = input[3]
            predictions = transformer(input, training=False)
            t_loss = loss_function(target, predictions, target_lengths)
            test_accuracy.update_state(target, predictions)
            return t_loss

        # `experimental_run_v2` replicates the provided computation and runs it
        # with the distributed input.
        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        @tf.function
        def distributed_test_step(dataset_inputs):
            per_replica_losses = strategy.experimental_run_v2(test_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        def distributed_batch(batch_gen):
            return [([input_data, target_data[:,:-1], input_len, target_len], target_data[:,1:])
                            for input_data, target_data, input_len, target_len in
                                [next(batch_gen) for _ in range(strategy.num_replicas_in_sync)]]

        batch_gen_train = batch_gen.next_train()
        batch_gen_val = batch_gen.next_val()

        for epoch in range(20):
            total_loss = 0.0
            num_batches = 0
            batch_gen.on_epoch_begin()
            for batch in tqdm(range(batches_train // strategy.num_replicas_in_sync), desc='Training', ncols=0):
                tf.summary.experimental.set_step(optimizer.iterations)
                batch_input = distributed_batch(batch_gen_train)
                batch_loss = distributed_train_step(batch_input)
                if epoch == 0 and batch == 0:
                    transformer.summary()
                num_batches += 1
                total_loss += batch_loss
                train_loss = total_loss / num_batches
                tf.summary.scalar("loss", batch_loss)
                tf.summary.scalar("lr", learning_rate(optimizer.iterations.numpy().astype(np.float32)))
                if batch % val_rate == 0:
                    batch_input = distributed_batch(batch_gen_val)
                    batch_loss = distributed_test_step(batch_input)
                    tf.summary.scalar("val_loss", batch_loss)
                    tf.summary.scalar('train_accuracy', train_accuracy.result())
                    tf.summary.scalar("val_accuracy", test_accuracy.result())
                    train_accuracy.reset_states()
                    test_accuracy.reset_states()

            print("Epoch {}: train loss: {}; accuracy: {}".format(epoch,
                        train_loss,
                        test_accuracy.result()))
            ckpt_manager.save()
