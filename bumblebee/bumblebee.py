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
import os, argparse
import tensorflow as tf
from tqdm import tqdm
from tf_transformer import Transformer
from tf_transformer_util import BatchGeneratorSim, TransformerLRS




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BumbleBee basecaller")
    parser.add_argument("model", help="pore model")
    args = parser.parse_args()

    input_max_len = 1000
    target_max_len = 100
    batches_train = 1000
    batches_val = 100
    batch_size = 32
    batch_gen = BatchGeneratorSim(args.model, target_len=target_max_len,
                                  batches_train=batches_train, batches_val=batches_val,
                                  minibatch_size=batch_size)

    d_input = batch_gen.input_dim
    d_output = batch_gen.target_dim
    d_model = 64
    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for d in physical_devices:
        tf.config.experimental.set_memory_growth(d, True)
    transformer_hparams = {'d_input' : d_input,
                           'd_output' : d_output,
                           'd_model' : d_model,
                           'dff' : d_model * 4,
                           'num_heads' : 8,
                           'max_iterations' : 16,
                           'encoder_time_scale' : 10000,
                           'decoder_time_scale' : 1000,
                           'random_shift' : True,
                           'input_memory_comp' : 8,
                           'target_memory_comp' : None}

    strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0'])
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    with strategy.scope():
        learning_rate = TransformerLRS(d_model, warmup_steps=2000)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9,amsgrad=True)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        def loss_function(real, pred, target_lengths):
            mask = tf.sequence_mask(target_lengths, target_max_len+2, dtype=tf.float32)
            loss_ = loss_object(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask
            return tf.nn.compute_average_loss(loss_, global_batch_size=batch_size)
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        with tf.device('/cpu:0'):
            transformer = Transformer(hparams=transformer_hparams)
        #transformer.summary()
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=transformer)

        def train_step(inputs):
            input, target = inputs
            target_lengths = input[3]
            with tf.GradientTape() as tape:
                predictions = transformer(input, training=True)
                loss = loss_function(target, predictions, target_lengths)
            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
            train_accuracy.update_state(target, predictions)
            return loss

        def test_step(inputs):
            input, target = inputs
            target_lengths = input[3]
            predictions = transformer(input, training=False)
            t_loss = loss_function(target, predictions, target_lengths)
            test_loss.update_state(t_loss)
            test_accuracy.update_state(target, predictions)

        # `experimental_run_v2` replicates the provided computation and runs it
        # with the distributed input.
        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        @tf.function
        def distributed_test_step(dataset_inputs):
            return strategy.experimental_run_v2(test_step, args=(dataset_inputs,))

        batch_gen_train = batch_gen.next_train()
        batch_gen_val = batch_gen.next_val()
        for epoch in range(20):
            total_loss = 0.0
            num_batches = 0
            batch_gen.on_epoch_begin()
            for batch in tqdm(range(batches_train), desc='Training'):
                input_data, target_data, input_len, target_len = next(batch_gen_train)
                total_loss += distributed_train_step((
                        [input_data, target_data[:,:-1], input_len, target_len],
                        target_data[:,1:]))
                num_batches += 1
            train_loss = total_loss / num_batches
            for batch in tqdm(range(batches_val), desc='Testing'):
                input_data, target_data, input_len, target_len = next(batch_gen_val)
                distributed_test_step((
                        [input_data, target_data[:,:-1], input_len, target_len],
                        target_data[:,1:]))
            print("Epoch {}: train loss: {}; test loss: {}; accuracy: {}".format(epoch,
                        train_loss,
                        test_loss.result(),
                        test_accuracy.result()))
            test_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()
