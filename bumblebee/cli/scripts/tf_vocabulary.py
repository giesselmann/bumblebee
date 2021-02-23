import glob
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm




def tf_parse(eg):
    example = tf.io.parse_example(
        eg[tf.newaxis], {
            'title': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'abstract': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'journal' : tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'year' : tf.io.FixedLenFeature(shape=(), dtype=tf.string)})
    return (example['abstract'][0], example['title'][0])



if __name__ == "__main__":
    record_files = glob.glob("tfrecords/*.tfrec")
    ds = tf.data.Dataset.from_tensor_slices(record_files)
    ds = ds.interleave(lambda x: tf.data.TFRecordDataset(filenames=x).map(tf_parse, num_parallel_calls=8), cycle_length=16, block_length=16)
    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus((abstract.numpy() for abstract, title in tqdm(ds)), target_vocab_size=2**15)
    tokenizer_en.save_to_file('tf_vocabulary_32k')
