import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import argparse
import warnings
import os 
warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

ELMO = "https://tfhub.dev/google/elmo/2"
NNLM = "https://tfhub.dev/google/nnlm-en-dim128/1"


def execute(tensor):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        return sess.run(tensor)


def embed(model_name, sentences):
    if model_name == "elmo":
        elmo = hub.Module(ELMO, trainable=True)
        executable = elmo(
            sentences,
            signature="default",
            as_dict=True)["elmo"]

    elif model_name == "nnlm":
        nnlm = hub.Module(NNLM)
        executable = nnlm(sentences)

    else:
        raise NotImplementedError

    return execute(executable)


def word_to_sentence(embeddings):
    return embeddings.sum(axis=1)


def get_embeddings_elmo_nnlm(sentences):
    return word_to_sentence(embed("elmo", sentences)), embed("nnlm", sentences)
