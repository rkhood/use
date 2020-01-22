import tensorflow as tf
import tensorflow_hub as hub


def get_model(module="https://tfhub.dev/google/universal-sentence-encoder/4"):
    sess = tf.Session()
    model = hub.load(module)
    print ("module %s loaded" % module_url)
    return model


def embed(tokens):
    return model(tokens)


def get_vectors(data):
    encoded = tf.nn.l2_normalize(embed(tf.constant(data)), axis=1)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        vectors = session.run(encoded)
    return vectors



