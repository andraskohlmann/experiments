import tensorflow as tf
from tensorflow.contrib.optimizer_v2.gradient_descent import GradientDescentOptimizer
from tensorflow.python.ops.metrics_impl import mean_squared_error
import tqdm
from data_processing.batch_generation import random_images_from_spectograms
from model.autoencoder import conv_ae_2d

encoder_setup = [
    (128, 3, 2),
    (256, 3, 1),
    (256, 3, 2),
    (128, 3, 2)
]

decoder_setup = [
    (128, 3, 2),
    (256, 3, 2),
    (256, 3, 1),
    (128, 3, 2)
]

def train():
    resolution = [128, 128]
    batch_size = 2
    # Create data inputs
    training_dataset = random_images_from_spectograms('data/spectograms', resolution=resolution, batch_size=batch_size)

    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                               training_dataset.output_shapes)
    sampled_images = iterator.get_next()

    training_init_op = iterator.make_initializer(training_dataset)
    # validation_init_op = iterator.make_initializer(validation_dataset)

    # Create tensorflow graph
    autoencoded_images = conv_ae_2d(sampled_images, encoder_setup, decoder_setup)
    loss = mean_squared_error(labels=sampled_images, predictions=autoencoded_images)
    optimizer = GradientDescentOptimizer(learning_rate=1e-3)
    train_op = optimizer.minimize(loss)

    # Run 20 epochs in which the training dataset is traversed, followed by the
    # validation dataset.
    with tf.Session() as sess:
        epochs = 100
        for epoch in range(epochs):
            # Training loop
            sess.run(training_init_op)
            with tqdm.trange(10) as t:
                for i in t:
                    # Description will be displayed on the left
                    t.set_description('GEN %i' % i)
                    _, loss = sess.run([train_op, loss])
                    t.set_postfix(loss=loss)
            print("epoch {}".format(epoch))

        # Save weights


if __name__ == '__main__':
    train()