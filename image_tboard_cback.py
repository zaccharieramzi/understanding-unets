"""Inspired by https://stackoverflow.com/a/49363251/4332585"""
import io

from keras.callbacks import Callback
from PIL import Image
from skimage.util import img_as_ubyte
import tensorflow as tf

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    _, height, width, channel = tensor.shape
    tensor = tensor[0]
    tensor_normalized = tensor - tensor.min()
    tensor_normalized /= tensor_normalized.max()
    tensor_normalized = img_as_ubyte(tensor_normalized)
    image = Image.fromarray(tensor_normalized)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    summary = tf.Summary.Image(
        height=height,
        width=width,
        colorspace=channel,
        encoded_image_string=image_string,
    )
    return summary

class TensorBoardImage(Callback):
    def __init__(self, log_dir, image, noisy_image):
        super().__init__()
        self.log_dir = log_dir
        self.noisy_image = noisy_image
        self.write_image(image, 'Original Image', 0)

    def write_image(self, image, tag, epoch):
        image = make_image(image)
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=image)])
        writer = tf.summary.FileWriter(self.log_dir)
        writer.add_summary(summary, epoch)
        writer.close()

    def on_epoch_end(self, epoch, logs={}):
        denoised_image = self.model.predict_on_batch(self.noisy_image)
        self.write_image(denoised_image, 'Denoised Image', epoch)
