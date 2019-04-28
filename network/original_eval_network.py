import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import LeakyReLU

class LeakyReLU(LeakyReLU):
    def __init__(self, **kwargs):
        self.__name__ = "LeakyReLU"
        super(LeakyReLU, self).__init__(**kwargs)

#Hyper parameter
LEARKYRELU_ALPHA = -1/5.5


class OriginalEvalModel(keras.Model):
    def __init__(self):
        super(OriginalEvalModel,self).__init__(name='')
        self.drop_out = keras.layers.Dropout(rate=0.5)
        self.LReLU = LeakyReLU(alpha=LEARKYRELU_ALPHA)
        self.flatten = keras.layers.Flatten()
        self.conv2d_1 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='valid')
        self.conv2d_2 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='valid')
        self.conv2d_3 = keras.layers.Conv2D(filters=1, kernel_size=1, strides=(1,1), padding='same')
        self.dense_4 = keras.layers.Dense(128)
        self.dense_5 = keras.layers.Dense(64)
        self.dense_6_eval = keras.layers.Dense(2, activation='softmax')

    def call(self, input_tensor):
        output_1 = self.conv2d_1(input_tensor)
        output_2 = self.conv2d_2(output_1)
        output_3 = self.flatten(self.conv2d_3(output_2))
        
        output_4 = self.drop_out(self.dense_4(output_3))
        output_5 = self.drop_out(self.dense_5(output_4))
        eval_output = self.dense_6_eval(output_5)
        print(eval_output)
        return eval_output

def create_model():
    return OriginalEvalModel()