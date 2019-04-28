import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import LeakyReLU

class LeakyReLU(LeakyReLU):
    def __init__(self, **kwargs):
        self.__name__ = "LeakyReLU"
        super(LeakyReLU, self).__init__(**kwargs)

#Hyper parameter
LEARKYRELU_ALPHA = -1/5.5

class PolicyNetworkModel(keras.Model):

    def __init__(self):
        super(PolicyNetworkModel,self).__init__(name='')
        self.drop_out = keras.layers.Dropout(rate=0.2)
        self.LReLU = LeakyReLU(alpha=LEARKYRELU_ALPHA)
        self.flatten = keras.layers.Flatten()
        self.conv2d_1 = keras.layers.Conv2D(filters=32, kernel_size=2, strides=(1,1), padding='valid', activation=self.LReLU)
        
        self.conv2d_2 = keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), 
                        padding='same', activation=self.LReLU, name='conv_2')

        self.conv2d_3 = keras.layers.Conv2D(filters=48, kernel_size=2, strides=(1,1), 
                        padding='valid', activation=self.LReLU, name='conv_3')

        self.conv2d_4 = keras.layers.Conv2D(filters=48, kernel_size=3, strides=(2,2), 
                        padding='same', activation=self.LReLU, name='conv_4')

        self.conv2d_5 = keras.layers.Conv2D(filters=64, kernel_size=2, strides=(1,1), 
                        padding='valid', activation=self.LReLU, name='conv_5')

        self.conv2d_6 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2),
                        padding='same', activation=self.LReLU, name='conv_6')

        self.conv2d_7 = keras.layers.Conv2D(filters=64, kernel_size=1, strides=(1,1), 
                        padding='same', activation=self.LReLU, name='conv_7') 
        
        self.conv2d_8_policy = keras.layers.Conv2D(filters=4, kernel_size=1, strides=(1,1),
                        padding='same', activation=self.LReLU)
        
        self.global_average_pool = keras.layers.GlobalAveragePooling2D()
        self.dense_9_policy = keras.layers.Dense(4, activation='softmax')

    def conv_drop(self, conv_layer, input_tensor):
        output = conv_layer(input_tensor)
        output = self.drop_out(output)
        return output

    def call(self, input_tensor):
        output_1 = self.conv_drop(self.conv2d_1, input_tensor)
        output_2 = self.conv_drop(self.conv2d_2, output_1)
        output_3 = self.conv_drop(self.conv2d_3, output_2)
        output_4 = self.conv_drop(self.conv2d_4, output_3)
        output_5 = self.conv_drop(self.conv2d_5, output_4)
        output_6 = self.conv_drop(self.conv2d_6, output_5)
        output_7 = self.conv_drop(self.conv2d_7, output_6)
        output_8_policy = self.global_average_pool(self.conv_drop(self.conv2d_8_policy, output_7))
        policy_output = self.dense_9_policy(self.flatten(output_8_policy))
        return policy_output

def create_model():
    return PolicyNetworkModel()
