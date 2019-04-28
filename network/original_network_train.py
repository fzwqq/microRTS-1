import original_eval_network
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

#Hyper parameter
LEARNING_RATE = 1e-4
BATCH_SZIE = 256
ADAM_BEAT1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPLISION = 1e-8


def get_dataset():
    data = np.random.randint(low=0, high=2, size=(1000, 8,8,25))
    labels = np.random.randint(low=0, high=2,size=(1000, 2))
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32).repeat()
    return dataset


def start_to_run():

    dataset = get_dataset()

    eval_model = original_eval_network.create_model()

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,beta1=ADAM_BEAT1,
                beta2=ADAM_BETA2, epsilon=ADAM_EPLISION)

    # 论文没有给出 loss function 这里暂时选用 mean square 作为 loss function
    eval_model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())

    eval_model.fit(dataset, batch_size=32, epochs=1000, steps_per_epoch=30) 

    eval_model.save_weights('./weights/my_model')    
    
if __name__ == "__main__":
    start_to_run()