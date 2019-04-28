import eval_network
import policy_network
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os


#Hyper parameter
LEARNING_RATE = 1e-4
BATCH_SZIE = 256
ADAM_BEAT1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPLISION = 1e-8


def get_dataset_eval():
    data = np.random.randint(low=0, high=2, size=(100, 8,8,25))
    labels = np.random.randint(low=0, high=2,size=(100, 2))
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32).repeat()
    return dataset

def get_dataset_policy():
    data = np.random.randint(low=0, high=2, size=(100, 8,8,26))
    labels = np.random.randint(low=0, high=2,size=(100, 4))
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32).repeat()
    return dataset

def start_to_run():
    # 获得 dataset
    dataset_eval = get_dataset_eval()

    dataset_policy = get_dataset_policy()
    
    eval_model = eval_network.create_model()

    policy_model = policy_network.create_model()

    # 根据论文给出的参数
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,beta1=ADAM_BEAT1,
                beta2=ADAM_BETA2, epsilon=ADAM_EPLISION)

    # 论文没有给出 loss function 这里暂时选用 mean square 作为 loss function
    eval_model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())

    policy_model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())

    # 在没有 fit 之前 神经网络的结构还未被实际构建
    eval_model.fit(dataset_eval, batch_size=32, epochs=1,steps_per_epoch=30)
    if not os.path.exists('./weights/eval_model.h5'):
        eval_model.save_weights('./weights/eval_model.h5',save_format='h5')

    policy_model.fit(dataset_policy, batch_size=32, epochs=1,steps_per_epoch=30)
    if not os.path.exists('./weights/policy_model.h5'):
        policy_model.save_weights('./weights/policy_model.h5',save_format='h5')

    # 论文中没有提到参数是否共享
    # 这里假设共享它们
    # 等得到真实数据再做具体测试 
    for i in range(100):
        print('eval')
        eval_model.load_weights('./weights/eval_model.h5')
        eval_model.load_weights('./weights/policy_model.h5',by_name=True)
        eval_model.fit(dataset_eval, batch_size=32, epochs=5,steps_per_epoch=30)
        eval_model.save_weights('./weights/eval_model.h5',save_format='h5')

        print('policy')
        policy_model.load_weights('./weights/policy_model.h5')
        policy_model.load_weights('./weights/eval_model.h5',by_name=True)
        policy_model.fit(dataset_policy, batch_size=32, epochs=5,steps_per_epoch=30)
        policy_model.save_weights('./weights/policy_model.h5',save_format='h5')
       
if __name__ == "__main__":
    start_to_run()
