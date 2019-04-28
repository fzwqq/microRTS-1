# microRTS

## network

* **network/eval_network** and **network/policy_network** 实现了 [1] 论文的神经网络结构，使用随机生成的数据集可以运行。进一步的测试还需要得到真实的数据

* **network/original_eval_network** 实现了 [2] 论文中的神经网络结构，[1] 论文基于 [2] 论文的结构引入 policy network

## Reference

[1]:Barriga, N. A., Stanescu, M., & Buro, M. (2017). Combining Strategic Learning and Tactical Search in Real-Time Strategy Games, (October). Retrieved from http://arxiv.org/abs/1709.03480

[2]:Stanescu, M., Barriga, N. A., Hess, A., & Buro, M. (2017). Evaluating real-time strategy game states using convolutional neural networks. IEEE Conference on Computatonal Intelligence and Games, CIG, (September). https://doi.org/10.1109/CIG.2016.7860439
