# microRTS

## network

### 2017_champion_cnn

实现了 [1] 论文的神经网络结构，使用随机生成的数据集可以运行。进一步的测试还需要得到真实的数据

### original_cnn

实现了 [2] 论文中的神经网络结构，[1] 论文基于 [2] 论文的网络结构引入了 policy network

## datagenerator

* 用于实现对 microRTS 平台生成的 trace 文件的解析
* 替换文件中的 encode_game_state 和 reward 函数来实现自定义的 state 和 reward 的解析
* 当前将解析出来的数据转换为论文 [2] 中的数据格式，可以用来训练 original_cnn 神经网络

# Reference

[1]:Barriga, N. A., Stanescu, M., & Buro, M. (2017). Combining Strategic Learning and Tactical Search in Real-Time Strategy Games, (October). Retrieved from http://arxiv.org/abs/1709.03480

[2]:Stanescu, M., Barriga, N. A., Hess, A., & Buro, M. (2017). Evaluating real-time strategy game states using convolutional neural networks. IEEE Conference on Computatonal Intelligence and Games, CIG, (September). https://doi.org/10.1109/CIG.2016.7860439
