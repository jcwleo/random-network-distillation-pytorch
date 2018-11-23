# Random Network Distillation
#### Intrinsic Reward Graph
![Alt text](https://github.com/jcwleo/random-network-distillation-pytorch/blob/master/asset/venture.gif)

- [x] Advantage Actor critic [[1]](#references)
- [x] Parallel Advantage Actor critic [[2]](#references)
- [x] Exploration by Random Network Distillation [[3]](#references)
- [x] Proximal Policy Optimization Algorithms [[4]](#references)

 
## 1. Setup
####  Requirements

------------

- python3.6
- gym
- [OpenCV Python](https://pypi.python.org/pypi/opencv-python)
- [PyTorch](http://pytorch.org/)
- [tensorboardX](https://github.com/lanpa/tensorboardX)


## 2. How to Train
Modify the parameters in `config.conf` as you like.
```
python train.py
```

## 3. How to Eval
```
python eval.py
```

## 4. Loss/Reward Graph
- Venture Env
![image](https://user-images.githubusercontent.com/23333028/48773457-c37cec00-ed0a-11e8-8c20-f9c35effc42d.png)



References
----------

[1] [Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)    
[2] [Efficient Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1705.04862)  
[3] [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)   
[4] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
  
