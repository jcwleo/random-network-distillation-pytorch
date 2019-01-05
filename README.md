# Random Network Distillation
#### Intrinsic Reward Graph with play
| Venture             |  Montezuma's Revenge |
:-------------------------:|:-------------------------:
![](https://github.com/jcwleo/random-network-distillation-pytorch/blob/master/asset/venture.gif)  |  [![Video Label](https://github.com/jcwleo/random-network-distillation-pytorch/blob/master/asset/montezuma.gif)](https://youtu.be/eEoHnkDUw4k?t=0s)
 ~ | [New model for Montezuma](https://www.youtube.com/watch?v=IxXHQDHvGoE&feature=youtu.be)

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
- Montezuma's Revenge Env
![image](https://user-images.githubusercontent.com/23333028/50719328-de9a9400-10dd-11e9-8c8c-29f7709cdf1d.png)
- Venture Env
![image](https://user-images.githubusercontent.com/23333028/48773457-c37cec00-ed0a-11e8-8c20-f9c35effc42d.png)



References
----------

[1] [Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)    
[2] [Efficient Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1705.04862)  
[3] [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)   
[4] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
  
