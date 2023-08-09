# A Generalized Framework for Self-Play

## 简介

想弄一个统一的博弈学习理论框架。

## 安装方法

### Github

```bazaar
git clone https://github.com/Zealoter/GWPFEFG.git
pip install -r requirements.txt 
```



## 用法

虽然可以弄成一个框架，但是把框架分成采样和非采样有助于代码的实现。运行TrainGFSP和运行TrainGFSP_Sampling即可

GAME_FULL和GAME_Sampling是游戏的实现，可以参考里面的内容做自己的游戏。