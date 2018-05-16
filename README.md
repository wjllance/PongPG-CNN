# PongPG-CNN
Using policy gradient to play atari game pong(deterministic-v4) on gym


Based on
------------
* [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)  Karpathy' Blog（1）
* [gameofdimension/policy-gradient-pong](https://github.com/gameofdimension/policy-gradient-pong)（2）



### Dependencies & Usage
  refer to (2)
  
### 主要改动
##### 1. 预处理：
80x80 -> 40x80，取出图像中间160x160并采样为40x80（这样采样后球刚好还占有1x1的像素，如果采成40x40，在某些帧中球就消失了）
##### 2. 网络输入：
(cur_x-prev_x) -> [prev_x, cur_x], 前后帧差分改为前后帧一起输入（没有做实验比较差别。。）
##### 3. 网络结构： 
* 改动前： 200个隐单元的全连接层加输出层
* 改动后：
  * 输入：  40x80x2,
  * 第一层：5x2, 4 channel, 
  * 第二层：3x1, 8 channel,
  * 第三层：100, 全连接，
  * 输出层：1,  动作概率，这里是向上的概率（这里只考虑两个动作）
##### 4.batch：
* 完成一场完整比赛（21分），为一个episode，
* （2）中每n=10个episode训练一次，一个episode大约有300~1000帧数据
* 本方法中每次获得reward（+1/-1）作为一个step，每个step只取结束前60帧，每40step训练一次，一次训练的输入最大2400帧（还可自行调整）
    
### 训练结果 
  网络规模较小，直接在CPU训练，大约14小时reward可以到0左右【（2）在pong-v0上训练50小时到达0附近】
