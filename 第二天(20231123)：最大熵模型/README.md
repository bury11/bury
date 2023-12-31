# 最大熵模型
## 最大熵原理

***熵的定义***  
设 $X\$ 是一个取有限个值的离散随机变量，其概率分布为

$$
{\rm{P}}(X = {x_i}) = {p_i},  i = 1,2,...,n\
$$

则随机变量 $X\$ 的熵定义为

$$
H(X) =  - \sum\limits_{i = 1}^n {{p_i}} \log {p_i}\tag{1}
$$

在式(1)中，若 ${p_i} = 0\$ ，则定义 $0log0=0\$ 。通常其中对数以2为底或以e为底，这时熵的单位分别被称为比特(bit)，或纳特(nat)。

***条件熵***  
设有随机变量 $H(X,Y)\$ 表示在已知随机变量 $X\$ 的条件下随机变量Y的不确定性。  
随机变量 $X\$ 给定的条件下随机变量 $Y\$ 的条件熵(conditional entropy) $H(Y|X)\$ ，定义为 $X\$ 给定条件下 $Y\$ 的条件概率分布的熵对 $X\$ 的数学期望

$$
H(Y|X) = E{p_i}(H(Y|X = {x_i})) = \sum\limits_{i = 1}^n {{p_i}H(Y|X = {x_i})}  = \sum\limits_{i = 1}^n {P(X = {x_i})H(Y|X = {x_i})} \
$$

这里， ${p_i} = P(X = {x_i}), i = 1,2, \cdots ,n\$ 。

***最大熵原理***  
由定义可知，熵只依赖于 $X\$ 的分布，而与 $X\$ 的取值无关，所以也将 $X\$ 的熵记作 $H(p)\$ ，即

$$
H(p) =  - \sum\limits_{i = 1}^n {{p_i}} \log {p_i}\
$$

熵越大，随机变量的不确定性就越大。从定义可验证

$$
0 \le H(p) \le \log n\
$$

式中， $n\$ 时 $X\$ 的取值个数，当且仅当 $X\$ 的分布是均匀分布时右边等号成立。这也就是说，当 $X\$ 服从均匀分布时，熵最大。

推导：  
求熵的最大值即求 $\max  H =  - \sum\limits_{i = 1}^n {{p_i}} \log {p_i} + \lambda (\sum\limits_{i = 1}^n {{p_i}}  - 1)\$ ，其中 $\lambda (\sum\limits_{i = 1}^n {{p_i}}  - 1)\$ 为 $H\$ 的约束条件，这个约束条件由 $\sum\limits_{i = 1}^n {{p_i}}  = 1\$ 得来

我们使用拉格朗日乘数法对每个 ${p_i}\$ 求偏导可得同样的结果

$$
-(\log {p_i} + 1) + \lambda  = 0\
$$

$$
{p_i} = {2^{\lambda  - 1}}\
$$

又因为 $\sum\limits_{i = 1}^n {{p_i}}  = 1\$ ，故 $\sum\limits_{i = 1}^n {{p_i}}  = n \cdot {2^{\lambda  - 1}} = 1\$ ，由此可得 $p_i = \frac{1}{n}\$ 

即 $0 \le H(p) \le \log n\$ 
连续分布中正态分布熵最大，具体推导过程见[b站简博士](https://www.bilibili.com/video/BV1No4y1o7ac?p=65&vd_source=17517435653aa14cfea6edfa3d9b5f96)  
**举例**  
假设X有5个取值，满足以下约束条件（先验信息）： $P(A)+P(B)+P(C)+P(D)+P(E)=1\$   
满足这个约束条件的概率分布有无穷多个,在没有任何其他信息的情况下，我们还是需要对概率分布进行估计，此时我们可以认为这个分布中取各个值的概率是相等的，即 $P(A)=P(B)=P(C)=P(D)=P(E)=1/5\$ ，同样也是因为没有其它的信息（约束条件），因此等概率的判断是合理的。

直观地，最大熵原理认为要选择的概率模型首先必须满足已有的事实，即约束条件。在没有更多信息的情况下，那些不确定的部分都是“等可能的”。最大熵原理通过熵的最大化来表示等可能性。“等可能”不容易操作，而熵则是一个可优化的数值指标。
## 最大熵模型的定义

最大熵原理是统计学习的一般原理，将它应用到分类得到最大熵模型。


假设分类模型是一个条件概率分布 $P(Y|X)\$ ， $X\$ 表示输入， $Y\$ 表示输出。这个模型表示的是对于给定的输入 $X\$ ，以条件概率 $P(Y|X)\$ 输出 $Y\$ 。
给定一个训练数据集：
$T = \{ ({x_1},{y_1}),({x_2},{y_2}),...,({x_N},{y_N})\} \$ 
，学习的目标是用最大熵原理选择最好的分类模型。  

我们可以首先考虑模型应该满足的条件，即约束条件。给定训练数据集，可以确定联合分布P(X,Y）的经验分布和边缘分布P(X）的经验分布，两者都可以通过训练集算出来，分别以 $\widetilde{P}(X,Y)$ 和 $\widetilde{P}(X)$ 表示。

$$
\widetilde P(X = x,Y = y) = \frac{{v(X = x,Y = y)}}{N}\
$$

$$
\widetilde P(X = x) = \frac{{v(X = x)}}{N}\
$$

其中， $v(X = a,Y =y)\$ 表示训练数据中样本 $(x, y)\$ 出现的频数， $v(X = x)\$ 表示训练数据中输入 $x\$ 出现的频数， $N\$ 表示训练样本容量。

用特征函数(feature function)  $f(x, y)\$ 描述输入 $x\$ 和输出 $y\$ 之间的某一个事实。其定义是

$$
 f(x,y) =
  \begin{cases}
    1,       & \quad \text{x and y satisfy a certain fact}\\
    0,  & \quad \text{otherwise}
  \end{cases}
\
$$

它是一个二值函数，当 $x\$ 和 $y\$ 满足这个事实时取值为1，否则取值为0。

特征函数 $f(x,y)\$ 在训练集上关于联合经验分布的期望值：

$$
{E_{\widetilde P}}(f) = \sum\limits_{x,y} {\widetilde P(x,y)f(x,y)} = \sum\limits_{x,y} {\widetilde P(x)\widetilde P(x|y)f(x,y)}\
$$

上式中出现的分布都是经验分布，说明 ${E_{\widetilde P}}(f)$ 仅与训练集有关，并不是真实情况。那么真实情况应该是什么样的呢？其实只需要将经验分布转变成真实的分布就行了：

$$
{E_P}(f) = \sum\limits_{x,y} {P(x,y)f(x,y)} = \sum\limits_{x,y} {P(x)P(y|x)f(x,y)}\
$$

上式是特征函数 $f(x,y)\$ 在模型上关于模型 $P(X|Y)\$ 与 $P(x)\$ 的期望值，我们需要求的模型即 $P(X|Y)\$ ，但是其中还存在着一个未知分布 $P(x)\$ 。由于我们并不知道真实情况的 $P(x)\$ 是什么，我们可以使用 $P(x)\$ 的经验分布代替真实的 $P(x)\$ 来对真实情况进行近似表示，于是上式转变为：

$$
{E_P}(f) = \sum\limits_{x,y} {\widetilde P(x)P(y|x)f(x,y)}\
$$

现在我们有了一个针对训练集的期望 ${E_{\widetilde P}}(f)$ ，和针对模型的期望 ${E_{P}}(f)$ ，此时，只需要让两式相等（使在总体中出现的概率等于在样本中出现的概率），就能够让模型拟合训练集，并最终求得我们的模型：
如果模型能够获取训练数据中的信息，那么就可以假设这两个期望值相等，即

$$
{E_P}(f) = {E_{\widetilde P}}(f)\
$$

或

$$
\sum\limits_{x,y} {\widetilde P(x)P(y|x)f(x,y)}  = \sum\limits_{x,y} {\widetilde P(x,y)f(x,y)} \
$$

上式中 $P(y|x)\$ 为模型，两个经验分布可以从训练集中得到， $f(x,y)\$ 是特征函数。  
上式即为一个约束条件，假如有n个特征函数 ${f_i}(x,y),i = 1,2, \cdots ,n\$ ，那么就有n个约束条件。除此之外还有一个必定存在的约束，即模型概率之和等于1： $\sum\limits_y {P(y|x) = 1} \$ 

现在我们有了所有的约束条件，接着写出模型的熵的公式，就可以根据最大熵规则，在约束条件下得到模型。

定义在条件概率分布 $P(Y|X)\$ 上的条件熵为：

$$
\begin{align*}
H(P) = H(Y|X) & =  \sum\limits_{i = 1}^n {P(X = {x_i})H(Y|X = {x_i})} \\
 & =  - \sum\limits_{i = 1}^n {P(X = {x_i})} \sum\limits_{j = m}^m {P(Y = {y_j}|X = {x_i})\log P(Y = {y_j}|X = {x_i})} \\
 & =  - \sum\limits_{x,y} {P(x)P(y|x)\log P(y|x)} \\
 & =  - \sum\limits_{x,y} {\widetilde P(x)P(y|x)\log P(y|x)} 
\end{align*}\
$$

条件熵中依然使用了 $P(x)\$ 的经验分布代替真实分布，式中的对数为自然对数（以e为底）。那么求解模型的问题转换为求得最大化的条件熵问题。

**定义**(最大熵模型) 假设满足所有约束条件的模型集合为

$$
{\cal C} \equiv \\{ P \in {\cal P}|{E_P}({f_i}) = {E_{\tilde P}}({f_i}),i = 1,2, \cdots ,n \\} \
$$

定义在条件概率分布 $P(Y|X)\$ 上的条件熵为：

$$
H(P) =  - \sum\limits_{x,y} {\widetilde P(x)P(y|x)\log P(y|x)} \
$$

则模型集合c中条件熵h最大的模型成为最大熵模型。式中的对数为自然对数（以e为底）
## 最大熵模型的学习
