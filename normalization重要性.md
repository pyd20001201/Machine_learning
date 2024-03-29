归一化（normalization）：

  为了数据处理的方便，将数据映射为(0,1)区间或者(-1,1)区间之间的小数。特别地，为了数据处理的方便，更多时候我们倾向于将数据映射到0~1范围之间进行处理。


归一化处理的重要性：

  1) 在使用梯度下降方法求解时，进行归一化能够提高梯度下降的速度。

  假设现在的模型有两个特征x1和x2，特征x1的区间范围是(0,10)，而x2的区间范围是(0,2000),那么其形成的等高线偏椭圆就会非常尖，那么在梯度下降的时候，模

型很有可能走的是“之”字型，即每次都朝着垂直等高线的方向走，这样一来就会导致模型需要迭代很多次才能收敛，那么梯度下降的速度就会非常慢。

  但如果我们使用了归一化，特征的值就都被映射到了(0,1)区间之内，形成的等高线形状就会比较圆，在梯度下降时就会很快的收敛。

  对于机器学习中几乎所有使用梯度下降进行求解的模型，都需要进行归一化处理，否则会导致模型很难收敛甚至不能收敛。

  ![image](https://github.com/pyd20001201/Machine_learning/blob/master/2880006-762f0ef1c74dcd75.png)
  
  
  2) 归一化有可能提高精度
  一些分类器需要计算样本之间的距离（如欧式距离），例如KNN。如果一个特征值域范围非常大，那么距离计算就会主要取决于这个特征，从而有可能与实际情况相悖。
  
  3) 实现无量纲化
  使得不同度量之间的特征具有可比性，将不同量纲的特征转化为无量纲的纯数值，便于不同单位或量级的指标能够进行比较和加权。
  
  4) 避免数值问题
  避免了数值太大带来的溢出等问题


归一化的方法：

  1) Min_max Normalization
     x = (x - x_min) / (x_max - x_min)
     
     
  2) 平均归一化
     x = (x - u) / (x_max - x_min)
     
     
标准化：
  Z_score standardization:
     x = (x - u) / σ
     σ为标准差
     u为均值
