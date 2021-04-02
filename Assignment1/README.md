# 1. 配置问题

考虑到很多朋友使用Windows系统（包括我这个VegetableBird...），这就直接导致数据集无法通过官网所给的教程下载，因此这里首先要解决作业的配置问题。
想要运行官网上的那段命令，第一步需要下载数据集。点击[这里](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)进行下载。下载后不需要解压，将这个压缩包放在*Assignment1/CS231n/datasets*目录下。第二步下载Git。在Git Bash中通过cd命令进入到*Assignment1/CS231n/datasets*这个文件夹下，输入*./get_datasets.sh*这个命令，配置完成！
当然，还有些朋友可能会缺少某些包，那就缺少哪些下载哪些就好。

# 2 代码分析

## 2.1 KNN

在KNN中，需要完成k_nearest_neighbor.py和knn.ipynb中相应的代码。

## 2.1.1  训练数据的可视化

```python
# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show();
```

这段代码的主要目的在于将每个类别中的一些训练数据可视化。*idxs*通过np.flatnonzero获得第y类的索引值，在随机选出*samples_per_class*个样本绘制出来。

## 2.1.2 compute_distances_two_loops

这个函数在k_nearest_neighbor.py中，使用两个循环计算第i个测试数据与第j个训练数据的L2 distance，并放入$dist_{ij}$中，这个是非常简单的。

```python
# 直接计算即可
dists[i][j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
```

## 2.1.3 predict_labels

这个函数的目的是通过计算出的dists求得第i个测试数据到训练数据最近的K个样本类别。
第一部分代码：

```python
# 获得排序后的索引序列
sorted_index = np.argsort(dists[i, :])
top_K = sorted_index[:k]
closest_y = self.y_train[top_K]
```

这部分代码的意义是对于第i个测试数据，先将dists的第i行进行排序，并获得排序后的索引序列，那么top_K = sorted_index[:k]。c此时最近的K个类别就是在top_K索引下的y_train的值。

第二部分代码：

```python
y_pred[i] = np.argmax(np.bincount(closest_y))
```

这里使用了np.bincount函数，这个函数是产生一个元素数量比closet_y最大值大一的numpy矩阵。也就是说若closet_y中最大值为9，通过这个函数产生的矩阵元素个数为10个，每个索引对应的元素值就是该索引值在closet_y中出现的个数。所以前K个类别中最多的类即为最大值的索引。

## 2.1.4  compute_distances_one_loop

这里用到了广播原则。

```python
# 使用广播原则 注意每行的和是L2的值
dists[i, :] = np.sqrt(np.sum((X[i] - self.X_train) ** 2, axis = 1))
```

X[i]维度为（D,）X_train的维度是（num_train, D），所以X[i]会按行扩展，即扩展成每一行的元素值都与X[i]相同。

## 2.1.5 compute_distances_no_loop

个人觉得还是挺难的。
朴素想法：

```python
X = X.reshape(num_test, 1, X.shape[1])
dists = np.sqrt(np.sum((X - self.X_train) ** 2), axis = 2)
```

思路：$dists_{ij} = \sqrt{(\sum_{k=0}^{D-1}(X_{ik}-X\_train_{jk})^2)}$，这里的D是X_train的列数，也就是输入的特征维度数。根据这个公式，可以发现如果不使用循环，可以把X和X_train扩展为（num_test,num_train,D）的numpy矩阵。这样扩展的目的是使X在第i行第j列z轴方向上是X[i,:]，使X_train在第i行第j列z轴方向上是X[j,:]。这就使得当扩展后相减后获得的新矩阵$new_{ijk}=X_{ik}-X\_train_{jk}$，所以将new平方后在z轴方向上求和在开方即为$dists_{ij}$。可能这样说依旧有朋友不太理解，我在换种解释方法。X原本为一个(num_test,D)的矩阵，我们把它沿行方向立起来，这时它就成为了一个(num_test,1,D)维的矩阵，此时将它按列方向扩展。X_train原本为一个(num_train,D)的矩阵，把它也沿行方向立起来，这时它就成为了一个(num_train,1,D)维的矩阵。不过这里略有不同的在于把它的行当作列，也就是说此时它是一个(1，num_train,D)的矩阵。将它按行扩展，就是上述所扩展出的矩阵。
下面考虑以下如何使用广播原则实现这一过程。X是(num_test,D)的矩阵，X_train是(num_train,D)的矩阵，将这连个都扩展成（num_test,num_train,D）的矩阵，所以可以把X reshape成(num_test,1,D)的矩阵，这样通过广播就可以获得获得相应的矩阵了。不过遗憾的是这种做法会导致内存超限，不过这还是提供了一个思路。

进阶想法：

```python
# 方案二
d1 = np.sum(X ** 2, axis = 1)
d2 = np.sum(self.X_train ** 2, axis = 1)
d = d1.reshape(-1, 1) + d2.reshape(1, -1)
dists = np.sqrt(d - 2 * np.dot(X, self.X_train.T))np.sum()
```

思路：这个方案的主要思路就是利用了$(a-b)^2=a^2+b^2-2ab$。回到这个问题上，$dist_{ij}=\Sigma_{k=0}^{D-1} (X_{ik}^2+X\_train_{jk}^2)-2\Sigma_{k=0}^{D-1}X_{ik}X\_train_{jk}$。所以，这就需要











