# Course_CS231n

## StandFord大学CS231n课程的作业代码及相应的分析

&emsp;&emsp;现在网上关于CS231n资料非常多，但是学习这些资料远远比不上自己亲手实现一遍代码。

# 1 配置问题

考虑到很多朋友使用Windows系统（包括我这个VegetableBird...），这就直接导致数据集无法通过官网所给的教程下载，因此这里首先要解决作业的配置问题。

想要运行官网上的那段命令，第一步需要下载数据集。点击[这里](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)进行下载。下载后不需要解压，将这个压缩包放在*Assignment1/CS231n/datasets*目录下。第二步下载Git。在Git Bash中通过cd命令进入到*Assignment1/CS231n/datasets*这个文件夹下，输入*./get_datasets.sh*这个命令，配置完成！

当然，还有些朋友可能会缺少某些包，那就缺少哪些下载哪些就好。
