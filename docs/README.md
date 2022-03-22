# Fair Rotated Target Recognition Framework, FRTRF

## 数据

## 安装
- PyTorch : 可按照[PyTorch官网](http://pytorch.org)的指南，根据自己的平台安装指定的版本
- 安装指定依赖：

```
# if git has not been installed, run "sudo apt-get install git" before the following command.
pip install -r requirements.txt
```

## 训练
必须首先启动visdom：

```
python -m visdom.server
```

然后使用如下命令启动训练：

```
# 在gpu0上训练,并把可视化结果保存在visdom 的classifier env上
python main.py train --train-data-root=./data/train --use-gpu --env=classifier
```


详细的使用命令 可使用
```
python main.py help
```

## 测试

```
python main.py test --data-root=./data/test  --batch-size=256 --load-path='checkpoints/squeezenet.pth'
```
