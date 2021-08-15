# BiSeNetV2-RoadSegmentation-Paddle
This is a project aiming at training an AI model for semantic segmentation purpose. The specific target is for extracting road segmentation from an image or a video on edge devices.

# [AI达人训练营] 基于BiSeNetV2的实时路面分割模型


本项目属于我自己的一个项目My Buddy下面的一个sub-project. My Buddy是一个通过运用AI视觉可以自动跟随主人的小robot.
而这个sub project的主要目的就是训练出一个可以用在Edge devices上的小型轻量化的计算机视觉模型用来实时的分割地面.


# 一、项目背景

*项目的起因是因为年迈的老父母和喜欢跑步的自己....*

**场景一**

爸妈出去散布,爸爸腿脚不方便需要一个电动车,还好政府给他发了一个电动小车.但是他也需要运动来保持身体技能不要退化的太快.
所以当爸爸走路的时候 妈妈就要帮他推着车,或者开着电动小车.这样以来 妈妈就不能好好散布,也就是说始终只能有一个人在舒服的散布锻炼身体.

**场景二**

爸妈,甚至是我自己买菜的时候总是需要拉着购物小车.墨尔本这边的习惯是去一次超市买半个星期或者一周的事物.大包小包的攧着对于我这个身强体壮的人来说不是问题,但是逛街买东西要好久呢,所以还是挺不方便. 对于爸妈来说就更不方便啦.
其实购物的时候拉着小购物车在澳洲非常流行,我又专门去"蹲点"了几个市场,发现了至少又30%的人拉车.90%以上的老年人都喜欢拉着小车购物,年轻人也会时常拉车来超市或者菜市场.
<br />
<div style="width:90%; margin:10px; display: flex; justify-content:center">
    <div style="width:40%; float:left; margin:10px;">
  		<img src="https://ai-studio-static-online.cdn.bcebos.com/f2ea9de30ceb418682fc3c6aaa852e88e045d5451cd0447a9971942849216165" />
	</div>
    <div style="width:40%; float:right">
  		<img src="https://ai-studio-static-online.cdn.bcebos.com/dc87d0976e604a849dd08da9b0211755113a5831c9fc4c3cb3dcf32374edcc2c" />
	</div>
</div>

<div style="width:90%; margin:10px; display: flex; justify-content:center">
    <div style="width:40%; float:left; margin:10px;">
  		<img src="https://ai-studio-static-online.cdn.bcebos.com/a5eec0a4ec2d4bcca15b253e8637eaeef4c6d2f1861a4fcdbd24a99797fd3179" />
	</div>
    <div style="width:40%; float:right">
  		<img src="https://ai-studio-static-online.cdn.bcebos.com/f061fa37a60248e39b49adab97e9b1297b50ffe630e84fa5a7959f73f6a362bc" />
	</div>
</div>


<br />

**场景三**

我不想背包...包能不能跟着我...?我喜欢跑步,健身,游泳,所以经常一个书包塞的鼓鼓囊囊的,背起来很不爽啊.能不能....有个东西跟着我到天涯海角并且帮我带着东西呢?

**场景四**

澳洲的各家超市门口都摆满了trolleys(购物时用的小车),像Coles, Woolworth少则几百,但如美国来的Costco大型购物超市 门前的购物车更是数不胜数. 而超市内路面由于澳洲安全法规严格,路面都是超级平整易行.是非常好的大规模使用My Buddy的场所. 所以我在想....要不要.....
![Costco](https://ai-studio-static-online.cdn.bcebos.com/121a93a218444d178a080f4af19937e36d1046e40c944337badce34279a3eeee)

甚至,这边的AusPost的邮递员 都是很多老年人推着小车


![postman](https://ai-studio-static-online.cdn.bcebos.com/881f4f3ad099415caf76a2887f8177fc98a3b5af631b46299adf51689a2177f9)



好了....我想我已经成功说服我自己了....


# 二、数据集简介

我的项目使用了407张(还在增加更多场景中....)从我居住的本地墨尔本拍摄到的视频里提取出来的照片.

## 0. 怎样收集数据集
这里采用了2种方式.室外采用遥控小车,室内手持(减少别人对数据采集的反感度....)

<div style="width:90%; margin:10px; display: flex; justify-content:center">
   <div style="width:40%; float:left; margin:10px;">
      <img src="https://ai-studio-static-online.cdn.bcebos.com/57d2ed5d4f5e4353a2da0c82c9b5deafce4d5673507b4d3ea333609a73a7a1c7" />
  </div>

  <div style="width:40%; float:left; margin:10px;">
      <img src="https://ai-studio-static-online.cdn.bcebos.com/db79c083b7674b3a8e011a2ef95eec4951147807dd014d8a86c84e3c35f71644" />
  </div>
</div>
  


## 1.数据加载和预处理


```python
import paddle.vision.transforms as T

# 数据的加载和预处理
train_transforms = [
    T.RandomPaddingCrop(crop_size=(384,384)),
    T.RandomHorizontalFlip(prob=0.3),
    T.Resize(target_size=(416,416)),
    T.Normalize()
]

val_transforms = [
    T.Resize(target_size=(416,416)),
    T.Normalize()
]

# 训练数据集
train_dataset = Dataset(
    dataset_root = dataset_dir,
    train_path = train_path,
    num_classes = 2,
    transforms = train_transforms,
    edge = True,
    separator=' ',
    ignore_index=255,
    mode = 'train'
)

# 评估数据集
val_dataset = Dataset(
    dataset_root = dataset_dir,
    val_path = test_path,
    num_classes = 2,
    transforms = val_transforms,
    separator=' ',
    ignore_index=255,
    mode = 'val'
)

```

需要注意的地方是,在做transform的时候 有些步骤不能颠倒.比如说Normalize()一定放在最后.


## 2.数据集查看


```python
print('图片：')
print(type(train_dataset[0][0]))
print(train_dataset[0][0])
print('标签：')
print(type(train_dataset[0][1]))
print(train_dataset[0][1])

# 可视化展示
from matplotlib import pyplot as plt

view = train_dataset[0][0].transpose(1,2,0)

plt.figure()
plt.imshow(view)
plt.show()

```
![](https://ai-studio-static-online.cdn.bcebos.com/96b3d864f62247f699f4cac2da1cd71ad8afeda370854e4aa97b60d0e2fb982d)


# 三、模型选择和开发

详细说明你使用的算法。此处可细分，如下所示：

一开始想着用Deeplabv3p + MobileNetV3_Small__075 (我就叫它 DLV3PMNV3)来做路面分割. 
原因很简单, 我看到Deeplabv3p效果很棒, 又想着mobilenet的轻量化 应该很适合在edge devices e.g. RPi, Edgeboard, OAK-D上面用.

但是后来我还是换成了用BiSeNetV2 来做路面分割.原因是因为:

> 通过实验发现， Deeplabv3plus + MobileNetV3（DLV3PMNV3）并不能得到很好的精度。原因是因为 MobileNetV3作为骨干网络本身层数较少 不能学习足够多的特征，导致在Deeplabv3plus的网络中得不到较好的特征值 进而导致整体准确度非常低。 换用BiSeNetV2做实验之后 发现效果远远好于DLV3PMNV3网络。 并且发现 BiseNetV2有着很高的 FPS 这符合项目的需求.(虽然在下图种BiSeNetV2的mIoU看起来比Deeplabv3要低很多, 但是那个Deeplabv3是以ResNet为backbone的,而且 BiSeNetV2在实验过程中 证明它的准确率足够用.
> 

![](https://ai-studio-static-online.cdn.bcebos.com/b24ff93561464deeb1c681afe3578a3c49c1cd6fc68d47f2acb6c729b82de1b3)



## 1.模型组网

![](https://ai-studio-static-online.cdn.bcebos.com/0d743126153243a5b9925213449e250252682ef9718c4205872a58cedf0a9e45)



```python
model = BiSeNetV2(
    num_classes=2,
    lambd=0.25
)

```

## 2.模型网络结构可视化


```python
# 模型可视化
model.full_name
```

```
   <bound method Layer.full_name of BiSeNetV2(
  (db): DetailBranch(
    (convs): Sequential(
      (0): ConvBNReLU(
        (_conv): Conv2D(3, 64, kernel_size=[3, 3], stride=[2, 2], padding=same, data_format=NCHW)
        (_batch_norm): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
      )
      (1): ConvBNReLU(
        (_conv): Conv2D(64, 64, kernel_size=[3, 3], padding=same, data_format=NCHW)
        (_batch_norm): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
      )
      (2): ConvBNReLU(
        (_conv): Conv2D(64, 64, kernel_size=[3, 3], stride=[2, 2], padding=same, data_format=NCHW)
        (_batch_norm): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
      )
      (3): ConvBNReLU(
        (_conv): Conv2D(64, 64, kernel_size=[3, 3], padding=same, data_format=NCHW)
        (_batch_norm): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
      )
      (4): ConvBNReLU(
        (_conv): Conv2D(64, 64, kernel_size=[3, 3], padding=same, data_format=NCHW)
        (_batch_norm): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
      )
      (5): ConvBNReLU(
        (_conv): Conv2D(64, 128, kernel_size=[3, 3], stride=[2, 2], padding=same, data_format=NCHW)
        (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
      )
      (6): ConvBNReLU(
        (_conv): Conv2D(128, 128, kernel_size=[3, 3], padding=same, data_format=NCHW)
        (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
      )
      (7): ConvBNReLU(
        (_conv): Conv2D(128, 128, kernel_size=[3, 3], padding=same, data_format=NCHW)
        (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
      )
    )
  )
  (sb): SemanticBranch(
    (stem): StemBlock(
      (conv): ConvBNReLU(
        (_conv): Conv2D(3, 16, kernel_size=[3, 3], stride=[2, 2], padding=same, data_format=NCHW)
        (_batch_norm): BatchNorm2D(num_features=16, momentum=0.9, epsilon=1e-05)
      )
      (left): Sequential(
        (0): ConvBNReLU(
          (_conv): Conv2D(16, 8, kernel_size=[1, 1], padding=same, data_format=NCHW)
          (_batch_norm): BatchNorm2D(num_features=8, momentum=0.9, epsilon=1e-05)
        )
        (1): ConvBNReLU(
          (_conv): Conv2D(8, 16, kernel_size=[3, 3], stride=[2, 2], padding=same, data_format=NCHW)
          (_batch_norm): BatchNorm2D(num_features=16, momentum=0.9, epsilon=1e-05)
        )
      )
      (right): MaxPool2D(kernel_size=3, stride=2, padding=1)
      (fuse): ConvBNReLU(
        (_conv): Conv2D(32, 16, kernel_size=[3, 3], padding=same, data_format=NCHW)
        (_batch_norm): BatchNorm2D(num_features=16, momentum=0.9, epsilon=1e-05)
      )
    )
    (stage3): Sequential(
      (0): GatherAndExpansionLayer2(
        (branch_1): Sequential(
          (0): ConvBNReLU(
            (_conv): Conv2D(16, 16, kernel_size=[3, 3], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=16, momentum=0.9, epsilon=1e-05)
          )
          (1): DepthwiseConvBN(
            (depthwise_conv): ConvBN(
              (_conv): Conv2D(16, 96, kernel_size=[3, 3], stride=[2, 2], padding=same, groups=16, data_format=NCHW)
              (_batch_norm): BatchNorm2D(num_features=96, momentum=0.9, epsilon=1e-05)
            )
          )
          (2): DepthwiseConvBN(
            (depthwise_conv): ConvBN(
              (_conv): Conv2D(96, 96, kernel_size=[3, 3], padding=same, groups=96, data_format=NCHW)
              (_batch_norm): BatchNorm2D(num_features=96, momentum=0.9, epsilon=1e-05)
            )
          )
          (3): ConvBN(
            (_conv): Conv2D(96, 32, kernel_size=[1, 1], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=32, momentum=0.9, epsilon=1e-05)
          )
        )
        (branch_2): Sequential(
          (0): DepthwiseConvBN(
            (depthwise_conv): ConvBN(
              (_conv): Conv2D(16, 16, kernel_size=[3, 3], stride=[2, 2], padding=same, groups=16, data_format=NCHW)
              (_batch_norm): BatchNorm2D(num_features=16, momentum=0.9, epsilon=1e-05)
            )
          )
          (1): ConvBN(
            (_conv): Conv2D(16, 32, kernel_size=[1, 1], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=32, momentum=0.9, epsilon=1e-05)
          )
        )
      )
      (1): GatherAndExpansionLayer1(
        (conv): Sequential(
          (0): ConvBNReLU(
            (_conv): Conv2D(32, 32, kernel_size=[3, 3], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=32, momentum=0.9, epsilon=1e-05)
          )
          (1): DepthwiseConvBN(
            (depthwise_conv): ConvBN(
              (_conv): Conv2D(32, 192, kernel_size=[3, 3], padding=same, groups=32, data_format=NCHW)
              (_batch_norm): BatchNorm2D(num_features=192, momentum=0.9, epsilon=1e-05)
            )
          )
          (2): ConvBN(
            (_conv): Conv2D(192, 32, kernel_size=[1, 1], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=32, momentum=0.9, epsilon=1e-05)
          )
        )
      )
    )
    (stage4): Sequential(
      (0): GatherAndExpansionLayer2(
        (branch_1): Sequential(
          (0): ConvBNReLU(
            (_conv): Conv2D(32, 32, kernel_size=[3, 3], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=32, momentum=0.9, epsilon=1e-05)
          )
          (1): DepthwiseConvBN(
            (depthwise_conv): ConvBN(
              (_conv): Conv2D(32, 192, kernel_size=[3, 3], stride=[2, 2], padding=same, groups=32, data_format=NCHW)
              (_batch_norm): BatchNorm2D(num_features=192, momentum=0.9, epsilon=1e-05)
            )
          )
          (2): DepthwiseConvBN(
            (depthwise_conv): ConvBN(
              (_conv): Conv2D(192, 192, kernel_size=[3, 3], padding=same, groups=192, data_format=NCHW)
              (_batch_norm): BatchNorm2D(num_features=192, momentum=0.9, epsilon=1e-05)
            )
          )
          (3): ConvBN(
            (_conv): Conv2D(192, 64, kernel_size=[1, 1], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
          )
        )
        (branch_2): Sequential(
          (0): DepthwiseConvBN(
            (depthwise_conv): ConvBN(
              (_conv): Conv2D(32, 32, kernel_size=[3, 3], stride=[2, 2], padding=same, groups=32, data_format=NCHW)
              (_batch_norm): BatchNorm2D(num_features=32, momentum=0.9, epsilon=1e-05)
            )
          )
          (1): ConvBN(
            (_conv): Conv2D(32, 64, kernel_size=[1, 1], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
          )
        )
      )
      (1): GatherAndExpansionLayer1(
        (conv): Sequential(
          (0): ConvBNReLU(
            (_conv): Conv2D(64, 64, kernel_size=[3, 3], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
          )
          (1): DepthwiseConvBN(
            (depthwise_conv): ConvBN(
              (_conv): Conv2D(64, 384, kernel_size=[3, 3], padding=same, groups=64, data_format=NCHW)
              (_batch_norm): BatchNorm2D(num_features=384, momentum=0.9, epsilon=1e-05)
            )
          )
          (2): ConvBN(
            (_conv): Conv2D(384, 64, kernel_size=[1, 1], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
          )
        )
      )
    )
    (stage5_4): Sequential(
      (0): GatherAndExpansionLayer2(
        (branch_1): Sequential(
          (0): ConvBNReLU(
            (_conv): Conv2D(64, 64, kernel_size=[3, 3], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
          )
          (1): DepthwiseConvBN(
            (depthwise_conv): ConvBN(
              (_conv): Conv2D(64, 384, kernel_size=[3, 3], stride=[2, 2], padding=same, groups=64, data_format=NCHW)
              (_batch_norm): BatchNorm2D(num_features=384, momentum=0.9, epsilon=1e-05)
            )
          )
          (2): DepthwiseConvBN(
            (depthwise_conv): ConvBN(
              (_conv): Conv2D(384, 384, kernel_size=[3, 3], padding=same, groups=384, data_format=NCHW)
              (_batch_norm): BatchNorm2D(num_features=384, momentum=0.9, epsilon=1e-05)
            )
          )
          (3): ConvBN(
            (_conv): Conv2D(384, 128, kernel_size=[1, 1], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
          )
        )
        (branch_2): Sequential(
          (0): DepthwiseConvBN(
            (depthwise_conv): ConvBN(
              (_conv): Conv2D(64, 64, kernel_size=[3, 3], stride=[2, 2], padding=same, groups=64, data_format=NCHW)
              (_batch_norm): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
            )
          )
          (1): ConvBN(
            (_conv): Conv2D(64, 128, kernel_size=[1, 1], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
          )
        )
      )
      (1): GatherAndExpansionLayer1(
        (conv): Sequential(
          (0): ConvBNReLU(
            (_conv): Conv2D(128, 128, kernel_size=[3, 3], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
          )
          (1): DepthwiseConvBN(
            (depthwise_conv): ConvBN(
              (_conv): Conv2D(128, 768, kernel_size=[3, 3], padding=same, groups=128, data_format=NCHW)
              (_batch_norm): BatchNorm2D(num_features=768, momentum=0.9, epsilon=1e-05)
            )
          )
          (2): ConvBN(
            (_conv): Conv2D(768, 128, kernel_size=[1, 1], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
          )
        )
      )
      (2): GatherAndExpansionLayer1(
        (conv): Sequential(
          (0): ConvBNReLU(
            (_conv): Conv2D(128, 128, kernel_size=[3, 3], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
          )
          (1): DepthwiseConvBN(
            (depthwise_conv): ConvBN(
              (_conv): Conv2D(128, 768, kernel_size=[3, 3], padding=same, groups=128, data_format=NCHW)
              (_batch_norm): BatchNorm2D(num_features=768, momentum=0.9, epsilon=1e-05)
            )
          )
          (2): ConvBN(
            (_conv): Conv2D(768, 128, kernel_size=[1, 1], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
          )
        )
      )
      (3): GatherAndExpansionLayer1(
        (conv): Sequential(
          (0): ConvBNReLU(
            (_conv): Conv2D(128, 128, kernel_size=[3, 3], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
          )
          (1): DepthwiseConvBN(
            (depthwise_conv): ConvBN(
              (_conv): Conv2D(128, 768, kernel_size=[3, 3], padding=same, groups=128, data_format=NCHW)
              (_batch_norm): BatchNorm2D(num_features=768, momentum=0.9, epsilon=1e-05)
            )
          )
          (2): ConvBN(
            (_conv): Conv2D(768, 128, kernel_size=[1, 1], padding=same, data_format=NCHW)
            (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
          )
        )
      )
    )
    (ce): ContextEmbeddingBlock(
      (gap): AdaptiveAvgPool2D(output_size=1)
      (bn): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
      (conv_1x1): ConvBNReLU(
        (_conv): Conv2D(128, 128, kernel_size=[1, 1], padding=same, data_format=NCHW)
        (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
      )
      (conv_3x3): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
    )
  )
  (bga): BGA(
    (db_branch_keep): Sequential(
      (0): DepthwiseConvBN(
        (depthwise_conv): ConvBN(
          (_conv): Conv2D(128, 128, kernel_size=[3, 3], padding=same, groups=128, data_format=NCHW)
          (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        )
      )
      (1): Conv2D(128, 128, kernel_size=[1, 1], data_format=NCHW)
    )
    (db_branch_down): Sequential(
      (0): ConvBN(
        (_conv): Conv2D(128, 128, kernel_size=[3, 3], stride=[2, 2], padding=same, data_format=NCHW)
        (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
      )
      (1): AvgPool2D(kernel_size=3, stride=2, padding=1)
    )
    (sb_branch_keep): Sequential(
      (0): DepthwiseConvBN(
        (depthwise_conv): ConvBN(
          (_conv): Conv2D(128, 128, kernel_size=[3, 3], padding=same, groups=128, data_format=NCHW)
          (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        )
      )
      (1): Conv2D(128, 128, kernel_size=[1, 1], data_format=NCHW)
      (2): Activation(
        (act_func): Sigmoid()
      )
    )
    (sb_branch_up): ConvBN(
      (_conv): Conv2D(128, 128, kernel_size=[3, 3], padding=same, data_format=NCHW)
      (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
    )
    (conv): ConvBN(
      (_conv): Conv2D(128, 128, kernel_size=[3, 3], padding=same, data_format=NCHW)
      (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
    )
  )
  (aux_head1): SegHead(
    (conv_3x3): Sequential(
      (0): ConvBNReLU(
        (_conv): Conv2D(16, 16, kernel_size=[3, 3], padding=same, data_format=NCHW)
        (_batch_norm): BatchNorm2D(num_features=16, momentum=0.9, epsilon=1e-05)
      )
      (1): Dropout(p=0.1, axis=None, mode=upscale_in_train)
    )
    (conv_1x1): Conv2D(16, 2, kernel_size=[1, 1], data_format=NCHW)
  )
  (aux_head2): SegHead(
    (conv_3x3): Sequential(
      (0): ConvBNReLU(
        (_conv): Conv2D(32, 32, kernel_size=[3, 3], padding=same, data_format=NCHW)
        (_batch_norm): BatchNorm2D(num_features=32, momentum=0.9, epsilon=1e-05)
      )
      (1): Dropout(p=0.1, axis=None, mode=upscale_in_train)
    )
    (conv_1x1): Conv2D(32, 2, kernel_size=[1, 1], data_format=NCHW)
  )
  (aux_head3): SegHead(
    (conv_3x3): Sequential(
      (0): ConvBNReLU(
        (_conv): Conv2D(64, 64, kernel_size=[3, 3], padding=same, data_format=NCHW)
        (_batch_norm): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
      )
      (1): Dropout(p=0.1, axis=None, mode=upscale_in_train)
    )
    (conv_1x1): Conv2D(64, 2, kernel_size=[1, 1], data_format=NCHW)
  )
  (aux_head4): SegHead(
    (conv_3x3): Sequential(
      (0): ConvBNReLU(
        (_conv): Conv2D(128, 128, kernel_size=[3, 3], padding=same, data_format=NCHW)
        (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
      )
      (1): Dropout(p=0.1, axis=None, mode=upscale_in_train)
    )
    (conv_1x1): Conv2D(128, 2, kernel_size=[1, 1], data_format=NCHW)
  )
  (head): SegHead(
    (conv_3x3): Sequential(
      (0): ConvBNReLU(
        (_conv): Conv2D(128, 128, kernel_size=[3, 3], padding=same, data_format=NCHW)
        (_batch_norm): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
      )
      (1): Dropout(p=0.1, axis=None, mode=upscale_in_train)
    )
    (conv_1x1): Conv2D(128, 2, kernel_size=[1, 1], data_format=NCHW)
  )
)>
```

## 3.模型训练&模型评估测试


```python
# 配置优化器、损失函数、评估指标
loss_types = [
    CrossEntropyLoss(),  # 像素级优化
    CrossEntropyLoss(),
    CrossEntropyLoss(),
    DiceLoss(),  # 整体/局部的优化
    DiceLoss()
]
loss_coefs = [1.0] * 5
loss_dict = {'types': loss_types, 'coef': loss_coefs}

lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01, decay_steps=3000, end_lr=0.0001)
opt_choice = paddle.optimizer.Momentum(learning_rate=lr, momentum=0.9, parameters=model.parameters())
              
# 启动模型全流程训练
train(
    model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=opt_choice,
    save_dir='output',
    iters=6000,
    batch_size=8,
    save_interval=100,
    log_iters=20,
    num_workers=0,
    use_vdl=False,
    losses=loss_dict,
    keep_checkpoint_max=5
)
```
> 
> 2021-08-14 08:05:32 [INFO]	[TRAIN] epoch: 1, iter: 20/6000, loss: 2.5284, lr: 0.009937, batch_cost: 0.5262, reader_cost: 0.41899, ips: 15.2046 samples/sec | ETA 00:52:26
> 
> 82/82 [==============================] - 3s 33ms/step - batch_cost: 0.0321 - reader cost: 3.6484e-
> 
> 2021-08-14 08:06:16 [INFO]	[EVAL] #Images: 82 mIoU: 0.8397 Acc: 0.9263 Kappa: 0.8238 
> 2021-08-14 08:06:16 [INFO]	[EVAL] Class IoU: 
> [0.9005 0.7789]
> 2021-08-14 08:06:16 [INFO]	[EVAL] Class Acc: 
> [0.9187 0.9466]
> 2021-08-14 08:06:16 [INFO]	[EVAL] The model with the best validation mIoU (0.8397) was saved at iter 100.
> 
> 2021-08-14 08:32:16 [INFO]	[TRAIN] epoch: 78, iter: 3100/6000, loss: 0.7283, lr: 0.000100, batch_cost: 0.5105, reader_cost: 0.40670, ips: 15.6712 samples/sec | ETA 00:24:40
> 2021-08-14 08:32:16 [INFO]	Start evaluating (total_samples: 82, total_iters: 82)...
> 
> 82/82 [==============================] - 3s 32ms/step - batch_cost: 0.0315 - reader cost: 3.0198e-
> 
> 2021-08-14 08:32:18 [INFO]	[EVAL] #Images: 82 mIoU: 0.9361 Acc: 0.9713 Kappa: 0.9337 
> 2021-08-14 08:32:18 [INFO]	[EVAL] Class IoU: 
> [0.9589 0.9133]
> 2021-08-14 08:32:18 [INFO]	[EVAL] Class Acc: 
> [0.9762 0.9606]
> 2021-08-14 08:32:18 [INFO]	[EVAL] The model with the best validation mIoU (0.9396) was saved at iter 2200.
    

> 可以看出 大约在 2900-3000 iters的时候 loss就已经下不去了. 所以不用再继续做training了. 之后想要有更好的精度提升, 可以有以下措施:
> 
> 1. 更多更干净准确的数据
> 2. 更多的参数微调
> 3. 试一试更多其他的网络




## 4.模型预测

### 4.1 批量预测

使用model.predict接口来完成对大量数据集的批量预测。


```python
# 进行预测操作
from paddleseg.core.predict import predict

image_list,image_dir = get_image_list('./data/test')

predict(
        model,
        model_path='/home/aistudio/work/PaddleSeg/output/best_model/model.pdparams',# 模型路径
        transforms= T.Compose(val_transforms), #记得用 T.Compose() 不然会报错
        image_list=image_list, #list,待预测的图像路径列表。
        image_dir=image_dir, #str，待预测的图片所在目录
        save_dir='/home/aistudio/work/PaddleSeg/output/predict_pngs' #str，结果输出路径
)
```


![](https://ai-studio-static-online.cdn.bcebos.com/157433c09c0b4669afe896c0902e500fd32b31660d5c4703984c87c37933d021)

![](https://ai-studio-static-online.cdn.bcebos.com/a3b07380bf7b4e7da09201ff666fb045512509c75952485c99b6fd9126d154dc)

![](https://ai-studio-static-online.cdn.bcebos.com/4fac172310b94d4797579e167cfcddb3a91cd11efbef4f749ad4103a33ab36ed)

可以看出, 还是有些部分没有标注准确, 不过整体来说 还是挺满意的了.

而且模型下载下来后小于10MB, 很适合用在算力有限的Edge Device上面,完美^^


让我们来看一下室外的效果

<div style="width:90%; margin:10px; display: flex; justify-content:center">
   <div style="width:40%; float:left; margin:10px;">
      <b>Image with maske</b>
      <img src="https://ai-studio-static-online.cdn.bcebos.com/51e7f9339def4dd3bbf4fc7fccff087e6b7a638f7ebe444ea9c508cce8e078e2" />
  </div>

  <div style="width:40%; float:left; margin:10px;">
      <b>Just the maske</b>
      <img src="https://ai-studio-static-online.cdn.bcebos.com/74c85a984e274ea5ab9f34a7fe8ad6af0fb86c3ed32a4b16aa3ffc89c8c1a573" />
  </div>
</div>


# 四、总结与升华

> PaddleSeg心得：
> 
> 最一开始注意到的是 num_classes, 因为我的这个模型只需要把路面识别出来 所以我一位这里用1就好了. 但是发现效果很不理想.如果想了想 其实 地面是一个类别, 那么所有非地面的就是另外一个类别.所以改成了2. 这和我之前做的 鸟类识别模型遇到的是一个问题.
> 
> 之后注意到的是 loss_types, 这里BiSeNetV2 需要5个损失函数, 所以一开始选择了 5个都是 CEL(CrossEntropyLoss), 之后想用BCE(BinaryCrossEntropy)但是想到 图片里有大量的天空,人物,建筑物,只有小部分地面,这样就造成了 highly unbalanced label distribution.
> 
> 再来谈谈CEL, 在CEL里面 损失是通过计算平均每一个像素的损失值来获得的,而且每个像素的损失值都是分散计算的.这样做在计算每个像素的损失值的时候就完全没有考虑到它周围相邻像素值的情况.这就导致了CEL只注重微观视野,而忽略了整体,所以对于图像级别的预测效果将不会太好.(reference: Understanding Dice Loss for Crisp Boundary Detection)
> 
> 混合进了2个DiceLoss(DL). 相较于CEL的"近视", DL在计算损失值的时候可以做到"具体"与"宏观"兼备,因为大幅度底稿准确率. 具体原理也请看上面的reference.
> 
> lr的decay_steps我选择了和train()里面的iters是一样的.(虽然用了3000,那是因为之前iters是用3000,后来加到6000忘记改变这个decay_steps啦....)
> 
> 由于怕算法陷入local maxima而跑不出来,导致找不到global maxima 所以用了 momentum.
> 
> batch size(BS)是我这次学到最大的知识点之一. 之前对BS的了解就是它指定了每一批训练时用到的图片的数量.感觉这个和训练精度没有关系,只和训练速度快慢有关系. 但是听完解释之后 才知道如果BS选的太大了 那么即使每次做修正的时候对错误的修正也不会太强, 但是选小了的话,那么会有很大影响 不一定是好事. 所以这个BS的选择真的是要 好好炼丹...(这里要特别感谢 助教:红白黑, 的细心解释.点赞!!!)
> 

> PaddleHub心得
> 
> 在构建URL格式的时候是个大坑....当你运行了
> `hub serving start -m bisenetv2_predict`
> 命令行后 它并不会告诉你正确的endpoint的URL
> 
> 它只会告诉你 'http://0.0.0.0:8866/'.....
> 
> 而且官方文档上会说URL的格式是如下: base_url+/predict/Category/module_name
> 
> 这里Category的值如下 Category: text, image
> 
> Module Name: 就是你自己给你的module起得名字 @moduleinfo(name="bisenetv2_predict",...)
> 但是!! 你加了Category它反而会报错
> JSONDecodeError: Expecting value: line 1 column 1 (char 0)
> 
> 真是深远大坑啊~!! 我的七夕一整天就是在这个坑里渡过了.....
> 
> 所以真正正确的URL格式是: baseurl+/predict/modulename
> 
> 对于从强类型程序转到若类型程序的developers一点心得: Python不像C#一样是强类型语言,所以好用.但是....却有时让我们不知道传参的时候 互相之间传的参数应该是什么类型,这个也是花了我很多时间的地方.所以我在下面的代码种都标注出了每个重要关键点变量的参数类型,方便大家在debug的时候发现问题.

# 个人简介

按规矩来链接一下我的AI Studio Link:
https://aistudio.baidu.com/aistudio/personalcenter/thirdview/710090