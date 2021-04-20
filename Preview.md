# GAN的初体验

大家好，欢迎来到新手入门课程，本次课程将会带领大家进入深度学习中最有趣的领域--GAN, Generative Adversarial Networks（生成式对抗网络）的学习，帮助大家掌握基础理论知识，为后续的课程学习打下夯实的基础。

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/4f70994aa198469091fa25efc60d6f85405ddac2c4b541bf968ab7483b936d50' width='700'/>
</div>



## GAN的基础概念


GAN被“卷积网络之父”Yann LeCun（杨立昆）誉为「过去十年计算机科学领域最有趣的想法之一」，是近年来火遍全网，AI研究者最为关注的深度学习技术方向之一。

生成式对抗网络，简称GAN，是一种近年来大热的深度学习模型，该模型由两个基础神经网络即**生成器神经网络（Generator Neural Network）** 和**判别器神经网络（Discriminator Neural Network）** 所组成，其中一个用于生成内容，另一个则用于判别生成的内容。

GAN受博弈论中的零和博弈启发，将生成问题视作**判别器和生成器这两个网络的对抗和博弈**：生成器从给定噪声中（一般是指均匀分布或者正态分布）产生合成数据，判别器分辨生成器的的输出和真实数据。前者试图产生更接近真实的数据，相应地，后者试图更完美地分辨真实数据与生成数据。**由此，两个网络在对抗中进步，在进步后继续对抗，由生成式网络得的数据也就越来越完美，逼近真实数据，从而可以生成想要得到的数据（图片、序列、视频等）。**

## GAN的花样“玩”法

首先，带大家领略下GAN的魅力，为什么称其为最有趣的应用呢？请看下面GAN的才艺展示！

### 图像/视频领域

**Text to Image Generation：根据文字描述生成对应图像**

[参考论文](https://arxiv.org/abs/1605.05396)

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/2f165cff92ff42d6b7c3714e0eb051831c529355816048cbacab2a5cf1eb755f' width='500'/>
</div>

**Image to Image Translation：图像到图像的转化**

如将黑白图像转换为彩色图像、将航拍图像变成地图形式、将白天的照片转换为黑夜的照片、甚至可以根据物体的轮廓、边缘信息，来生成实体包包的形式。

[参考论文](https://arxiv.org/abs/1611.07004)

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/2c8c515fc05b45e3ac9fdec34aaa93cdbf9c032304b3407c90f70f76adbbee9f' width='500'/>
</div>

**Super resolution：图片/视频超分**

[参考论文](https://arxiv.org/abs/1609.04802)

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/7b1a0797eaec41d5a9da79d6b61abf68094d4a4095104fd7b42ce78aca7cf02b' width='800'/>
</div>

**Photo to Cartoon：人物/实景动漫化**

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/b1e71d88d89c4d37800046004b21f740809f40c10c204c429f9f1c096dbab446' width='800'/>
</div>

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/6232aa99120246c183c88354f170d944fc04417fc6514787be0152c0e5706542' width='800'/>
</div>

**Motion Driving：人脸表情动作迁移**

涉及模型：[First Order Motion](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/tutorials/motion_driving.md)

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/b514caa0ad52409eb315143681238105a5ef8a9c29574a68b87975c8efa266ac' width='500'/>
</div>


**Lip Sythesis：唇形动作合成**

涉及模型：[Wav2lip](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/tutorials/wav2lip.md)

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/16d0b24fdc5c451395b3b308cf27b59bd4b024366b41457dbb80d0105f938849' width='500'/>
</div>

#### 以上应用均在[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/README_cn.md)中有完整实现及教程。

### 人机交互领域

Santana等人实现了利用GAN 的辅助自动驾驶。首先，生成与真实交通场景图像分布一致的图像，然后，训练一个基于循环神经网络的转移模型来预测下一个交通场景。

另外，GAN还可以用于对抗神经机器翻译，将神经机器翻译（neural machine translation, NMT）作为GAN 的生成器，采用策略梯度方法训练判别器，通过最小化人类翻译和神经机器翻译的差别生成高质量的翻译。

### 小结
GAN最直接的应用在于数据的生成，也就是通过GAN的建模能力生成图像、语音、文字、视频等等。目前，GAN最成功的应用领域主要是计算机视觉，包括图像、视频的生成，如图像翻译、图像上色、图像修复、视频生成等。此外GAN在自然语言处理，人机交互领域也略有拓展和应用。

近期，GAN在娱乐方向的应用一直在不断增长，比如人脸表情迁移、肢体动作迁移、动漫化等等，皆为娱乐界提供了人工智能的助力！

## GAN的原理

举个简单的例子，我们将成内容的网络称为G（Generator），将鉴别内容的网络称为D（Discriminator），下图中枯叶蝶进化的例子可以很好的说明GAN的工作原理。（引自李宏毅老师GAN课程）

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/fae74179f6b74f6499124b2f0debe1bfabc0412374d24472811c3486f58e2e37' width='500'/>
</div>

图中的枯叶蝶扮演Generator的角色，相应的其天敌之一的麻雀扮演Discriminator的角色。起初，枯叶蝶的翅膀与其他的蝴蝶别无二致，都是色彩斑斓；

* 第一阶段：麻雀为了识别并捕杀蝴蝶升级自己的判别标准为非棕色翅膀；

* 第二阶段：为了躲避麻雀，枯叶蝶的翅膀进化为棕色；

* 第三阶段：麻雀更加聪明，识别枯叶蝶的标准升级为所看到的物体是否具有纹路；

* 第四阶段：枯叶蝶的翅膀进化出纹路更像枯叶；

* ……

如此不断的进行下去，伴随着枯叶蝶的不断进化和麻雀判别标准的不断升级，二者不断地相互博弈，最终导致的结果就是枯叶蝶的翅膀（输出）无限接近于真实的枯叶（真实物体）。

### 用数学语言描述上述原理

* G是一个生成图片的网络，它接收一个随机的噪声z，通过这个噪声生成图片，记做G(z)。
* D是一个判别网络，判别一张图片是不是“真实的”。它的输入参数是x，x代表一张图片，输出D（x）代表x为真实图片的概率，如果为1，就代表100%是真实的图片，而输出为0，就代表不可能是真实的图片。

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/a587a4dca23f42a5b9ece7ab37f0deb5bab1fb4c112b4c198891e3b9936f4bc8' width='500'/>
</div>

上图中的标记符号：

* Pdata(x) → 真实数据的分布
* X → pdata(x)的样本（真实图片）
* P(z) →生成器的分布
* Z → p(z)的样本（噪声）

在训练过程中，生成网络G的目标就是尽量生成真实的图片去欺骗判别网络D。而D的目标就是尽量把G生成的图片和真实的图片分别开来。这样，G和D构成了一个动态的“博弈过程”。

最后博弈的结果是什么？在最理想的状态下，G可以生成足以“以假乱真”的图片G(z)。对于D来说，它难以判定G生成的图片究竟是不是真实的，因此D(G(z)) = 0.5。

用公式表示如下：

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/e7118494404a45f0b3e064450346902164c4199d38c241629a7dcd218c062efe' width='500'/>
</div>

整个式子由两项构成。X表示真实图片，Z表示输入G网络的噪声，而G(z)表示G网络生成的图片。D(x)表示D网络判断真实图片是否真实的概率（因为x就是真实的，所以对于D来说，这个值越接近1越好）。而D(G(z))是D网络判断G生成的图片的是否真实的概率。

G的目的：上面提到过，D(G(z))是D网络判断G生成的图片是否真实的概率，G应该希望自己生成的图片“越接近真实越好”。也就是说，G希望D(G(z))尽可能得大，这时V(D, G)会变小。因此我们看到式子的最前面的记号是（min_G）。

D的目的：D的能力越强，D(x)应该越大，D(G(x))应该越小。这时V(D,G)会变大。因此式子对于D来说是求最大(max_D)。

最终通过不断的训练，生成的图片会相当真实。（如下图）

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/ee508c5b78e44528a98541b696a549f6b1600e34293e416c9119571a65559470' width='500'/>
</div>



## GAN的发展历史

GAN最早是由Ian J. Goodfellow等人于2014年10月提出的，他的《Generative Adversarial Nets》可以说是这个领域的开山之作，论文一经发表，就引起了热议。而随着GAN在理论与模型上的高速发展，它在计算机视觉、自然语言处理、人机交互等领域有着越来越深入的应用，并不断向着其它领域继续延伸。

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/7d9da15cd7ac4643a557af811c1ee858b6ea02915d854fd8a9c5bb061afbe59b' width='500'/>
</div>
图自李宏毅老师的GAN课程



### **下面将按照时间顺序，简单介绍GAN的演进历史中的代表性网络**

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/2bc2c3fab7cc4e38882df932ea7c9cb09f34a8eb2da040cba47cccdb89276ed8' width='500'/>
</div>

**DCGAN**

顾名思义，DCGAN[3]主要讨论 CNN 与 GAN 如何结合使用并给出了一系列建议。由于卷积神经网络(Convolutional neural network, CNN)比MLP有更强的拟合与表达能力，并在判别式模型中取得了很大的成果。因此，Alec等人将CNN引入生成器和判别器，称作深度卷积对抗神经网络（Deep Convolutional GAN, DCGAN）。另外还讨论了 GAN 特征的可视化、潜在空间插值等问题。

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/7ac07e9e0b28408395236d8f5c7aa4d00435ffb6c86f495db1129879f4264f1b' width='500'/>
</div>

DCGAN生成的动漫头像：

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/711e56f8c0c84449a0331ec1334b0daef54e2a0e37d6465f914c8dca087b64cd' width='500'/>
</div>

**ImprovedGAN**

Ian Goodfellow 等人[4]提供了诸多训练稳定 GAN 的建议，包括特征匹配、mini-batch 识别、历史平均、单边标签平滑以及虚拟批标准化等技巧。讨论了 GAN 不稳定性的最佳假设。

**PACGAN**

PACGAN[5]讨论的是的如何分析 model collapse，以及提出了 PAC 判别器的方法用于解决 model collapse。思想其实就是将判别器的输入改成多个样本，这样判别器可以同时看到多个样本可以从一定程度上防止 model collapse。

**WGAN**

WGAN[6]首先从理论上分析了原始 GAN 模型存在的训练不稳定、生成器和判别器的 loss 无法只是训练进程、生成样本缺乏多样性等问题，并通过改进算法流程针对性的给出了改进要点。

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/74dcd22ed8454781bf9f980cd7a4c42aad32fbf44d3a48128d820fc69088a2b1' width='500'/>
</div>


**CycleGAN**

CycleGAN[7]讨论的是 image2image 的转换问题，提出了 Cycle consistency loss 来处理缺乏成对训练样本来做 image2image 的转换问题。Cycle Consistency Loss 背后的主要想法，图片 A 转化得到图片 B，再从图片 B 转换得到图片 A’，那么图片 A 和图片 A’应该是图一张图片。

**Vid2Vid**

Vid2Vid[8]通过在生成器中加入光流约束，判别器中加入光流信息以及对前景和背景分别建模重点解决了视频转换过程中前后帧图像的不一致性问题。

**PGGAN**

PGGAN[9]创造性地提出了以一种渐进增大（Progressive growing）的方式训练 GAN，利用逐渐增大的 PGGAN 网络实现了效果令人惊叹的生成图像。“Progressive Growing” 指的是先训练 4x4 的网络，然后训练 8x8，不断增大，最终达到 1024x1024。这既加快了训练速度，又大大稳定了训练速度，并且生成的图像质量非常高。

**StackGAN**

StackGAN[10]是由文本生成图像，StackGAN 模型与 PGGAN 工作的原理很像，StackGAN 首先输出分辨率为 64×64 的图像，然后将其作为先验信息生成一个 256×256 分辨率的图像。

**BigGAN**

BigGAN[11]模型是基于 ImageNet 生成图像质量最高的模型之一。该模型很难在本地机器上实现，而且 有许多组件，如 Self-Attention、 Spectral Normalization 和带有投影鉴别器的 cGAN 等。

**StyleGAN**

StyleGAN[12]应该是截至目前最复杂的 GAN 模型，该模型借鉴了一种称为自适应实例标准化 (AdaIN) 的机制来控制潜在空间向量 z。虽然很难自己实现一个 StyleGAN，但是它提供了很多有趣的想法。

**参考文献**

[1] [Must-Read Papers on GANs/ 必读！生成对抗网络GAN论文TOP 10](https://towardsdatascience.com/must-read-papers-on-gans-b665bbae3317)

[2] Generative Adversarial Networks

[3] Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

[4] Improved Techniques for Training GANs

[5] PacGAN: The power of two samples in generative adversarial networks

[6] Wasserstein GAN

[7] Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

[8] Video-to-Video Synthesis

[9] [深度推荐系统](https://zhuanlan.zhihu.com/p/70033932)


## 总结

在新手课程中，我们学习了GAN的基础概念、发展历史、原理及应用，希望同学们能对GAN有一定的基础认知，在后续的课程里，我们也会结合原理和应用，深入讲解GAN的知识，让大家通过7日课程的学习，基本掌握GAN的使用。

## 相关模型资料

以下是课程中所涉及到的所有模型简介、代码链接及论文。

*注意：实际代码请参考Config文件进行配置。

### Wasserstein GAN

论文：[Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)

简介：本文从理论上分析了原始 GAN 模型存在的训练不稳定、生成器和判别器的 loss 无法只是训练进程、生成样本缺乏多样性等问题，并通过改进算法流程针对性的给出了改进要点。

代码链接：https://github.com/PaddlePaddle/PaddleGAN/blob/develop/configs/wgan_mnist.yaml

### DCGAN

论文：[UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1511.06434.pdf)

简介：由于卷积神经网络(Convolutional neural network, CNN)比MLP有更强的拟合与表达能力，并在判别式模型中取得了很大的成果。因此，本文将CNN引入生成器和判别器，称作深度卷积对抗神经网络（Deep Convolutional GAN, DCGAN）。

代码链接：https://github.com/PaddlePaddle/PaddleGAN/blob/develop/configs/dcgan_mnist.yaml

### Least Squares GAN

论文：[Least Squares Generative Adversarial Networks](https://arxiv.org/pdf/1611.04076.pdf)

简介：本文主要将交叉熵损失函数换做了最小二乘损失函数，改善了传统 GAN 生成的图片质量不高，且训练过程十分不稳定的问题。

### Progressive Growing of GAN

论文：[PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION](https://arxiv.org/pdf/1710.10196.pdf)

简介：本文提出了一种用来训练生成对抗网络的新方法：渐进式地增加生成器和判别器的规模，同时，提出了一种提高生成图像多样性的方法以及给出一种新的关于图像生成质量和多样性的评价指标。

### StyleGAN

论文：[A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf)

简介：本文是NVIDIA继ProGAN之后提出的新的生成网络，其主要通过分别修改每一层级的输入，在不影响其他层级的情况下，来控制该层级所表示的视觉特征。 这些特征可以是粗的特征（如姿势、脸型等），也可以是一些细节特征（如瞳色、发色等）。

### StyleGAN2

论文：[Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/pdf/1912.04958.pdf)

简介：本文主要解决StyleGAN生成图像伪影的同时还能得到细节更好的高质量图像。新的改进方案也不会带来更高的计算成本。不管是在现有的分布质量指标上，还是在人所感知的图像质量上，新提出的模型都实现了无条件图像建模任务上新的 SOTA。

代码链接：https://github.com/PaddlePaddle/PaddleGAN/blob/develop/configs/stylegan_v2_256_ffhq.yaml

### Conditional GAN

论文：[Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)

简介：本文提出在利用 GAN（对抗网络）的方法时，在生成模型G和判别模型D中都加入条件信息来引导模型的训练，并将这种方法应用于跨模态问题，例如图像自动标注等。

代码链接：https://github.com/PaddlePaddle/PaddleGAN/blob/develop/configs/cond_dcgan_mnist.yaml

### CycleGAN

论文：[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)

简介：CycleGAN本质上是两个镜像对称的GAN，构成了一个环形网络。 两个GAN共享两个生成器，并各自带一个判别器，即共有两个判别器和两个生成器。 一个单向GAN两个loss，两个即共四个loss。 可以实现无配对的两个图片集的训练是CycleGAN与Pixel2Pixel相比的一个典型优点。

代码链接：https://github.com/PaddlePaddle/PaddleGAN/blob/develop/configs/cyclegan_horse2zebra.yaml

### Pix2Pix

论文：[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)

简介：本文在GAN的基础上提供一个通用方法，完成成对的图像转换。

代码链接：https://github.com/PaddlePaddle/PaddleGAN/blob/develop/configs/pix2pix_cityscapes_2gpus.yaml

### U-GAT-IT

论文：[U-GAT-IT: UNSUPERVISED GENERATIVE ATTENTIONAL NETWORKS WITH ADAPTIVE LAYERINSTANCE NORMALIZATION FOR IMAGE-TO-IMAGE TRANSLATION](https://arxiv.org/pdf/1907.10830.pdf)

简介：本文主要研究无监督的image-to-image translation。在风格转换中引入了注意力模块，并且提出了一种新的可学习的normalization方法。注意力模块根据辅助分类器获得的attention map，使得模型聚能更好地区分源域和目标域的重要区域。同时，AdaLIN（自适应层实例归一化）帮助注意力指导模型根据所学习的数据集灵活地控制形状和纹理的变化量。

代码链接：https://github.com/PaddlePaddle/PaddleGAN/blob/develop/configs/ugatit_selfie2anime_light.yaml

### Super Resolution GAN

论文：[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802.pdf)

简介：本文主要讲解如何利用卷积神经网络实现单影像的超分辨率，其瓶颈仍在于如何恢复图像的细微纹理信息。

### Enhanced Super Resolution GAN

论文：[ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/pdf/1809.00219.pdf)

简介：本文在SRGAN的基础上进行了改进，包括改进网络的结构，判决器的判决形式，以及更换了一个用于计算感知域损失的预训练网络。

代码链接：https://github.com/PaddlePaddle/PaddleGAN/blob/develop/configs/esrgan_x4_div2k.yaml

### Residual Channel Attention Networks（RCAN）

论文：[Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://arxiv.org/pdf/1807.02758.pdf)

简介：本文提出了一个深度残差通道注意力网络（RCAN）解决过深的网络难以训练、网络的表示能力较弱的问题。

### EDVR

论文：[EDVR: Video Restoration with Enhanced Deformable Convolutional Networks](https://arxiv.org/pdf/1905.02716.pdf)

简介：本文主要介绍基于可形变卷积的视频恢复、去模糊、超分的网络。

代码链接：https://github.com/PaddlePaddle/PaddleGAN/blob/develop/configs/edvr.yaml

### First Order Motion

论文：[First Order Motion Model for Image Animation](https://arxiv.org/pdf/2003.00196.pdf)

简介：本文介绍的是image animation，给定一张源图片，给定一个驱动视频，生成一段视频，其中主角是源图片，动作是驱动视频中的动作。如下图所示，源图像通常包含一个主体，驱动视频包含一系列动作。

### Wav2lip

论文：[A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild](https://arxiv.org/pdf/2008.10010.pdf)

简介：本文主要介绍如何将任意说话的面部视频与任意语音进行唇形同步。

代码链接：https://github.com/PaddlePaddle/PaddleGAN/blob/develop/configs/wav2lip.yaml

### 参考资料

[1] [一文看懂生成式对抗网络GANs：介绍指南及前景展望](http://36kr.com/p/5086889.html)

[2] [PaddleGAN GitHub项目](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/README_cn.md)

[3]  [宋代诗人‘开口’念诗、蒙娜丽莎‘唱’rap-PaddleGAN唇形合成的应用](https://aistudio.baidu.com/aistudio/projectdetail/1463208)

[6] [老北京城影像修复-PaddleGAN上色、超分、插帧的应用](https://aistudio.baidu.com/aistudio/projectdetail/1161285)

[7] [一键生成多人版‘脸部动作迁移🐜’-PaddleGAN表情动作迁移（First Order Motion）顶级应用](https://aistudio.baidu.com/aistudio/projectdetail/1603391)

[8] [生成式对抗网络学习笔记](https://www.cnblogs.com/dereen/p/gan.html)
