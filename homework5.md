# 七日课 大作业：实现超分

经过这几天的学习，相信大家对GAN已经有了一定的了解了，也在前面的作业中体验过GAN的一些应用了。那现在大家是不是想要升级一下难度，自己动手来训练一个模型呢？

需要自己动手训练的大作业来啦，大作业内容为基于PaddleGAN中的超分模型，实现卡通画超分。


## 安装PaddleGAN

PaddleGAN的安装目前支持Clone GitHub和Gitee两种方式：


```python
# 安装ppgan
# 当前目录在: /home/aistudio/, 这个目录也是左边文件和文件夹所在的目录
# 克隆最新的PaddleGAN仓库到当前目录
# !git clone https://github.com/PaddlePaddle/PaddleGAN.git
# 如果从github下载慢可以从gitee clone：
#!git clone https://gitee.com/paddlepaddle/PaddleGAN.git
# 安装Paddle GAN
%cd PaddleGAN/
#!pip config set global.index-url https://pypi.douban.com/simple
!pip install -v -e .
```

    /home/aistudio/PaddleGAN
    Created temporary directory: /tmp/pip-ephem-wheel-cache-5qu1zgf6
    Created temporary directory: /tmp/pip-req-tracker-1_1uogbh
    Created requirements tracker '/tmp/pip-req-tracker-1_1uogbh'
    Created temporary directory: /tmp/pip-install-0ag9ytn8
    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Obtaining file:///home/aistudio/PaddleGAN
      Added file:///home/aistudio/PaddleGAN to build tracker '/tmp/pip-req-tracker-1_1uogbh'
        Running setup.py (path:/home/aistudio/PaddleGAN/setup.py) egg_info for package from file:///home/aistudio/PaddleGAN
        Running command python setup.py egg_info
        running egg_info
        writing ppgan.egg-info/PKG-INFO
        writing dependency_links to ppgan.egg-info/dependency_links.txt
        writing entry points to ppgan.egg-info/entry_points.txt
        writing requirements to ppgan.egg-info/requires.txt
        writing top-level names to ppgan.egg-info/top_level.txt
        reading manifest file 'ppgan.egg-info/SOURCES.txt'
        writing manifest file 'ppgan.egg-info/SOURCES.txt'
      Source in /home/aistudio/PaddleGAN has version 0.1.0, which satisfies requirement ppgan==0.1.0 from file:///home/aistudio/PaddleGAN
      Removed ppgan==0.1.0 from file:///home/aistudio/PaddleGAN from build tracker '/tmp/pip-req-tracker-1_1uogbh'
    Requirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ppgan==0.1.0) (4.36.1)
    Requirement already satisfied: PyYAML>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ppgan==0.1.0) (5.1.2)
    Requirement already satisfied: scikit-image>=0.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ppgan==0.1.0) (0.18.1)
    Requirement already satisfied: scipy>=1.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ppgan==0.1.0) (1.3.0)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ppgan==0.1.0) (4.1.1.26)
    Requirement already satisfied: imageio-ffmpeg in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ppgan==0.1.0) (0.3.0)
    Requirement already satisfied: librosa==0.7.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ppgan==0.1.0) (0.7.0)
    Requirement already satisfied: numba==0.48 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ppgan==0.1.0) (0.48.0)
    Requirement already satisfied: easydict in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ppgan==0.1.0) (1.9)
    Requirement already satisfied: tifffile>=2019.7.26 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->ppgan==0.1.0) (2021.4.8)
    Requirement already satisfied: numpy>=1.16.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->ppgan==0.1.0) (1.20.2)
    Requirement already satisfied: matplotlib!=3.0.0,>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->ppgan==0.1.0) (2.2.3)
    Requirement already satisfied: networkx>=2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->ppgan==0.1.0) (2.4)
    Requirement already satisfied: PyWavelets>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->ppgan==0.1.0) (1.1.1)
    Requirement already satisfied: imageio>=2.3.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->ppgan==0.1.0) (2.6.1)
    Requirement already satisfied: pillow!=7.1.0,!=7.1.1,>=4.3.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->ppgan==0.1.0) (7.1.2)
    Requirement already satisfied: six>=1.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->ppgan==0.1.0) (1.15.0)
    Requirement already satisfied: audioread>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->ppgan==0.1.0) (2.1.8)
    Requirement already satisfied: joblib>=0.12 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->ppgan==0.1.0) (0.14.1)
    Requirement already satisfied: resampy>=0.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->ppgan==0.1.0) (0.2.2)
    Requirement already satisfied: decorator>=3.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->ppgan==0.1.0) (4.4.0)
    Requirement already satisfied: scikit-learn!=0.19.0,>=0.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->ppgan==0.1.0) (0.22.1)
    Requirement already satisfied: soundfile>=0.9.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->ppgan==0.1.0) (0.10.3.post1)
    Requirement already satisfied: llvmlite<0.32.0,>=0.31.0dev0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from numba==0.48->ppgan==0.1.0) (0.31.0)
    Requirement already satisfied: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from numba==0.48->ppgan==0.1.0) (41.4.0)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.14.0->ppgan==0.1.0) (0.10.0)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.14.0->ppgan==0.1.0) (2.8.0)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.14.0->ppgan==0.1.0) (2019.3)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.14.0->ppgan==0.1.0) (1.1.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.14.0->ppgan==0.1.0) (2.4.2)
    Requirement already satisfied: cffi>=1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from soundfile>=0.9.0->librosa==0.7.0->ppgan==0.1.0) (1.14.0)
    Requirement already satisfied: pycparser in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from cffi>=1.0->soundfile>=0.9.0->librosa==0.7.0->ppgan==0.1.0) (2.19)
    Installing collected packages: ppgan
      Found existing installation: ppgan 0.1.0
        Uninstalling ppgan-0.1.0:
          Created temporary directory: /tmp/pip-uninstall-lr76cc2o
          Removing file or directory /opt/conda/envs/python35-paddle120-env/bin/paddlegan
          Created temporary directory: /tmp/pip-uninstall-psdy_g83
          Removing file or directory /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ppgan.egg-link
          Removing pth entries from /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/easy-install.pth:
          Removing entry: /home/aistudio/PaddleGAN
          Successfully uninstalled ppgan-0.1.0
      Running setup.py develop for ppgan
        Running command /opt/conda/envs/python35-paddle120-env/bin/python -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'/home/aistudio/PaddleGAN/setup.py'"'"'; __file__='"'"'/home/aistudio/PaddleGAN/setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' develop --no-deps
        running develop
        running egg_info
        writing ppgan.egg-info/PKG-INFO
        writing dependency_links to ppgan.egg-info/dependency_links.txt
        writing entry points to ppgan.egg-info/entry_points.txt
        writing requirements to ppgan.egg-info/requires.txt
        writing top-level names to ppgan.egg-info/top_level.txt
        reading manifest file 'ppgan.egg-info/SOURCES.txt'
        writing manifest file 'ppgan.egg-info/SOURCES.txt'
        running build_ext
        Creating /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ppgan.egg-link (link to .)
        Adding ppgan 0.1.0 to easy-install.pth file
        Installing paddlegan script to /opt/conda/envs/python35-paddle120-env/bin
    
        Installed /home/aistudio/PaddleGAN
    Successfully installed ppgan
    Cleaning up...
    Removed build tracker '/tmp/pip-req-tracker-1_1uogbh'


### 数据准备
我们为大家准备了处理好的超分数据集[卡通画超分数据集](https://aistudio.baidu.com/aistudio/datasetdetail/80790)


```python
# 回到/home/aistudio/下
%cd /home/aistudio
# 解压数据
!unzip -q data/data80790/animeSR.zip -d data/
# 将解压后的数据链接到` /home/aistudio/PaddleGAN/data `目录下
!mv data/animeSR PaddleGAN/data/
```


### 数据集的组成形式
```
    PaddleGAN
      ├── data
          ├── animeSR
                ├── train
                ├── train_X4
                ├── test
                └── test_X4
  ```

训练数据集包括400张卡通画，其中``` train ```中是高分辨率图像，``` train_X4 ```中是对应的4倍缩小的低分辨率图像。测试数据集包括20张卡通画，其中``` test ```中是高分辨率图像，``` test_X4 ```中是对应的4倍缩小的低分辨率图像。


```python
%cd /home/aistudio
```

    /home/aistudio


### 数据可视化


```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# 训练数据统计
train_names = os.listdir('PaddleGAN/data/animeSR/train')
print(f'训练集数据量: {len(train_names)}')

# 测试数据统计
test_names = os.listdir('PaddleGAN/data/animeSR/test')
print(f'测试集数据量: {len(test_names)}')

# 训练数据可视化
img = cv2.imread('PaddleGAN/data/animeSR/train/Anime_1.jpg')
img = img[:,:,::-1]
plt.figure()
plt.imshow(img)
plt.show()
```

    训练集数据量: 400
    测试集数据量: 20



![png](images/output_8_1.png)


### 选择超分模型

PaddleGAN中提供的超分模型包括RealSR, ESRGAN, LESRCNN, DRN等，详情可见[超分模型](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/super_resolution.md)。

接下来以ESRGAN为例进行演示。

### 修改配置文件
 所有模型的配置文件均在``` /home/aistudio/PaddleGAN/configs ```目录下。
 
 找到你需要的模型的配置文件，修改模型参数，一般修改迭代次数，num_workers，batch_size以及数据集路径。有能力的同学也可以尝试修改其他参数，或者基于现有模型进行二次开发，模型代码在``` /home/aistudio/PaddleGAN/ppgan/models ```目录下。
 
 以ESRGAN为例，这里将将配置文件``esrgan_psnr_x4_div2k.yaml``中的
 
 参数``total_iters``改为50000
 
 参数``dataset：train：num_workers``改为12
 
 参数``dataset：train：batch_size``改为48
 
 参数``dataset：train：gt_folder``改为data/animeSR/train
 
 参数``dataset：train：lq_folder``改为data/animeSR/train_X4
 
 参数``dataset：test：gt_folder``改为data/animeSR/test
 
 参数``dataset：test：lq_folder``改为data/animeSR/test_X4
 

### 训练模型
以ESRGAN为例，运行以下代码训练ESRGAN模型。

如果希望使用其他模型训练，可以修改配置文件名字。


```python
%cd /home/aistudio/PaddleGAN/
# !python -u tools/main.py --config-file configs/esrgan_psnr_x4_div2k.yaml
!python -u tools/main.py --config-file configs/drn_psnr_x4_div2k_hw4.yaml
```

    /home/aistudio/PaddleGAN
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/scipy/fftpack/__init__.py:103: DeprecationWarning: The module numpy.dual is deprecated.  Instead of using dual, use the functions directly from numpy or scipy.
      from numpy.dual import register_func
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/scipy/special/orthogonal.py:81: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around, int,
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/scipy/io/matlab/mio5.py:98: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      from .mio5_utils import VarReader5
    [04/19 01:13:57] ppgan INFO: Configs: {'total_iters': 60000, 'output_dir': 'output_dir/drn_psnr_x4_div2k_hw4-2021-04-19-01-13', 'min_max': (0.0, 255.0), 'model': {'name': 'DRN', 'generator': {'name': 'DRNGenerator', 'scale': (2, 4), 'n_blocks': 30, 'n_feats': 16, 'n_colors': 3, 'rgb_range': 255, 'negval': 0.2}, 'pixel_criterion': {'name': 'L1Loss'}}, 'dataset': {'train': {'name': 'SRDataset', 'gt_folder': 'data/animeSR/train', 'lq_folder': 'data/animeSR/train_X4', 'num_workers': 2, 'batch_size': 4, 'scale': 4, 'preprocess': [{'name': 'LoadImageFromFile', 'key': 'lq'}, {'name': 'LoadImageFromFile', 'key': 'gt'}, {'name': 'Transforms', 'input_keys': ['lq', 'gt'], 'output_keys': ['lq', 'lqx2', 'gt'], 'pipeline': [{'name': 'SRPairedRandomCrop', 'gt_patch_size': 384, 'scale': 4, 'scale_list': True, 'keys': ['image', 'image']}, {'name': 'PairedRandomHorizontalFlip', 'keys': ['image', 'image', 'image']}, {'name': 'PairedRandomVerticalFlip', 'keys': ['image', 'image', 'image']}, {'name': 'PairedRandomTransposeHW', 'keys': ['image', 'image', 'image']}, {'name': 'Transpose', 'keys': ['image', 'image', 'image']}, {'name': 'Normalize', 'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0], 'keys': ['image', 'image', 'image']}]}]}, 'test': {'name': 'SRDataset', 'gt_folder': 'data/animeSR/test', 'lq_folder': 'data/animeSR/test_X4', 'scale': 4, 'preprocess': [{'name': 'LoadImageFromFile', 'key': 'lq'}, {'name': 'LoadImageFromFile', 'key': 'gt'}, {'name': 'Transforms', 'input_keys': ['lq', 'gt'], 'pipeline': [{'name': 'Transpose', 'keys': ['image', 'image']}, {'name': 'Normalize', 'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0], 'keys': ['image', 'image']}]}]}}, 'lr_scheduler': {'name': 'CosineAnnealingRestartLR', 'learning_rate': 0.0001, 'periods': [60000], 'restart_weights': [1], 'eta_min': 1e-07}, 'optimizer': {'optimG': {'name': 'Adam', 'net_names': ['generator'], 'weight_decay': 0.0, 'beta1': 0.9, 'beta2': 0.999}, 'optimD': {'name': 'Adam', 'net_names': ['dual_model_0', 'dual_model_1'], 'weight_decay': 0.0, 'beta1': 0.9, 'beta2': 0.999}}, 'validate': {'interval': 2000, 'save_img': False, 'metrics': {'psnr': {'name': 'PSNR', 'crop_border': 4, 'test_y_channel': True}, 'ssim': {'name': 'SSIM', 'crop_border': 4, 'test_y_channel': True}}}, 'log_config': {'interval': 20, 'visiual_interval': 500}, 'snapshot_config': {'interval': 1000}, 'is_train': True, 'timestamp': '-2021-04-19-01-13'}
    W0419 01:13:57.842463  8325 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0419 01:13:57.848598  8325 device_context.cc:372] device: 0, cuDNN Version: 7.6.
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:143: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if data.dtype == np.object:
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:14:17] ppgan.engine.trainer INFO: Iter: 20/60000 lr: 1.000e-04 loss_promary: 63.478 loss_dual: 166.031 loss_total: 229.509 batch_cost: 0.65151 sec reader_cost: 0.06580 sec ips: 1.53490 images/s eta: 10:51:16
    [04/19 01:14:29] ppgan.engine.trainer INFO: Iter: 40/60000 lr: 1.000e-04 loss_promary: 41.273 loss_dual: 108.360 loss_total: 149.633 batch_cost: 0.57928 sec reader_cost: 0.00555 sec ips: 1.72629 images/s eta: 9:38:52
    [04/19 01:14:40] ppgan.engine.trainer INFO: Iter: 60/60000 lr: 1.000e-04 loss_promary: 51.886 loss_dual: 128.109 loss_total: 179.994 batch_cost: 0.57813 sec reader_cost: 0.00543 sec ips: 1.72972 images/s eta: 9:37:32
    [04/19 01:14:52] ppgan.engine.trainer INFO: Iter: 80/60000 lr: 1.000e-04 loss_promary: 46.061 loss_dual: 108.375 loss_total: 154.437 batch_cost: 0.57925 sec reader_cost: 0.00579 sec ips: 1.72638 images/s eta: 9:38:27
    [04/19 01:15:03] ppgan.engine.trainer INFO: Iter: 100/60000 lr: 1.000e-04 loss_promary: 48.136 loss_dual: 107.703 loss_total: 155.839 batch_cost: 0.57982 sec reader_cost: 0.00565 sec ips: 1.72469 images/s eta: 9:38:50
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:15:16] ppgan.engine.trainer INFO: Iter: 120/60000 lr: 1.000e-04 loss_promary: 48.097 loss_dual: 41.616 loss_total: 89.713 batch_cost: 0.64170 sec reader_cost: 0.06733 sec ips: 1.55837 images/s eta: 10:40:24
    [04/19 01:15:28] ppgan.engine.trainer INFO: Iter: 140/60000 lr: 1.000e-04 loss_promary: 46.597 loss_dual: 31.976 loss_total: 78.572 batch_cost: 0.57988 sec reader_cost: 0.00608 sec ips: 1.72451 images/s eta: 9:38:30
    [04/19 01:15:39] ppgan.engine.trainer INFO: Iter: 160/60000 lr: 1.000e-04 loss_promary: 39.048 loss_dual: 34.754 loss_total: 73.803 batch_cost: 0.57644 sec reader_cost: 0.00526 sec ips: 1.73480 images/s eta: 9:34:53
    [04/19 01:15:51] ppgan.engine.trainer INFO: Iter: 180/60000 lr: 1.000e-04 loss_promary: 38.973 loss_dual: 24.884 loss_total: 63.857 batch_cost: 0.57866 sec reader_cost: 0.00545 sec ips: 1.72813 images/s eta: 9:36:54
    [04/19 01:16:03] ppgan.engine.trainer INFO: Iter: 200/60000 lr: 1.000e-04 loss_promary: 29.008 loss_dual: 21.845 loss_total: 50.853 batch_cost: 0.58122 sec reader_cost: 0.00605 sec ips: 1.72051 images/s eta: 9:39:16
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:16:15] ppgan.engine.trainer INFO: Iter: 220/60000 lr: 1.000e-04 loss_promary: 37.386 loss_dual: 28.817 loss_total: 66.203 batch_cost: 0.64657 sec reader_cost: 0.06987 sec ips: 1.54662 images/s eta: 10:44:11
    [04/19 01:16:27] ppgan.engine.trainer INFO: Iter: 240/60000 lr: 1.000e-04 loss_promary: 34.371 loss_dual: 26.080 loss_total: 60.451 batch_cost: 0.58132 sec reader_cost: 0.00600 sec ips: 1.72021 images/s eta: 9:38:59
    [04/19 01:16:39] ppgan.engine.trainer INFO: Iter: 260/60000 lr: 1.000e-04 loss_promary: 30.836 loss_dual: 23.207 loss_total: 54.043 batch_cost: 0.57901 sec reader_cost: 0.00524 sec ips: 1.72708 images/s eta: 9:36:29
    [04/19 01:16:50] ppgan.engine.trainer INFO: Iter: 280/60000 lr: 9.999e-05 loss_promary: 25.883 loss_dual: 19.010 loss_total: 44.894 batch_cost: 0.58051 sec reader_cost: 0.00587 sec ips: 1.72262 images/s eta: 9:37:47
    [04/19 01:17:02] ppgan.engine.trainer INFO: Iter: 300/60000 lr: 9.999e-05 loss_promary: 31.875 loss_dual: 23.183 loss_total: 55.058 batch_cost: 0.57955 sec reader_cost: 0.00544 sec ips: 1.72548 images/s eta: 9:36:38
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:17:15] ppgan.engine.trainer INFO: Iter: 320/60000 lr: 9.999e-05 loss_promary: 24.372 loss_dual: 19.296 loss_total: 43.668 batch_cost: 0.64438 sec reader_cost: 0.06954 sec ips: 1.55189 images/s eta: 10:40:55
    [04/19 01:17:26] ppgan.engine.trainer INFO: Iter: 340/60000 lr: 9.999e-05 loss_promary: 27.434 loss_dual: 22.789 loss_total: 50.223 batch_cost: 0.58068 sec reader_cost: 0.00592 sec ips: 1.72211 images/s eta: 9:37:22
    [04/19 01:17:38] ppgan.engine.trainer INFO: Iter: 360/60000 lr: 9.999e-05 loss_promary: 24.257 loss_dual: 21.213 loss_total: 45.470 batch_cost: 0.58172 sec reader_cost: 0.00581 sec ips: 1.71905 images/s eta: 9:38:12
    [04/19 01:17:50] ppgan.engine.trainer INFO: Iter: 380/60000 lr: 9.999e-05 loss_promary: 33.485 loss_dual: 29.422 loss_total: 62.907 batch_cost: 0.58148 sec reader_cost: 0.00585 sec ips: 1.71976 images/s eta: 9:37:47
    [04/19 01:18:01] ppgan.engine.trainer INFO: Iter: 400/60000 lr: 9.999e-05 loss_promary: 22.259 loss_dual: 17.849 loss_total: 40.108 batch_cost: 0.58146 sec reader_cost: 0.00565 sec ips: 1.71981 images/s eta: 9:37:34
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:18:14] ppgan.engine.trainer INFO: Iter: 420/60000 lr: 9.999e-05 loss_promary: 22.394 loss_dual: 20.318 loss_total: 42.712 batch_cost: 0.64407 sec reader_cost: 0.06775 sec ips: 1.55264 images/s eta: 10:39:32
    [04/19 01:18:26] ppgan.engine.trainer INFO: Iter: 440/60000 lr: 9.999e-05 loss_promary: 19.131 loss_dual: 16.371 loss_total: 35.503 batch_cost: 0.58086 sec reader_cost: 0.00597 sec ips: 1.72159 images/s eta: 9:36:35
    [04/19 01:18:37] ppgan.engine.trainer INFO: Iter: 460/60000 lr: 9.999e-05 loss_promary: 24.926 loss_dual: 27.474 loss_total: 52.400 batch_cost: 0.58073 sec reader_cost: 0.00625 sec ips: 1.72197 images/s eta: 9:36:16
    [04/19 01:18:49] ppgan.engine.trainer INFO: Iter: 480/60000 lr: 9.998e-05 loss_promary: 27.643 loss_dual: 23.709 loss_total: 51.352 batch_cost: 0.58106 sec reader_cost: 0.00619 sec ips: 1.72100 images/s eta: 9:36:23
    [04/19 01:19:01] ppgan.engine.trainer INFO: Iter: 500/60000 lr: 9.998e-05 loss_promary: 21.496 loss_dual: 20.045 loss_total: 41.541 batch_cost: 0.58150 sec reader_cost: 0.00593 sec ips: 1.71970 images/s eta: 9:36:38
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:19:14] ppgan.engine.trainer INFO: Iter: 520/60000 lr: 9.998e-05 loss_promary: 21.784 loss_dual: 19.803 loss_total: 41.587 batch_cost: 0.64773 sec reader_cost: 0.07244 sec ips: 1.54385 images/s eta: 10:42:06
    [04/19 01:19:25] ppgan.engine.trainer INFO: Iter: 540/60000 lr: 9.998e-05 loss_promary: 15.900 loss_dual: 12.675 loss_total: 28.574 batch_cost: 0.58150 sec reader_cost: 0.00622 sec ips: 1.71969 images/s eta: 9:36:15
    [04/19 01:19:37] ppgan.engine.trainer INFO: Iter: 560/60000 lr: 9.998e-05 loss_promary: 18.628 loss_dual: 15.933 loss_total: 34.561 batch_cost: 0.58184 sec reader_cost: 0.00628 sec ips: 1.71868 images/s eta: 9:36:24
    [04/19 01:19:49] ppgan.engine.trainer INFO: Iter: 580/60000 lr: 9.998e-05 loss_promary: 23.034 loss_dual: 21.614 loss_total: 44.647 batch_cost: 0.58080 sec reader_cost: 0.00576 sec ips: 1.72177 images/s eta: 9:35:10
    [04/19 01:20:00] ppgan.engine.trainer INFO: Iter: 600/60000 lr: 9.998e-05 loss_promary: 21.321 loss_dual: 23.301 loss_total: 44.622 batch_cost: 0.58076 sec reader_cost: 0.00582 sec ips: 1.72189 images/s eta: 9:34:56
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:20:13] ppgan.engine.trainer INFO: Iter: 620/60000 lr: 9.997e-05 loss_promary: 18.906 loss_dual: 18.048 loss_total: 36.954 batch_cost: 0.64588 sec reader_cost: 0.07019 sec ips: 1.54828 images/s eta: 10:39:11
    [04/19 01:20:25] ppgan.engine.trainer INFO: Iter: 640/60000 lr: 9.997e-05 loss_promary: 15.333 loss_dual: 14.594 loss_total: 29.927 batch_cost: 0.58124 sec reader_cost: 0.00594 sec ips: 1.72047 images/s eta: 9:35:01
    [04/19 01:20:36] ppgan.engine.trainer INFO: Iter: 660/60000 lr: 9.997e-05 loss_promary: 18.379 loss_dual: 17.390 loss_total: 35.769 batch_cost: 0.58108 sec reader_cost: 0.00625 sec ips: 1.72094 images/s eta: 9:34:40
    [04/19 01:20:48] ppgan.engine.trainer INFO: Iter: 680/60000 lr: 9.997e-05 loss_promary: 18.336 loss_dual: 18.486 loss_total: 36.822 batch_cost: 0.58038 sec reader_cost: 0.00586 sec ips: 1.72302 images/s eta: 9:33:47
    [04/19 01:21:00] ppgan.engine.trainer INFO: Iter: 700/60000 lr: 9.997e-05 loss_promary: 17.497 loss_dual: 19.230 loss_total: 36.727 batch_cost: 0.58100 sec reader_cost: 0.00605 sec ips: 1.72116 images/s eta: 9:34:12
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:21:13] ppgan.engine.trainer INFO: Iter: 720/60000 lr: 9.996e-05 loss_promary: 15.596 loss_dual: 14.743 loss_total: 30.339 batch_cost: 0.64569 sec reader_cost: 0.06962 sec ips: 1.54872 images/s eta: 10:37:56
    [04/19 01:21:24] ppgan.engine.trainer INFO: Iter: 740/60000 lr: 9.996e-05 loss_promary: 29.212 loss_dual: 30.057 loss_total: 59.269 batch_cost: 0.58191 sec reader_cost: 0.00650 sec ips: 1.71848 images/s eta: 9:34:43
    [04/19 01:21:36] ppgan.engine.trainer INFO: Iter: 760/60000 lr: 9.996e-05 loss_promary: 15.733 loss_dual: 17.481 loss_total: 33.214 batch_cost: 0.58094 sec reader_cost: 0.00644 sec ips: 1.72134 images/s eta: 9:33:34
    [04/19 01:21:47] ppgan.engine.trainer INFO: Iter: 780/60000 lr: 9.996e-05 loss_promary: 20.356 loss_dual: 18.265 loss_total: 38.621 batch_cost: 0.58135 sec reader_cost: 0.00606 sec ips: 1.72012 images/s eta: 9:33:47
    [04/19 01:21:59] ppgan.engine.trainer INFO: Iter: 800/60000 lr: 9.996e-05 loss_promary: 19.118 loss_dual: 18.280 loss_total: 37.398 batch_cost: 0.58114 sec reader_cost: 0.00600 sec ips: 1.72074 images/s eta: 9:33:23
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:22:12] ppgan.engine.trainer INFO: Iter: 820/60000 lr: 9.995e-05 loss_promary: 25.516 loss_dual: 26.115 loss_total: 51.632 batch_cost: 0.64687 sec reader_cost: 0.06963 sec ips: 1.54590 images/s eta: 10:38:01
    [04/19 01:22:24] ppgan.engine.trainer INFO: Iter: 840/60000 lr: 9.995e-05 loss_promary: 21.598 loss_dual: 20.981 loss_total: 42.579 batch_cost: 0.58135 sec reader_cost: 0.00603 sec ips: 1.72014 images/s eta: 9:33:11
    [04/19 01:22:35] ppgan.engine.trainer INFO: Iter: 860/60000 lr: 9.995e-05 loss_promary: 17.393 loss_dual: 17.348 loss_total: 34.741 batch_cost: 0.58114 sec reader_cost: 0.00557 sec ips: 1.72075 images/s eta: 9:32:48
    [04/19 01:22:47] ppgan.engine.trainer INFO: Iter: 880/60000 lr: 9.995e-05 loss_promary: 19.336 loss_dual: 21.729 loss_total: 41.066 batch_cost: 0.58131 sec reader_cost: 0.00608 sec ips: 1.72026 images/s eta: 9:32:46
    [04/19 01:22:59] ppgan.engine.trainer INFO: Iter: 900/60000 lr: 9.994e-05 loss_promary: 16.413 loss_dual: 16.858 loss_total: 33.271 batch_cost: 0.58095 sec reader_cost: 0.00601 sec ips: 1.72131 images/s eta: 9:32:13
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:23:11] ppgan.engine.trainer INFO: Iter: 920/60000 lr: 9.994e-05 loss_promary: 18.339 loss_dual: 23.648 loss_total: 41.986 batch_cost: 0.64540 sec reader_cost: 0.06825 sec ips: 1.54943 images/s eta: 10:35:29
    [04/19 01:23:23] ppgan.engine.trainer INFO: Iter: 940/60000 lr: 9.994e-05 loss_promary: 23.629 loss_dual: 21.328 loss_total: 44.958 batch_cost: 0.58056 sec reader_cost: 0.00612 sec ips: 1.72247 images/s eta: 9:31:27
    [04/19 01:23:35] ppgan.engine.trainer INFO: Iter: 960/60000 lr: 9.994e-05 loss_promary: 18.920 loss_dual: 18.754 loss_total: 37.674 batch_cost: 0.57873 sec reader_cost: 0.00571 sec ips: 1.72793 images/s eta: 9:29:27
    [04/19 01:23:46] ppgan.engine.trainer INFO: Iter: 980/60000 lr: 9.993e-05 loss_promary: 19.748 loss_dual: 17.703 loss_total: 37.452 batch_cost: 0.57723 sec reader_cost: 0.00514 sec ips: 1.73241 images/s eta: 9:27:47
    [04/19 01:23:58] ppgan.engine.trainer INFO: Iter: 1000/60000 lr: 9.993e-05 loss_promary: 15.214 loss_dual: 17.409 loss_total: 32.623 batch_cost: 0.58224 sec reader_cost: 0.00630 sec ips: 1.71752 images/s eta: 9:32:31
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:24:12] ppgan.engine.trainer INFO: Iter: 1020/60000 lr: 9.993e-05 loss_promary: 19.737 loss_dual: 26.476 loss_total: 46.213 batch_cost: 0.65192 sec reader_cost: 0.07388 sec ips: 1.53394 images/s eta: 10:40:49
    [04/19 01:24:24] ppgan.engine.trainer INFO: Iter: 1040/60000 lr: 9.993e-05 loss_promary: 28.158 loss_dual: 31.846 loss_total: 60.004 batch_cost: 0.58347 sec reader_cost: 0.00663 sec ips: 1.71389 images/s eta: 9:33:20
    [04/19 01:24:35] ppgan.engine.trainer INFO: Iter: 1060/60000 lr: 9.992e-05 loss_promary: 21.971 loss_dual: 25.383 loss_total: 47.354 batch_cost: 0.58207 sec reader_cost: 0.00604 sec ips: 1.71802 images/s eta: 9:31:46
    [04/19 01:24:47] ppgan.engine.trainer INFO: Iter: 1080/60000 lr: 9.992e-05 loss_promary: 18.228 loss_dual: 19.767 loss_total: 37.995 batch_cost: 0.58206 sec reader_cost: 0.00652 sec ips: 1.71804 images/s eta: 9:31:34
    [04/19 01:24:59] ppgan.engine.trainer INFO: Iter: 1100/60000 lr: 9.992e-05 loss_promary: 27.621 loss_dual: 28.304 loss_total: 55.925 batch_cost: 0.58203 sec reader_cost: 0.00626 sec ips: 1.71813 images/s eta: 9:31:20
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:25:12] ppgan.engine.trainer INFO: Iter: 1120/60000 lr: 9.991e-05 loss_promary: 18.168 loss_dual: 18.356 loss_total: 36.524 batch_cost: 0.65070 sec reader_cost: 0.07452 sec ips: 1.53680 images/s eta: 10:38:32
    [04/19 01:25:23] ppgan.engine.trainer INFO: Iter: 1140/60000 lr: 9.991e-05 loss_promary: 18.123 loss_dual: 15.457 loss_total: 33.581 batch_cost: 0.58061 sec reader_cost: 0.00598 sec ips: 1.72232 images/s eta: 9:29:34
    [04/19 01:25:35] ppgan.engine.trainer INFO: Iter: 1160/60000 lr: 9.991e-05 loss_promary: 18.048 loss_dual: 21.205 loss_total: 39.252 batch_cost: 0.58227 sec reader_cost: 0.00623 sec ips: 1.71742 images/s eta: 9:31:00
    [04/19 01:25:46] ppgan.engine.trainer INFO: Iter: 1180/60000 lr: 9.990e-05 loss_promary: 15.891 loss_dual: 17.062 loss_total: 32.953 batch_cost: 0.58132 sec reader_cost: 0.00596 sec ips: 1.72022 images/s eta: 9:29:52
    [04/19 01:25:58] ppgan.engine.trainer INFO: Iter: 1200/60000 lr: 9.990e-05 loss_promary: 17.870 loss_dual: 16.845 loss_total: 34.715 batch_cost: 0.58025 sec reader_cost: 0.00555 sec ips: 1.72338 images/s eta: 9:28:38
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:26:11] ppgan.engine.trainer INFO: Iter: 1220/60000 lr: 9.990e-05 loss_promary: 14.491 loss_dual: 13.567 loss_total: 28.057 batch_cost: 0.64817 sec reader_cost: 0.07213 sec ips: 1.54279 images/s eta: 10:34:59
    [04/19 01:26:23] ppgan.engine.trainer INFO: Iter: 1240/60000 lr: 9.989e-05 loss_promary: 16.536 loss_dual: 22.039 loss_total: 38.575 batch_cost: 0.58085 sec reader_cost: 0.00581 sec ips: 1.72162 images/s eta: 9:28:50
    [04/19 01:26:34] ppgan.engine.trainer INFO: Iter: 1260/60000 lr: 9.989e-05 loss_promary: 21.102 loss_dual: 24.230 loss_total: 45.332 batch_cost: 0.58176 sec reader_cost: 0.00624 sec ips: 1.71893 images/s eta: 9:29:31
    [04/19 01:26:46] ppgan.engine.trainer INFO: Iter: 1280/60000 lr: 9.989e-05 loss_promary: 16.054 loss_dual: 16.370 loss_total: 32.424 batch_cost: 0.58136 sec reader_cost: 0.00604 sec ips: 1.72011 images/s eta: 9:28:56
    [04/19 01:26:58] ppgan.engine.trainer INFO: Iter: 1300/60000 lr: 9.988e-05 loss_promary: 14.028 loss_dual: 22.018 loss_total: 36.046 batch_cost: 0.58098 sec reader_cost: 0.00561 sec ips: 1.72124 images/s eta: 9:28:22
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:27:10] ppgan.engine.trainer INFO: Iter: 1320/60000 lr: 9.988e-05 loss_promary: 21.019 loss_dual: 16.765 loss_total: 37.784 batch_cost: 0.64732 sec reader_cost: 0.07163 sec ips: 1.54483 images/s eta: 10:33:04
    [04/19 01:27:22] ppgan.engine.trainer INFO: Iter: 1340/60000 lr: 9.988e-05 loss_promary: 19.218 loss_dual: 22.091 loss_total: 41.310 batch_cost: 0.58044 sec reader_cost: 0.00626 sec ips: 1.72282 images/s eta: 9:27:28
    [04/19 01:27:34] ppgan.engine.trainer INFO: Iter: 1360/60000 lr: 9.987e-05 loss_promary: 18.593 loss_dual: 20.358 loss_total: 38.951 batch_cost: 0.58194 sec reader_cost: 0.00618 sec ips: 1.71838 images/s eta: 9:28:44
    [04/19 01:27:45] ppgan.engine.trainer INFO: Iter: 1380/60000 lr: 9.987e-05 loss_promary: 14.360 loss_dual: 16.396 loss_total: 30.756 batch_cost: 0.58051 sec reader_cost: 0.00543 sec ips: 1.72261 images/s eta: 9:27:09
    [04/19 01:27:57] ppgan.engine.trainer INFO: Iter: 1400/60000 lr: 9.987e-05 loss_promary: 12.998 loss_dual: 14.094 loss_total: 27.091 batch_cost: 0.58009 sec reader_cost: 0.00551 sec ips: 1.72387 images/s eta: 9:26:32
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:28:10] ppgan.engine.trainer INFO: Iter: 1420/60000 lr: 9.986e-05 loss_promary: 16.090 loss_dual: 19.258 loss_total: 35.348 batch_cost: 0.64621 sec reader_cost: 0.07017 sec ips: 1.54748 images/s eta: 10:30:54
    [04/19 01:28:22] ppgan.engine.trainer INFO: Iter: 1440/60000 lr: 9.986e-05 loss_promary: 17.972 loss_dual: 17.946 loss_total: 35.918 batch_cost: 0.58064 sec reader_cost: 0.00569 sec ips: 1.72225 images/s eta: 9:26:41
    [04/19 01:28:33] ppgan.engine.trainer INFO: Iter: 1460/60000 lr: 9.985e-05 loss_promary: 19.243 loss_dual: 23.714 loss_total: 42.956 batch_cost: 0.58023 sec reader_cost: 0.00557 sec ips: 1.72346 images/s eta: 9:26:05
    [04/19 01:28:45] ppgan.engine.trainer INFO: Iter: 1480/60000 lr: 9.985e-05 loss_promary: 19.743 loss_dual: 19.099 loss_total: 38.842 batch_cost: 0.58233 sec reader_cost: 0.00593 sec ips: 1.71725 images/s eta: 9:27:57
    [04/19 01:28:56] ppgan.engine.trainer INFO: Iter: 1500/60000 lr: 9.985e-05 loss_promary: 17.139 loss_dual: 17.380 loss_total: 34.519 batch_cost: 0.58122 sec reader_cost: 0.00609 sec ips: 1.72052 images/s eta: 9:26:40
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:29:09] ppgan.engine.trainer INFO: Iter: 1520/60000 lr: 9.984e-05 loss_promary: 16.820 loss_dual: 18.175 loss_total: 34.996 batch_cost: 0.64873 sec reader_cost: 0.07214 sec ips: 1.54148 images/s eta: 10:32:16
    [04/19 01:29:21] ppgan.engine.trainer INFO: Iter: 1540/60000 lr: 9.984e-05 loss_promary: 12.631 loss_dual: 12.592 loss_total: 25.223 batch_cost: 0.58133 sec reader_cost: 0.00636 sec ips: 1.72019 images/s eta: 9:26:24
    [04/19 01:29:33] ppgan.engine.trainer INFO: Iter: 1560/60000 lr: 9.983e-05 loss_promary: 13.175 loss_dual: 20.308 loss_total: 33.484 batch_cost: 0.58168 sec reader_cost: 0.00598 sec ips: 1.71916 images/s eta: 9:26:32
    [04/19 01:29:44] ppgan.engine.trainer INFO: Iter: 1580/60000 lr: 9.983e-05 loss_promary: 18.930 loss_dual: 19.853 loss_total: 38.783 batch_cost: 0.58197 sec reader_cost: 0.00613 sec ips: 1.71831 images/s eta: 9:26:37
    [04/19 01:29:56] ppgan.engine.trainer INFO: Iter: 1600/60000 lr: 9.983e-05 loss_promary: 14.343 loss_dual: 20.500 loss_total: 34.843 batch_cost: 0.58134 sec reader_cost: 0.00602 sec ips: 1.72017 images/s eta: 9:25:49
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:30:09] ppgan.engine.trainer INFO: Iter: 1620/60000 lr: 9.982e-05 loss_promary: 18.351 loss_dual: 13.379 loss_total: 31.730 batch_cost: 0.64652 sec reader_cost: 0.07037 sec ips: 1.54674 images/s eta: 10:29:03
    [04/19 01:30:21] ppgan.engine.trainer INFO: Iter: 1640/60000 lr: 9.982e-05 loss_promary: 19.793 loss_dual: 14.459 loss_total: 34.252 batch_cost: 0.58159 sec reader_cost: 0.00603 sec ips: 1.71942 images/s eta: 9:25:41
    [04/19 01:30:32] ppgan.engine.trainer INFO: Iter: 1660/60000 lr: 9.981e-05 loss_promary: 15.872 loss_dual: 22.678 loss_total: 38.550 batch_cost: 0.58108 sec reader_cost: 0.00560 sec ips: 1.72092 images/s eta: 9:24:59
    [04/19 01:30:44] ppgan.engine.trainer INFO: Iter: 1680/60000 lr: 9.981e-05 loss_promary: 16.093 loss_dual: 16.705 loss_total: 32.798 batch_cost: 0.58221 sec reader_cost: 0.00580 sec ips: 1.71760 images/s eta: 9:25:53
    [04/19 01:30:55] ppgan.engine.trainer INFO: Iter: 1700/60000 lr: 9.980e-05 loss_promary: 15.498 loss_dual: 20.680 loss_total: 36.179 batch_cost: 0.58201 sec reader_cost: 0.00586 sec ips: 1.71819 images/s eta: 9:25:30
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:31:08] ppgan.engine.trainer INFO: Iter: 1720/60000 lr: 9.980e-05 loss_promary: 15.076 loss_dual: 15.571 loss_total: 30.647 batch_cost: 0.64810 sec reader_cost: 0.07159 sec ips: 1.54296 images/s eta: 10:29:30
    [04/19 01:31:20] ppgan.engine.trainer INFO: Iter: 1740/60000 lr: 9.979e-05 loss_promary: 14.478 loss_dual: 19.964 loss_total: 34.441 batch_cost: 0.58186 sec reader_cost: 0.00629 sec ips: 1.71863 images/s eta: 9:24:58
    [04/19 01:31:32] ppgan.engine.trainer INFO: Iter: 1760/60000 lr: 9.979e-05 loss_promary: 20.613 loss_dual: 24.444 loss_total: 45.057 batch_cost: 0.57716 sec reader_cost: 0.00519 sec ips: 1.73262 images/s eta: 9:20:13
    [04/19 01:31:43] ppgan.engine.trainer INFO: Iter: 1780/60000 lr: 9.978e-05 loss_promary: 19.604 loss_dual: 24.439 loss_total: 44.043 batch_cost: 0.58093 sec reader_cost: 0.00582 sec ips: 1.72137 images/s eta: 9:23:41
    [04/19 01:31:55] ppgan.engine.trainer INFO: Iter: 1800/60000 lr: 9.978e-05 loss_promary: 14.005 loss_dual: 17.798 loss_total: 31.803 batch_cost: 0.58775 sec reader_cost: 0.00673 sec ips: 1.70139 images/s eta: 9:30:06
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:32:08] ppgan.engine.trainer INFO: Iter: 1820/60000 lr: 9.977e-05 loss_promary: 15.376 loss_dual: 18.488 loss_total: 33.864 batch_cost: 0.65517 sec reader_cost: 0.07702 sec ips: 1.52632 images/s eta: 10:35:17
    [04/19 01:32:20] ppgan.engine.trainer INFO: Iter: 1840/60000 lr: 9.977e-05 loss_promary: 12.263 loss_dual: 18.700 loss_total: 30.963 batch_cost: 0.58171 sec reader_cost: 0.00597 sec ips: 1.71906 images/s eta: 9:23:51
    [04/19 01:32:31] ppgan.engine.trainer INFO: Iter: 1860/60000 lr: 9.976e-05 loss_promary: 17.384 loss_dual: 13.716 loss_total: 31.100 batch_cost: 0.58100 sec reader_cost: 0.00569 sec ips: 1.72118 images/s eta: 9:22:58
    [04/19 01:32:43] ppgan.engine.trainer INFO: Iter: 1880/60000 lr: 9.976e-05 loss_promary: 14.017 loss_dual: 17.111 loss_total: 31.128 batch_cost: 0.58044 sec reader_cost: 0.00610 sec ips: 1.72282 images/s eta: 9:22:14
    [04/19 01:32:55] ppgan.engine.trainer INFO: Iter: 1900/60000 lr: 9.975e-05 loss_promary: 18.486 loss_dual: 18.842 loss_total: 37.327 batch_cost: 0.58149 sec reader_cost: 0.00576 sec ips: 1.71973 images/s eta: 9:23:03
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:33:08] ppgan.engine.trainer INFO: Iter: 1920/60000 lr: 9.975e-05 loss_promary: 19.816 loss_dual: 18.322 loss_total: 38.138 batch_cost: 0.68269 sec reader_cost: 0.07337 sec ips: 1.46480 images/s eta: 11:00:49
    [04/19 01:33:23] ppgan.engine.trainer INFO: Iter: 1940/60000 lr: 9.974e-05 loss_promary: 17.275 loss_dual: 24.194 loss_total: 41.468 batch_cost: 0.73015 sec reader_cost: 0.00580 sec ips: 1.36958 images/s eta: 11:46:31
    [04/19 01:33:40] ppgan.engine.trainer INFO: Iter: 1960/60000 lr: 9.974e-05 loss_promary: 21.456 loss_dual: 21.470 loss_total: 42.926 batch_cost: 0.86193 sec reader_cost: 0.00583 sec ips: 1.16018 images/s eta: 13:53:45
    [04/19 01:33:57] ppgan.engine.trainer INFO: Iter: 1980/60000 lr: 9.973e-05 loss_promary: 16.088 loss_dual: 18.566 loss_total: 34.654 batch_cost: 0.82053 sec reader_cost: 0.00627 sec ips: 1.21873 images/s eta: 13:13:26
    [04/19 01:34:13] ppgan.engine.trainer INFO: Iter: 2000/60000 lr: 9.973e-05 loss_promary: 17.592 loss_dual: 20.415 loss_total: 38.007 batch_cost: 0.84356 sec reader_cost: 0.00600 sec ips: 1.18546 images/s eta: 13:35:25
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dataloader/dataloader_iter.py:89: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if isinstance(slot[0], (np.ndarray, np.bool, numbers.Number)):
    [04/19 01:34:18] ppgan.engine.trainer INFO: Test iter: [0/20]
    [04/19 01:35:43] ppgan.engine.trainer INFO: Metric psnr: 24.4917
    [04/19 01:35:43] ppgan.engine.trainer INFO: Metric ssim: 0.7276
    [04/19 01:35:57] ppgan.engine.trainer INFO: Iter: 2020/60000 lr: 9.972e-05 loss_promary: 15.361 loss_dual: 14.017 loss_total: 29.377 batch_cost: 0.65480 sec reader_cost: 0.07924 sec ips: 1.52719 images/s eta: 10:32:44
    [04/19 01:36:08] ppgan.engine.trainer INFO: Iter: 2040/60000 lr: 9.972e-05 loss_promary: 16.256 loss_dual: 16.448 loss_total: 32.704 batch_cost: 0.58092 sec reader_cost: 0.00617 sec ips: 1.72141 images/s eta: 9:21:09
    [04/19 01:36:20] ppgan.engine.trainer INFO: Iter: 2060/60000 lr: 9.971e-05 loss_promary: 12.447 loss_dual: 10.349 loss_total: 22.796 batch_cost: 0.58116 sec reader_cost: 0.00638 sec ips: 1.72069 images/s eta: 9:21:11
    [04/19 01:36:32] ppgan.engine.trainer INFO: Iter: 2080/60000 lr: 9.970e-05 loss_promary: 20.355 loss_dual: 17.976 loss_total: 38.331 batch_cost: 0.58165 sec reader_cost: 0.00579 sec ips: 1.71924 images/s eta: 9:21:28
    [04/19 01:36:43] ppgan.engine.trainer INFO: Iter: 2100/60000 lr: 9.970e-05 loss_promary: 16.370 loss_dual: 18.246 loss_total: 34.616 batch_cost: 0.58119 sec reader_cost: 0.00578 sec ips: 1.72060 images/s eta: 9:20:50
    [04/19 01:36:56] ppgan.engine.trainer INFO: Iter: 2120/60000 lr: 9.969e-05 loss_promary: 10.624 loss_dual: 16.487 loss_total: 27.111 batch_cost: 0.65813 sec reader_cost: 0.08121 sec ips: 1.51945 images/s eta: 10:34:52
    [04/19 01:37:08] ppgan.engine.trainer INFO: Iter: 2140/60000 lr: 9.969e-05 loss_promary: 19.127 loss_dual: 27.915 loss_total: 47.043 batch_cost: 0.58115 sec reader_cost: 0.00625 sec ips: 1.72072 images/s eta: 9:20:24
    [04/19 01:37:20] ppgan.engine.trainer INFO: Iter: 2160/60000 lr: 9.968e-05 loss_promary: 12.573 loss_dual: 15.697 loss_total: 28.270 batch_cost: 0.58221 sec reader_cost: 0.00648 sec ips: 1.71760 images/s eta: 9:21:14
    [04/19 01:37:31] ppgan.engine.trainer INFO: Iter: 2180/60000 lr: 9.968e-05 loss_promary: 16.201 loss_dual: 23.092 loss_total: 39.293 batch_cost: 0.58117 sec reader_cost: 0.00630 sec ips: 1.72066 images/s eta: 9:20:02
    [04/19 01:37:43] ppgan.engine.trainer INFO: Iter: 2200/60000 lr: 9.967e-05 loss_promary: 13.940 loss_dual: 21.044 loss_total: 34.985 batch_cost: 0.58085 sec reader_cost: 0.00579 sec ips: 1.72161 images/s eta: 9:19:32
    [04/19 01:37:56] ppgan.engine.trainer INFO: Iter: 2220/60000 lr: 9.966e-05 loss_promary: 17.898 loss_dual: 15.674 loss_total: 33.573 batch_cost: 0.66084 sec reader_cost: 0.08364 sec ips: 1.51322 images/s eta: 10:36:22
    [04/19 01:38:08] ppgan.engine.trainer INFO: Iter: 2240/60000 lr: 9.966e-05 loss_promary: 15.978 loss_dual: 14.785 loss_total: 30.763 batch_cost: 0.58083 sec reader_cost: 0.00609 sec ips: 1.72167 images/s eta: 9:19:08
    [04/19 01:38:19] ppgan.engine.trainer INFO: Iter: 2260/60000 lr: 9.965e-05 loss_promary: 16.599 loss_dual: 18.691 loss_total: 35.290 batch_cost: 0.58059 sec reader_cost: 0.00590 sec ips: 1.72239 images/s eta: 9:18:42
    [04/19 01:38:31] ppgan.engine.trainer INFO: Iter: 2280/60000 lr: 9.964e-05 loss_promary: 14.358 loss_dual: 11.582 loss_total: 25.940 batch_cost: 0.58038 sec reader_cost: 0.00540 sec ips: 1.72302 images/s eta: 9:18:18
    [04/19 01:38:43] ppgan.engine.trainer INFO: Iter: 2300/60000 lr: 9.964e-05 loss_promary: 13.929 loss_dual: 19.567 loss_total: 33.496 batch_cost: 0.58178 sec reader_cost: 0.00588 sec ips: 1.71887 images/s eta: 9:19:27
    [04/19 01:38:56] ppgan.engine.trainer INFO: Iter: 2320/60000 lr: 9.963e-05 loss_promary: 13.047 loss_dual: 16.969 loss_total: 30.016 batch_cost: 0.65920 sec reader_cost: 0.08344 sec ips: 1.51700 images/s eta: 10:33:41
    [04/19 01:39:07] ppgan.engine.trainer INFO: Iter: 2340/60000 lr: 9.963e-05 loss_promary: 14.619 loss_dual: 16.012 loss_total: 30.631 batch_cost: 0.58107 sec reader_cost: 0.00618 sec ips: 1.72096 images/s eta: 9:18:23
    [04/19 01:39:19] ppgan.engine.trainer INFO: Iter: 2360/60000 lr: 9.962e-05 loss_promary: 12.596 loss_dual: 18.904 loss_total: 31.499 batch_cost: 0.58188 sec reader_cost: 0.00657 sec ips: 1.71856 images/s eta: 9:18:59
    [04/19 01:39:31] ppgan.engine.trainer INFO: Iter: 2380/60000 lr: 9.961e-05 loss_promary: 15.707 loss_dual: 14.290 loss_total: 29.997 batch_cost: 0.57872 sec reader_cost: 0.00521 sec ips: 1.72796 images/s eta: 9:15:45
    [04/19 01:39:42] ppgan.engine.trainer INFO: Iter: 2400/60000 lr: 9.961e-05 loss_promary: 8.631 loss_dual: 14.338 loss_total: 22.969 batch_cost: 0.57854 sec reader_cost: 0.00520 sec ips: 1.72850 images/s eta: 9:15:23
    [04/19 01:39:55] ppgan.engine.trainer INFO: Iter: 2420/60000 lr: 9.960e-05 loss_promary: 16.557 loss_dual: 16.560 loss_total: 33.117 batch_cost: 0.66264 sec reader_cost: 0.08479 sec ips: 1.50912 images/s eta: 10:35:54
    [04/19 01:40:07] ppgan.engine.trainer INFO: Iter: 2440/60000 lr: 9.959e-05 loss_promary: 22.001 loss_dual: 17.571 loss_total: 39.572 batch_cost: 0.58166 sec reader_cost: 0.00605 sec ips: 1.71922 images/s eta: 9:17:59
    [04/19 01:40:19] ppgan.engine.trainer INFO: Iter: 2460/60000 lr: 9.959e-05 loss_promary: 15.869 loss_dual: 13.675 loss_total: 29.545 batch_cost: 0.58116 sec reader_cost: 0.00564 sec ips: 1.72069 images/s eta: 9:17:19
    [04/19 01:40:30] ppgan.engine.trainer INFO: Iter: 2480/60000 lr: 9.958e-05 loss_promary: 19.945 loss_dual: 14.853 loss_total: 34.798 batch_cost: 0.58149 sec reader_cost: 0.00596 sec ips: 1.71972 images/s eta: 9:17:26
    [04/19 01:40:42] ppgan.engine.trainer INFO: Iter: 2500/60000 lr: 9.957e-05 loss_promary: 13.003 loss_dual: 14.029 loss_total: 27.031 batch_cost: 0.58156 sec reader_cost: 0.00604 sec ips: 1.71951 images/s eta: 9:17:19
    [04/19 01:40:56] ppgan.engine.trainer INFO: Iter: 2520/60000 lr: 9.957e-05 loss_promary: 16.438 loss_dual: 10.901 loss_total: 27.339 batch_cost: 0.65903 sec reader_cost: 0.08210 sec ips: 1.51737 images/s eta: 10:31:20
    [04/19 01:41:08] ppgan.engine.trainer INFO: Iter: 2540/60000 lr: 9.956e-05 loss_promary: 14.245 loss_dual: 10.355 loss_total: 24.600 batch_cost: 0.58233 sec reader_cost: 0.00611 sec ips: 1.71723 images/s eta: 9:17:40
    [04/19 01:41:19] ppgan.engine.trainer INFO: Iter: 2560/60000 lr: 9.955e-05 loss_promary: 14.098 loss_dual: 11.672 loss_total: 25.770 batch_cost: 0.58284 sec reader_cost: 0.00680 sec ips: 1.71573 images/s eta: 9:17:57
    [04/19 01:41:31] ppgan.engine.trainer INFO: Iter: 2580/60000 lr: 9.955e-05 loss_promary: 17.174 loss_dual: 11.959 loss_total: 29.133 batch_cost: 0.58176 sec reader_cost: 0.00656 sec ips: 1.71892 images/s eta: 9:16:44
    [04/19 01:41:43] ppgan.engine.trainer INFO: Iter: 2600/60000 lr: 9.954e-05 loss_promary: 18.211 loss_dual: 10.048 loss_total: 28.259 batch_cost: 0.58261 sec reader_cost: 0.00617 sec ips: 1.71641 images/s eta: 9:17:21
    [04/19 01:41:56] ppgan.engine.trainer INFO: Iter: 2620/60000 lr: 9.953e-05 loss_promary: 11.370 loss_dual: 7.060 loss_total: 18.430 batch_cost: 0.66024 sec reader_cost: 0.08330 sec ips: 1.51460 images/s eta: 10:31:23
    [04/19 01:42:08] ppgan.engine.trainer INFO: Iter: 2640/60000 lr: 9.952e-05 loss_promary: 13.381 loss_dual: 10.162 loss_total: 23.543 batch_cost: 0.58341 sec reader_cost: 0.00644 sec ips: 1.71406 images/s eta: 9:17:43
    [04/19 01:42:19] ppgan.engine.trainer INFO: Iter: 2660/60000 lr: 9.952e-05 loss_promary: 15.670 loss_dual: 11.880 loss_total: 27.550 batch_cost: 0.58049 sec reader_cost: 0.00563 sec ips: 1.72267 images/s eta: 9:14:44
    [04/19 01:42:31] ppgan.engine.trainer INFO: Iter: 2680/60000 lr: 9.951e-05 loss_promary: 13.269 loss_dual: 8.850 loss_total: 22.119 batch_cost: 0.58059 sec reader_cost: 0.00608 sec ips: 1.72239 images/s eta: 9:14:38
    [04/19 01:42:43] ppgan.engine.trainer INFO: Iter: 2700/60000 lr: 9.950e-05 loss_promary: 18.722 loss_dual: 13.943 loss_total: 32.665 batch_cost: 0.58168 sec reader_cost: 0.00590 sec ips: 1.71917 images/s eta: 9:15:29
    [04/19 01:42:56] ppgan.engine.trainer INFO: Iter: 2720/60000 lr: 9.949e-05 loss_promary: 19.900 loss_dual: 10.433 loss_total: 30.333 batch_cost: 0.66458 sec reader_cost: 0.08641 sec ips: 1.50471 images/s eta: 10:34:26
    [04/19 01:43:07] ppgan.engine.trainer INFO: Iter: 2740/60000 lr: 9.949e-05 loss_promary: 15.552 loss_dual: 8.821 loss_total: 24.373 batch_cost: 0.58221 sec reader_cost: 0.00594 sec ips: 1.71760 images/s eta: 9:15:36
    [04/19 01:43:19] ppgan.engine.trainer INFO: Iter: 2760/60000 lr: 9.948e-05 loss_promary: 14.496 loss_dual: 14.597 loss_total: 29.094 batch_cost: 0.58107 sec reader_cost: 0.00608 sec ips: 1.72097 images/s eta: 9:14:19
    [04/19 01:43:31] ppgan.engine.trainer INFO: Iter: 2780/60000 lr: 9.947e-05 loss_promary: 14.764 loss_dual: 14.157 loss_total: 28.921 batch_cost: 0.58211 sec reader_cost: 0.00611 sec ips: 1.71790 images/s eta: 9:15:07
    [04/19 01:43:42] ppgan.engine.trainer INFO: Iter: 2800/60000 lr: 9.946e-05 loss_promary: 13.649 loss_dual: 12.618 loss_total: 26.267 batch_cost: 0.58097 sec reader_cost: 0.00576 sec ips: 1.72125 images/s eta: 9:13:51
    [04/19 01:43:56] ppgan.engine.trainer INFO: Iter: 2820/60000 lr: 9.946e-05 loss_promary: 13.505 loss_dual: 13.115 loss_total: 26.620 batch_cost: 0.66120 sec reader_cost: 0.08474 sec ips: 1.51239 images/s eta: 10:30:07
    [04/19 01:44:07] ppgan.engine.trainer INFO: Iter: 2840/60000 lr: 9.945e-05 loss_promary: 14.352 loss_dual: 9.050 loss_total: 23.402 batch_cost: 0.58132 sec reader_cost: 0.00585 sec ips: 1.72023 images/s eta: 9:13:47
    [04/19 01:44:19] ppgan.engine.trainer INFO: Iter: 2860/60000 lr: 9.944e-05 loss_promary: 12.959 loss_dual: 14.089 loss_total: 27.048 batch_cost: 0.58150 sec reader_cost: 0.00634 sec ips: 1.71968 images/s eta: 9:13:46
    [04/19 01:44:31] ppgan.engine.trainer INFO: Iter: 2880/60000 lr: 9.943e-05 loss_promary: 18.652 loss_dual: 13.910 loss_total: 32.562 batch_cost: 0.58164 sec reader_cost: 0.00635 sec ips: 1.71927 images/s eta: 9:13:42
    [04/19 01:44:42] ppgan.engine.trainer INFO: Iter: 2900/60000 lr: 9.943e-05 loss_promary: 15.142 loss_dual: 12.142 loss_total: 27.285 batch_cost: 0.58235 sec reader_cost: 0.00624 sec ips: 1.71718 images/s eta: 9:14:11
    [04/19 01:44:55] ppgan.engine.trainer INFO: Iter: 2920/60000 lr: 9.942e-05 loss_promary: 12.317 loss_dual: 6.153 loss_total: 18.470 batch_cost: 0.65899 sec reader_cost: 0.08170 sec ips: 1.51747 images/s eta: 10:26:54
    [04/19 01:45:07] ppgan.engine.trainer INFO: Iter: 2940/60000 lr: 9.941e-05 loss_promary: 17.240 loss_dual: 8.534 loss_total: 25.774 batch_cost: 0.58189 sec reader_cost: 0.00632 sec ips: 1.71854 images/s eta: 9:13:22
    [04/19 01:45:19] ppgan.engine.trainer INFO: Iter: 2960/60000 lr: 9.940e-05 loss_promary: 12.092 loss_dual: 7.342 loss_total: 19.434 batch_cost: 0.58115 sec reader_cost: 0.00604 sec ips: 1.72073 images/s eta: 9:12:28
    [04/19 01:45:30] ppgan.engine.trainer INFO: Iter: 2980/60000 lr: 9.939e-05 loss_promary: 15.261 loss_dual: 14.130 loss_total: 29.392 batch_cost: 0.58151 sec reader_cost: 0.00623 sec ips: 1.71968 images/s eta: 9:12:36
    [04/19 01:45:42] ppgan.engine.trainer INFO: Iter: 3000/60000 lr: 9.939e-05 loss_promary: 14.025 loss_dual: 8.749 loss_total: 22.774 batch_cost: 0.58096 sec reader_cost: 0.00577 sec ips: 1.72129 images/s eta: 9:11:54
    [04/19 01:45:57] ppgan.engine.trainer INFO: Iter: 3020/60000 lr: 9.938e-05 loss_promary: 14.293 loss_dual: 9.987 loss_total: 24.280 batch_cost: 0.65870 sec reader_cost: 0.08094 sec ips: 1.51814 images/s eta: 10:25:32
    [04/19 01:46:09] ppgan.engine.trainer INFO: Iter: 3040/60000 lr: 9.937e-05 loss_promary: 13.811 loss_dual: 8.890 loss_total: 22.701 batch_cost: 0.58263 sec reader_cost: 0.00629 sec ips: 1.71637 images/s eta: 9:13:05
    [04/19 01:46:20] ppgan.engine.trainer INFO: Iter: 3060/60000 lr: 9.936e-05 loss_promary: 15.104 loss_dual: 8.828 loss_total: 23.932 batch_cost: 0.58170 sec reader_cost: 0.00630 sec ips: 1.71909 images/s eta: 9:12:01
    [04/19 01:46:32] ppgan.engine.trainer INFO: Iter: 3080/60000 lr: 9.935e-05 loss_promary: 14.856 loss_dual: 12.739 loss_total: 27.596 batch_cost: 0.58225 sec reader_cost: 0.00645 sec ips: 1.71746 images/s eta: 9:12:21
    [04/19 01:46:43] ppgan.engine.trainer INFO: Iter: 3100/60000 lr: 9.934e-05 loss_promary: 11.349 loss_dual: 12.099 loss_total: 23.448 batch_cost: 0.58279 sec reader_cost: 0.00624 sec ips: 1.71587 images/s eta: 9:12:40
    [04/19 01:46:57] ppgan.engine.trainer INFO: Iter: 3120/60000 lr: 9.934e-05 loss_promary: 15.433 loss_dual: 10.784 loss_total: 26.217 batch_cost: 0.66053 sec reader_cost: 0.08301 sec ips: 1.51394 images/s eta: 10:26:10
    [04/19 01:47:08] ppgan.engine.trainer INFO: Iter: 3140/60000 lr: 9.933e-05 loss_promary: 17.642 loss_dual: 8.609 loss_total: 26.251 batch_cost: 0.58074 sec reader_cost: 0.00601 sec ips: 1.72193 images/s eta: 9:10:20
    [04/19 01:47:20] ppgan.engine.trainer INFO: Iter: 3160/60000 lr: 9.932e-05 loss_promary: 14.461 loss_dual: 10.638 loss_total: 25.099 batch_cost: 0.58170 sec reader_cost: 0.00606 sec ips: 1.71909 images/s eta: 9:11:03
    [04/19 01:47:32] ppgan.engine.trainer INFO: Iter: 3180/60000 lr: 9.931e-05 loss_promary: 15.401 loss_dual: 7.740 loss_total: 23.141 batch_cost: 0.57842 sec reader_cost: 0.00552 sec ips: 1.72885 images/s eta: 9:07:45
    [04/19 01:47:43] ppgan.engine.trainer INFO: Iter: 3200/60000 lr: 9.930e-05 loss_promary: 10.359 loss_dual: 7.368 loss_total: 17.727 batch_cost: 0.58063 sec reader_cost: 0.00571 sec ips: 1.72227 images/s eta: 9:09:39
    [04/19 01:47:56] ppgan.engine.trainer INFO: Iter: 3220/60000 lr: 9.929e-05 loss_promary: 15.986 loss_dual: 10.090 loss_total: 26.076 batch_cost: 0.66185 sec reader_cost: 0.08387 sec ips: 1.51091 images/s eta: 10:26:19
    [04/19 01:48:08] ppgan.engine.trainer INFO: Iter: 3240/60000 lr: 9.928e-05 loss_promary: 12.029 loss_dual: 8.194 loss_total: 20.222 batch_cost: 0.58408 sec reader_cost: 0.00673 sec ips: 1.71211 images/s eta: 9:12:31
    [04/19 01:48:20] ppgan.engine.trainer INFO: Iter: 3260/60000 lr: 9.927e-05 loss_promary: 17.556 loss_dual: 10.413 loss_total: 27.969 batch_cost: 0.58233 sec reader_cost: 0.00614 sec ips: 1.71724 images/s eta: 9:10:40
    [04/19 01:48:31] ppgan.engine.trainer INFO: Iter: 3280/60000 lr: 9.927e-05 loss_promary: 14.315 loss_dual: 7.516 loss_total: 21.832 batch_cost: 0.58236 sec reader_cost: 0.00598 sec ips: 1.71714 images/s eta: 9:10:31
    [04/19 01:48:43] ppgan.engine.trainer INFO: Iter: 3300/60000 lr: 9.926e-05 loss_promary: 12.701 loss_dual: 7.517 loss_total: 20.218 batch_cost: 0.58223 sec reader_cost: 0.00617 sec ips: 1.71754 images/s eta: 9:10:11
    [04/19 01:48:56] ppgan.engine.trainer INFO: Iter: 3320/60000 lr: 9.925e-05 loss_promary: 10.371 loss_dual: 5.356 loss_total: 15.727 batch_cost: 0.66191 sec reader_cost: 0.08517 sec ips: 1.51079 images/s eta: 10:25:16
    [04/19 01:49:08] ppgan.engine.trainer INFO: Iter: 3340/60000 lr: 9.924e-05 loss_promary: 19.273 loss_dual: 8.842 loss_total: 28.115 batch_cost: 0.58188 sec reader_cost: 0.00635 sec ips: 1.71858 images/s eta: 9:09:28
    [04/19 01:49:20] ppgan.engine.trainer INFO: Iter: 3360/60000 lr: 9.923e-05 loss_promary: 13.212 loss_dual: 8.035 loss_total: 21.247 batch_cost: 0.58134 sec reader_cost: 0.00620 sec ips: 1.72017 images/s eta: 9:08:46
    [04/19 01:49:31] ppgan.engine.trainer INFO: Iter: 3380/60000 lr: 9.922e-05 loss_promary: 18.137 loss_dual: 11.269 loss_total: 29.406 batch_cost: 0.58239 sec reader_cost: 0.00632 sec ips: 1.71707 images/s eta: 9:09:34
    [04/19 01:49:43] ppgan.engine.trainer INFO: Iter: 3400/60000 lr: 9.921e-05 loss_promary: 8.742 loss_dual: 6.220 loss_total: 14.962 batch_cost: 0.58153 sec reader_cost: 0.00607 sec ips: 1.71961 images/s eta: 9:08:33
    [04/19 01:49:56] ppgan.engine.trainer INFO: Iter: 3420/60000 lr: 9.920e-05 loss_promary: 16.886 loss_dual: 9.576 loss_total: 26.463 batch_cost: 0.65945 sec reader_cost: 0.08213 sec ips: 1.51642 images/s eta: 10:21:50
    [04/19 01:50:08] ppgan.engine.trainer INFO: Iter: 3440/60000 lr: 9.919e-05 loss_promary: 12.232 loss_dual: 10.275 loss_total: 22.508 batch_cost: 0.58086 sec reader_cost: 0.00567 sec ips: 1.72159 images/s eta: 9:07:32
    [04/19 01:50:19] ppgan.engine.trainer INFO: Iter: 3460/60000 lr: 9.918e-05 loss_promary: 11.938 loss_dual: 8.808 loss_total: 20.746 batch_cost: 0.58198 sec reader_cost: 0.00584 sec ips: 1.71826 images/s eta: 9:08:24
    [04/19 01:50:31] ppgan.engine.trainer INFO: Iter: 3480/60000 lr: 9.917e-05 loss_promary: 14.303 loss_dual: 6.294 loss_total: 20.597 batch_cost: 0.58255 sec reader_cost: 0.00584 sec ips: 1.71659 images/s eta: 9:08:45
    [04/19 01:50:43] ppgan.engine.trainer INFO: Iter: 3500/60000 lr: 9.916e-05 loss_promary: 11.199 loss_dual: 6.567 loss_total: 17.766 batch_cost: 0.58199 sec reader_cost: 0.00587 sec ips: 1.71825 images/s eta: 9:08:01
    [04/19 01:50:57] ppgan.engine.trainer INFO: Iter: 3520/60000 lr: 9.915e-05 loss_promary: 10.825 loss_dual: 6.085 loss_total: 16.910 batch_cost: 0.65896 sec reader_cost: 0.08105 sec ips: 1.51755 images/s eta: 10:20:17
    [04/19 01:51:08] ppgan.engine.trainer INFO: Iter: 3540/60000 lr: 9.914e-05 loss_promary: 11.761 loss_dual: 8.049 loss_total: 19.810 batch_cost: 0.58461 sec reader_cost: 0.00647 sec ips: 1.71055 images/s eta: 9:10:06
    [04/19 01:51:20] ppgan.engine.trainer INFO: Iter: 3560/60000 lr: 9.914e-05 loss_promary: 15.792 loss_dual: 7.255 loss_total: 23.047 batch_cost: 0.58202 sec reader_cost: 0.00619 sec ips: 1.71816 images/s eta: 9:07:28
    [04/19 01:51:32] ppgan.engine.trainer INFO: Iter: 3580/60000 lr: 9.913e-05 loss_promary: 19.550 loss_dual: 10.711 loss_total: 30.261 batch_cost: 0.58216 sec reader_cost: 0.00609 sec ips: 1.71775 images/s eta: 9:07:24
    [04/19 01:51:43] ppgan.engine.trainer INFO: Iter: 3600/60000 lr: 9.912e-05 loss_promary: 16.224 loss_dual: 9.150 loss_total: 25.375 batch_cost: 0.58303 sec reader_cost: 0.00605 sec ips: 1.71519 images/s eta: 9:08:02
    [04/19 01:51:57] ppgan.engine.trainer INFO: Iter: 3620/60000 lr: 9.911e-05 loss_promary: 19.393 loss_dual: 9.433 loss_total: 28.826 batch_cost: 0.66593 sec reader_cost: 0.08857 sec ips: 1.50166 images/s eta: 10:25:44
    [04/19 01:52:08] ppgan.engine.trainer INFO: Iter: 3640/60000 lr: 9.910e-05 loss_promary: 16.884 loss_dual: 9.102 loss_total: 25.986 batch_cost: 0.58224 sec reader_cost: 0.00584 sec ips: 1.71751 images/s eta: 9:06:54
    [04/19 01:52:20] ppgan.engine.trainer INFO: Iter: 3660/60000 lr: 9.909e-05 loss_promary: 11.678 loss_dual: 6.671 loss_total: 18.350 batch_cost: 0.58223 sec reader_cost: 0.00608 sec ips: 1.71753 images/s eta: 9:06:42
    [04/19 01:52:32] ppgan.engine.trainer INFO: Iter: 3680/60000 lr: 9.908e-05 loss_promary: 13.693 loss_dual: 8.099 loss_total: 21.792 batch_cost: 0.58256 sec reader_cost: 0.00632 sec ips: 1.71656 images/s eta: 9:06:49
    [04/19 01:52:43] ppgan.engine.trainer INFO: Iter: 3700/60000 lr: 9.907e-05 loss_promary: 11.155 loss_dual: 7.069 loss_total: 18.224 batch_cost: 0.58204 sec reader_cost: 0.00616 sec ips: 1.71810 images/s eta: 9:06:08
    [04/19 01:52:57] ppgan.engine.trainer INFO: Iter: 3720/60000 lr: 9.906e-05 loss_promary: 9.190 loss_dual: 9.003 loss_total: 18.193 batch_cost: 0.65984 sec reader_cost: 0.08177 sec ips: 1.51552 images/s eta: 10:18:55
    [04/19 01:53:08] ppgan.engine.trainer INFO: Iter: 3740/60000 lr: 9.905e-05 loss_promary: 13.700 loss_dual: 12.651 loss_total: 26.351 batch_cost: 0.58334 sec reader_cost: 0.00662 sec ips: 1.71427 images/s eta: 9:06:58
    [04/19 01:53:20] ppgan.engine.trainer INFO: Iter: 3760/60000 lr: 9.904e-05 loss_promary: 10.330 loss_dual: 10.118 loss_total: 20.448 batch_cost: 0.58213 sec reader_cost: 0.00627 sec ips: 1.71784 images/s eta: 9:05:38
    [04/19 01:53:32] ppgan.engine.trainer INFO: Iter: 3780/60000 lr: 9.903e-05 loss_promary: 9.282 loss_dual: 8.993 loss_total: 18.275 batch_cost: 0.58233 sec reader_cost: 0.00622 sec ips: 1.71723 images/s eta: 9:05:38
    [04/19 01:53:43] ppgan.engine.trainer INFO: Iter: 3800/60000 lr: 9.902e-05 loss_promary: 14.506 loss_dual: 8.278 loss_total: 22.784 batch_cost: 0.58297 sec reader_cost: 0.00613 sec ips: 1.71534 images/s eta: 9:06:02
    [04/19 01:53:56] ppgan.engine.trainer INFO: Iter: 3820/60000 lr: 9.900e-05 loss_promary: 13.400 loss_dual: 8.013 loss_total: 21.413 batch_cost: 0.65957 sec reader_cost: 0.08143 sec ips: 1.51614 images/s eta: 10:17:33
    [04/19 01:54:08] ppgan.engine.trainer INFO: Iter: 3840/60000 lr: 9.899e-05 loss_promary: 13.391 loss_dual: 9.413 loss_total: 22.805 batch_cost: 0.58234 sec reader_cost: 0.00654 sec ips: 1.71720 images/s eta: 9:05:03
    [04/19 01:54:20] ppgan.engine.trainer INFO: Iter: 3860/60000 lr: 9.898e-05 loss_promary: 9.350 loss_dual: 8.794 loss_total: 18.143 batch_cost: 0.58369 sec reader_cost: 0.00661 sec ips: 1.71325 images/s eta: 9:06:07
    [04/19 01:54:31] ppgan.engine.trainer INFO: Iter: 3880/60000 lr: 9.897e-05 loss_promary: 13.253 loss_dual: 6.582 loss_total: 19.835 batch_cost: 0.58274 sec reader_cost: 0.00663 sec ips: 1.71604 images/s eta: 9:05:02
    [04/19 01:54:43] ppgan.engine.trainer INFO: Iter: 3900/60000 lr: 9.896e-05 loss_promary: 17.851 loss_dual: 7.513 loss_total: 25.363 batch_cost: 0.58180 sec reader_cost: 0.00619 sec ips: 1.71881 images/s eta: 9:03:58
    [04/19 01:54:56] ppgan.engine.trainer INFO: Iter: 3920/60000 lr: 9.895e-05 loss_promary: 13.704 loss_dual: 6.403 loss_total: 20.107 batch_cost: 0.66146 sec reader_cost: 0.08463 sec ips: 1.51180 images/s eta: 10:18:14
    [04/19 01:55:08] ppgan.engine.trainer INFO: Iter: 3940/60000 lr: 9.894e-05 loss_promary: 14.670 loss_dual: 7.629 loss_total: 22.299 batch_cost: 0.58228 sec reader_cost: 0.00635 sec ips: 1.71740 images/s eta: 9:04:01
    [04/19 01:55:20] ppgan.engine.trainer INFO: Iter: 3960/60000 lr: 9.893e-05 loss_promary: 12.772 loss_dual: 10.192 loss_total: 22.965 batch_cost: 0.58212 sec reader_cost: 0.00644 sec ips: 1.71786 images/s eta: 9:03:41
    [04/19 01:55:31] ppgan.engine.trainer INFO: Iter: 3980/60000 lr: 9.892e-05 loss_promary: 14.273 loss_dual: 9.824 loss_total: 24.097 batch_cost: 0.57773 sec reader_cost: 0.00532 sec ips: 1.73090 images/s eta: 8:59:24
    [04/19 01:55:43] ppgan.engine.trainer INFO: Iter: 4000/60000 lr: 9.891e-05 loss_promary: 13.173 loss_dual: 8.718 loss_total: 21.891 batch_cost: 0.57984 sec reader_cost: 0.00562 sec ips: 1.72461 images/s eta: 9:01:10
    [04/19 01:55:47] ppgan.engine.trainer INFO: Test iter: [0/20]
    [04/19 01:57:07] ppgan.engine.trainer INFO: Metric psnr: 24.6712
    [04/19 01:57:07] ppgan.engine.trainer INFO: Metric ssim: 0.7347
    Traceback (most recent call last):
      File "tools/main.py", line 56, in <module>
        main(args, cfg)
      File "tools/main.py", line 46, in main
        trainer.train()
      File "/home/aistudio/PaddleGAN/ppgan/engine/trainer.py", line 175, in train
        self.model.train_iter(self.optimizers)
      File "/home/aistudio/PaddleGAN/ppgan/models/drn_model.py", line 128, in train_iter
        loss_total.backward()
      File "</opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/decorator.py:decorator-gen-114>", line 2, in backward
      File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/wrapped_decorator.py", line 25, in __impl__
        return wrapped_func(*args, **kwargs)
      File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/framework.py", line 225, in __impl__
        return func(*args, **kwargs)
      File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/varbase_patch_methods.py", line 177, in backward
        self._run_backward(framework._dygraph_tracer(), retain_graph)
      File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/multiprocess_utils.py", line 134, in __handler__
        core._throw_error_if_process_failed()
    SystemError: (Fatal) DataLoader process (pid   1. If run DataLoader by DataLoader.from_generator(...), queue capacity is set by from_generator(..., capacity=xx, ...).
      2. If run DataLoader by DataLoader(dataset, ...), queue capacity is set as 2 times of the max value of num_workers and len(places).
      3. If run by DataLoader(dataset, ..., use_shared_memory=True), set use_shared_memory=False for not using shared memory.) exited is killed by signal: 13388.
      It may be caused by insufficient shared storage space. This problem usually occurs when using docker as a development environment.
      Please use command `df -h` to check the storage space of `/dev/shm`. Shared storage space needs to be greater than (DataLoader Num * DataLoader queue capacity * 1 batch data size).
      You can solve this problem by increasing the shared storage space or reducing the queue capacity appropriately.
    Bus error (at /paddle/paddle/fluid/imperative/data_loader.cc:161)
    


### 测试模型
以ESRGAN为例，模型训练好后，运行以下代码测试ESRGAN模型。

其中``/home/aistudio/pretrained_model/ESRGAN_PSNR_50000_weight.pdparams``是刚才ESRGAN训练的模型参数，同学们需要换成自己的模型参数。

如果希望使用其他模型测试，可以修改配置文件名字。


```python
%cd /home/aistudio/PaddleGAN/
# !python tools/main.py --config-file configs/esrgan_psnr_x4_div2k.yaml --evaluate-only --load /home/aistudio/pretrained_model/ESRGAN_PSNR_50000_weight.pdparams
!python tools/main.py --config-file configs/drn_psnr_x4_div2k_hw4.yaml --evaluate-only --load /home/aistudio/pretrained_model/DRN_PSNR_60000_weight.pdparams
```

### 实验结果展示及模型下载
这里使用ESRGAN模型训练了一个基于PSNR指标的预测模型和一个基于GAN的预测模型。

数值结果展示及模型下载

| 方法 | 数据集 | 迭代次数 | 训练时长 | PSNR | SSIM | 模型下载 |
|---|---|---|---|---|---|---|
| ESRGAN_PSNR  | 卡通画超分数据集 | 50000 | 13.5h | 25.4782 | 0.7608 |[ESRGAN_PSNR](./pretrained_model/ESRGAN_PSNR_50000_weight.pdparams)|
| ESRGAN_GAN  | 卡通画超分数据集 | 50000 | 11h | 21.4148 | 0.6176 |[ESRGAN_GAN](./pretrained_model/ESRGAN_GAN_50000_weight.pdparams)|
| ESRGAN_GAN  | 卡通画超分数据集 | 50000 | 11h | 21.4148 | 0.6176 |[ESRGAN_GAN](./pretrained_model/ESRGAN_GAN_50000_weight.pdparams)|
| **DRN_PSNR**  | 卡通画超分数据集 | 50000 | 15h | 25.4905 | 0.7619 |[DRN_PSNR](./pretrained_model/DRN_PSNR_50000_weight.pdparams)|

*DRN_PSNR*结果可视化
| 低分辨率 | DRN_PSNR | GT |
|---|---|---|
|![](images/Anime_419_lq.png)|![](images/Anime_419_output.png)|![](images/Anime_419_gt.png)|
|![](images/Anime_401_lq.png)|![](images/Anime_401_output.png)|![](images/Anime_401_gt.png)|
|![](images/Anime_407_lq.png)|![](images/Anime_407_output.png)|![](images/Anime_407_gt.png)|

| 低分辨率 | ESRGAN_PSNR | ESRGAN_GAN | GT |
|---|---|---|---|
|![](./image/Anime_401_lq.png)|![](./image/Anime_401_psnr.png)|![](./image/Anime_401_gan.png)|![](./image/Anime_401_gt.png)|
|![](./image/Anime_407_lq.png)|![](./image/Anime_407_psnr.png)|![](./image/Anime_407_gan.png)|![](./image/Anime_407_gt.png)|



## 本地运行

Docker运行记录备忘

```shell
docker run --rm --gpus '"device=1"' -it -v $(pwd):/paddlegan -w /paddlegan --privileged --ipc=host -p 127.0.0.1:6006:6006 paddlepaddle/paddle:2.0.2-gpu-cuda10.1-cudnn7 bash

## 修改/dev/shm大小，防止dataloader报错
## fstab文件中的内容如下：
# tmpfs                   /dev/shm                tmpfs   defaults,size=2048M         0  0
cp fstab /etc/
mount -o remount /dev/shm
df -h /dev/shm/
```

对于容器中的numba依赖包版本0.48无法成功安装到Python3.5，可能要考虑新装Python3.7到容器中，然后更新pip、paddlepaddle-gpu。

```shell
# 训练
python3 -u tools/main.py --config-file configs/drn_psnr_x4_div2k_hw4.yaml
# 恢复训练
python3 -u tools/main.py --config-file configs/drn_psnr_x4_div2k_hw4.yaml --resume pre_weights/iter_4000_checkpoint.pdparams

# 测试
python3 tools/main.py --config-file configs/drn_psnr_x4_div2k_hw4.yaml --evaluate-only --load output_dir/drn_psnr_x4_div2k_hw4-2021-04-19-08-41/iter_50000_weight.pdparams
```

*训练过程中可用VisualDL监控loss等数据的变化情况。*

drn_psnr_x4_div2k_hw4.yaml详情如下：

```
total_iters: 50000
output_dir: output_dir
# tensor range for function tensor2img
min_max:
  (0., 255.)

model:
  name: DRN
  generator:
    name: DRNGenerator
    scale: (2, 4)
    n_blocks: 30
    n_feats: 16
    n_colors: 3
    rgb_range: 255
    negval: 0.2
  pixel_criterion:
    name: L1Loss

dataset:
  train:
    name: SRDataset
    gt_folder: data/animeSR/train
    lq_folder: data/animeSR/train_X4
    num_workers: 4
    batch_size: 8
    scale: 4
    preprocess:
      - name: LoadImageFromFile
        key: lq
      - name: LoadImageFromFile
        key: gt
      - name: Transforms
        input_keys: [lq, gt]
        output_keys: [lq, lqx2, gt]
        pipeline:
          - name: SRPairedRandomCrop
            gt_patch_size: 384
            scale: 4
            scale_list: True
            keys: [image, image]
          - name: PairedRandomHorizontalFlip
            keys: [image, image, image]
          - name: PairedRandomVerticalFlip
            keys: [image, image, image]
          - name: PairedRandomTransposeHW
            keys: [image, image, image]
          - name: Transpose
            keys: [image, image, image]
          - name: Normalize
            mean: [0., 0., 0.]
            std: [1., 1., 1.]
            keys: [image, image, image]
  test:
    name: SRDataset
    gt_folder: data/animeSR/test
    lq_folder: data/animeSR/test_X4
    scale: 4
    preprocess:
      - name: LoadImageFromFile
        key: lq
      - name: LoadImageFromFile
        key: gt
      - name: Transforms
        input_keys: [lq, gt]
        pipeline:
          - name: Transpose
            keys: [image, image]
          - name: Normalize
            mean: [0., 0., 0.]
            std: [1., 1., 1.]
            keys: [image, image]

lr_scheduler:
  name: CosineAnnealingRestartLR
  learning_rate: 0.0001
  periods: [50000]
  restart_weights: [1]
  eta_min: !!float 1e-7

optimizer:
  optimG:
    name: Adam
    net_names:
      - generator
    weight_decay: 0.0
    beta1: 0.9
    beta2: 0.999
  optimD:
    name: Adam
    net_names:
      - dual_model_0
      - dual_model_1
    weight_decay: 0.0
    beta1: 0.9
    beta2: 0.999

validate:
  interval: 2000
  save_img: false

  metrics:
    psnr: # metric name, can be arbitrary
      name: PSNR
      crop_border: 4
      test_y_channel: True
    ssim:
      name: SSIM
      crop_border: 4
      test_y_channel: True

log_config:
  interval: 20
  visiual_interval: 500

snapshot_config:
  interval: 1000
```  
