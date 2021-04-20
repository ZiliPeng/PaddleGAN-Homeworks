# 万众期待的【多人版】蚂蚁呀嘿🐜 来了！--PaddleGAN良心之作

大家是不是玩「单人版」的蚂蚁呀嘿玩的很爽了呢？开始觉得不被满足？想要尝试与别人共舞「蚂蚁呀嘿」？想要的永远比拥有的更多？
![](https://ai-studio-static-online.cdn.bcebos.com/9e17be2b3a5f49ce90cc42d6f05d3638490dd4c09f3246ec9fec2a5e0b1d3112)

别慌！！[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)良心制作，在单人版被大家疯传之后，今天，[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)带来了多人版本的实现！！

**无需切换任何软件或安装包，手把手教你如何一键实现多人乱舞蚂蚁呀嘿**🤩🤩🤩

🧡💛🧡💛🧡🧡💛🧡💛🧡🧡💛🧡💛🧡🧡💛🧡💛🧡🧡💛🧡💛🧡🧡💛🧡💛🧡

**🥳🥳 更绝的是，此次升级版无论是单人版还是多人版都可通用！🥳🥳**

🧡💛🧡💛🧡🧡💛🧡💛🧡🧡💛🧡💛🧡🧡💛🧡💛🧡🧡💛🧡💛🧡🧡💛🧡💛🧡

接下来，快和[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)一起来动手实现吧！

整个实现步骤还是老样子，分为三步：
1. 下载[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)
2. 运行First Order Motion命令
3. 配上音乐🎵

然后，优秀的你就拥有了一个顶级宝藏技能，一键实现多人蚂蚁呀嘿！！快去和你身边朋友炫耀吧~~

不要忘记给[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)点Star支持噢~~![](https://ai-studio-static-online.cdn.bcebos.com/dadbafa290874810bb3ff387f7c76ff250ff97aef1534e418b24ee0a4c97bee1)

## 接下来，我们来让百年前油画中的人物一起「蚂蚁呀嘿」吧~

PaddleGAN施魔法前：

<div align='left'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/32c666383dc44666b76effb76a094e4ffc57e3d5b1104fe8a7f3759378b188f7' width='600'/>
</div>

PaddleGAN施魔法后：

![](https://ai-studio-static-online.cdn.bcebos.com/fb3b30f51a3545fab0508ac72c4f9c5d9e943563f5e74466a8b5dbf2f77aed6c)


## 【多人版】蚂蚁呀嘿技术流程
整体流程分为三步：
1. 将照片中的多人脸使用人脸检测模型S3FD框出并抠出
2. 对抠出的人脸用First Order Motion进行脸部表情迁移
3. 将迁移好的人脸放回对应的原位置

**注意，由于人脸检测后，需要确定位置将人脸框出来并抠出，如果人脸太近，会框到除了其他的脸，导致效果较差，如下图**

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/b5d61c87269d4725ab194879343e56ce232b497ec6974f9e84881e4c199f6a40' width='300'/>
</div>

**所以尽量选取人脸间距较大的照片，同时，框的大小需要根据照片中人脸间距情况进行调节（参数--ratio）**

## 下载PaddleGAN


```python
# 从github上克隆PaddleGAN代码
#!git clone https://github.com/PaddlePaddle/PaddleGAN
!git clone https://gitee.com/paddlepaddle/PaddleGAN.git
%cd PaddleGAN
!git checkout develop
```

    /home/aistudio/PaddleGAN
    fatal: Not a git repository (or any parent up to mount point /home/aistudio)
    Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).



```python
%cd PaddleGAN
!git checkout develop
```

    [Errno 2] No such file or directory: 'PaddleGAN'
    /home/aistudio/PaddleGAN
    fatal: Not a git repository (or any parent up to mount point /home/aistudio)
    Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).



```python
# 安装所需安装包
%cd PaddleGAN/
!pip install -r requirements.txt
!pip install imageio-ffmpeg
%cd applications/
```

    [Errno 2] No such file or directory: 'PaddleGAN/'
    /home/aistudio/PaddleGAN
    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Requirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 1)) (4.36.1)
    Requirement already satisfied: PyYAML>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 2)) (5.1.2)
    Collecting scikit-image>=0.14.0 (from -r requirements.txt (line 3))
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/fe/01/3a830f3df578ea3ed94ee7fd9f91e85c3dec2431d8548ab1c91869e51450/scikit_image-0.18.1-cp37-cp37m-manylinux1_x86_64.whl (29.2MB)
    [K     |████████████████████████████████| 29.2MB 8.9MB/s eta 0:00:012
    [?25hRequirement already satisfied: scipy>=1.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 4)) (1.3.0)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 5)) (4.1.1.26)
    Requirement already satisfied: imageio-ffmpeg in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 6)) (0.3.0)
    Collecting librosa==0.7.0 (from -r requirements.txt (line 7))
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/ad/6e/0eb0de1c9c4e02df0b40e56f258eb79bd957be79b918511a184268e01720/librosa-0.7.0.tar.gz (1.6MB)
    [K     |████████████████████████████████| 1.6MB 31.0MB/s eta 0:00:01
    [?25hRequirement already satisfied: numba==0.48 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 8)) (0.48.0)
    Requirement already satisfied: easydict in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 9)) (1.9)
    Collecting munch (from -r requirements.txt (line 10))
      Downloading https://mirror.baidu.com/pypi/packages/cc/ab/85d8da5c9a45e072301beb37ad7f833cd344e04c817d97e0cc75681d248f/munch-2.5.0-py2.py3-none-any.whl
    Requirement already satisfied: matplotlib!=3.0.0,>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->-r requirements.txt (line 3)) (2.2.3)
    Requirement already satisfied: pillow!=7.1.0,!=7.1.1,>=4.3.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->-r requirements.txt (line 3)) (7.1.2)
    Collecting PyWavelets>=1.1.1 (from scikit-image>=0.14.0->-r requirements.txt (line 3))
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/62/bd/592c7242fdd1218a96431512e77265c50812315ef72570ace85e1cfae298/PyWavelets-1.1.1-cp37-cp37m-manylinux1_x86_64.whl (4.4MB)
    [K     |████████████████████████████████| 4.4MB 13.5MB/s eta 0:00:01
    [?25hCollecting numpy>=1.16.5 (from scikit-image>=0.14.0->-r requirements.txt (line 3))
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/73/ef/8967d406f3f85018ceb5efab50431e901683188f1741ceb053efcab26c87/numpy-1.20.2-cp37-cp37m-manylinux2010_x86_64.whl (15.3MB)
    [K     |████████████████████████████████| 15.3MB 8.3MB/s eta 0:00:011
    [?25hRequirement already satisfied: imageio>=2.3.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->-r requirements.txt (line 3)) (2.6.1)
    Collecting tifffile>=2019.7.26 (from scikit-image>=0.14.0->-r requirements.txt (line 3))
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/c0/9b/db2b4777156c755ea589cb93ae50fc12a39119623bd7eca9bb8eaab523fc/tifffile-2021.4.8-py3-none-any.whl (165kB)
    [K     |████████████████████████████████| 174kB 26.0MB/s eta 0:00:01
    [?25hRequirement already satisfied: networkx>=2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->-r requirements.txt (line 3)) (2.4)
    Requirement already satisfied: audioread>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->-r requirements.txt (line 7)) (2.1.8)
    Requirement already satisfied: scikit-learn!=0.19.0,>=0.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->-r requirements.txt (line 7)) (0.22.1)
    Requirement already satisfied: joblib>=0.12 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->-r requirements.txt (line 7)) (0.14.1)
    Requirement already satisfied: decorator>=3.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->-r requirements.txt (line 7)) (4.4.0)
    Requirement already satisfied: six>=1.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->-r requirements.txt (line 7)) (1.15.0)
    Requirement already satisfied: resampy>=0.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->-r requirements.txt (line 7)) (0.2.2)
    Requirement already satisfied: soundfile>=0.9.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->-r requirements.txt (line 7)) (0.10.3.post1)
    Requirement already satisfied: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from numba==0.48->-r requirements.txt (line 8)) (41.4.0)
    Requirement already satisfied: llvmlite<0.32.0,>=0.31.0dev0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from numba==0.48->-r requirements.txt (line 8)) (0.31.0)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.14.0->-r requirements.txt (line 3)) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.14.0->-r requirements.txt (line 3)) (1.1.0)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.14.0->-r requirements.txt (line 3)) (2.8.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.14.0->-r requirements.txt (line 3)) (2.4.2)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.14.0->-r requirements.txt (line 3)) (2019.3)
    Requirement already satisfied: cffi>=1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from soundfile>=0.9.0->librosa==0.7.0->-r requirements.txt (line 7)) (1.14.0)
    Requirement already satisfied: pycparser in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from cffi>=1.0->soundfile>=0.9.0->librosa==0.7.0->-r requirements.txt (line 7)) (2.19)
    Building wheels for collected packages: librosa
      Building wheel for librosa (setup.py) ... [?25ldone
    [?25h  Created wheel for librosa: filename=librosa-0.7.0-cp37-none-any.whl size=1598346 sha256=db2fa6d938937f19b35fbe39000e0731636bf5ac2c436afc7f1f08e5bc2db55d
      Stored in directory: /home/aistudio/.cache/pip/wheels/81/7e/60/c27574fffbf2f28075dbf4b28c00d3fe3240fefb51d597932e
    Successfully built librosa
    [31mERROR: blackhole 0.3.2 has requirement xgboost==1.1.0, but you'll have xgboost 1.3.3 which is incompatible.[0m
    Installing collected packages: numpy, PyWavelets, tifffile, scikit-image, librosa, munch
      Found existing installation: numpy 1.16.4
        Uninstalling numpy-1.16.4:
          Successfully uninstalled numpy-1.16.4
      Found existing installation: librosa 0.7.2
        Uninstalling librosa-0.7.2:
          Successfully uninstalled librosa-0.7.2
    Successfully installed PyWavelets-1.1.1 librosa-0.7.0 munch-2.5.0 numpy-1.20.2 scikit-image-0.18.1 tifffile-2021.4.8
    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Requirement already satisfied: imageio-ffmpeg in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (0.3.0)
    /home/aistudio/PaddleGAN/applications


## 执行命令

大家可以上传自己准备的视频和图片，并在如下命令中的source_image参数和driving_video参数分别换成自己的图片和视频路径，然后运行如下命令，就可以完成动作表情迁移，程序运行成功后，会在ouput文件夹生成名为result.mp4的视频文件，该文件即为动作迁移后的视频。

同时，根据自己上传的照片中人脸的间距，

本项目中提供了原始图片和驱动视频供展示使用。具体的各参数使用说明如下
- driving_video: 驱动视频，视频中人物的表情动作作为待迁移的对象
- source_image: 原始图片，视频中人物的表情动作将迁移到该原始图片中的人物上
- relative: 指示程序中使用视频和图片中人物关键点的相对坐标还是绝对坐标，建议使用相对坐标，若使用绝对坐标，会导致迁移后人物扭曲变形
- adapt_scale: 根据关键点凸包自适应运动尺度
- ratio：将框出来的人脸贴回原图时的区域占宽高的比例，默认为0.4，范围为【0.4，0.5】





```python
!export PYTHONPATH=$PYTHONPATH:/home/aistudio/PaddleGAN && python -u tools/first-order-demo.py  --driving_video ~/work/2.MP4  --source_image ~/work/1.jpg --ratio 0.4 --relative --adapt_scale 
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/scipy/linalg/__init__.py:217: DeprecationWarning: The module numpy.dual is deprecated.  Instead of using dual, use the functions directly from numpy or scipy.
      from numpy.dual import register_func
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/scipy/special/orthogonal.py:81: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around, int,
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/scipy/io/matlab/mio5.py:98: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      from .mio5_utils import VarReader5
    [04/19 22:21:04] ppgan INFO: Found /home/aistudio/.cache/ppgan/vox-cpk.pdparams
    W0419 22:21:04.113543   783 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0419 22:21:04.118865   783 device_context.cc:372] device: 0, cuDNN Version: 7.6.
    2021-04-19 22:21:11,599 - INFO - unique_endpoints {''}
    2021-04-19 22:21:11,599 - INFO - Found /home/aistudio/.cache/paddle/hapi/weights/s3fd_paddle.pdparams
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:143: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if data.dtype == np.object:
    100%|█████████████████████████████████████████| 222/222 [00:06<00:00, 33.08it/s]
    100%|█████████████████████████████████████████| 222/222 [00:06<00:00, 33.33it/s]
    100%|█████████████████████████████████████████| 222/222 [00:06<00:00, 32.48it/s]
    100%|█████████████████████████████████████████| 222/222 [00:06<00:00, 32.86it/s]
    100%|█████████████████████████████████████████| 222/222 [00:06<00:00, 32.73it/s]
    2021-04-19 22:22:07,362 - WARNING - IMAGEIO FFMPEG_WRITER WARNING: input image is not divisible by macro_block_size=16, resizing from (1200, 900) to (1200, 912) to ensure video compatibility with most codecs and players. To prevent resizing, make your input image divisible by the macro_block_size or set the macro_block_size to 1 (risking incompatibility).


## 最后一步：使用moviepy为生成的视频加上音乐


```python
# add audio
!pip install moviepy
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Requirement already satisfied: moviepy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (1.0.1)
    Requirement already satisfied: proglog<=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from moviepy) (0.1.9)
    Requirement already satisfied: decorator<5.0,>=4.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from moviepy) (4.4.0)
    Requirement already satisfied: imageio<3.0,>=2.5; python_version >= "3.4" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from moviepy) (2.6.1)
    Requirement already satisfied: requests<3.0,>=2.8.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from moviepy) (2.22.0)
    Requirement already satisfied: imageio-ffmpeg>=0.2.0; python_version >= "3.4" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from moviepy) (0.3.0)
    Requirement already satisfied: tqdm<5.0,>=4.11.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from moviepy) (4.36.1)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from moviepy) (1.20.2)
    Requirement already satisfied: pillow in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from imageio<3.0,>=2.5; python_version >= "3.4"->moviepy) (7.1.2)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests<3.0,>=2.8.1->moviepy) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests<3.0,>=2.8.1->moviepy) (1.25.6)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests<3.0,>=2.8.1->moviepy) (2.8)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests<3.0,>=2.8.1->moviepy) (2019.9.11)



```python
#为生成的视频加上音乐
from moviepy.editor import *

videoclip_1 = VideoFileClip("/home/aistudio/work/2.MP4")
videoclip_2 = VideoFileClip("./output/result.mp4")

audio_1 = videoclip_1.audio

videoclip_3 = videoclip_2.set_audio(audio_1)
videoclip_3.write_videofile("./output/finalout.mp4", audio_codec="aac")
```

    t:   0%|          | 0/223 [00:00<?, ?it/s, now=None]                

    Moviepy - Building video ./output/finalout.mp4.
    MoviePy - Writing audio in finaloutTEMP_MPY_wvf_snd.mp4
    MoviePy - Done.
    Moviepy - Writing video ./output/finalout.mp4
    


    t:  12%|█▏        | 27/223 [00:00<00:01, 129.00it/s, now=None]                                                              


    Moviepy - Done !
    Moviepy - video ready ./output/finalout.mp4


# 多人版的蚂蚁呀嘿完成！

至此，多人版本的蚂蚁呀嘿就制作完成啦~

**大家快来动手尝试吧！记住，选择人物间距较大的合照效果更佳噢！**

当然，[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)的应用也不会止步于此，[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)还能提供各类不同的能力，包括**唇形合成（对嘴型）、视频/照片修复（上色、超分、插帧）、人脸动漫化、照片动漫化**等等！！一个比一个更厉害！

强烈鼓励大家玩起来，激发[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)的潜能！

记得点个Star收藏噢~~
![](https://ai-studio-static-online.cdn.bcebos.com/e2a76e0a1b864259a11b3556dda2f6838503e476637a408f946b333d38438455)

