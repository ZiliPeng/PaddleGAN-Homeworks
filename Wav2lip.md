# 宋代诗人念诗的秘密--PaddleGAN实现精准唇形合成

## 宋代著名诗人苏轼「动起来」的秘密
就在上周，坐拥百万粉丝的**独立艺术家大谷Spitzer老师**利用深度学习技术使**宋代诗人苏轼活过来，穿越千年，为屏幕前的你们亲自朗诵其著名古诗~** [点击量](https://www.bilibili.com/video/BV1mt4y1z7W8)近百万，同时激起百万网友热议，到底是什么技术这么牛气？

![](https://ai-studio-static-online.cdn.bcebos.com/c21d8a1de3084b6ca599bc2cda373d3fef4b1a0ae98646f4b629dae14c9bb1f4)


## PaddleGAN的唇形迁移能力--Wav2lip
**铛铛铛！！飞桨[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)这就来给大家揭秘，手把手教大家如何实现唇型的迁移，学习过本项目的你们，从此不仅能让苏轼念诗，还能让蒙娜丽莎播新闻、新闻主播唱Rap... 只有你想不到的，没有[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)做不到的！**

本教程是基于[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)实现的视频唇形同步模型**Wav2lip**, 它实现了人物口型与输入语音同步，俗称「对口型」。 比如这样：
![](https://ai-studio-static-online.cdn.bcebos.com/16d0b24fdc5c451395b3b308cf27b59bd4b024366b41457dbb80d0105f938849)

**不仅仅让静态图像会「说话」，Wav2lip还可以直接将动态的视频，进行唇形转换，输出与目标语音相匹配的视频，自制视频配音不是梦！**

本次教程包含四个部分：

- Wav2lip原理讲解
- 下载PaddleGAN代码
- 唇形动作合成命令使用说明
- 成果展示

**若是大家喜欢这个教程，欢迎到[Github PaddleGAN主页](https://github.com/PaddlePaddle/PaddleGAN)点击star呀！下面就让我们一起动手实现吧！**
<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/47cea097a0284dd39fc2804a53aa8ee6dad16ffe104641258046eb05af49cd64' width='1000'/>
</div>

## Wav2lip模型原理
Wav2lip实现唇形与语音精准同步突破的关键在于，它采用了**唇形同步判别器，以强制生成器持续产生准确而逼真的唇部运动。**

此外，该研究通过在鉴别器中，使用**多个连续帧而不是单个帧，并使用视觉质量损失（而不仅仅是对比损失）来考虑时间相关性，从而改善了视觉质量。**

该wav2lip模型几乎是**万能**的，适用于任何**人脸**、**任何语音**、**任何语言**，对任意视频都能达到很高的准确率，可以无缝地与原始视频融合，还可以用于**转换动画人脸，并且导入合成语音**也是可行的

## 下载PaddleGAN代码


```python
# 下载PaddlePaddle安装包
%cd /home/aistudio/work
```

    /home/aistudio/work



```python
# 从github上克隆PaddleGAN代码（如下载速度过慢，可用gitee源）
!git clone https://gitee.com/PaddlePaddle/PaddleGAN
#!git clone https://github.com/PaddlePaddle/PaddleGAN


```

    Cloning into 'PaddleGAN'...
    remote: Enumerating objects: 2995, done.[K
    remote: Counting objects: 100% (2995/2995), done.[K
    remote: Compressing objects: 100% (1759/1759), done.[K
    remote: Total 2995 (delta 1910), reused 1838 (delta 1146), pack-reused 0[K
    Receiving objects: 100% (2995/2995), 153.76 MiB | 15.57 MiB/s, done.
    Resolving deltas: 100% (1910/1910), done.
    Checking connectivity... done.



```python
%cd /home/aistudio/work/PaddleGAN
!pip install -r requirements.txt
%cd applications/
```

    /home/aistudio/work/PaddleGAN
    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Requirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 1)) (4.36.1)
    Requirement already satisfied: PyYAML>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 2)) (5.1.2)
    Collecting scikit-image>=0.14.0 (from -r requirements.txt (line 3))
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/fe/01/3a830f3df578ea3ed94ee7fd9f91e85c3dec2431d8548ab1c91869e51450/scikit_image-0.18.1-cp37-cp37m-manylinux1_x86_64.whl (29.2MB)
    [K     |████████████████████████████████| 29.2MB 8.7MB/s eta 0:00:011
    [?25hRequirement already satisfied: scipy>=1.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 4)) (1.3.0)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 5)) (4.1.1.26)
    Requirement already satisfied: imageio-ffmpeg in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 6)) (0.3.0)
    Collecting librosa==0.7.0 (from -r requirements.txt (line 7))
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/ad/6e/0eb0de1c9c4e02df0b40e56f258eb79bd957be79b918511a184268e01720/librosa-0.7.0.tar.gz (1.6MB)
    [K     |████████████████████████████████| 1.6MB 10.5MB/s eta 0:00:01
    [?25hRequirement already satisfied: numba==0.48 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 8)) (0.48.0)
    Requirement already satisfied: easydict in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r requirements.txt (line 9)) (1.9)
    Collecting tifffile>=2019.7.26 (from scikit-image>=0.14.0->-r requirements.txt (line 3))
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/c0/9b/db2b4777156c755ea589cb93ae50fc12a39119623bd7eca9bb8eaab523fc/tifffile-2021.4.8-py3-none-any.whl (165kB)
    [K     |████████████████████████████████| 174kB 36.3MB/s eta 0:00:01
    [?25hCollecting numpy>=1.16.5 (from scikit-image>=0.14.0->-r requirements.txt (line 3))
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/73/ef/8967d406f3f85018ceb5efab50431e901683188f1741ceb053efcab26c87/numpy-1.20.2-cp37-cp37m-manylinux2010_x86_64.whl (15.3MB)
    [K     |████████████████████████████████| 15.3MB 11.1MB/s eta 0:00:01
    [?25hCollecting PyWavelets>=1.1.1 (from scikit-image>=0.14.0->-r requirements.txt (line 3))
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/62/bd/592c7242fdd1218a96431512e77265c50812315ef72570ace85e1cfae298/PyWavelets-1.1.1-cp37-cp37m-manylinux1_x86_64.whl (4.4MB)
    [K     |████████████████████████████████| 4.4MB 14.2MB/s eta 0:00:01
    [?25hRequirement already satisfied: matplotlib!=3.0.0,>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->-r requirements.txt (line 3)) (2.2.3)
    Requirement already satisfied: pillow!=7.1.0,!=7.1.1,>=4.3.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->-r requirements.txt (line 3)) (7.1.2)
    Requirement already satisfied: networkx>=2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->-r requirements.txt (line 3)) (2.4)
    Requirement already satisfied: imageio>=2.3.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-image>=0.14.0->-r requirements.txt (line 3)) (2.6.1)
    Requirement already satisfied: audioread>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->-r requirements.txt (line 7)) (2.1.8)
    Requirement already satisfied: scikit-learn!=0.19.0,>=0.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->-r requirements.txt (line 7)) (0.22.1)
    Requirement already satisfied: joblib>=0.12 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->-r requirements.txt (line 7)) (0.14.1)
    Requirement already satisfied: decorator>=3.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->-r requirements.txt (line 7)) (4.4.0)
    Requirement already satisfied: six>=1.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->-r requirements.txt (line 7)) (1.15.0)
    Requirement already satisfied: resampy>=0.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->-r requirements.txt (line 7)) (0.2.2)
    Requirement already satisfied: soundfile>=0.9.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from librosa==0.7.0->-r requirements.txt (line 7)) (0.10.3.post1)
    Requirement already satisfied: llvmlite<0.32.0,>=0.31.0dev0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from numba==0.48->-r requirements.txt (line 8)) (0.31.0)
    Requirement already satisfied: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from numba==0.48->-r requirements.txt (line 8)) (41.4.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.14.0->-r requirements.txt (line 3)) (1.1.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.14.0->-r requirements.txt (line 3)) (2.4.2)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.14.0->-r requirements.txt (line 3)) (2019.3)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.14.0->-r requirements.txt (line 3)) (0.10.0)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image>=0.14.0->-r requirements.txt (line 3)) (2.8.0)
    Requirement already satisfied: cffi>=1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from soundfile>=0.9.0->librosa==0.7.0->-r requirements.txt (line 7)) (1.14.0)
    Requirement already satisfied: pycparser in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from cffi>=1.0->soundfile>=0.9.0->librosa==0.7.0->-r requirements.txt (line 7)) (2.19)
    Building wheels for collected packages: librosa
      Building wheel for librosa (setup.py) ... [?25ldone
    [?25h  Created wheel for librosa: filename=librosa-0.7.0-cp37-none-any.whl size=1598346 sha256=1321a16b95630823e889c6e5309ec9fdcc9b5a980cfbd963259a5ca5679e00cc
      Stored in directory: /home/aistudio/.cache/pip/wheels/81/7e/60/c27574fffbf2f28075dbf4b28c00d3fe3240fefb51d597932e
    Successfully built librosa
    [31mERROR: blackhole 0.3.2 has requirement xgboost==1.1.0, but you'll have xgboost 1.3.3 which is incompatible.[0m
    Installing collected packages: numpy, tifffile, PyWavelets, scikit-image, librosa
      Found existing installation: numpy 1.16.4
        Uninstalling numpy-1.16.4:
          Successfully uninstalled numpy-1.16.4
      Found existing installation: librosa 0.7.2
        Uninstalling librosa-0.7.2:
          Successfully uninstalled librosa-0.7.2
    Successfully installed PyWavelets-1.1.1 librosa-0.7.0 numpy-1.20.2 scikit-image-0.18.1 tifffile-2021.4.8
    /home/aistudio/work/PaddleGAN/applications


## 唇形动作合成命令使用说明

重点来啦！！本项目支持大家上传自己准备的视频和音频， 合成任意想要的**逼真的配音视频**！！![](https://ai-studio-static-online.cdn.bcebos.com/731e8683ff9d415b872981887563621186ea193f251b452183b20b4e7c2c1e4f)



只需在如下命令中的**face参数**和**audio参数**分别换成自己的视频和音频路径，然后运行如下命令，就可以生成和音频同步的视频。

程序运行完成后，会在当前文件夹下生成文件名为**outfile**参数指定的视频文件，该文件即为和音频同步的视频文件。本项目中提供了demo展示所用到的视频和音频文件。具体的参数使用说明如下：
- face: 原始视频，视频中的人物的唇形将根据音频进行唇形合成--通俗来说，想让谁说话
- audio：驱动唇形合成的音频，视频中的人物将根据此音频进行唇形合成--通俗来说，想让这个人说什么


```python
!export PYTHONPATH=$PYTHONPATH:/home/aistudio/work/PaddleGAN && python tools/wav2lip.py --face /home/aistudio/work/1.jpeg --audio /home/aistudio/work/2.m4a --outfile /home/aistudio/work/pp_put.mp4
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:26: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      def convert_to_list(value, n, name, dtype=np.int):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/scipy/linalg/__init__.py:217: DeprecationWarning: The module numpy.dual is deprecated.  Instead of using dual, use the functions directly from numpy or scipy.
      from numpy.dual import register_func
    /home/aistudio/work/PaddleGAN/ppgan/models/base_model.py:52: DeprecationWarning: invalid escape sequence \/
      """
    /home/aistudio/work/PaddleGAN/ppgan/modules/init.py:70: DeprecationWarning: invalid escape sequence \s
      """
    /home/aistudio/work/PaddleGAN/ppgan/modules/init.py:134: DeprecationWarning: invalid escape sequence \m
      """
    /home/aistudio/work/PaddleGAN/ppgan/modules/init.py:159: DeprecationWarning: invalid escape sequence \m
      """
    /home/aistudio/work/PaddleGAN/ppgan/modules/init.py:190: DeprecationWarning: invalid escape sequence \m
      """
    /home/aistudio/work/PaddleGAN/ppgan/modules/init.py:227: DeprecationWarning: invalid escape sequence \m
      """
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/scipy/special/orthogonal.py:81: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around, int,
    /home/aistudio/work/PaddleGAN/ppgan/modules/dense_motion.py:116: DeprecationWarning: invalid escape sequence \h
      """
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/scipy/io/matlab/mio5.py:98: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      from .mio5_utils import VarReader5
    Number of frames available for inference: 1
    Extracting raw audio...
    ffmpeg version 2.8.15-0ubuntu0.16.04.1 Copyright (c) 2000-2018 the FFmpeg developers
      built with gcc 5.4.0 (Ubuntu 5.4.0-6ubuntu1~16.04.10) 20160609
      configuration: --prefix=/usr --extra-version=0ubuntu0.16.04.1 --build-suffix=-ffmpeg --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --cc=cc --cxx=g++ --enable-gpl --enable-shared --disable-stripping --disable-decoder=libopenjpeg --disable-decoder=libschroedinger --enable-avresample --enable-avisynth --enable-gnutls --enable-ladspa --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libmodplug --enable-libmp3lame --enable-libopenjpeg --enable-libopus --enable-libpulse --enable-librtmp --enable-libschroedinger --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvorbis --enable-libvpx --enable-libwavpack --enable-libwebp --enable-libx265 --enable-libxvid --enable-libzvbi --enable-openal --enable-opengl --enable-x11grab --enable-libdc1394 --enable-libiec61883 --enable-libzmq --enable-frei0r --enable-libx264 --enable-libopencv
      libavutil      54. 31.100 / 54. 31.100
      libavcodec     56. 60.100 / 56. 60.100
      libavformat    56. 40.101 / 56. 40.101
      libavdevice    56.  4.100 / 56.  4.100
      libavfilter     5. 40.101 /  5. 40.101
      libavresample   2.  1.  0 /  2.  1.  0
      libswscale      3.  1.101 /  3.  1.101
      libswresample   1.  2.101 /  1.  2.101
      libpostproc    53.  3.100 / 53.  3.100
    Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '/home/aistudio/work/2.m4a':
      Metadata:
        major_brand     : mp42
        minor_version   : 0
        compatible_brands: isommp42
        creation_time   : 2020-12-31 01:31:27
      Duration: 00:00:46.42, start: 0.000000, bitrate: 150 kb/s
        Stream #0:0(eng): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 148 kb/s (default)
        Metadata:
          creation_time   : 2020-12-31 01:31:27
          handler_name    : SoundHandle
    Output #0, wav, to 'temp/temp.wav':
      Metadata:
        major_brand     : mp42
        minor_version   : 0
        compatible_brands: isommp42
        ISFT            : Lavf56.40.101
        Stream #0:0(eng): Audio: pcm_s16le ([1][0][0][0] / 0x0001), 48000 Hz, stereo, s16, 1536 kb/s (default)
        Metadata:
          creation_time   : 2020-12-31 01:31:27
          handler_name    : SoundHandle
          encoder         : Lavc56.60.100 pcm_s16le
    Stream mapping:
      Stream #0:0 -> #0:0 (aac (native) -> pcm_s16le (native))
    Press [q] to stop, [?] for help
    size=    8704kB time=00:00:46.42 bitrate=1536.0kbits/s    
    video:0kB audio:8704kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.000875%
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/numba/types/__init__.py:110: DeprecationWarning: `np.long` is a deprecated alias for `np.compat.long`. To silence this warning, use `np.compat.long` by itself. In the likely event your code does not need to work on Python 2 you can use the builtin `int` for which `np.compat.long` is itself an alias. Doing this will not modify any behaviour and is safe. When replacing `np.long`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      long_ = _make_signed(np.long)
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/numba/types/__init__.py:111: DeprecationWarning: `np.long` is a deprecated alias for `np.compat.long`. To silence this warning, use `np.compat.long` by itself. In the likely event your code does not need to work on Python 2 you can use the builtin `int` for which `np.compat.long` is itself an alias. Doing this will not modify any behaviour and is safe. When replacing `np.long`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      ulong = _make_unsigned(np.long)
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/sklearn/linear_model/_least_angle.py:30: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      method='lar', copy_X=True, eps=np.finfo(np.float).eps,
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/sklearn/linear_model/_least_angle.py:169: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      method='lar', copy_X=True, eps=np.finfo(np.float).eps,
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/sklearn/linear_model/_least_angle.py:286: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      eps=np.finfo(np.float).eps, copy_Gram=True, verbose=0,
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/sklearn/linear_model/_least_angle.py:858: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      eps=np.finfo(np.float).eps, copy_X=True, fit_path=True):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/sklearn/linear_model/_least_angle.py:1094: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      eps=np.finfo(np.float).eps, copy_X=True, fit_path=True,
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/sklearn/linear_model/_least_angle.py:1120: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      eps=np.finfo(np.float).eps, positive=False):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/sklearn/linear_model/_least_angle.py:1349: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      max_n_alphas=1000, n_jobs=None, eps=np.finfo(np.float).eps,
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/sklearn/linear_model/_least_angle.py:1590: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      max_n_alphas=1000, n_jobs=None, eps=np.finfo(np.float).eps,
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/sklearn/linear_model/_least_angle.py:1723: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      eps=np.finfo(np.float).eps, copy_X=True, positive=False):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/sklearn/decomposition/_lda.py:29: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      EPS = np.finfo(np.float).eps
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/sklearn/feature_extraction/image.py:167: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      dtype=np.int):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/numba/ir_utils.py:1512: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if (hasattr(numpy, value)
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/numba/ir_utils.py:1513: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      and def_val == getattr(numpy, value)):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/numba/ir_utils.py:1512: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if (hasattr(numpy, value)
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/numba/ir_utils.py:1513: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      and def_val == getattr(numpy, value)):
    (80, 3714)
    Length of mel chunks: 1157
    2021-04-20 21:32:20,639 - INFO - unique_endpoints {''}
    2021-04-20 21:32:20,640 - INFO - Downloading wav2lip_hq.pdparams from https://paddlegan.bj.bcebos.com/models/wav2lip_hq.pdparams
    100%|████████████████████████████████| 141910/141910 [00:03<00:00, 44966.16it/s]
    Model loaded
      0%|                                                    | 0/10 [00:00<?, ?it/s]2021-04-20 21:32:24,233 - INFO - unique_endpoints {''}
    2021-04-20 21:32:24,233 - INFO - Downloading s3fd_paddle.pdparams from https://paddlegan.bj.bcebos.com/models/s3fd_paddle.pdparams
    
      0%|                                                | 0/109119 [00:00<?, ?it/s][A
      0%|                                    | 323/109119 [00:00<00:34, 3192.52it/s][A
      1%|▍                                  | 1549/109119 [00:00<00:26, 4102.71it/s][A
      4%|█▍                                 | 4405/109119 [00:00<00:18, 5521.09it/s][A
      9%|███                                | 9485/109119 [00:00<00:13, 7536.19it/s][A
     14%|████▌                            | 14989/109119 [00:00<00:09, 10169.18it/s][A
     19%|██████▏                          | 20626/109119 [00:00<00:06, 13484.73it/s][A
     24%|███████▉                         | 26079/109119 [00:00<00:04, 17417.92it/s][A
     29%|█████████▋                       | 31886/109119 [00:00<00:03, 22048.27it/s][A
     34%|███████████▎                     | 37406/109119 [00:00<00:02, 26893.49it/s][A
     40%|█████████████                    | 43237/109119 [00:01<00:02, 32070.13it/s][A
     45%|██████████████▊                  | 49092/109119 [00:01<00:01, 37103.67it/s][A
     50%|████████████████▍                | 54544/109119 [00:01<00:01, 40731.49it/s][A
     55%|██████████████████▏              | 59944/109119 [00:01<00:01, 43967.57it/s][A
     60%|███████████████████▊             | 65543/109119 [00:01<00:00, 46994.33it/s][A
     65%|█████████████████████▌           | 71465/109119 [00:01<00:00, 50096.78it/s][A
     71%|███████████████████████▎         | 77257/109119 [00:01<00:00, 52212.12it/s][A
     76%|█████████████████████████▏       | 83126/109119 [00:01<00:00, 53998.25it/s][A
     82%|██████████████████████████▉      | 88963/109119 [00:01<00:00, 55235.75it/s][A
     87%|████████████████████████████▋    | 94880/109119 [00:01<00:00, 56358.83it/s][A
     92%|█████████████████████████████▌  | 100819/109119 [00:02<00:00, 57230.20it/s][A
    100%|████████████████████████████████| 109119/109119 [00:02<00:00, 50942.63it/s][A
    
      0%|                                                     | 0/1 [00:00<?, ?it/s][A/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:143: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if data.dtype == np.object:
    
    100%|█████████████████████████████████████████████| 1/1 [00:02<00:00,  2.53s/it][A


# 总结
**首先帮大家总结一波：让图片会说话、视频花式配音的魔法--Wav2lip的使用只用三步**：
1. 安装Paddle环境并下载[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)
2. 选择想要「配音/对口型」的对象以及音频内容
3. 运行代码并保存制作完成的对口型视频分享惊艳众人

贴心的送上项目传送门：[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN) 记得点Star关注噢~~
<div align='left'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/c7e2bcd255574e32b10061e0c4a1003a244bb7bd60ad43d394b23183f7390175' width='300'/>
</div>

# 除了嘴型同步，PaddleGAN还有哪些魔法？

PaddleGAN是只能做「对口型」的应用么？NONONO！当然不是！！
<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/f3b7e65df22a4e0fb771db150886dfd93ff602ebf8374fe0bf20e2083f5b1213' width='100'/>
</div>


接下来就给大家展示下PaddleGAN另外的花式应用，如各类**图形影像生成、处理能力**。

**人脸属性编辑能力**能够在人脸识别和人脸生成基础上，操纵面部图像的单个或多个属性，实现换妆、变老、变年轻、变换性别、发色等，一键换脸成为可能；

**动作迁移**，能够实现肢体动作变换、人脸表情动作迁移等等等等。

强烈鼓励大家玩起来，激发PaddleGAN的潜能！

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/461d1f34cf5242fca07d4e333e41f51c099a96017e324531b575a775d0679fc6' width='700'/>
</div>
<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/7d2cc83c689c474e8f3c0fa85e58e12b9885b47333d94d4dba4c66e622acf47e' width='700'/>
</div>

欢迎加入官方QQ群（1058398620）与各路技术高手交流~~

<div align='center'>
  <img src='https://ai-studio-static-online.cdn.bcebos.com/eb4d10d066c547f19cb373eb72458b12703e1c5b2ea34457b225d958925c2c83' width='250' height='300'/>
</div>
