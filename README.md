# AnimeGAN
A Tensorflow implementation of AnimeGAN for fast photo animation  !  
The paper can be accessed [here](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/doc/Chen2020_Chapter_AnimeGAN.pdf) or on the [website](https://link.springer.com/chapter/10.1007/978-981-15-5577-0_18).  
  
**Online access**:  Be grateful to [@TonyLianLong](https://github.com/TonyLianLong/AnimeGAN.js) for developing an online access project, you can implement photo animation through a browser without installing anything, [click here to have a try](https://animegan.js.org/).  
  
**Good news**:  tensorflow-1.15.0 is compatible with the code of this repository. In this version, you can run this code without any modification. [The premise is that the CUDA and cudnn corresponding to the tf version are correctly installed](https://tensorflow.google.cn/install/source#gpu). Maybe the versions between tf-1.8.0 and tf-1.15.0 are also supported and compatible with this repository, but I didn’t make too many extra attempts.  

  
-----  
This is the Open source of the paper <AnimeGAN: a novel lightweight GAN for photo animation>, which uses the GAN framwork to transform real-world photos into anime images.  
  
**Some suggestions:**   
1. since the real photos in the training set are all landscape photos, if you want to stylize the photos with people as the main body, you may as well add at least 3000 photos of people in the training set and retrain to obtain a new model.  
2. In order to obtain a better face animation effect, when using 2 images as data pairs for training, it is suggested that the faces in the photos and the faces in the anime style data should be consistent in terms of gender as much as possible.  
3. The generated stylized images will be affected by the overall brightness and tone of the style data, so try not to select the anime images of night as the style data, and it is necessary to make an exposure compensation for the overall style data to promote the consistency of brightness and darkness of the entire style data.  

**News:**  AnimeGAN+ is expected to be released this summer. After some simple tricks were added to AnimeGAN, the obtained AnimeGAN+ has better animation effects. When I return to school to graduate, more pre-trained models and video animation test code will also be released in this repository.  

___  

## Requirements  
- python 3.6.8  
- tensorflow-gpu 1.8  
- opencv  
- tqdm  
- numpy  
- glob  
- argparse  
  
## Usage 
### 1. Show final program
  eg. `python realtime.py`
  > 需要先解压`checkpoint.zip`和`realtime.py`放在同一目录下。  
  > 由于一直git不成功就放百度yun上了
  > 链接：https://pan.baidu.com/s/1SjbiKouBMTttKzRJWpVVog 
  > 提取码：ckpt  
  > 直接运行调用的是文件读写的展示方式，调用函数write_file()即使用`JanpanStreet.jpg`生成结果`Gan_resultH.png`并通过matplot显示效果。  
  > 如果想要修改其他图片，读取的文件只能是jpg格式，在realtime.py的函数write_file()中修改变量imgpath即可。    
  > 如果想要使用摄像头的实时画面风格迁移在realtime.py的main函数中调用start_capture()即可。  
  > 各项依赖文件完整后运行会显示`[*] Success to read AnimeGAN.model-60`等待图片处理完成后显示效果即可。  
  > 没有文件夹`checkpoint`会报错`[*] Failed to find a checkpoint`。  
  ![](ProgramRes.png)
  
### 2. Download vgg19 or Pretrained model  
> [vgg19.npy](https://github.com/TachibanaYoshino/AnimeGAN/releases/tag/vgg16%2F19.npy)  
  
> [Pretrained model](https://github.com/TachibanaYoshino/AnimeGAN/releases/tag/Haoyao-style_V1.0)  

### 3. Download dataset  
> [Link](https://github.com/TachibanaYoshino/AnimeGAN/releases/tag/dataset-1)  

### 4. Do edge_smooth  
  eg. `python edge_smooth.py --dataset Hayao --img_size 256`  
  
### 5. Train  
  eg. `python main.py --phase train --dataset Hayao --epoch 101 --init_epoch 1`  
  
### 6. Test  
  eg. `python main.py --phase test --dataset Hayao`  
  or `python test.py --checkpoint_dir checkpoint/AnimeGAN_Hayao_lsgan_300_300_1_3_10 --test_dir dataset/test/real --style_name H`  
  

 
  
____  
## Results  
:blush:  pictures from the paper 'AnimeGAN: a novel lightweight GAN for photo animation'  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/doc/sota.png)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/doc/e2.png)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/doc/e3.png)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/doc/e4.png)  
  
:heart_eyes:  Photo  to  Hayao  Style  
  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(37).jpg)![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(37).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(1).jpg)![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(1).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(20).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(20).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(21).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(21).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(22).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(22).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(23).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(23).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(24).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(24).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(46).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(46).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(30).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(30).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(28).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(28).jpg)  
![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo/1%20(38).jpg) ![](https://github.com/TachibanaYoshino/AnimeGAN/blob/master/result/Hayao/photo_result/1%20(38).jpg)  
____  
## Acknowledgment  
This code is based on the [CartoonGAN-Tensorflow](https://github.com/taki0112/CartoonGAN-Tensorflow/blob/master/CartoonGAN.py) and [Anime-Sketch-Coloring-with-Swish-Gated-Residual-UNet](https://github.com/pradeeplam/Anime-Sketch-Coloring-with-Swish-Gated-Residual-UNet). Thanks to the contributors of this project.  

