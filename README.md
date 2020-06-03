# DeepFake v1 
#### * I do not allow malicious video production through this source code. This is just a practice code. (해당 소스 코드를 통한 악의적인 영상 제작을 불허합니다.)

### version Tensorflow 2.0

## 1. Make dataset


<img src= "https://user-images.githubusercontent.com/52908154/79987173-09357980-84e8-11ea-8663-a2e0c96a40f5.png" width="70%"></img>

* #### You have to get character images from youtube or other media.  


<img src= "https://user-images.githubusercontent.com/52908154/79987452-65989900-84e8-11ea-8c8a-60b753d43185.png" width="70%"></img>

* #### If you have a video editing tool, just bring the part where the face of the person is shown.



<img src="https://user-images.githubusercontent.com/52908154/79148660-cc3ef800-7e00-11ea-94f4-17c062df5b63.png" width="40%"></img>
![ezgif com-resize (3)](https://user-images.githubusercontent.com/52908154/79193092-e4486300-7e64-11ea-8e65-f1b5e0850a84.gif)
![ezgif com-resize (2)](https://user-images.githubusercontent.com/52908154/79193102-e90d1700-7e64-11ea-969b-4ff20e1efb6d.gif)    


* #### We will extract the landmarks of the characters through the dlib Library

<img src="https://user-images.githubusercontent.com/52908154/79988221-56feb180-84e9-11ea-9159-205814f31c02.png" width="20%"></img>
<img src="https://user-images.githubusercontent.com/52908154/79988263-641ba080-84e9-11ea-86a7-50daea14225e.png" width="20%"></img>
<img src="https://user-images.githubusercontent.com/52908154/79988407-94fbd580-84e9-11ea-90f7-021246f6d058.png" width="20%"></img>
<img src="https://user-images.githubusercontent.com/52908154/79988392-90cfb800-84e9-11ea-859f-41999886af32.png" width="20%"></img>

* #### You have to save both face and landmark images.


# Model
<img src="https://user-images.githubusercontent.com/52908154/79988633-d5f3ea00-84e9-11ea-990b-0af9eedc25a8.png" width="60%"></img>
* #### We will use an auto-encoder. 
* #### But if you look closely at the picture, there is one encoder and two decoder.  
* #### You have to share an encoder when you learn two characters. The reason is to compress the features of the face in the encoder well.  


<img src="https://user-images.githubusercontent.com/52908154/79994857-70a3f700-84f1-11ea-8788-abe6289bbe11.png" width="60%"></img>
* #### Introducing 'warping' in the learning process improves performance. 'Warping' is distorting the image. From this, when a new look comes in, it can produce better results.

<img src="https://user-images.githubusercontent.com/52908154/79994939-86b1b780-84f1-11ea-97fd-59e39507e0a9.png" width="20%"></img>

* #### 'warping' applies to the Landmark image, which is input data of the model. 
* #### Do not 'warping' on original face images other than Landmark images.

<img src="https://user-images.githubusercontent.com/52908154/79989157-8104a380-84ea-11ea-9693-a96b8a6a1d74.png" width="60%"></img>
* #### If you are good at restoring the two characters, try changing the decoder to add images.

# Image processing

<img src="https://user-images.githubusercontent.com/52908154/79990496-16ecfe00-84ec-11ea-914d-8ca655726d0c.png" width="60%"></img>

* #### If you have followed the process so far, the image above will be made. However, because of the background other than the face, it becomes unnatural.

* #### This is the part that we each need to modify to match the characteristics of the video.

<img src="https://user-images.githubusercontent.com/52908154/79149326-f218cc80-7e01-11ea-9f1e-acb05b0926c0.png" width="60%"></img>

* #### I detected the skin color and replaced the background with black. It also went through blending to blur the boundaries of skin color between characters in the synthesis process.

* #### If this process is complicated and cumbersome, there is another way. When you create a dataset, you crop an image.

<img src="https://user-images.githubusercontent.com/52908154/79996600-9a5e1d80-84f3-11ea-855d-c60ba2b2ac0a.png" width="40%"></img>


* #### You only need to bring in the face by setting the highest and lowest points of the landmark coordinates as shown above. I recommend this method and the implementation is [here](https://github.com/JunHyeok96/DeepFake/blob/master/make_landmark.py) . This does not completely bring only facial skin, but most of the time the background is removed.

# result 
![final3](https://user-images.githubusercontent.com/52908154/79148003-cbf22d00-7dff-11ea-8bc8-2e641bce2fa3.gif)  
![final2](https://user-images.githubusercontent.com/52908154/79148037-d6acc200-7dff-11ea-9823-1ad8355f166a.gif)  
![final](https://user-images.githubusercontent.com/52908154/79147964-bda41100-7dff-11ea-991d-86319ddc212b.gif)  
  
 * Source, Conversion Image, Image Processing Image
  
![ezgif com-crop](https://user-images.githubusercontent.com/52908154/79192536-da723000-7e63-11ea-8dc2-2ed7eab7bc94.gif)

 * Results for the entire image

![ezgif com-resize (7)](https://user-images.githubusercontent.com/52908154/79993799-1e160b00-84f0-11ea-84b8-6e7756ed9c4a.gif)
 
 * Actually, I didn't use my face to learn, but it's okay if the landmarks are similar.

### Data Information

* It used the video for about two to three minutes.

* 64 x 64 images were used.


# Quick Start

* dataset path

```
DeepFake
  dataset_video
    src
      video
    dst
      video
  dataset
    src
      img
      land
    dst
      img
      land

```

```
$ git clone https://github.com/JunHyeok96/DeepFake.git
$ cd DeepFake 
$ python make_landmark.py 
And follow the train.ipynb process.  
Once the learning is complete, 
$ python make_deepfake_video.py 
```




##### Image Source
https://medium.com/@jonathan_hui/how-deep-learning-fakes-videos-deepfakes-and-how-to-detect-it-c0b50fbf7cb9
