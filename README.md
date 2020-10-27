# Identify Captcha Image by Convolutional Recurrent Neural Network(CRNN)

![Python3.7](https://img.shields.io/badge/Python-3.7-blue.svg) ![TensorFlow2.3](https://img.shields.io/badge/TensorFlow-2.3-yellow.svg) ![Keras2.4](https://img.shields.io/badge/Keras-2.4-red.svg)

This project demonstrates how we can build a deep neural network with **Connectionist Temporal Classification** loss function for reading captcha. The model is based on the paper ["An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition (2015), Baoguang Shi et al."](http://arxiv.org/abs/1507.05717).


## Dataset 資料集  
Total 18,000 different 104 x 24 captcha images, with 4 maximum length for each image.   
Image sourse: [Taiwan Insurance Institute](http://insdb.tii.org.tw/pivot/)
<p align="center">
  <img src="https://github.com/YenLinWu/CRNN_with_CTC_Loss/blob/main/Dataset_Img.png">
</p>


## References  參考文獻
> ### Articles  文章  
[1] [A_K_Nain (2020), OCR model for reading Captchas](https://keras.io/examples/vision/captcha_ocr/)  
[2] [Awni Hannun (2017), Sequence Modeling with CTC](https://distill.pub/2017/ctc/)    
[3] [Understanding CTC loss for speech recognition in Keras](https://stackoverflow.com/questions/57292896/understanding-ctc-loss-for-speech-recognition-in-keras)  
> ### Papers  論文    
[4] [Baoguang Shi, Xiang Bai & Cong Yao (2015), An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717 'Paper')    
[5] [Alex Graves, Santiago Fernández, Faustino Gomez & Jürgen Schmidhuber (2006), Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](http://www.cs.toronto.edu/~graves/icml_2006.pdf 'Paper')    

## 作者 Author  
<span> - &copy; Tom Wu (<a href="https://github.com/YenLinWu">Github</a>) </span>  
