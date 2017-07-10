#REMEMBERING HISTORY WITH CONVOLUTIONAL LSTM FOR ANOMALY DETECTION

By Weixin Luo$^{*}$, Wen Liu$^{*}$, Shenghua Gao

## Introduction
All codes are based on [Caffe](https://github.com/BVLC/caffe).
The main modifications are following
* Adding [a convolutional lstm layer](https://github.com/StevenLiuWen/convlstm_anomaly_detection/blob/master/include/caffe/layers/conv_lstm_layer.hpp)
* Changing the original **Blob** datastructure from 4-dimension tensor $(N \times C \times H \times W)$  to 5-dimension tensor $(T \times  N \times C \times H \times W)$ tensor. **Be careful that do not use cuDNN engien in all activation function layers in this motidified Caffe version, such as ReLU, sigmoid, tanh and so on, because cuDNN do not support 5-dimension in these activation function (but Conv layer is ok). If do that, it will be wrong.**
* Adding a [video data layers](https://github.com/StevenLiuWen/convlstm_anomaly_detection/blob/master/include/caffe/layers/video_data_layer.hpp)

## Requirements for compiling
Since all the code are implemented on Caffe, so that in order to compile successfully, it must satisfy all requirements of Caffe ([see the installization instruction](http://caffe.berkeleyvision.org/installation.html)).
Whatmore, it also need to compile with Opencv 3.+, because the [video data layer](https://github.com/StevenLiuWen/convlstm_anomaly_detection/blob/master/include/caffe/layers/video_data_layer.hpp) uses opencv libraries. 

## Datasets
>* **CUHK Avenue**
>* **UCSD Pedestrians 1 & 2**
>* **Subway Enter & Exit**
 please to contact the orginal author to get the download links.


## Training 
As **CUHK Avenue** dataset for example, details are showd in the [zstorm_conv_lstm_deconv/solver.prototxt](https://github.com/StevenLiuWen/convlstm_anomaly_detection/blob/master/zstorm_conv_lstm_deconv/solver.prototxt) and [zstorm_conv_lstm_deconv/train.prototxt](https://github.com/StevenLiuWen/convlstm_anomaly_detection/blob/master/zstorm_conv_lstm_deconv/train.prototxt).

## Testing or evaluation
Details are in the python shell, [zstorm_conv_lstm_deconv/test.py](https://github.com/StevenLiuWen/convlstm_anomaly_detection/blob/master/zstorm_conv_lstm_deconv/test.py).

## License 

All code are following the license of Caffe, and Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
