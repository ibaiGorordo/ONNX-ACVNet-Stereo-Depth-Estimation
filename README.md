# ONNX-ACVNet-Stereo-Depth-Estimation
 About Python scripts form performing stereo depth estimation using the ACVNet model in ONNX.
 
![!ACVNet stereo detph estimation](https://github.com/ibaiGorordo/ONNX-ACVNet-Stereo-Depth-Estimation/blob/main/doc/img/out.jpg)

# Requirements

 * Check the **requirements.txt** file. 
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.
 * For OAK-D host inference, you will need the **depthai** library.
 * Additionally, **pafy** and **youtube-dl** are required for youtube video inference.
 
# Installation
```
git clone https://github.com/ibaiGorordo/ONNX-ACVNet-Stereo-Depth-Estimation.git
cd ONNX-ACVNet-Stereo-Depth-Estimation
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

### For youtube video inference
```
pip install youtube_dl
pip install git+https://github.com/zizo-pro/pafy@b8976f22c19e4ab5515cacbfae0a3970370c102b
```

### OAK-D Host inference:
```pip install depthai```

You might need additional installations, check the depthai reference below for more details.

# ONNX model
The original models were converted to different formats (including .onnx) by [PINTO0309](https://github.com/PINTO0309), download the models from the download script in [his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/266_ACVNet) and save them into the **[models](https://github.com/ibaiGorordo/ONNX-ACVNet-Stereo-Depth-Estimation/tree/main/models)** folder. 

# Original Pytorch model
The Pytorch pretrained model was taken from the [original repository](https://github.com/gangweiX/ACVNet).
 
# Examples

 * **Image inference**:
 ```
 python image_depth_estimation.py
 ```

 * **Video inference**:
 ```
 python video_depth_estimation.py
 ```
 
 * **Driving Stereo dataet inference**: https://youtu.be/Wh0UuIbSu8Q
 ![!ACVNet stereo depth estimation](https://github.com/ibaiGorordo/ONNX-ACVNet-Stereo-Depth-Estimation/blob/main/doc/img/onnx_acvnet.gif)
  
 *Original video: Driving stereo dataset, reference below*
  
 ```
 python driving_stereo_test.py
 ```
 
 * **Depthai inference**: 
 ```
 python depthai_host_depth_estimation.py
 ```

 * **Inference speed test**: 
 ```
 python test_speed.py
 ```

# Inference speed results
ONNX Runtime - CUDA (Nvidia 1660 Super)
- **240x320:** 175 ms
- **320x480:** 320 ms
- **384x640:** 490 ms
- **480x640:** 1896 - 12057 ms
- **544x960:** Memory overload
- **720x1280:** Memory overload

# References:
* ACVNet model: https://github.com/gangweiX/ACVNet
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow
* Driving Stereo dataset: https://drivingstereo-dataset.github.io/
* Depthai library: https://pypi.org/project/depthai/
* Original paper: https://arxiv.org/abs/2203.02146
 
