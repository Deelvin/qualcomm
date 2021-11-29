#OpenCLRunner

This app is used to experiment with extracted / generated kernels in standalone fashion

##mali_kernels branch
branch is created to localize the root cause for accuracy issue reported by Apache TVM customer

Through creation of simple network only with one convolution  

Conv1x1 (inp channels=2, out channels=32, dims = 56, 56)

issue was reproduced only with as a sequence of two kernels in OpenCLRunner/app/src/main/assets/localized_kernels.cl

first kernel - tvmgen_default_fused_nn_conv2d_kernel0 which performs calc of conv1x1 works fine
second kernel - tvmgen_default_fused_nn_conv2d_kernel1_bad which performs data layout change is the localized problematic place

kernel was modified to leave only problem lines. 
```
(int)(3136 * ((int)get_group_id(0) / 3136))
```

Issue is related with simple math related to idx calcualtion and produce incorrect values starting from idx = 65356. 
All value become zeros - see Deelvin::OpenCL Runner results of output1 in logcat
Another strange indicator is that if instead of same constant different values are used then everything is all right
for example 
```
(int)(3135 * ((int)get_group_id(0) / 3136)) or (int)(3137 * ((int)get_group_id(0) / 3136)) 
```

produces correct values
seems to be some incorrect compiler behavior

if we add cast to (float) in the middle then everything is okay after 65535
```
int val = (int)(3136 * (float)((int)get_group_id(0) / 3136));
```

Problem was reproduced on several Qualcomm based devices and was not reproduced on non-Qualcomm ones. 
Here is the list of devices used for verification

* Nokia 7.2, Qualcomm Snapdragon 660, Adreno 512
* Realme RMX2202, Qualcomm Snapdragon 888, Adreno 660
* Samsung A71, Qualcomm Snapdragon 730, Adreno 618
