#ifndef OPENCL_RUNNER_MACE_INCEPTIONV3_KERNEL_H
#define OPENCL_RUNNER_MACE_INCEPTIONV3_KERNEL_H

#include <jni.h>
#include <string>

#include <common_functions.h>

ExecTime tvm_resnet50v2_conv_kernel(JNIEnv* env, jobject assetManager);

#endif //OPENCL_RUNNER_MACE_INCEPTIONV3_KERNEL_H
