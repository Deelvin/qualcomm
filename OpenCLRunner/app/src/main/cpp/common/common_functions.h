#ifndef OPENCL_RUNNER_COMMON_FUNCTIONS_H
#define OPENCL_RUNNER_COMMON_FUNCTIONS_H

#include <jni.h>
#include <string>

std::string readKernel(JNIEnv* env, jobject assetManager, const std::string& name);

#endif //OPENCL_RUNNER_COMMON_FUNCTIONS_H
