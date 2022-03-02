#include <jni.h>
#include <string>

#include <android/log.h>

#include "implementations/vector_add.h"
#include "implementations/simple_mad.h"
#include "implementations/tvm_resnet50v2_conv_kernel.h"
#include "implementations/dense_textures.h"
#include "implementations/cuda_dense.h"
#include "implementations/cuda_dense_textires.h"
#include "implementations/dense_alter_op_weights.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_deelvin_openclrunner_MainActivity_runOpenCL(
        JNIEnv* env,
        jobject,
        jobject assetManager) {
    //std::string res = measureExecTime(vector_add, env, assetManager);
    //std::string res = measureExecTime(dense_textures, env, assetManager);
    //std::string res = measureExecTime(dense_textures, env, assetManager);
    std::string res = measureExecTime(dense_alter_op_weights, env, assetManager);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner", "Exec time: %s", res.c_str());
    return env->NewStringUTF(res.c_str());
}