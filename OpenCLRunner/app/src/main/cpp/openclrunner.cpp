#include <jni.h>
#include <string>

#include "implementations/vector_add.h"
#include "implementations/mace_inceptionv3_kernel.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_deelvin_openclrunner_MainActivity_runOpenCL(
        JNIEnv* env,
        jobject,
        jobject assetManager) {
    //std::string res = measureExecTime(vector_add, env, assetManager);
    std::string res = measureExecTime(mace_inceptionv3_kernel, env, assetManager);
    return env->NewStringUTF(res.c_str());
}