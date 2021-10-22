#include <jni.h>
#include <string>

#include "implementations/vector_add.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_deelvin_openclrunner_MainActivity_runOpenCL(
        JNIEnv* env,
        jobject,
        jobject assetManager) {
    std::string res = vector_add(env, assetManager);
    return env->NewStringUTF(res.c_str());
}