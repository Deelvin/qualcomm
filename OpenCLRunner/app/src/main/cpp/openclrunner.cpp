#include <jni.h>
#include <string>

#include "implementations/vector_add.h"
#include "implementations/simple_mad.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_deelvin_openclrunner_MainActivity_runOpenCL(
        JNIEnv* env,
        jobject,
        jobject assetManager) {
    //std::string res = measureExecTime(vector_add, env, assetManager);
    std::string res = measureExecTime(simple_mad, env, assetManager, 50);
    return env->NewStringUTF(res.c_str());
}