#include "common_functions.h"

#include <android/asset_manager_jni.h>

std::string readKernel(JNIEnv* env, jobject assetManager, const std::string& name) {
    std::string res;
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    AAsset* asset = AAssetManager_open(mgr, name.c_str(), AASSET_MODE_BUFFER);
    size_t assetLength = AAsset_getLength(asset);
    char* buffer = new char[assetLength+1];
    AAsset_read(asset, buffer, assetLength);
    AAsset_close(asset);
    buffer[assetLength] = '\0';
    res = buffer;
    delete [] buffer;

    return res;
}