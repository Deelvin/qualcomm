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

void prepareOpenCLDevice(cl_device_id& device_id, cl_context& ctx, cl_command_queue& cq) {
    cl_platform_id platform_id;
    cl_uint ret_num_platforms;
    cl_uint ret_num_devices;

    int err = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    assert(err == CL_SUCCESS);

    // Create context
    ctx = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
    cq = clCreateCommandQueue(ctx, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    assert(err == CL_SUCCESS);
}

std::string measureExecTime(Executor exec, JNIEnv* env, jobject assetManager, unsigned int repeat) {
    double cpuTime = 0;
    double kernelTime = 0;
    for (int i = 0; i < repeat; ++i) {
        auto time = exec(env, assetManager);
        cpuTime += time.cpuTime;
        kernelTime += time.kernelTime;
    }
    cpuTime /= repeat;
    kernelTime /= repeat;
    std::string str = "CPU: " + std::to_string(cpuTime) + ", OpenCL: " + std::to_string(kernelTime);
    return str;
}