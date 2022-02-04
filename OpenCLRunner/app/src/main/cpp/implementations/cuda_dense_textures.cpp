#include "dense_textures.h"

#include <android/log.h>

#include <chrono>
#include <string>
#include <vector>

namespace {
    struct InputShape {
        int b;
        int k;
    };
    struct FilterShape {
        int o;
        int k;
    };
}

ExecTime cuda_dense_textures(JNIEnv* env, jobject assetManager) {
    std::string kernelName = "cuda_dense_textures.cl";
    // pad, placeholder1, output, weights, bias
    InputShape is = {1, 4096};
    FilterShape fs = {4096, 4096};
    std::vector<float> input(is.b * is.k);
    std::vector<float> filter(fs.o * fs.k);
    std::vector<float> bias(fs.o);
    size_t gws0[3] = {262144, 1, 1};
    size_t lws0[3] = {64, 1, 1};

    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;
    int err;
    prepareOpenCLDevice(device_id, context, command_queue);

    std::string kernelSource = readKernel(env, assetManager, kernelName);
    const char* str = kernelSource.c_str();

    // ============ CREATE OpenCL IMAGES ============
    cl_image_format format;             // structure to define image format
    format.image_channel_data_type = CL_HALF_FLOAT;
    format.image_channel_order = CL_RGBA;

    // init image description
    cl_image_desc desc = { 0 };               // structure to define image description
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = 64;
    desc.image_height = 16;

    // input
    cl_mem input_img = clCreateImage(
            context,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            &format,
            &desc,
            input.data(),
            &err);
    assert(err == CL_SUCCESS);

    // filter
    desc.image_width = 1024; // cout%4 * cin
    desc.image_height = 4096; // kh * kw * cout/4
    cl_mem filter_img = clCreateImage(
            context,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            &format,
            &desc,
            filter.data(),
            &err);
    assert(err == CL_SUCCESS);

    // ============ CREATE OpenCL BUFFERS ============
    cl_mem bias_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bias.size() * sizeof(float), bias.data(), &err);
    assert(err == CL_SUCCESS);
    cl_mem output = clCreateBuffer(context, CL_MEM_READ_WRITE, bias.size() * sizeof(float), NULL, &err);
    assert(err == CL_SUCCESS);

    cl_program program = clCreateProgramWithSource(context, 1,  &str, NULL, &err);
    assert(err == CL_SUCCESS);
    auto cpuStart = std::chrono::high_resolution_clock::now();
    err = clBuildProgramWrapper(program, 1, &device_id);
    assert(err == CL_SUCCESS);
    cl_kernel kernel0 = clCreateKernel(program, "fused_nn_dense_add_nn_relu_1_kernel0", &err);
    assert(err == CL_SUCCESS);

    err = clSetKernelArg(kernel0, 0, sizeof(cl_mem), (void *)&input_img);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel0, 1, sizeof(cl_mem), (void *)&filter_img);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel0, 2, sizeof(cl_mem), (void *)&output);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel0, 3, sizeof(cl_mem), (void *)&bias_buf);
    assert(err == CL_SUCCESS);
    cl_event event0;
    err = clEnqueueNDRangeKernel(command_queue, kernel0, 3, NULL, gws0, lws0, 0, NULL, &event0);
    assert(err == CL_SUCCESS);
    err = clWaitForEvents(1, &event0);
    assert(err == CL_SUCCESS);
    err = clFinish(command_queue);
    assert(err == CL_SUCCESS);

    auto cpuEnd = std::chrono::high_resolution_clock::now();


    cl_ulong time_start;
    cl_ulong time_end;
    clGetEventProfilingInfo(event0, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event0, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double kernelTimeMS = (time_end - time_start) * 1e-6;

    auto cpuTimeMS = std::chrono::duration_cast<std::chrono::nanoseconds>(cpuEnd - cpuStart).count() * 1e-6;

    err = clReleaseEvent(event0);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(input_img);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(filter_img);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(bias_buf);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(output);
    assert(err == CL_SUCCESS);
    err = clReleaseKernel(kernel0);
    assert(err == CL_SUCCESS);
    err = clReleaseCommandQueue(command_queue);
    assert(err == CL_SUCCESS);
    err = clReleaseProgram(program);
    assert(err == CL_SUCCESS);
    err = clReleaseContext(context);
    assert(err == CL_SUCCESS);
    err = clReleaseDevice(device_id);
    assert(err == CL_SUCCESS);

    return {cpuTimeMS, kernelTimeMS};
}
