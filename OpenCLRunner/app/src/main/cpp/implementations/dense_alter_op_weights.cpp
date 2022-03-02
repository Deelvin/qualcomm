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

ExecTime dense_alter_op_weights(JNIEnv* env, jobject assetManager) {
    std::string kernelName = "dense_alter_op_weights.cl";
    // pad, placeholder1, output, weights, bias
    InputShape is = {1, 25088};
    FilterShape fs = {4096, 25088};
    std::vector<float> input(is.b * is.k);
    std::vector<float> filter(fs.o * fs.k);
    std::vector<float> bias(fs.o);
    size_t gws0[3] = {6272, 1, 1};
    size_t lws0[3] = {64, 1, 1};
    size_t gws1[3] = {256, 1, 1};
    size_t lws1[3] = {32, 1, 1};
    size_t gws2[3] = {1, 4096, 1};
    size_t lws2[3] = {1, 1, 1};

    // input img: HxW: 16x64
    // filter: HxW: 4096x1024

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
    desc.image_width = 56;
    desc.image_height = 112;

    // input
    cl_mem input_img = clCreateImage(
            context,
            CL_MEM_READ_WRITE,
            &format,
            &desc,
            NULL,
            &err);
    assert(err == CL_SUCCESS);

    // filter
    desc.image_width = 6272; // cout%4 * cin
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
    // input
    cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, input.size() * sizeof(float), input.data(), &err);
    assert(err == CL_SUCCESS);
    cl_mem bias_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bias.size() * sizeof(float), bias.data(), &err);
    assert(err == CL_SUCCESS);
    cl_mem compute = clCreateBuffer(context, CL_MEM_READ_WRITE, bias.size() * sizeof(float), NULL, &err);
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
    cl_kernel kernel1 = clCreateKernel(program, "fused_nn_dense_add_nn_relu_1_kernel1", &err);
    assert(err == CL_SUCCESS);
    cl_kernel kernel2 = clCreateKernel(program, "fused_nn_dense_add_nn_relu_1_kernel2", &err);
    assert(err == CL_SUCCESS);

    err = clSetKernelArg(kernel0, 0, sizeof(cl_mem), (void *)&input_img);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel0, 1, sizeof(cl_mem), (void *)&input_buf);
    assert(err == CL_SUCCESS);
    cl_event event0;
    err = clEnqueueNDRangeKernel(command_queue, kernel0, 3, NULL, gws0, lws0, 0, NULL, &event0);
    assert(err == CL_SUCCESS);
    err = clWaitForEvents(1, &event0);
    assert(err == CL_SUCCESS);
    err = clFinish(command_queue);
    assert(err == CL_SUCCESS);

    err = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&input_img);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void *)&filter_img);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel1, 2, sizeof(cl_mem), (void *)&compute);
    assert(err == CL_SUCCESS);
    cl_event event1;
    err = clEnqueueNDRangeKernel(command_queue, kernel1, 3, NULL, gws1, lws1, 0, NULL, &event1);
    assert(err == CL_SUCCESS);
    err = clWaitForEvents(1, &event1);
    assert(err == CL_SUCCESS);
    err = clFinish(command_queue);
    assert(err == CL_SUCCESS);

    err = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&output);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void *)&compute);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel2, 2, sizeof(cl_mem), (void *)&bias_buf);
    assert(err == CL_SUCCESS);
    cl_event event2;
    err = clEnqueueNDRangeKernel(command_queue, kernel2, 3, NULL, gws2, lws2, 0, NULL, &event2);
    assert(err == CL_SUCCESS);
    err = clWaitForEvents(1, &event2);
    assert(err == CL_SUCCESS);
    err = clFinish(command_queue);
    assert(err == CL_SUCCESS);

    auto cpuEnd = std::chrono::high_resolution_clock::now();


    cl_ulong time_start;
    cl_ulong time_end;
    clGetEventProfilingInfo(event0, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event0, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double kernel1TimeMS = (time_end - time_start) * 1e-6;
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner Kernel size", "Kernel0 : %f ms", kernel1TimeMS);

    clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event1, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double kernel2TimeMS = (time_end - time_start) * 1e-6;
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner Kernel size", "Kernel1 : %f ms", kernel2TimeMS);

    clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double kernel3TimeMS = (time_end - time_start) * 1e-6;
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner Kernel size", "Kernel2 : %f ms", kernel3TimeMS);

    double kernelTimeMS = kernel1TimeMS + kernel2TimeMS + kernel3TimeMS;
    auto cpuTimeMS = std::chrono::duration_cast<std::chrono::nanoseconds>(cpuEnd - cpuStart).count() * 1e-6;

    err = clReleaseEvent(event0);
    assert(err == CL_SUCCESS);
    err = clReleaseEvent(event1);
    assert(err == CL_SUCCESS);
    err = clReleaseEvent(event2);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(input_img);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(filter_img);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(input_buf);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(bias_buf);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(compute);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(output);
    assert(err == CL_SUCCESS);
    err = clReleaseKernel(kernel0);
    assert(err == CL_SUCCESS);
    err = clReleaseKernel(kernel1);
    assert(err == CL_SUCCESS);
    err = clReleaseKernel(kernel2);
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
