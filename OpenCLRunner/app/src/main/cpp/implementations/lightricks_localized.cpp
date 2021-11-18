//
// Created by iliya on 17.11.21.
//
#include "lightricks_localized.h"
#include <common_functions.h>

#include <chrono>

#include <android/log.h>

#define INPUT_BUFF_SIZE 1*2*56*56
#define OUTPUT_BUFF_SIZE 1*32*56*56
#define CONV_WEIGHTS_SIZE 1*32*2
#define DEBUG_IDXS_MAX_SIZE 128*28*784

ExecTime lightricks_localized_reproducer(JNIEnv* env, jobject assetManager)
{
    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;
    int err;
    prepareOpenCLDevice(device_id, context, command_queue);

    std::string kernelSource = readKernel(env, assetManager, "localized_kernels.cl");
    const char* str = kernelSource.c_str();

    // Create buffers
    cl_float *input0 = new cl_float[INPUT_BUFF_SIZE];
    cl_float *conv_weights = new cl_float[CONV_WEIGHTS_SIZE];


    cl_float *output0 = new cl_float[OUTPUT_BUFF_SIZE];
    cl_float *input1  = new cl_float[OUTPUT_BUFF_SIZE];
    cl_float *output1 = new cl_float[OUTPUT_BUFF_SIZE];

    cl_int *out0_idxs = new cl_int[DEBUG_IDXS_MAX_SIZE]; // non zero should be 439040 accordin to python script

    // Initialize input arrays
    for (int i(0); i < INPUT_BUFF_SIZE; ++i) {
        input0[i] = 1.0;
    }

    for (int i(0); i < CONV_WEIGHTS_SIZE; ++i) {
        conv_weights[i] = 1.0;
    }

    for (int i(0); i < DEBUG_IDXS_MAX_SIZE; ++i) {
        out0_idxs[i] = 0;
    }



    cl_mem input0_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, INPUT_BUFF_SIZE * sizeof(cl_float), input0, &err);
    assert(err == CL_SUCCESS);
    cl_mem conv_weights_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, CONV_WEIGHTS_SIZE * sizeof(cl_float), conv_weights, &err);
    assert(err == CL_SUCCESS);
    cl_mem output0_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, OUTPUT_BUFF_SIZE * sizeof(cl_float), NULL, &err);
    assert(err == CL_SUCCESS);
    cl_mem output1_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, OUTPUT_BUFF_SIZE * sizeof(cl_float), NULL, &err);
    assert(err == CL_SUCCESS);

    cl_mem out0_idxs_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, DEBUG_IDXS_MAX_SIZE * sizeof(cl_int), NULL, &err);
    assert(err == CL_SUCCESS);


    cl_program program = clCreateProgramWithSource(context, 1,  &str, NULL, &err);
    assert(err == CL_SUCCESS);

    auto cpuStart = std::chrono::high_resolution_clock::now();

    err = clBuildProgramWrapper(program, 1, &device_id);
    assert(err == CL_SUCCESS);

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "tvmgen_default_fused_nn_conv2d_kernel0", &err);
    assert(err == CL_SUCCESS);

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input0_mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&conv_weights_mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output0_mem);
    assert(err == CL_SUCCESS);


    // added argument for debug purposes
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&out0_idxs_mem);
    assert(err == CL_SUCCESS);

    // Run kernel
    size_t gws0[3] = { 784, 1, 1}; // Define global size of execution
    size_t lws0[3] = { 28, 1, 1}; // Define local size of execution

    cl_event event;
    err = clEnqueueNDRangeKernel(command_queue, kernel, 3, NULL, gws0, lws0, 0, NULL, &event);
    assert(err == CL_SUCCESS);
    err = clWaitForEvents(1, &event);
    assert(err == CL_SUCCESS);
    err = clFinish(command_queue);
    assert(err == CL_SUCCESS);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    // Read buffer with result of calculation
    err = clEnqueueReadBuffer(command_queue, output0_mem, CL_TRUE, 0, OUTPUT_BUFF_SIZE * sizeof(cl_float), output0, 0, NULL, NULL);
    assert(err == CL_SUCCESS);

    err = clEnqueueReadBuffer(command_queue, out0_idxs_mem, CL_TRUE, 0, OUTPUT_BUFF_SIZE * sizeof(cl_int), out0_idxs, 0, NULL, NULL);
    assert(err == CL_SUCCESS);


    std::string res = "{";
    float sum = 0;
    for (int i(0); i < OUTPUT_BUFF_SIZE; ++i) {
        //res += std::to_string(output0[i]) + ", ";
        sum += output0[i];
        input1[i] = output0[i];
    }

    int sum2 = 0;
    for (int i(0); i < DEBUG_IDXS_MAX_SIZE; ++i) {
        sum2 += out0_idxs[i];
    }
    res += std::to_string(sum2);
    res += "}";

    __android_log_print(ANDROID_LOG_DEBUG, "Deelvin::OpenCL Runner results of output0:", "%s", res.c_str());

    // Create kernel

    cl_mem input1_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, INPUT_BUFF_SIZE * sizeof(cl_float), input1, &err);
    assert(err == CL_SUCCESS);


    cl_kernel kernel1 = clCreateKernel(program, "tvmgen_default_fused_nn_conv2d_kernel1_good", &err);
    assert(err == CL_SUCCESS);

    // Set kernel arguments
    err = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&output1_mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void *)&output0_mem);
    assert(err == CL_SUCCESS);

    // Run kernel
    size_t gws1[3] = { 100352, 1, 1}; // Define global size of execution
    size_t lws1[3] = { 2, 1, 1}; // Define local size of execution

    err = clEnqueueNDRangeKernel(command_queue, kernel1, 3, NULL, gws1, lws1, 0, NULL, &event);
    assert(err == CL_SUCCESS);
    err = clWaitForEvents(1, &event);
    assert(err == CL_SUCCESS);
    err = clFinish(command_queue);
    assert(err == CL_SUCCESS);

    // Read buffer with result of calculation
    err = clEnqueueReadBuffer(command_queue, output1_mem, CL_TRUE, 0, OUTPUT_BUFF_SIZE * sizeof(cl_float), output1, 0, NULL, NULL);
    assert(err == CL_SUCCESS);



    std::string res1 = "{";
    sum = 0;
    for (int i(0); i < OUTPUT_BUFF_SIZE; ++i) {
        //res1 += std::to_string(output1[i]) + ", ";
        if (abs(output1[i] - 2.0) < 0.01)
            sum += output1[i];
    }

    res1 += std::to_string(sum);
    res1 += "}";

    __android_log_print(ANDROID_LOG_DEBUG, "Deelvin::OpenCL Runner results of output1:", "%s", res1.c_str());


    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double kernelTimeMS = (time_end - time_start)   * 1e-6; // from ns to ms
    auto cpuTimeMS = std::chrono::duration_cast<std::chrono::nanoseconds>(cpuEnd - cpuStart).count() * 1e-6;

    err = clReleaseMemObject(input0_mem);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(conv_weights_mem);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(output0_mem);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(input1_mem);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(output1_mem);
    assert(err == CL_SUCCESS);

    err = clReleaseMemObject(out0_idxs_mem);
    assert(err == CL_SUCCESS);

    delete [] input0;
    delete [] conv_weights;


    delete [] output0;
    delete [] input1;
    delete [] output1;

    delete [] out0_idxs;

    return {cpuTimeMS, kernelTimeMS};

}
