#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supportedby OpenCL implementation on your device."
#endif

// Work_size: 262144x1x1x64x1x1
__kernel void fused_nn_dense_add_nn_relu_1_kernel0(__read_only image2d_t placeholder, __read_only image2d_t placeholder1, __global half* restrict T_relu, __global half* restrict placeholder2) {
  half4 T_dense_rf[1];
  __local half4 red_buf0[64];
  __local half4 T_dense[1];
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)T_dense_rf + 0);
  for (int k_outer = 0; k_outer < 16; ++k_outer) {
    half4 _1 = read_imageh(placeholder, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((k_outer * 32), 0));
    half4 _2 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((k_outer * 128) + 0, ((((int)get_group_id(0)) * 128) + ((int)get_local_id(0)))));
    vstore4((vload4(0, (half*)T_dense_rf + 0) + (((half*)&_1)[0] * _2)), 0, (half*)T_dense_rf + 0);
    half4 _3 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((k_outer * 128) + 1, ((((int)get_group_id(0)) * 128) + ((int)get_local_id(0)))));
    vstore4((vload4(0, (half*)T_dense_rf + 0) + (((half*)&_1)[1] * _3)), 0, (half*)T_dense_rf + 0);
    half4 _4 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((k_outer * 128) + 2, ((((int)get_group_id(0)) * 128) + ((int)get_local_id(0)))));
    vstore4((vload4(0, (half*)T_dense_rf + 0) + (((half*)&_1)[2] * _4)), 0, (half*)T_dense_rf + 0);
    half4 _5 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((k_outer * 128) + 3, ((((int)get_group_id(0)) * 128) + ((int)get_local_id(0)))));
    vstore4((vload4(0, (half*)T_dense_rf + 0) + (((half*)&_1)[3] * _5)), 0, (half*)T_dense_rf + 0);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  ((volatile __local half4*)red_buf0)[(((int)get_local_id(0)))] = T_dense_rf[(0)];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)get_local_id(0)) < 32) {
    ((volatile __local half4*)red_buf0)[(((int)get_local_id(0)))] = (((volatile __local half4*)red_buf0)[(((int)get_local_id(0)))] + ((volatile __local half4*)red_buf0)[((((int)get_local_id(0)) + 32))]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)get_local_id(0)) < 16) {
    ((volatile __local half4*)red_buf0)[(((int)get_local_id(0)))] = (((volatile __local half4*)red_buf0)[(((int)get_local_id(0)))] + ((volatile __local half4*)red_buf0)[((((int)get_local_id(0)) + 16))]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)get_local_id(0)) < 8) {
    ((volatile __local half4*)red_buf0)[(((int)get_local_id(0)))] = (((volatile __local half4*)red_buf0)[(((int)get_local_id(0)))] + ((volatile __local half4*)red_buf0)[((((int)get_local_id(0)) + 8))]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)get_local_id(0)) < 4) {
    ((volatile __local half4*)red_buf0)[(((int)get_local_id(0)))] = (((volatile __local half4*)red_buf0)[(((int)get_local_id(0)))] + ((volatile __local half4*)red_buf0)[((((int)get_local_id(0)) + 4))]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)get_local_id(0)) < 2) {
    ((volatile __local half4*)red_buf0)[(((int)get_local_id(0)))] = (((volatile __local half4*)red_buf0)[(((int)get_local_id(0)))] + ((volatile __local half4*)red_buf0)[((((int)get_local_id(0)) + 2))]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)get_local_id(0)) < 1) {
    ((volatile __local half4*)red_buf0)[(((int)get_local_id(0)))] = (((volatile __local half4*)red_buf0)[(((int)get_local_id(0)))] + ((volatile __local half4*)red_buf0)[((((int)get_local_id(0)) + 1))]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)get_local_id(0)) == 0) {
    T_dense[(0)] = ((volatile __local half4*)red_buf0)[(0)];
  }
  if (((int)get_local_id(0)) == 0) {
    vstore4((max((half4)(T_dense[(0)] + vload4(0, (half*)placeholder2 + (int)get_group_id(0))), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)))), 0, (half*)T_relu + (((int)get_group_id(0))));
  }
}