#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supportedby OpenCL implementation on your device."
#endif

// Work_size: 262144x1x1x64x1x1
__kernel void fused_nn_dense_add_nn_relu_1_kernel0(__global half* restrict placeholder, __global half* restrict placeholder1, __global half* restrict T_relu, __global half* restrict placeholder2) {
  half T_dense_rf[1];
  __local half red_buf0[64];
  __local half T_dense[1];
  T_dense_rf[(0)] = (half)0.000000e+00f;
  for (int k_outer = 0; k_outer < 64; ++k_outer) {
    T_dense_rf[(0)] = (T_dense_rf[(0)] + (placeholder[(((k_outer * 64) + ((int)get_local_id(0))))] * placeholder1[((((((int)get_group_id(0)) * 4096) + (k_outer * 64)) + ((int)get_local_id(0))))]));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  ((volatile __local half*)red_buf0)[(((int)get_local_id(0)))] = T_dense_rf[(0)];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)get_local_id(0)) < 32) {
    ((volatile __local half*)red_buf0)[(((int)get_local_id(0)))] = (((volatile __local half*)red_buf0)[(((int)get_local_id(0)))] + ((volatile __local half*)red_buf0)[((((int)get_local_id(0)) + 32))]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)get_local_id(0)) < 16) {
    ((volatile __local half*)red_buf0)[(((int)get_local_id(0)))] = (((volatile __local half*)red_buf0)[(((int)get_local_id(0)))] + ((volatile __local half*)red_buf0)[((((int)get_local_id(0)) + 16))]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)get_local_id(0)) < 8) {
    ((volatile __local half*)red_buf0)[(((int)get_local_id(0)))] = (((volatile __local half*)red_buf0)[(((int)get_local_id(0)))] + ((volatile __local half*)red_buf0)[((((int)get_local_id(0)) + 8))]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)get_local_id(0)) < 4) {
    ((volatile __local half*)red_buf0)[(((int)get_local_id(0)))] = (((volatile __local half*)red_buf0)[(((int)get_local_id(0)))] + ((volatile __local half*)red_buf0)[((((int)get_local_id(0)) + 4))]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)get_local_id(0)) < 2) {
    ((volatile __local half*)red_buf0)[(((int)get_local_id(0)))] = (((volatile __local half*)red_buf0)[(((int)get_local_id(0)))] + ((volatile __local half*)red_buf0)[((((int)get_local_id(0)) + 2))]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)get_local_id(0)) < 1) {
    ((volatile __local half*)red_buf0)[(((int)get_local_id(0)))] = (((volatile __local half*)red_buf0)[(((int)get_local_id(0)))] + ((volatile __local half*)red_buf0)[((((int)get_local_id(0)) + 1))]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (((int)get_local_id(0)) == 0) {
    T_dense[(0)] = ((volatile __local half*)red_buf0)[(0)];
  }
  if (((int)get_local_id(0)) == 0) {
    T_relu[(((int)get_group_id(0)))] = max((half)(T_dense[(0)] + placeholder2[(((int)get_group_id(0)))]), (half)(half)0.000000e+00f);
  }
}