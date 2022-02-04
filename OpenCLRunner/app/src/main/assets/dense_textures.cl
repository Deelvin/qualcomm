// Work_size: 1024x1x1x32x1x1x
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supportedby OpenCL implementation on your device."
#endif

__kernel void fused_nn_dense_add_nn_relu_1_kernel0(__write_only image2d_t input_pack_texture, __global half* restrict placeholder0) {
  (void)write_imageh(input_pack_texture, (int2)((((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) & 63), (((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) >> 6)), vload4(0, placeholder0 + ((((int)get_group_id(0)) * 128) + (((int)get_local_id(0)) * 4))));
}

// Work_size: 4194304x1x1x32x1x1x
__kernel void fused_nn_dense_add_nn_relu_1_kernel1(__write_only image2d_t weight_pack_texture_nhwc, __global half* restrict placeholder1) {
  int4 _1 = (int4)(((((((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) >> 12) * 16384) + (((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) & 4095)))+(4096*0),
                   ((((((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) >> 12) * 16384) + (((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) & 4095)))+(4096*1),
                   ((((((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) >> 12) * 16384) + (((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) & 4095)))+(4096*2),
                   ((((((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) >> 12) * 16384) + (((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) & 4095)))+(4096*3));
  (void)write_imageh(weight_pack_texture_nhwc, (int2)((((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) & 1023), (((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) >> 10)), ((half4)(placeholder1[_1.s0],placeholder1[_1.s1],placeholder1[_1.s2],placeholder1[_1.s3])));
}

// Work_size: 1024x1x1x32x1x1x
__kernel void fused_nn_dense_add_nn_relu_1_kernel2(__read_only image2d_t input_pack_texture, __read_only image2d_t weight_pack_texture_nhwc, __global half* restrict compute) {
  half4 T_dense[1];
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)T_dense + 0);
  for (int kcc_outer = 0; kcc_outer < 2; ++kcc_outer) {
    for (int in_h = 0; in_h < 4; ++in_h) {
      for (int in_w = 0; in_w < 4; ++in_w) {
        for (int kcc_inner = 0; kcc_inner < 32; ++kcc_inner) {
          half4 _1 = read_imageh(input_pack_texture, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((kcc_outer * 32) + kcc_inner), ((in_h * 4) + in_w)));
          half4 _2 = read_imageh(weight_pack_texture_nhwc, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((in_w * 256) + (kcc_outer * 128)) + (kcc_inner * 4)), (((((int)get_group_id(0)) * 128) + (((int)get_local_id(0)) * 4)) + in_h)));
          vstore4((vload4(0, (half*)T_dense + 0) + (((half*)&_1)[0] * _2)), 0, (half*)T_dense + 0);
          half4 _3 = read_imageh(weight_pack_texture_nhwc, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((in_w * 256) + (kcc_outer * 128)) + (kcc_inner * 4)) + 1), (((((int)get_group_id(0)) * 128) + (((int)get_local_id(0)) * 4)) + in_h)));
          vstore4((vload4(0, (half*)T_dense + 0) + (((half*)&_1)[1] * _3)), 0, (half*)T_dense + 0);
          half4 _4 = read_imageh(weight_pack_texture_nhwc, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((in_w * 256) + (kcc_outer * 128)) + (kcc_inner * 4)) + 2), (((((int)get_group_id(0)) * 128) + (((int)get_local_id(0)) * 4)) + in_h)));
          vstore4((vload4(0, (half*)T_dense + 0) + (((half*)&_1)[2] * _4)), 0, (half*)T_dense + 0);
          half4 _5 = read_imageh(weight_pack_texture_nhwc, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((in_w * 256) + (kcc_outer * 128)) + (kcc_inner * 4)) + 3), (((((int)get_group_id(0)) * 128) + (((int)get_local_id(0)) * 4)) + in_h)));
          vstore4((vload4(0, (half*)T_dense + 0) + (((half*)&_1)[3] * _5)), 0, (half*)T_dense + 0);
        }
      }
    }
  }
  vstore4(vload4(0, (half*)T_dense + 0), 0, compute + ((((int)get_group_id(0)) * 128) + (((int)get_local_id(0)) * 4)));
}

// Work_size: 1x4096x1x1x1x1x
__kernel void fused_nn_dense_add_nn_relu_1_kernel3(__global half* restrict compute, __global half* restrict compute1, __global half* restrict placeholder2) {
  compute[(((int)get_group_id(1)))] = max((half)(compute1[(((int)get_group_id(1)))] + placeholder2[(((int)get_group_id(1)))]), (half)(half)0.000000e+00f);
}

