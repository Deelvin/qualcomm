#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supportedby OpenCL implementation on your device."
#endif

// Work_size: 6272x1x1x64x1x1x
__kernel void fused_nn_dense_add_nn_relu_1_kernel0(__write_only image2d_t input_pack_texture, __global half* restrict placeholder0) {
  (void)write_imageh(input_pack_texture, (int2)((((((int)get_group_id(0)) * 64) + ((int)get_local_id(0))) % 56), (((((int)get_group_id(0)) * 64) + ((int)get_local_id(0))) / 56)), vload4(0, placeholder0 + ((((int)get_group_id(0)) * 256) + (((int)get_local_id(0)) * 4))));
}

// Work_size: 256x1x1x32x1x1x
__kernel void fused_nn_dense_add_nn_relu_1_kernel1(__read_only image2d_t input_pack_texture, __read_only image2d_t placeholder1, __global half* restrict compute) {
  half4 T_dense[4];
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)T_dense + 0);
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)T_dense + 4);
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)T_dense + 8);
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)T_dense + 12);
  for (int kcc_outer = 0; kcc_outer < 28; ++kcc_outer) {
    for (int in_h = 0; in_h < 4; ++in_h) {
      for (int in_w = 0; in_w < 28; ++in_w) {
        for (int kcc_inner = 0; kcc_inner < 2; ++kcc_inner) {
          half4 _1 = read_imageh(input_pack_texture, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((kcc_outer * 2) + kcc_inner), ((in_h * 28) + in_w)));
          half4 _2 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((in_w * 224) + (kcc_outer * 8)) + (kcc_inner * 4)), (((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + in_h)));
          vstore4((vload4(0, (half*)T_dense + 0) + (((half*)&_1)[0] * _2)), 0, (half*)T_dense + 0);
          half4 _3 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((in_w * 224) + (kcc_outer * 8)) + (kcc_inner * 4)), ((((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + in_h) + 128)));
          vstore4((vload4(0, (half*)T_dense + 4) + (((half*)&_1)[0] * _3)), 0, (half*)T_dense + 4);
          half4 _4 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((in_w * 224) + (kcc_outer * 8)) + (kcc_inner * 4)), ((((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + in_h) + 256)));
          vstore4((vload4(0, (half*)T_dense + 8) + (((half*)&_1)[0] * _4)), 0, (half*)T_dense + 8);
          half4 _5 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((in_w * 224) + (kcc_outer * 8)) + (kcc_inner * 4)), ((((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + in_h) + 384)));
          vstore4((vload4(0, (half*)T_dense + 12) + (((half*)&_1)[0] * _5)), 0, (half*)T_dense + 12);
          half4 _6 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((in_w * 224) + (kcc_outer * 8)) + (kcc_inner * 4)) + 1), (((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + in_h)));
          vstore4((vload4(0, (half*)T_dense + 0) + (((half*)&_1)[1] * _6)), 0, (half*)T_dense + 0);
          half4 _7 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((in_w * 224) + (kcc_outer * 8)) + (kcc_inner * 4)) + 1), ((((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + in_h) + 128)));
          vstore4((vload4(0, (half*)T_dense + 4) + (((half*)&_1)[1] * _7)), 0, (half*)T_dense + 4);
          half4 _8 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((in_w * 224) + (kcc_outer * 8)) + (kcc_inner * 4)) + 1), ((((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + in_h) + 256)));
          vstore4((vload4(0, (half*)T_dense + 8) + (((half*)&_1)[1] * _8)), 0, (half*)T_dense + 8);
          half4 _9 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((in_w * 224) + (kcc_outer * 8)) + (kcc_inner * 4)) + 1), ((((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + in_h) + 384)));
          vstore4((vload4(0, (half*)T_dense + 12) + (((half*)&_1)[1] * _9)), 0, (half*)T_dense + 12);
          half4 _10 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((in_w * 224) + (kcc_outer * 8)) + (kcc_inner * 4)) + 2), (((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + in_h)));
          vstore4((vload4(0, (half*)T_dense + 0) + (((half*)&_1)[2] * _10)), 0, (half*)T_dense + 0);
          half4 _11 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((in_w * 224) + (kcc_outer * 8)) + (kcc_inner * 4)) + 2), ((((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + in_h) + 128)));
          vstore4((vload4(0, (half*)T_dense + 4) + (((half*)&_1)[2] * _11)), 0, (half*)T_dense + 4);
          half4 _12 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((in_w * 224) + (kcc_outer * 8)) + (kcc_inner * 4)) + 2), ((((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + in_h) + 256)));
          vstore4((vload4(0, (half*)T_dense + 8) + (((half*)&_1)[2] * _12)), 0, (half*)T_dense + 8);
          half4 _13 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((in_w * 224) + (kcc_outer * 8)) + (kcc_inner * 4)) + 2), ((((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + in_h) + 384)));
          vstore4((vload4(0, (half*)T_dense + 12) + (((half*)&_1)[2] * _13)), 0, (half*)T_dense + 12);
          half4 _14 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((in_w * 224) + (kcc_outer * 8)) + (kcc_inner * 4)) + 3), (((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + in_h)));
          vstore4((vload4(0, (half*)T_dense + 0) + (((half*)&_1)[3] * _14)), 0, (half*)T_dense + 0);
          half4 _15 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((in_w * 224) + (kcc_outer * 8)) + (kcc_inner * 4)) + 3), ((((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + in_h) + 128)));
          vstore4((vload4(0, (half*)T_dense + 4) + (((half*)&_1)[3] * _15)), 0, (half*)T_dense + 4);
          half4 _16 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((in_w * 224) + (kcc_outer * 8)) + (kcc_inner * 4)) + 3), ((((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + in_h) + 256)));
          vstore4((vload4(0, (half*)T_dense + 8) + (((half*)&_1)[3] * _16)), 0, (half*)T_dense + 8);
          half4 _17 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((in_w * 224) + (kcc_outer * 8)) + (kcc_inner * 4)) + 3), ((((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + in_h) + 384)));
          vstore4((vload4(0, (half*)T_dense + 12) + (((half*)&_1)[3] * _17)), 0, (half*)T_dense + 12);
        }
      }
    }
  }
  vstore4(vload4(0, (half*)T_dense + 0), 0, compute + ((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)));
  vstore4(vload4(0, (half*)T_dense + 4), 0, compute + (((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + 128));
  vstore4(vload4(0, (half*)T_dense + 8), 0, compute + (((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + 256));
  vstore4(vload4(0, (half*)T_dense + 12), 0, compute + (((((int)get_group_id(0)) * 512) + (((int)get_local_id(0)) * 4)) + 384));
}

// Work_size: 1x4096x1x1x1x1x
__kernel void fused_nn_dense_add_nn_relu_1_kernel2(__global half* restrict compute, __global half* restrict compute1, __global half* restrict placeholder2) {
  compute[(((int)get_group_id(1)))] = max((half)(compute1[(((int)get_group_id(1)))] + placeholder2[(((int)get_group_id(1)))]), (half)(half)0.000000e+00f);
}
