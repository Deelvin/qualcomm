# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,unused-variable,unused-argument
"""Winograd template for Adreno backend"""

import logging
import tvm
from tvm import te
from tvm import autotvm

from tvm.topi import nn
from tvm.topi.utils import get_const_int, get_const_tuple, traverse_inline
from ..nn.winograd_util import winograd_transform_matrices
from .utils import split_to_chunks, pack_input, pack_filter, expand_spatial_dimensions, add_pad, bind_data_copy, get_texture_storage


logger = logging.getLogger("conv2d_nchw_winograd")


def _infer_tile_size(data):
    if len(data.shape) == 4:
        N, CI, H, W = get_const_tuple(data.shape)
    else:
        N, CI, H, W, CB = get_const_tuple(data.shape)

    if H % 8 == 0:
        return 4
    return 2

@autotvm.register_topi_compute("conv2d_nchw_winograd.image2d")
def conv2d_nchw_winograd(cfg, data, kernel, strides, padding, dilation, out_dtype):
    args={"shared" : False, "accumulator" : "float16"}
    return conv2d_nchw_winograd_comp(
        cfg, data, kernel, strides, padding, dilation, out_dtype, args=args, pre_computed=False
    )

@autotvm.register_topi_compute("conv2d_nchw_winograd_acc32.image2d")
def conv2d_nchw_winograd_acc32(cfg, data, kernel, strides, padding, dilation, out_dtype):
    args={"shared" : False, "accumulator" : "float32"}
    return conv2d_nchw_winograd_comp(
        cfg, data, kernel, strides, padding, dilation, out_dtype, args=args, pre_computed=False
    )

@autotvm.register_topi_schedule("conv2d_nchw_winograd.image2d")
def schedule_conv2d_nchw_winograd(cfg, outs):
    return schedule_conv2d_nchw_winograd_impl(cfg, outs, tag="cast_from_acc16")

@autotvm.register_topi_schedule("conv2d_nchw_winograd_acc32.image2d")
def schedule_conv2d_nchw_winograd_acc32(cfg, outs):
    return schedule_conv2d_nchw_winograd_impl(cfg, outs, tag="cast_from_acc32")

@autotvm.register_topi_compute("conv2d_nchw_winograd_without_weight_transform.image2d")
def conv2d_nchw_winograd_without_weight_transform(cfg, data, kernel, strides, padding, dilation, out_dtype):
    args={"shared" : False, "accumulator" : "float16"}
    return conv2d_nchw_winograd_comp(
        cfg, data, kernel, strides, padding, dilation, out_dtype, args=args, pre_computed=True
    )

@autotvm.register_topi_compute("conv2d_nchw_winograd_without_weight_transform_acc32.image2d")
def conv2d_nchw_winograd_without_weight_transform_acc32(cfg, data, kernel, strides, padding, dilation, out_dtype):
    args={"shared" : False, "accumulator" : "float32"}
    return conv2d_nchw_winograd_comp(
        cfg, data, kernel, strides, padding, dilation, out_dtype, args=args, pre_computed=True
    )

@autotvm.register_topi_schedule("conv2d_nchw_winograd_without_weight_transform.image2d")
def schedule_conv2d_nchw_winograd_without_weight_transform(cfg, outs):
    return schedule_conv2d_nchw_winograd_impl(cfg, outs, tag="cast_from_acc16", pre_computed=True)

@autotvm.register_topi_schedule("conv2d_nchw_winograd_without_weight_transform_acc32.image2d")
def schedule_conv2d_nchw_winograd_without_weight_transform_acc32(cfg, outs):
    return schedule_conv2d_nchw_winograd_impl(cfg, outs, tag="cast_from_acc32", pre_computed=True)

def schedule_conv2d_nchw_winograd_impl(cfg, outs, tag, pre_computed=False):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == tag:
            schedule_conv2d_winograd(cfg, s, op.output(0), pre_computed=pre_computed)

    traverse_inline(s, outs[0].op, _callback)
    return s

def conv2d_nchw_winograd_comp(cfg, data, kernel, strides, padding, dilation, out_dtype, args, pre_computed):
    """Compute declaration for winograd"""
    tile_size = _infer_tile_size(data)

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    HSTR, WSTR = (strides, strides) if isinstance(strides, int) else strides

    convert_from4d = False
    if len(data.shape) == 4:
        N, DCI, H, W = get_const_tuple(data.shape)
        if not pre_computed:
            out_channels, CI, KH, KW = get_const_tuple(kernel.shape)
        else:
            alpha, _, CI, out_channels = get_const_tuple(kernel.shape)
            KH = KW = alpha + 1 - tile_size

        in_channel_chunks, in_channel_block, in_channel_tail = split_to_chunks(CI, 4)
        out_channel_chunks, out_channel_block, out_channel_tail = split_to_chunks(out_channels, 4)
        if autotvm.GLOBAL_SCOPE.in_tuning == True:
            dshape = (N, in_channel_chunks, H, W, in_channel_block)
            data = tvm.te.placeholder(dshape, data.dtype, name="data_placeholder")
            if not pre_computed:  # kernel tensor is raw tensor, do strict check
                kshape = (out_channel_chunks, CI, KH, KW, out_channel_block)
                kernel = tvm.te.placeholder(kshape, kernel.dtype, name="kernel_placeholder")
            else:
                kshape = (alpha, alpha, CI, out_channel_chunks, out_channel_block)
                kernel = tvm.te.placeholder(kshape, kernel.dtype, name="kernel_placeholder")
        else:
            convert_from4d = True
            data = pack_input(data, "NCHW", N, in_channel_chunks, in_channel_block, in_channel_tail, H, W)
            if not pre_computed:  # kernel tensor is raw tensor, do strict check
                kernel = pack_filter(kernel, "OIHW", out_channel_chunks, out_channel_block, out_channel_tail,
                                     CI, in_channel_chunks, in_channel_block, in_channel_tail, KH, KW)
            else:
                kernel = pack_filter(kernel, "HWIO", out_channel_chunks, out_channel_block, out_channel_tail,
                                     CI, in_channel_chunks, in_channel_block, in_channel_tail, alpha, alpha)
    N, DCI, H, W, CB = get_const_tuple(data.shape)
    if not pre_computed:  # kernel tensor is raw tensor, do strict check
        CO, CI, KH, KW, COB = get_const_tuple(kernel.shape)
        alpha = KW + tile_size - 1
        assert HSTR == 1 and WSTR == 1 and KH == KW
    else:
        alpha, _, CI, CO, COB = get_const_tuple(kernel.shape)
        KH = KW = alpha + 1 - tile_size
        assert HSTR == 1 and WSTR == 1 and dilation_h == 1 and dilation_w == 1

    if isinstance(N, tvm.tir.Any):
        N = tvm.te.size_var("n")

    if not isinstance(H, int) or not isinstance(W, int):
        raise RuntimeError(
            "adreno winograd conv2d doesn't support dynamic input\
                           height or width."
        )

    pt, pl, pb, pr = nn.get_pad_tuple(padding, (KH, KW))
    # 0.12 ms
    data_pad = nn.pad(data, (0, 0, pt, pl, 0), (0, 0, pb, pr, 0), name="data_pad")

    r = KW
    m = tile_size
    A, B, G = winograd_transform_matrices(m, r, args["accumulator"])

    H = (H + pt + pb - KH) // HSTR + 1
    W = (W + pl + pr - KW) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m

    P = N * nH * nW if isinstance(N, int) else nH * nW

    # transform kernel
    if not pre_computed:
        r_kh = te.reduce_axis((0, KH), name="r_kh")
        r_kw = te.reduce_axis((0, KW), name="r_kw")
        kernel_pack = te.compute(
            (alpha, alpha, CI, CO, COB),
            lambda eps, nu, ci, co, cob: te.sum(
                kernel[co][ci][r_kh][r_kw][cob].astype(args["accumulator"]) * G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]
            ),
            name="kernel_pack",
        )
    else:
        kernel_pack = kernel

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    N, CI, H, W, CB = get_const_tuple(data.shape)
    #input_tile = te.compute(
    #    (CI, P, alpha, alpha, CB),
    #    lambda c, p, eps, nu, cb: data_pad[idxdiv(p, (nH * nW))][c][
    #        idxmod(idxdiv(p, nW), nH) * m + eps
    #    ][idxmod(p, nW) * m + nu][cb].astype(args["accumulator"]),
    #    name="d",
    #)

    # transform data
    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_a")
    # 6.26 ms
    # compute op -> buffer ? How to force to be texture
    data_pack = te.compute(
        (alpha, alpha, CI, P, CB),
        lambda eps, nu, ci, p, cb: te.sum(
            data_pad[idxdiv(p, (nH * nW))][ci][idxmod(idxdiv(p, nW), nH) * m + r_a][idxmod(p, nW) * m + r_b][cb].astype(args["accumulator"]) * B[r_a][eps] * B[r_b][nu], axis=[r_a, r_b]
            #input_tile[ci][p][r_a][r_b][cb] * B[r_a][eps] * B[r_b][nu], axis=[r_a, r_b]
        ),
        name="data_pack",
    )

    # cache_read: 0.225 ms

    # do batch gemm
    ci = te.reduce_axis((0, CI), name="ci")
    cb = te.reduce_axis((0, CB), name="cb")
    # 2.84 ms
    bgemm = te.compute(
        (alpha, alpha, CO, P, COB),
        lambda eps, nu, co, p, cob: te.sum(
            kernel_pack[eps][nu][ci * CB + cb][co][cob].astype(args["accumulator"]) * data_pack[eps][nu][ci][p][cb], axis=[ci, cb]
        ),
        name="bgemm",
    )

    # inverse transform
    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_a")
    # 1.09 ms
    inverse = te.compute(
        (CO, P, m, m, COB),
        lambda co, p, vh, vw, cob: te.sum(
            bgemm[r_a][r_b][co][p][cob] * A[r_a][vh] * A[r_b][vw], axis=[r_a, r_b]
        ),
        name="inverse",
    )

    # output
    if convert_from4d and autotvm.GLOBAL_SCOPE.in_tuning == False:
        #dummy_cast = te.compute(
        #    (N, out_channel_chunks, H, W, out_channel_block),
        #    lambda n, co, h, w, cob: inverse[co][n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m)][idxmod(h, m)][idxmod(w, m)][cob].astype(out_dtype),
        #    tag="dummy_cast")
        #output = te.compute(
        #    (N, out_channels, H, W),
        #    lambda n, c, h, w: dummy_cast[c // CB][n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m)][idxmod(h, m)][idxmod(w, m)][c % CB],
        #    name="output",
        #    tag="cast_from_acc" + args["accumulator"][-2:],
        #)
        output = te.compute(
            (N, out_channels, H, W),
            lambda n, c, h, w: inverse[c // CB][n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m)][idxmod(h, m)][idxmod(w, m)][c % CB].astype(out_dtype),
            name="output",
            tag="cast_from_acc" + args["accumulator"][-2:],
        )
    else:
        output = te.compute(
            (N, CO, H, W, COB),
            lambda n, co, h, w, cob: inverse[co][n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m)][idxmod(h, m)][idxmod(w, m)][cob].astype(out_dtype),
            name="output",
            tag="cast_from_acc" + args["accumulator"][-2:],
        )

    if isinstance(N, int):
        cfg.add_flop(2 * N * CO * COB * H * W * CI * CB * KH * KW)

    return output


def schedule_conv2d_winograd(cfg, s, output, pre_computed):
    """Schedule winograd template"""
    # get stages
    ##latest = s.outputs[0].output(0)
    ##if len(latest.op.axis) == 4:
    ##  latest_blocked = dummy = output.op.input_tensors[0]
    ##  inverse = dummy.op.input_tensors[0]
    ##else:
    ##  inverse = output.op.input_tensors[0]
    ##  latest_blocked = latest
    inverse = s[output].op.input_tensors[0]
    bgemm, A = s[inverse].op.input_tensors
    kernel_pack, data_pack = s[bgemm].op.input_tensors
    pad_data, B = s[data_pack].op.input_tensors
    #input_tile, B = s[data_pack].op.input_tensors
    #pad_data = s[input_tile].op.input_tensors[0]

    # data transform
    s[B].compute_inline()
    s[A].compute_inline()

    # Padding to texture
    AA = s.cache_read(pad_data, get_texture_storage(pad_data.shape), [data_pack])
    bind_data_copy(s[AA])

    # Precalculate matrix
    BM = s.cache_read(B, "global", [data_pack])
    bind_data_copy(s[BM])

    OL = s.cache_write(data_pack, "local")
    eps, nu, c, p, cb = s[data_pack].op.axis
    p, pi = s[data_pack].split(p, 1)
    fused = s[data_pack].fuse(c, p)
    bx, tx = s[data_pack].split(fused, 128)
    #fused = s[data_pack].fuse(nu, eps)
    #cfg.define_split("tile_y", fused, num_outputs=2)
    #by, ty = cfg["tile_y"].apply(s, data_pack, fused)
    #by, ty = s[data_pack].split(fused, 8)
    by, ty = nu, eps
    #s[data_pack].reorder(bx, tx, pi, eps, nu, cb)
    s[data_pack].reorder(bx, by, tx, ty, pi, cb)
    s[data_pack].vectorize(cb)
    s[data_pack].bind(bx, te.thread_axis("blockIdx.x"))
    s[data_pack].bind(tx, te.thread_axis("threadIdx.x"))
    s[data_pack].bind(by, te.thread_axis("blockIdx.y"))
    s[data_pack].bind(ty, te.thread_axis("threadIdx.y"))


    eps, nu, c, p, cb = s[OL].op.axis
    r_a, r_b = s[OL].op.reduce_axis
    s[OL].reorder(eps, nu, c, p, r_a, r_b, cb)
    s[OL].vectorize(cb)
    s[OL].compute_at(s[data_pack], ty)
    #s[data_pack].bind(pi, te.thread_axis("vthread"))
    s[data_pack].set_scope(get_texture_storage(data_pack.shape))
    #s[input_tile].compute_at(s[data_pack], pi)

    # transform kernel
    if not pre_computed:
        kernel, G = s[kernel_pack].op.input_tensors
        eps, nu, ci, co, cob = s[kernel_pack].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # skip this part during tuning to make recrods accurate
            # this part will be pre-computed during pre-compute optimization pass
            s[G].pragma(s[G].op.axis[0], "debug_skip_region")
            s[kernel_pack].pragma(eps, "debug_skip_region")
        else:
            s[G].compute_inline()
            r_a, r_b = s[kernel_pack].op.reduce_axis
            for axis in [eps, nu, r_a, r_b]:
                s[kernel_pack].unroll(axis)

            fused = s[kernel_pack].fuse(ci, co)
            bb, tt = s[kernel_pack].split(fused, 128)
            s[kernel_pack].reorder(bb, tt, eps, nu, r_a, r_b, cob)
            s[kernel_pack].vectorize(cob)
            s[kernel_pack].bind(bb, te.thread_axis("blockIdx.x"))
            s[kernel_pack].bind(tt, te.thread_axis("threadIdx.x"))
    else:
        kernel = kernel_pack

    if isinstance(kernel.op, tvm.te.ComputeOp) and "filter_pack" in kernel.op.tag:
        # manage scheduling of datacopy
        pack_data = pad_data.op.input_tensors[0]
        bind_data_copy(s[pack_data])
        bind_data_copy(s[kernel])
    elif isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()
    s[pad_data].compute_inline()

    ##### space definition begin #####
    b1, b2, y, x, cb = s[bgemm].op.axis
    rcc = s[bgemm].op.reduce_axis[0]
    alpha = get_const_int(b1.dom.extent)

    cfg.define_split(
        "tile_b", cfg.axis(alpha * alpha), num_outputs=4, filter=lambda x: x.size[-3:] == [1, 1, 1]
    )
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rcc, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 128, 1500])
    target = tvm.target.Target.current()
    if target.kind.name in ["nvptx", "rocm"]:
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])
    ##### space definition end #####

    # batch gemm
    OL = s.cache_write(bgemm, "local")
    #AA = s.cache_read(data_pack, get_texture_storage(data_pack.shape), [OL])
    #bind_data_copy(s[AA])
    if (autotvm.GLOBAL_SCOPE.in_tuning or
        isinstance(kernel.op, tvm.te.ComputeOp) and "filter_pack" in kernel.op.tag):
        BB = s.cache_read(kernel_pack, get_texture_storage(kernel_pack.shape), [OL])
        bind_data_copy(s[BB])

    b = s[bgemm].fuse(b1, b2)

    # tile and bind spatial axes
    bgemm_scope, b = s[bgemm].split(b, nparts=1)
    bz, vz, tz, zi = cfg["tile_b"].apply(s, bgemm, b)
    by, vy, ty, yi = cfg["tile_y"].apply(s, bgemm, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, bgemm, x)
    s[bgemm].bind(bz, te.thread_axis("blockIdx.z"))
    s[bgemm].bind(by, te.thread_axis("blockIdx.y"))
    s[bgemm].bind(bx, te.thread_axis("blockIdx.x"))
    s[bgemm].bind(vz, te.thread_axis("vthread"))
    s[bgemm].bind(vy, te.thread_axis("vthread"))
    s[bgemm].bind(vx, te.thread_axis("vthread"))
    s[bgemm].bind(tz, te.thread_axis("threadIdx.z"))
    s[bgemm].bind(ty, te.thread_axis("threadIdx.y"))
    s[bgemm].bind(tx, te.thread_axis("threadIdx.x"))
    s[bgemm].reorder(bgemm_scope, bz, by, bx, vz, vy, vx, tz, ty, tx, zi, yi, xi, cb)
    s[bgemm].vectorize(cb)
    s[bgemm].set_scope(get_texture_storage(bgemm.shape))

    # tile reduction axes
    s[OL].compute_at(s[bgemm], tx)
    b1, b2, y, x, cb = s[OL].op.axis
    (rcc, rcb) = s[OL].op.reduce_axis
    b = s[OL].fuse(b1, b2)
    rco, rci = cfg["tile_rc"].apply(s, OL, rcc)
    s[OL].reorder(rco, rci, rcb, b, y, x, cb)
    s[OL].vectorize(cb)

    s[bgemm].pragma(bgemm_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[bgemm].pragma(bgemm_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    # schedule inverse, output and fusion
    if output.op in s.outputs:
        OL = None
    else:
        OL = output
        s[OL].set_scope("local")
        output = s.outputs[0]

    m = alpha - 3 + 1
    if len(s[output].op.axis) == 4:
        n, co, h, w = s[output].op.axis
    else:
        n, co, h, w, _ = s[output].op.axis
    ho, wo, hi, wi = s[output].tile(h, w, m, m)
    inverse_scope, n = s[output].split(n, nparts=1)

    fused = s[output].fuse(n, co, ho, wo)
    bb, tt = s[output].split(fused, 128)

    s[output].bind(bb, te.thread_axis("blockIdx.x"))
    s[output].bind(tt, te.thread_axis("threadIdx.x"))

    if OL is not None:
        s[OL].compute_at(s[output], tt)

    co, p, vh, vw, cb = s[inverse].op.axis
    r_a, r_b = s[inverse].op.reduce_axis
    for axis in [vh, vw, r_a, r_b]:
        s[inverse].unroll(axis)
    s[inverse].vectorize(cb)
    s[inverse].compute_at(s[output], tt)

    return s
