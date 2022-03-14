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
from ..nn.conv2d import conv2d_winograd_nhwc, _conv2d_winograd_nhwc_impl
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

# Tile. For NCHW4c H or W is not block and there is a gaps between items. Maybe I should vectorize by tile?
# Try to generate this code and understand
def conv2d_nchw_winograd_comp(cfg, data, kernel, strides, padding, dilation, out_dtype, pre_computed):
    """Compute declaration for winograd"""
    tile_size = _infer_tile_size(data)

    data_len = len(data.shape)
    kernel_len = len(kernel.shape)
    if data_len == 4:
        N, _, H, W = get_const_tuple(data.shape)
        CO, CI, KH, KW = get_const_tuple(kernel.shape)
    else:
        N, _, H, W, _ = get_const_tuple(data.shape)
        CO, CI, KH, KW, COB = get_const_tuple(kernel.shape)

    if isinstance(N, tvm.tir.Any):
        N = tvm.te.size_var("n")

    if not isinstance(H, int) or not isinstance(W, int):
        raise RuntimeError(
            "cuda winograd conv2d doesn't support dynamic input\
                           height or width."
        )

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    HSTR, WSTR = (strides, strides) if isinstance(strides, int) else strides

    alpha = KW + tile_size - 1
    assert HSTR == 1 and WSTR == 1 and KH == KW

    pt, pl, pb, pr = nn.get_pad_tuple(padding, (KH, KW))
    if data_len == 4:
        data_pad = nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")
    else:
        data_pad = nn.pad(data, (0, 0, pt, pl, 0), (0, 0, pb, pr, 0), name="data_pad")

    r = KW
    m = tile_size
    A, B, G = winograd_transform_matrices(m, r, out_dtype)

    H = (H + pt + pb - KH) // HSTR + 1
    W = (W + pl + pr - KW) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m

    P = N * nH * nW if isinstance(N, int) else nH * nW

    # transform kernel
    r_kh = te.reduce_axis((0, KH), name="r_kh")
    r_kw = te.reduce_axis((0, KW), name="r_kw")
    if kernel_len == 4:
        kernel_pack = te.compute(
            (alpha, alpha, CI, CO),
            lambda eps, nu, ci, co: te.sum(
                kernel[co][ci][r_kh][r_kw] * G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]
            ),
            name="kernel_pack",
        )
    else:
        kernel_pack = te.compute(
            (alpha, alpha, CI, CO, COB),
            lambda eps, nu, ci, co, cob: te.sum(
                kernel[co][ci][r_kh][r_kw][cob] * G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]
            ),
            name="kernel_pack",
        )

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    if data_len == 4:
        N, CI, H, W = get_const_tuple(data.shape)
        # pack input tile
        input_tile = te.compute(
            (CI, P, alpha, alpha),
            lambda c, p, eps, nu: data_pad[idxdiv(p, (nH * nW))][c][
                idxmod(idxdiv(p, nW), nH) * m + eps
            ][idxmod(p, nW) * m + nu],
            name="d",
        )

        # transform data
        r_a = te.reduce_axis((0, alpha), "r_a")
        r_b = te.reduce_axis((0, alpha), "r_a")
        data_pack = te.compute(
            (alpha, alpha, CI, P),
            lambda eps, nu, ci, p: te.sum(
                input_tile[ci][p][r_a][r_b] * B[r_a][eps] * B[r_b][nu], axis=[r_a, r_b]
            ),
            name="data_pack",
        )

        # do batch gemm
        ci = te.reduce_axis((0, CI), name="ci")
        bgemm = te.compute(
            (alpha, alpha, CO, P),
            lambda eps, nu, co, p: te.sum(
                kernel_pack[eps][nu][ci][co] * data_pack[eps][nu][ci][p], axis=[ci]
            ),
            name="bgemm",
        )

        # inverse transform
        r_a = te.reduce_axis((0, alpha), "r_a")
        r_b = te.reduce_axis((0, alpha), "r_a")
        inverse = te.compute(
            (CO, P, m, m),
            lambda co, p, vh, vw: te.sum(
                bgemm[r_a][r_b][co][p] * A[r_a][vh] * A[r_b][vw], axis=[r_a, r_b]
            ),
            name="inverse",
        )

        # output
        output = te.compute(
            (N, CO, H, W),
            lambda n, co, h, w: inverse[
                co, n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m), idxmod(h, m), idxmod(w, m)
            ],
            name="output",
            tag="conv2d_nchw_winograd",
        )
    else:
        N, CI, H, W, CB = get_const_tuple(data.shape)
        # pack input tile
        input_tile = te.compute(
            (CI, P, alpha, alpha, CB),
            lambda c, p, eps, nu, cb: data_pad[idxdiv(p, (nH * nW))][c][
                idxmod(idxdiv(p, nW), nH) * m + eps
            ][idxmod(p, nW) * m + nu][cb],
            name="d",
        )

        # transform data
        r_a = te.reduce_axis((0, alpha), "r_a")
        r_b = te.reduce_axis((0, alpha), "r_a")
        data_pack = te.compute(
            (alpha, alpha, CI, P, CB),
            lambda eps, nu, ci, p, cb: te.sum(
                input_tile[ci][p][r_a][r_b][cb] * B[r_a][eps] * B[r_b][nu], axis=[r_a, r_b]
            ),
            name="data_pack",
        )

        # do batch gemm
        ci = te.reduce_axis((0, CI), name="ci")
        cb = te.reduce_axis((0, CB), name="cb")
        bgemm = te.compute(
            (alpha, alpha, CO, P, COB),
            lambda eps, nu, co, p, cob: te.sum(
                kernel_pack[eps][nu][ci * CB + cb][co][cob] * data_pack[eps][nu][ci][p][cb], axis=[ci, cb]
            ),
            name="bgemm",
        )

        # inverse transform
        r_a = te.reduce_axis((0, alpha), "r_a")
        r_b = te.reduce_axis((0, alpha), "r_a")
        inverse = te.compute(
            (CO, P, m, m, COB),
            lambda co, p, vh, vw, cob: te.sum(
                bgemm[r_a][r_b][co][p][cob] * A[r_a][vh] * A[r_b][vw], axis=[r_a, r_b]
            ),
            name="inverse",
        )

        # output
        output = te.compute(
            (N, CO, H, W, COB),
            lambda n, co, h, w, cob: inverse[co][n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m)][idxmod(h, m)][idxmod(w, m)][cob],
            name="output",
            tag="conv2d_nchw_winograd",
        )

    if isinstance(N, int):
        if data_len == 4:
            cfg.add_flop(2 * N * CO * H * W * CI * KH * KW)
        else:
            cfg.add_flop(2 * N * CO * COB * H * W * CI * CB * KH * KW)

    return output


def schedule_conv2d_nchw_winograd_impl(cfg, s, output, pre_computed):
    """Schedule winograd template"""
    # get stages
    inverse = s[output].op.input_tensors[0]
    bgemm, A = s[inverse].op.input_tensors
    kernel_pack, data_pack = s[bgemm].op.input_tensors
    input_tile, B = s[data_pack].op.input_tensors
    pad_data = s[input_tile].op.input_tensors[0]

    # data transform
    s[B].compute_inline()

    data_l = s.cache_write(data_pack, "local")
    len_axis = len(s[data_l].op.axis)
    if len_axis == 4:
        eps, nu, c, p = s[data_l].op.axis
    else:
        eps, nu, c, p, _ = s[data_l].op.axis
    r_a, r_b = s[data_l].op.reduce_axis
    for axis in [eps, nu, r_a, r_b]:
        s[data_l].unroll(axis)

    if len_axis == 4:
        eps, nu, c, p = s[data_pack].op.axis
    else:
        eps, nu, c, p, cb = s[data_pack].op.axis
    p, pi = s[data_pack].split(p, 1)
    fused = s[data_pack].fuse(c, p)
    bb, tt = s[data_pack].split(fused, 128)
    if len_axis == 4:
        s[data_pack].reorder(bb, tt, pi, eps, nu)
    else:
        s[data_pack].reorder(bb, tt, pi, eps, nu, cb)
        s[data_pack].vectorize(cb)
    s[data_pack].bind(bb, te.thread_axis("blockIdx.x"))
    s[data_pack].bind(tt, te.thread_axis("threadIdx.x"))

    s[data_l].compute_at(s[data_pack], pi)
    s[input_tile].compute_at(s[data_pack], pi)
    s[pad_data].compute_inline()

    # transform kernel
    kernel, G = s[kernel_pack].op.input_tensors
    if len_axis == 4:
        eps, nu, ci, co = s[kernel_pack].op.axis
    else:
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
        if len_axis == 4:
            s[kernel_pack].reorder(bb, tt, eps, nu, r_a, r_b)
        else:
            s[kernel_pack].reorder(bb, tt, eps, nu, r_a, r_b, cob)
            s[kernel_pack].vectorize(cob)
        s[kernel_pack].bind(bb, te.thread_axis("blockIdx.x"))
        s[kernel_pack].bind(tt, te.thread_axis("threadIdx.x"))

    if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()

    ##### space definition begin #####
    if len_axis == 4:
        b1, b2, y, x = s[bgemm].op.axis
    else:
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
    C = bgemm
    A0, B0 = kernel_pack, data_pack

    OL = s.cache_write(C, "local")
    AA = s.cache_read(A0, "shared", [OL])
    BB = s.cache_read(B0, "shared", [OL])

    b = s[bgemm].fuse(b1, b2)

    # tile and bind spatial axes
    bgemm_scope, b = s[bgemm].split(b, nparts=1)
    bz, vz, tz, zi = cfg["tile_b"].apply(s, C, b)
    by, vy, ty, yi = cfg["tile_y"].apply(s, C, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, C, x)
    s[C].bind(bz, te.thread_axis("blockIdx.z"))
    s[C].bind(by, te.thread_axis("blockIdx.y"))
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(vz, te.thread_axis("vthread"))
    s[C].bind(vy, te.thread_axis("vthread"))
    s[C].bind(vx, te.thread_axis("vthread"))
    s[C].bind(tz, te.thread_axis("threadIdx.z"))
    s[C].bind(ty, te.thread_axis("threadIdx.y"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))
    if len_axis == 4:
        s[C].reorder(bgemm_scope, bz, by, bx, vz, vy, vx, tz, ty, tx, zi, yi, xi)
    else:
        s[C].reorder(bgemm_scope, bz, by, bx, vz, vy, vx, tz, ty, tx, zi, yi, xi, cb)
        s[C].vectorize(cb)

    # tile reduction axes
    s[OL].compute_at(s[C], tx)
    if len_axis == 4:
        b1, b2, y, x = s[OL].op.axis
        (rcc,) = s[OL].op.reduce_axis
    else:
        b1, b2, y, x, cb = s[OL].op.axis
        (rcc, rcb) = s[OL].op.reduce_axis
    b = s[OL].fuse(b1, b2)
    #rco, rci = cfg["tile_rc"].apply(s, OL, rcc)
    if len_axis == 4:
        #s[OL].reorder(rco, rci, b, y, x)
        s[OL].reorder(rcc, b, y, x)
    else:
        #s[OL].reorder(rco, rci, rb, b, y, x, cb)
        s[OL].reorder(rcc, rcb, b, y, x, cb)
        s[OL].vectorize(cb)

    s[AA].compute_at(s[OL], rcc)
    s[BB].compute_at(s[OL], rcc)

    # cooperative fetching
    for load in [AA, BB]:
        fused = s[load].fuse(*list(s[load].op.axis))
        fused, tx = s[load].split(fused, cfg["tile_x"].size[2])
        fused, ty = s[load].split(fused, cfg["tile_y"].size[2])
        fused, tz = s[load].split(fused, cfg["tile_b"].size[2])
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))

    s[C].pragma(bgemm_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[C].pragma(bgemm_scope, "unroll_explicit", cfg["unroll_explicit"].val)

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
        n, co, h, w, cb = s[output].op.axis
    ho, wo, hi, wi = s[output].tile(h, w, m, m)
    inverse_scope, n = s[output].split(n, nparts=1)

    fused = s[output].fuse(n, co, ho, wo)
    bb, tt = s[output].split(fused, 128)

    s[output].bind(bb, te.thread_axis("blockIdx.x"))
    s[output].bind(tt, te.thread_axis("threadIdx.x"))

    if OL is not None:
        s[OL].compute_at(s[output], tt)

    s[A].compute_inline()
    if len(s[inverse].op.axis) == 4:
        co, p, vh, vw = s[inverse].op.axis
    else:
        co, p, vh, vw, cb = s[inverse].op.axis
    r_a, r_b = s[inverse].op.reduce_axis
    for axis in [vh, vw, r_a, r_b]:
        s[inverse].unroll(axis)
    if len(s[inverse].op.axis) != 4:
        s[inverse].vectorize(cb)
    s[inverse].compute_at(s[output], tt)

    return s


@autotvm.register_topi_compute("conv2d_nchw_winograd.image2d")
def conv2d_nchw_winograd(cfg, data, kernel, strides, padding, dilation, out_dtype):
    return conv2d_nchw_winograd_comp(
        cfg, data, kernel, strides, padding, dilation, out_dtype, pre_computed=False
    )


@autotvm.register_topi_schedule("conv2d_nchw_winograd.image2d")
def schedule_conv2d_nchw_winograd(cfg, outs):
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv2d_nchw_winograd" in op.tag:
            schedule_conv2d_nchw_winograd_impl(cfg, s, op.output(0), pre_computed=False)

    traverse_inline(s, outs[0].op, _callback)
    return s
