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
# pylint: disable=invalid-name, unused-argument
"""Schedule for dense operator"""
import logging
from tvm import te, tir
import tvm.autotvm as autotvm
from tvm.autotvm.task.space import SplitEntity
from tvm.contrib import cublas
from .. import nn
from .. import tag
from .. import generic
from ..utils import traverse_inline, get_const_tuple

logger = logging.getLogger("topi")


@autotvm.register_topi_compute("dense_small_batch.adreno")
def dense_small_batch(cfg, data, weight, bias=None, out_dtype=None):
    args = {"accumulator": "float16"}
    return dense_small_batch_comp(cfg, data, weight, bias, out_dtype, args)


def dense_small_batch_comp(cfg, data, weight, bias=None, out_dtype=None, args={}):
    """Dense operator on Adreno"""
    """The default implementation of dense in topi.

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.te.Tensor
        2-D with shape [out_dim, in_dim]

    bias : Optional[tvm.te.Tensor]
        1-D with shape [out_dim]

    out_dtype : Optional[str]
        The output type. This is used for mixed precision.

    auto_scheduler_rewritten_layout: str = ""
        The layout after auto-scheduler's layout rewrite pass.

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = data.shape

    out_dim, red_dim = weight.shape
    assert in_dim == red_dim

    channel_block = 4
    channel_chunk = in_dim // channel_block
    out_chunk = out_dim // channel_block
    width = 1
    height = 1

    data = te.compute(
        [batch, height, width, channel_chunk, channel_block],
        lambda nn, yy, xx, icc, icb: data[nn, icc * channel_block + icb],
        name="input_pack",
        tag="input_pack",
    )

    weight = te.compute(
        [out_chunk, height, width, red_dim, channel_block],
        lambda oc, yy, xx, red_dim, ob: weight[oc * channel_block + ob, red_dim],
        name="weight_pack",
        tag="weight_pack",
    )

    #k = te.reduce_axis((0, channel_chunk), name="k")
    kcc = te.reduce_axis((0, channel_chunk), name="rc")
    kcb = te.reduce_axis((0, channel_block), name="rc")
    matmul = te.compute(
        (batch, height, width, out_chunk, channel_block),
        lambda b, h, w, oc, ob: te.sum(data[b, h, w, kcc, kcb].astype(out_dtype) * weight[oc, h, w, kcc * channel_block + kcb, ob].astype(out_dtype), axis=[kcc, kcb]),
        name="T_dense",
        tag="dense",
        #attrs={"layout_free_placeholders": [weight]},
    )
    #if bias is not None:
    #    matmul = te.compute(
    #        (batch, height, width, out_dim, channel_block),
    #        lambda b, h, w, o, cb: matmul[b, h, w, o, cb] + bias[o].astype(out_dtype),
    #        tag=tag.BROADCAST,
    #    )
    dummy_cast = te.compute((batch, height, width, out_chunk, channel_block), lambda b,h,w,o,cb: matmul[b,h,w,o,cb].astype(out_dtype), tag="dummy_cast")
    return te.compute((batch, out_dim), lambda n,o: dummy_cast[n,0,0,o//channel_block,o%channel_block], tag="cast_from_acc" + args["accumulator"][-2:])


@autotvm.register_topi_schedule("dense_small_batch.adreno")
def schedule_dense_small_batch(cfg, outs):
    return schedule_dense_small_batch_impl(cfg, outs, tag="cast_from_acc16")

@autotvm.register_topi_schedule("dense_small_batch_acc32.adreno")
def schedule_dense_small_batch_acc32(cfg, outs):
    return schedule_dense_small_batch_impl(cfg, outs, tag="cast_from_acc32")

def schedule_dense_small_batch_impl(cfg, outs, tag):
    """Schedule float32/64 dense with small batch size"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == tag:
            _schedule_dense_small_batch(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_dense_small_batch(cfg, s, output):
    dummy = output.op.input_tensors[0]
    matmul = dummy.op.input_tensors[0]
    data, weights = s[matmul].op.input_tensors

    s[data].compute_inline()
    s[weights].compute_inline()
    _, _, _, o, _ = s[matmul].op.axis
    kcc, kcb = s[matmul].op.reduce_axis
    #cfg.define_split("tile_fc", o, num_outputs=3)
    cfg.define_split("tile_fc", o, num_outputs=3,
                filter=lambda entity: entity.size[1] <= 16 and entity.size[2] >= 8 and entity.size[2] < 512 )
    cfg.define_split("tile_k", kcc, num_outputs=2)

    latest = s.outputs[0].output(0)

    # create cache stage
    def get_texture_storage(shape):
        limit = 16384
        if shape[0] * shape[1] * shape[2] < limit and shape[3] < limit:
            return "texture"
        elif shape[0] * shape[1] < limit and shape[2] * shape[3] < limit:
            return "texture:nhwc"
        else:
            return "texture:weight"

    AT = s.cache_read(data, get_texture_storage(data.shape), [matmul])
    WT = s.cache_read(weights, get_texture_storage(weights.shape), [matmul])
    def copy_to_texture(stage):
        axes = s[stage].op.axis
        fused = s[stage].fuse(*axes[:-1])
        block, thread = s[stage].split(fused, factor=32)
        s[stage].vectorize(axes[-1])
        s[stage].bind(block, te.thread_axis("blockIdx.x"))
        s[stage].bind(thread, te.thread_axis("threadIdx.x"))
    copy_to_texture(AT)
    copy_to_texture(WT)

    b, h, w, o, ob = s[dummy].op.axis
    bo, vo, to = cfg["tile_fc"].apply(s, dummy, o)

    b = s[dummy].fuse(b, h, w)
    s[dummy].bind(b, te.thread_axis("blockIdx.y"))
    s[dummy].bind(bo, te.thread_axis("blockIdx.x"))
    s[dummy].bind(to, te.thread_axis("threadIdx.x"))
    s[dummy].bind(vo, te.thread_axis("vthread"))
    s[dummy].reorder(b, bo, vo, to, ob)
    s[dummy].vectorize(ob)

    s[matmul].compute_at(s[dummy], to)

    b, h, w, o, ob = s[matmul].op.axis

    #if cfg.is_fallback:
    #    cfg["tile_k"] = SplitEntity([-1, 64] if k > 64 else [1, 64])
    ko, kf = cfg["tile_k"].apply(s, matmul, kcc)
    s[matmul].reorder(b, h, w, o, ko, kf, kcb, ob)
    s[matmul].vectorize(ob)
    s[matmul].unroll(kcb)

    s[latest].compute_root()

    b, o = s[latest].op.axis
    s[latest].bind(b, te.thread_axis("blockIdx.x"))
    s[latest].bind(o, te.thread_axis("blockIdx.y"))
    #s[latest].bind(o, te.thread_axis("threadIdx.x"))
    if output != latest:
        s[output].compute_inline()


#@autotvm.register_topi_compute("dense_large_batch.adreno")
#def dense_large_batch(cfg, data, weight, bias=None, out_dtype=None):
#    """Dense operator on CUDA"""
#    return nn.dense(data, weight, bias, out_dtype)
#
#
#@autotvm.register_topi_schedule("dense_large_batch.adreno")
#def schedule_dense_large_batch(cfg, outs):
#    """Schedule float32/64 dense with large batch size"""
#    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
#    s = te.create_schedule([x.op for x in outs])
#
#    def _callback(op):
#        if op.tag == "dense":
#            _schedule_dense_large_batch(cfg, s, op.output(0))
#
#    traverse_inline(s, outs[0].op, _callback)
#    return s
#
#
#def _schedule_dense_large_batch(cfg, s, C):
#    """Schedule float32/64 dense with large batch size"""
#    A, B = C.op.input_tensors
#    batch, in_dim = get_const_tuple(A.shape)
#    out_dim, _ = get_const_tuple(B.shape)
#    k = C.op.reduce_axis[0]
#
#    # create tuning space
#    try:
#        block_cand = [64, 128]
#        vthread_cand = [2 ** x for x in range(1, 7)]
#        n_thread_cand = [2 ** x for x in range(3, 7)]
#        cfg.define_split(
#            "tile_x",
#            batch,
#            num_outputs=4,
#            filter=lambda x: (
#                x.size[1] in vthread_cand
#                and x.size[2] in n_thread_cand
#                and (x.size[1] * x.size[2] * x.size[3]) in block_cand
#            ),
#        )
#        cfg.define_split(
#            "tile_y",
#            out_dim,
#            num_outputs=4,
#            filter=lambda x: (
#                x.size[1] in vthread_cand
#                and x.size[2] in n_thread_cand
#                and (x.size[1] * x.size[2] * x.size[3]) in block_cand
#            ),
#        )
#        cfg.define_split("tile_k", in_dim, num_outputs=3, filter=lambda x: x.size[0] > 2)
#    except IndexError:
#        # Index error happens when no entities left after filtering, which was designed
#        # to prune tuning space for better search efficiency.
#        logger.debug("Tuning space was created without pruning due to unfit shapes")
#        cfg.define_split("tile_x", batch, num_outputs=4)
#        cfg.define_split("tile_y", out_dim, num_outputs=4)
#        cfg.define_split("tile_k", in_dim, num_outputs=3)
#
#    if cfg.is_fallback:
#        if batch > 1:
#            cfg["tile_x"] = SplitEntity([-1, 2, 16, 2])
#        else:
#            cfg["tile_x"] = SplitEntity([1, 1, 1, 1])
#        if out_dim > 1:
#            cfg["tile_y"] = SplitEntity([-1, 2, 16, 2])
#        else:
#            cfg["tile_y"] = SplitEntity([1, 1, 1, 1])
#        if in_dim > 8:
#            cfg["tile_k"] = SplitEntity([-1, 8, 1])
#        else:
#            cfg["tile_k"] = SplitEntity([-1, 1, 1])
#
#    # Explicit memory access
#    AA = s.cache_read(A, "shared", [C])
#    BB = s.cache_read(B, "shared", [C])
#    AL = s.cache_read(AA, "local", [C])
#    BL = s.cache_read(BB, "local", [C])
#    CC = s.cache_write(C, "local")
#
#    # Deal with op fusion
#    if C.op not in s.outputs:
#        s[C].compute_inline()
#        C = s.outputs[0].output(0)
#
#    # Split and reorder computation
#    bx, txz, tx, xi = cfg["tile_x"].apply(s, C, C.op.axis[0])
#    by, tyz, ty, yi = cfg["tile_y"].apply(s, C, C.op.axis[1])
#    s[C].reorder(by, bx, tyz, txz, ty, tx, yi, xi)
#    s[CC].compute_at(s[C], tx)
#
#    # Binding
#    s[C].bind(by, te.thread_axis("blockIdx.y"))
#    s[C].bind(bx, te.thread_axis("blockIdx.x"))
#    s[C].bind(tyz, te.thread_axis("vthread"))
#    s[C].bind(txz, te.thread_axis("vthread"))
#    s[C].bind(ty, te.thread_axis("threadIdx.y"))
#    s[C].bind(tx, te.thread_axis("threadIdx.x"))
#
#    # Split reduction
#    yo, xo = CC.op.axis
#    ko, kt, ki = cfg["tile_k"].apply(s, CC, k)
#    s[CC].reorder(ko, kt, ki, yo, xo)
#    s[AA].compute_at(s[CC], ko)
#    s[BB].compute_at(s[CC], ko)
#    s[CC].unroll(kt)
#    s[AL].compute_at(s[CC], kt)
#    s[BL].compute_at(s[CC], kt)
#
#    # Schedule for A's shared memory load
#    num_thread_x = cfg["tile_x"].size[2]
#    ty, _ = s[AA].split(s[AA].op.axis[0], nparts=num_thread_x)
#    _, xi = s[AA].split(s[AA].op.axis[1], factor=num_thread_x * 4)
#    tx, xi = s[AA].split(xi, nparts=num_thread_x)
#    s[AA].bind(ty, te.thread_axis("threadIdx.y"))
#    s[AA].bind(tx, te.thread_axis("threadIdx.x"))
#    s[AA].double_buffer()
#
#    # Schedule for B' shared memory load
#    num_thread_y = cfg["tile_y"].size[2]
#    ty, _ = s[BB].split(s[BB].op.axis[0], nparts=num_thread_y)
#    _, xi = s[BB].split(s[BB].op.axis[1], factor=num_thread_y * 4)
#    tx, xi = s[BB].split(xi, nparts=num_thread_y)
#    s[BB].bind(ty, te.thread_axis("threadIdx.y"))
#    s[BB].bind(tx, te.thread_axis("threadIdx.x"))
#    s[BB].double_buffer()
#
#
