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


@autotvm.register_topi_compute("dense.adreno")
def dense(cfg, data, weight, bias=None, out_dtype=None):
    args = {"accumulator": "float16"}
    return dense_comp(cfg, data, weight, bias, out_dtype, args)


@autotvm.register_topi_compute("dense_acc32.adreno")
def dense_acc32(cfg, data, weight, bias=None, out_dtype=None):
    args = {"accumulator": "float32"}
    return dense_comp(cfg, data, weight, bias, out_dtype, args)


def getDiv(value, start):
    div = 1
    for d in range(start,0,-1):
        if (value % d) == 0:
            div = d
            break
    return div


def dense_comp(cfg, data, weight, bias=None, out_dtype=None, args={}):
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

    def split_dim(dim_shape):
        h = getDiv(dim_shape, 4)
        dim_shape = dim_shape // h
        ch = getDiv(dim_shape, 64)
        w = dim_shape // ch
        return ch, h, w
    channel_chunk, height, width = split_dim(channel_chunk)
    red_dim = channel_chunk * channel_block

    data = te.compute(
        [batch, height, width, channel_chunk, channel_block],
        lambda nn, yy, xx, icc, icb: data[nn, yy*channel_block*channel_chunk*width + xx*channel_block*channel_chunk + icc*channel_block + icb].astype(args["accumulator"]),
        name="input_pack",
        tag="input_pack",
    )

    weight = te.compute(
        [out_chunk, height, width, red_dim, channel_block],
        lambda oc, yy, xx, rd, ob: weight[oc * channel_block + ob, yy*red_dim*width + xx*red_dim + rd].astype(args["accumulator"]),
        name="weight_pack",
        tag="weight_pack",
    )

    kcc = te.reduce_axis((0, channel_chunk), name="kcc")
    kcb = te.reduce_axis((0, channel_block), name="kcb")
    in_w = te.reduce_axis((0, width), name="in_w")
    in_h = te.reduce_axis((0, height), name="in_h")
    matmul = te.compute(
        (batch, out_chunk, channel_block),
        lambda b, oc, ob: te.sum(data[b, in_h, in_w, kcc, kcb] * weight[oc, in_h, in_w, kcc * channel_block + kcb, ob], axis=[in_h, in_w, kcc, kcb]),
        name="T_dense",
        tag="dense",
        attrs={"layout_free_placeholders": [weight]},
    )
    dummy_cast = te.compute((batch, out_chunk, channel_block), lambda b,o,cb: matmul[b,o,cb].astype(out_dtype), tag="dummy_cast")
    #return te.compute((batch, out_dim), lambda n,o: dummy_cast[n,o//(channel_block*channel_chunk*width),(o//(channel_block*channel_chunk))%width,(o//channel_block)%channel_chunk,o%channel_block], tag="cast_from_acc" + args["accumulator"][-2:])
    return te.compute((batch, out_dim), lambda n,o: dummy_cast[n,o//channel_block,o%channel_block], tag="cast_from_acc" + args["accumulator"][-2:])


@autotvm.register_topi_schedule("dense.adreno")
def schedule_dense(cfg, outs):
    return schedule_dense_impl(cfg, outs, tag="cast_from_acc16")

@autotvm.register_topi_schedule("dense_acc32.adreno")
def schedule_dense_acc32(cfg, outs):
    return schedule_dense_impl(cfg, outs, tag="cast_from_acc32")

def schedule_dense_impl(cfg, outs, tag):
    """Schedule float32/64 dense with small batch size"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == tag:
            _schedule_dense(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_dense(cfg, s, output):
    dummy = output.op.input_tensors[0]
    matmul = dummy.op.input_tensors[0]
    data, weights = s[matmul].op.input_tensors

    s[data].compute_inline()
    s[weights].compute_inline()
    _, _, kcc, kcb = s[matmul].op.reduce_axis
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

    b, o, ob = s[dummy].op.axis
    cfg.define_split("tile_fc", o, num_outputs=3)
    #cfg.define_split("tile_fc", o, num_outputs=3,
    #            filter=lambda entity: entity.size[1] <= 16 and entity.size[2] >= 8 and entity.size[2] < 512 )
    bo, vo, to = cfg["tile_fc"].apply(s, dummy, o)

    #b = s[dummy].fuse(b, h, w)
    s[dummy].bind(b, te.thread_axis("blockIdx.y"))
    s[dummy].bind(bo, te.thread_axis("blockIdx.x"))
    s[dummy].bind(to, te.thread_axis("threadIdx.x"))
    s[dummy].bind(vo, te.thread_axis("vthread"))
    s[dummy].reorder(b, bo, vo, to, ob)
    s[dummy].vectorize(ob)

    s[matmul].compute_at(s[dummy], to)

    b, o, ob = s[matmul].op.axis

    #if cfg.is_fallback:
    #    cfg["tile_k"] = SplitEntity([-1, 64] if k > 64 else [1, 64])
    ko, kf = cfg["tile_k"].apply(s, matmul, kcc)
    s[matmul].reorder(b, o, ko, kf, kcb, ob)
    s[matmul].vectorize(ob)
    s[matmul].unroll(kcb)
    #CF = s.rfactor(matmul, kf)

    s[latest].compute_root()

    b, o = s[latest].op.axis
    s[latest].bind(b, te.thread_axis("blockIdx.x"))
    s[latest].bind(o, te.thread_axis("blockIdx.y"))
    #s[latest].bind(o, te.thread_axis("threadIdx.x"))
    if output != latest:
        s[output].compute_inline()
