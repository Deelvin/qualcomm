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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Dense alter op and legalize functions for adreno"""

import logging

import re
import tvm
from tvm import te
from tvm import relay
from tvm import autotvm
from ..utils import get_const_tuple
from ..nn import dense_alter_layout
from .utils import getDiv, split_dim

logger = logging.getLogger("topi")


@dense_alter_layout.register("adreno")
def _alter_dense_layout(attrs, inputs, tinfos, out_type):
    print("In def _alter_dense_layout", flush=True)
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current
    data_tensor, weight_tensor = tinfos
    out_dtype = out_type.dtype
    B, K = get_const_tuple(data_tensor.shape)
    N, _ = get_const_tuple(weight_tensor.shape)

    channel_block = 4
    channel_chunk = K // channel_block

    channel_chunk, height, width = split_dim(channel_chunk)
    red_dim = channel_chunk * channel_block

    if isinstance(dispatch_ctx, autotvm.task.ApplyGraphBest):
        cfg = dispatch_ctx.query(target, None)
        workload = cfg.workload
    else:
        impl, outs = relay.backend.compile_engine.select_implementation(
            relay.op.get("nn.dense"), attrs, tinfos, out_type, target
        )
        workload = autotvm.task.get_workload(outs)
    if workload:
        cfg = dispatch_ctx.query(target, workload)
        topi_impl = workload[0]
        if topi_impl == "dense.image2d":
            new_weight = te.placeholder(
                (N // channel_block, height, width, red_dim, channel_block),
                dtype=weight_tensor.dtype,
            )
            # Relay dense doesn't have bias.
            new_workload = autotvm.task.args_to_workload(
                [
                    data_tensor,
                    new_weight,
                    None,
                    out_dtype,
                ],
                topi_impl,
            )
            print("New workload: ", new_workload)
            dispatch_ctx.update(target, new_workload, cfg)
            layout = "NK%dn" % channel_block
            weight_transform = relay.layout_transform(inputs[1], "NK", layout)
            layout = "NK%dkB" % red_dim
            weight_transform = relay.layout_transform(weight_transform, "NKB", layout)
            layout = "NH%dhRB" % width
            weight_transform = relay.layout_transform(weight_transform, "NHRB",layout) # OHWC4o
            layout = "NC{}c{}c{}n".format(width, red_dim, channel_block)
            print("Layout: ", layout)

            #layout = "NK%dk" % channel_block
            #data_transform = relay.layout_transform(inputs[0], "NK", layout)
            #layout = "NK%dkB" % channel_chunk
            #data_transform = relay.layout_transform(data_transform, "NKB", layout)
            #layout = "NH%dhCB" % width
            #data_transform = relay.layout_transform(data_transform, "NHCB",layout) # NHWC4c
            #return relay.nn.dense(data_transform, weight_transform, None, out_dtype, "NHWC4c")
            #return relay.nn.dense(inputs[0], weight_transform, None, out_dtype, "NHWC4c")
            return relay.nn.dense(inputs[0], inputs[1], None, out_dtype, layout)

    return None


