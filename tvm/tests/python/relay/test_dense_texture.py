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

import os
import tvm
import numpy as np
from tvm import relay
from tvm.relay import testing
from tvm.relay.transform import recast
from tvm.relay.transform import recast
from tvm.contrib import graph_runtime

def get_reference(mod, params1, input_shape, inputs):
    mod_fp32 = recast(mod, "float32", "float32", ops = ["nn.conv2d", "add", "nn.relu"])
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(
            mod_fp32, "llvm", params=params1
        )
    ctx = tvm.cpu()
    m = graph_runtime.create(graph, lib, ctx)
    if isinstance(input_shape, dict):
        for key in input_shape:
            m.set_input(key, inputs[-1])
    else:
        m.set_input("data", inputs[-1])
    m.set_input(**params)
    m.run()
    return [m.get_output(0).asnumpy(),]


# build module run with opencl and cpu, compare results
def build_run_compare(
    tvm_mod,
    params1,
    input_shape,
    dtype="float32",
    target="llvm"):

    rpc_tracker_host = os.environ["TVM_TRACKER_HOST"]
    rpc_tracker_port = os.environ["TVM_TRACKER_PORT"]
    if rpc_tracker_host:
        run_on_host = 0
        target_host = "llvm -mtriple=arm64-linux-android"
        rpc_tracker_port = int(rpc_tracker_port)
    else:
        run_on_host = 1
        target_host="llvm"

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(
            tvm_mod, target_host=target_host, target=target, params=params1
        )
    if run_on_host:
        ctx = tvm.opencl()
        m = graph_runtime.create(graph, lib, ctx)
    else:
        from tvm import rpc
        from tvm.contrib import utils, ndk
        rpc_key = "android"
        tracker = rpc.connect_tracker(rpc_tracker_host, rpc_tracker_port)
        remote = tracker.request(
            rpc_key, priority=0, session_timeout=600
        )
        temp = utils.tempdir()
        dso_binary = "dev_lib_cl.so"
        dso_binary_path = temp.relpath(dso_binary)
        ctx = remote.cl(0)
        lib.export_library(dso_binary_path, ndk.create_shared)
        remote.upload(dso_binary_path)
        rlib = remote.load_module(dso_binary)
        m = graph_runtime.create(graph, rlib, ctx)
    m.set_input(**params)
    inputs = []
    if isinstance(input_shape, dict):
        for key in input_shape:
            inputs.append(np.random.normal(size=input_shape[key]).astype(dtype))
            m.set_input(key, inputs[-1])
    else:
        inputs.append(np.random.normal(size=input_shape).astype(dtype))
        m.set_input("data", inputs[-1])
    m.run()

    ref_outputs = get_reference(tvm_mod, params1, input_shape, inputs)
    for i, ref_output in enumerate(ref_outputs):
        tvm_output = m.get_output(i)
        output = tvm_output.asnumpy()
        # for index, x in np.ndenumerate(ref_output):
        #     if abs(output[index] - x) > 0.01:
        #         print(index, output[index], x)

        np.testing.assert_allclose(output, ref_output, rtol=1e-1, atol=1e-1)


def test_dense_1x25088_4096x25088():
    target="opencl --device=adreno"
    dtype="float16"

    input_shape = (1, 25088)
    weights_shape = (4096, 25088)
    bias_shape = (weights_shape[0],)
    units = weights_shape[0]
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=weights_shape, dtype=dtype)
    C = relay.var("bias", shape=bias_shape, dtype=dtype)
    D = relay.nn.dense(A, B, units)
    D = relay.add(D, C)
    D = relay.nn.relu(D)
    mod = relay.Function([A, B, C], D)

    np.random.seed(1)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(weights_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target)


def test_dense_1x4096_4096x4096():
    target="opencl --device=adreno"
    dtype="float16"

    input_shape = (1, 4096)
    weights_shape = (4096, 4096)
    bias_shape = (weights_shape[0],)
    units = weights_shape[0]
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=weights_shape, dtype=dtype)
    C = relay.var("bias", shape=bias_shape, dtype=dtype)
    D = relay.nn.dense(A, B, units)
    D = relay.add(D, C)
    D = relay.nn.relu(D)
    mod = relay.Function([A, B, C], D)

    np.random.seed(1)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(weights_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target)


def test_dense_1x4096_1000x4096():
    target="opencl --device=adreno"
    dtype="float16"

    input_shape = (1, 4096)
    weights_shape = (1000, 4096)
    bias_shape = (weights_shape[0],)
    units = weights_shape[0]
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=weights_shape, dtype=dtype)
    C = relay.var("bias", shape=bias_shape, dtype=dtype)
    D = relay.nn.dense(A, B, units)
    D = relay.add(D, C)
    D = relay.nn.relu(D)
    mod = relay.Function([A, B, C], D)

    np.random.seed(1)
    initializer = relay.testing.init.Xavier()
    filter_data = np.ones(weights_shape).astype(dtype)
    bias_data = np.ones(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target)


def test_dense_1x4096_25088x4096():
    target="opencl --device=adreno"
    dtype="float16"

    input_shape = (1, 4096)
    weights_shape = (25088, 4096)
    bias_shape = (weights_shape[0],)
    units = weights_shape[0]
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=weights_shape, dtype=dtype)
    C = relay.var("bias", shape=bias_shape, dtype=dtype)
    D = relay.nn.dense(A, B, units)
    D = relay.add(D, C)
    D = relay.nn.relu(D)
    mod = relay.Function([A, B, C], D)

    np.random.seed(1)
    initializer = relay.testing.init.Xavier()
    filter_data = np.ones(weights_shape).astype(dtype)
    bias_data = np.ones(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target)


if __name__ == "__main__":
    test_dense_1x25088_4096x25088()
    test_dense_1x4096_4096x4096()
    test_dense_1x4096_1000x4096()
