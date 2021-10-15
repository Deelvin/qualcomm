import numpy as np
import tvm
import os
from tvm import rpc
from tvm import relay, autotvm
from tvm.contrib import ndk
from tvm.contrib import graph_runtime
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

TRACKER_HOST = "0.0.0.0"
TRACKER_PORT = 9190
TRACKER_KEY = "android"
LIB_NAME = "conv_from_inception_v3"
target_host = tvm.target.Target('llvm -mtriple=arm64-linux-android')
target = tvm.target.Target('opencl --device=adreno')
LOG_FILE = "inception_v3_conv_tuned.log"
WITH_TUNING = True

def _get_model(input_shape, filter_shape, dtype, var_names):
    """Return a model and any parameters it may have."""
    params = {}
    data = relay.var(next(var_names), shape=input_shape, dtype=dtype)
    kernel_size = (filter_shape[2], filter_shape[3])
    w_shape = (int(filter_shape[0] / 4), filter_shape[1], *kernel_size, 4)
    print("dtype: ", w_shape)
    w = tvm.nd.array(np.random.uniform(-128, 127, w_shape).astype(dtype))
    weights = relay.const(w, dtype)
    params["w"] = w
    b_shape = (1, int(filter_shape[0] / 4), 1, 1, 4)
    b = tvm.nd.array(np.random.uniform(-128, 127, b_shape).astype(dtype))
    bias = relay.const(b, dtype)
    params["b"] = b
    out = relay.nn.conv2d(data, weights, padding=(1,1), kernel_size=kernel_size, data_layout="NCHW4c", kernel_layout="OIHW4o")
    out = relay.add(out, bias)
    out = relay.nn.relu(out)
    if isinstance(out, tvm.relay.expr.Call):
        out = tvm.IRModule.from_expr(out)
    print('*' * 50)
    print(out)
    print('*' * 50)
    return out, params


def create_module(mod, params, log_file=None):
    if log_file is not None:
        with autotvm.apply_history_best(log_file):
            print("Compile...")
            with tvm.transform.PassContext(opt_level=3):
                graph_module = relay.build(mod['main'], target=target, target_host=target_host, params=params)
    else:
        with tvm.transform.PassContext(opt_level=3):
            graph_module = relay.build(mod['main'], target=target, target_host=target_host, params=params)
    lib_name = LIB_NAME + ".so"
    graph_module.export_library(lib_name, ndk.create_shared)
    return lib_name


def prepare_tvm(remote, ctx, lib_file):
    remote.upload(lib_file)
    func = remote.load_module(lib_file)
    return graph_runtime.GraphModule(func["default"](ctx))


def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        print("Task: ", tsk)
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    #os.remove(tmp_log_file)


def tune_model(mod, params):
    if WITH_TUNING is False:
        return None
    if os.path.exists(LOG_FILE):
        #return '/home/echuraev/Workspace/OctoML/qualcomm/logs/inceptionv3.texture.float16.acc16.autotvm.log'
        return LOG_FILE
    print("Extract tasks...")
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, target_host=target_host, params=params
    )
    tuning_opt = {
        "log_filename": LOG_FILE,
        "tuner": "xgb",
        "n_trial": 1024,
        "early_stopping": 600,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10, build_func=ndk.create_shared),
            runner=autotvm.RPCRunner(
                TRACKER_KEY,
                TRACKER_HOST,
                TRACKER_PORT,
                number=20,
                repeat=3,
                timeout=4,
                min_repeat_ms=150,
            ),
        ),
    }
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)
    return LOG_FILE


if __name__ == '__main__':
    input_shape = (1, 8, 147, 147, 4)
    filter_shape = (64, 32, 3, 3)
    dtype = 'float16'
    inputs = {
        "data": tvm.nd.array(np.random.uniform(-128, 127, input_shape).astype(dtype)),
    }
    mod, params = _get_model(input_shape, filter_shape, dtype, iter(inputs))
    tuned_info = tune_model(mod, params)
    lib_name = create_module(mod, params, tuned_info)

    remote = rpc.TrackerSession((TRACKER_HOST, TRACKER_PORT)).request(TRACKER_KEY)
    ctx = remote.cl()

    m = prepare_tvm(remote, ctx, lib_name)
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    m.set_input("data", data_tvm)
    print("Evaluate inference time cost...")
    ftimer = m.module.time_evaluator("run", ctx, number=1, repeat=100)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print(
        "Mean inference time (std dev): %.2f ms (%.2f ms)"
        % (np.mean(prof_res), np.std(prof_res))
    )
