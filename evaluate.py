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
import numpy as np

import mxnet.gluon as gluon
import tvm
from tvm import relay
from tvm.relay import testing
from tvm import autotvm
from tvm.contrib import utils, ndk
from tvm.topi import testing

# DEELVIN-207
# from tvm.relay.op import register_mixed_precision_conversion
# Pick a priority > 10 to overwrite defaults, higher priorities take precedence
# @register_mixed_precision_conversion("nn.conv2d", level=11)
# def conv2d_mixed_precision_rule(call_node: "relay.Call", mixed_precision_type: str):
#     return [
#         # always do main calculation in mixed_precision_type
#         relay.transform.mixed_precision.MIXED_PRECISION_ALWAYS,
#         # the dtype for the accumulator
#         "float32",
#         # the output dtype for the operation (usually fp16)
#         mixed_precision_type,
#     ]



class ModelImporter(object):
    def available_models(self):
        import inspect
        models = []
        for method in inspect.getmembers(type(self)):
            if "import_" in method[0]:
                models.append(method[0].split("import_")[1])
        return models

    def __call__(self, model, *args, **kwargs):
        import inspect

        for method in inspect.getmembers(type(self)):
            if "import_" + model == method[0]:
                return method[1](self, *args, **kwargs)
        raise ValueError("import_" + model + " not found.")


    def get_onnx_from_tf1(self, model_url, filename, input_names, output_names, shape_override = None):
        tf_model_file = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + "/models/{}.pb".format(filename)
        )

        from tvm.contrib import download
        download.download(model_url, tf_model_file)
        # converted using command line:
        # python -m tf2onnx.convert --graphdef mace_resnet-v2-50.pb --output mace_resnet-v2-50.onnx --inputs input:0[1,224,224,3] --outputs resnet_v2_50/predictions/Reshape_1:0
        onnx_model_file = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + "/models/{}.onnx".format(filename))
        if os.path.exists(onnx_model_file) == False:
            import tf2onnx
            import tensorflow as tf
            try:
                tf_compat_v1 = tf.compat.v1
            except ImportError:
                tf_compat_v1 = tf
            # Tensorflow utility functions
            import tvm.relay.testing.tf as tf_testing

            with tf_compat_v1.gfile.GFile(tf_model_file, "rb") as f:
                graph_def = tf_compat_v1.GraphDef()
                graph_def.ParseFromString(f.read())
                #graph = tf.import_graph_def(graph_def, name="")
                # Call the utility to import the graph definition into default graph.
                graph_def = tf_testing.ProcessGraphDefParam(graph_def)

                model_proto, external_tensor_storage = tf2onnx.convert.from_graph_def(graph_def,
                    name=filename, input_names=input_names, output_names=output_names,
                    shape_override = shape_override,
                    output_path=onnx_model_file)

        return onnx_model_file


    def get_graphdef_from_tf1(self, model_url, filename):
        graph_def = None
        tf_model_file = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + "/models/{}.pb".format(filename)
        )

        from tvm.contrib import download
        download.download(model_url, tf_model_file)
        # converted using command line:
        # python -m tf2onnx.convert --graphdef mace_resnet-v2-50.pb --output mace_resnet-v2-50.onnx --inputs input:0[1,224,224,3] --outputs resnet_v2_50/predictions/Reshape_1:0
        onnx_model_file = os.path.abspath(
            os.path.dirname(os.path.realpath(__file__))
            + "/../models/{}.onnx".format(filename))
        import tensorflow as tf
        try:
            tf_compat_v1 = tf.compat.v1
        except ImportError:
            tf_compat_v1 = tf
        # Tensorflow utility functions
        import tvm.relay.testing.tf as tf_testing

        with tf_compat_v1.gfile.GFile(tf_model_file, "rb") as f:
            graph_def = tf_compat_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        return graph_def

    def import_mace_mobilenetv1_nhwc(self, target="llvm", dtype="float32"):
        model_url = "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/mobilenet-v1/mobilenet-v1-1.0.pb"
        filename = "mace_mobilenet-v1-1.0"
        graph_def = self.get_graphdef_from_tf1(model_url, filename)
        shape_dict = {"input": (1, 224, 224, 3)}
        mod, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict,
                                        outputs=["MobilenetV1/Predictions/Reshape_1"])

        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        return (mod, params, shape_dict, dtype, target, ImageNetValidator(shape_dict, "NHWC", preproc="keras_mobilenetv1"))

    def import_mace_mobilenetv1_nchw(self, target="llvm", dtype="float32"):
        model_url = "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/mobilenet-v1/mobilenet-v1-1.0.pb"
        filename = "mace_mobilenet-v1-1.0"
        input_names = ["input:0"]
        output_names = ["MobilenetV1/Predictions/Reshape_1:0"]
        onnx_model_file = self.get_onnx_from_tf1(model_url, filename, input_names, output_names)
        import onnx
        model = onnx.load(onnx_model_file)
        shape_dict = {'input:0': [1, 224, 224, 3]}
        mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)

        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        return (mod, params, shape_dict, dtype, target, ImageNetValidator(shape_dict, "NHWC", preproc="keras_mobilenetv1"))

    def import_mace_resnet50_v2(self, target="llvm", dtype="float32"):
        model_url = "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/resnet-v2-50/resnet-v2-50.pb"
        filename = "mace_resnet-v2-50"
        input_names = ["input:0"]
        shape_override = {"input:0": [1, 224, 224, 3]}
        output_names = ["resnet_v2_50/predictions/Reshape_1:0"]
        onnx_model_file = self.get_onnx_from_tf1(model_url, filename, input_names, output_names, shape_override)
        import onnx
        model = onnx.load(onnx_model_file)
        shape_dict = {'input:0': [1, 224, 224, 3]}
        mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)

        # DEELVIN-207
        # mod = relay.transform.InferType()(mod)
        # mod = relay.transform.ToMixedPrecision()(mod)
        # print(mod)

        mod = relay.quantize.prerequisite_optimize(mod, params)

        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        return (mod, params, shape_dict, dtype, target, ImageNetValidator(shape_dict, "NHWC", preproc="keras_mobilenetv1"))


    def import_ac_resnet50_tf(self, target="llvm", dtype="float32"):
        model_url = "https://download.01.org/opencv/public_models/012020/resnet-50-tf/resnet_v1-50.pb"
        filename = "resnet_v1-50"
        input_names = ["map/TensorArrayStack/TensorArrayGatherV3:0"]
        shape_override = {"map/TensorArrayStack/TensorArrayGatherV3:0": [1, 224, 224, 3]}
        output_names = ["softmax_tensor:0"]
        onnx_model_file = self.get_onnx_from_tf1(model_url, filename, input_names, output_names, shape_override)
        import onnx
        model = onnx.load(onnx_model_file)
        shape_dict = {'map/TensorArrayStack/TensorArrayGatherV3:0': [1, 224, 224, 3]}
        mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)

        mod = relay.quantize.prerequisite_optimize(mod, params)

        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        return (mod, params, shape_dict, dtype, target)


    def import_mace_inceptionv3(self, target="llvm", dtype="float32"):
        model_url = "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/inception-v3/inception-v3.pb"
        filename = "mace_inception-v3"
        input_names = ["input:0"]
        output_names = ["InceptionV3/Predictions/Reshape_1:0"]
        onnx_model_file = self.get_onnx_from_tf1(model_url, filename, input_names, output_names)
        import onnx
        model = onnx.load(onnx_model_file)
        shape_dict = {'input:0': [1, 299, 299, 3]}
        mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)

        mod = relay.quantize.prerequisite_optimize(mod, params)
        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        return (mod, params, shape_dict, dtype, target, ImageNetValidator(shape_dict, "NHWC", preproc="keras"))

    def import_mxnet_vgg16(self, target="llvm", dtype="float32"):
        model, input_shape = gluon_model("vgg16", batch_size=1)
        shape_dict = {"data": input_shape}
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
        mod = relay.quantize.prerequisite_optimize(mod, params)

        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        return (mod, params, shape_dict, dtype, target, ImageNetValidator(shape_dict, preproc="mxnet"))

    def import_mace_deeplabv3(self, target="llvm", dtype="float32"):
        model_url = "https://cnbj1.fds.api.xiaomi.com/mace/miai-models/deeplab-v3-plus/deeplab-v3-plus-mobilenet-v2.pb"
        filename = "mace_deeplab-v3-plus-mobilenet-v2"
        graph_def = self.get_graphdef_from_tf1(model_url, filename)
        shape_dict = {"sub_7": (1, 513, 513, 3)}
        mod, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict,
                                        outputs=["ResizeBilinear_2"])

        mod = relay.quantize.prerequisite_optimize(mod, params)

        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
            mod = relay.quantize.prerequisite_optimize(mod, params)

        mod = relay.quantize.prerequisite_optimize(mod, params)
        return (mod, params, shape_dict, dtype, target)



    def import_mace_yolov3(self, target="llvm", dtype="float32"):
        model_url = "http://cnbj1.fds.api.xiaomi.com/mace/miai-models/yolo-v3/yolo-v3.pb"
        filename = "mace_yolo-v3"
        graph_def = self.get_graphdef_from_tf1(model_url, filename)
        shape_dict = {"input_1": (1, 416, 416, 3)}
        mod, params = relay.frontend.from_tensorflow(graph_def, shape=shape_dict,
                                        outputs=["conv2d_59/BiasAdd","conv2d_67/BiasAdd","conv2d_75/BiasAdd"])


        # model_url = "http://cnbj1.fds.api.xiaomi.com/mace/miai-models/yolo-v3/yolo-v3.pb"
        # model_path = os.path.abspath(
        #     os.path.dirname(os.path.realpath(__file__))
        #     + "/../models/mace_yolov3/yolo-v3.pb"
        # )

        # from tvm.contrib import download
        # download.download(model_url, model_path)

        # import tensorflow as tf
        # try:
        #     tf_compat_v1 = tf.compat.v1
        # except ImportError:
        #     tf_compat_v1 = tf
        # # Tensorflow utility functions
        # import tvm.relay.testing.tf as tf_testing

        # with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
        #     graph_def = tf_compat_v1.GraphDef()
        #     graph_def.ParseFromString(f.read())
        #     #graph = tf.import_graph_def(graph_def, name="")
        #     # Call the utility to import the graph definition into default graph.
        #     graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        # input_shape = {"input_1": (1, 416, 416, 3)}
        # mod, params = relay.frontend.from_tensorflow(graph_def, shape=input_shape,
        #                                 outputs=["conv2d_59/BiasAdd","conv2d_67/BiasAdd","conv2d_75/BiasAdd"])

        from tvm.relay import transform
        #mod = transform.DynamicToStatic()(mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)

        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
            mod = relay.quantize.prerequisite_optimize(mod, params)

        return (mod, params, shape_dict, dtype, target)


    def import_resnet50(self, target="llvm", dtype="float32"):
        model, input_shape = gluon_model("resnet50_v1", batch_size=1)
        shape_dict = {"data": input_shape}
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
        mod = relay.quantize.prerequisite_optimize(mod, params)

        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        return (mod, params, shape_dict, dtype, target, ImageNetValidator(shape_dict, preproc="mxnet"))


    def import_yolov3_mxnet(self, target="llvm", dtype="float32"):
        model, input_shape = gluoncv_model("yolo3_darknet53_voc", batch_size=1)
        shape_dict = {"data": input_shape}
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
        mod = relay.quantize.prerequisite_optimize(mod, params)

        # downcast to float16
        if dtype == "float16":
            mod = downcast_fp16(mod["main"], mod)
        mod = relay.quantize.prerequisite_optimize(mod, params)
        print(mod)
        return (mod, params, shape_dict, dtype, target, VOCValidator(shape_dict, preproc="gluoncv"))


def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Tune and/or evaluate a curated set of models"
    )
    models = ModelImporter().available_models()

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        required=True,
        help="Model to tune and/or evaluate",
        choices=models,
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="float16",
        choices=["float32", "float16"],
        help="Specify whether the model should be run with single or half precision floating point values",
    )
    parser.add_argument(
        "-l", "--log", type=str, default=None, help="AutoTVM tuning logfile name"
    )
    parser.add_argument(
        "-k", "--rpc_key", type=str, default="android", help="RPC key to use"
    )
    parser.add_argument(
        "-r",
        "--rpc_tracker_host",
        type=str,
        default=os.environ["TVM_TRACKER_HOST"],
        help="RPC tracker host IP address",
    )
    parser.add_argument(
        "-p",
        "--rpc_tracker_port",
        type=str,
        default=os.environ["TVM_TRACKER_PORT"],
        help="RPC tracker host port",
    )
    parser.add_argument(
        "-T",
        "--target",
        type=str,
        default="opencl --device=mali",
        help="Compilation target",
    )
    parser.add_argument(
        "--tune", action="store_true", help="Whether or not to run autotuning"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use graph runtime debugger to output per layer perf. data and other statistics",
    )

    args = parser.parse_args()
    if args.log == None:
        args.log = "logs/" + args.model + "." + args.type + ".autotvm.log"
    if args.rpc_tracker_port != None:
        args.rpc_tracker_port = int(args.rpc_tracker_port)
    args.tuning_options = {
        "log_filename": args.log,
        "early_stopping": None,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func=ndk.create_shared, timeout=15),
            runner=autotvm.RPCRunner(
                args.rpc_key,
                host=args.rpc_tracker_host,
                port=args.rpc_tracker_port,
                number=50,
                timeout=15,
                #min_repeat_ms=150,
                #cooldown_interval=150
            ),
        ),
    }
    return args


args = get_args()


def main():
    if "opencl" in args.target:
        executor = Executor(use_tracker="android")
    else:
        executor = Executor()
    executor.schedule(args.model, target=args.target, dtype=args.type)
    if args.tune:
        executor.tune_pending_benchmarks()
    else:
        executor.tune_pending_benchmarks(apply_previous_tune=True)
    executor.run_pending_benchmarks()


def downcast_fp16(func, module):
    from tvm.relay.expr_functor import ExprMutator
    from tvm.relay.expr import Call, Var, Constant, TupleGetItem
    from tvm.relay import transform as _transform
    from tvm.relay import cast
    from tvm.ir import IRModule
    from tvm.relay import function as _function

    """Downcast to fp16 mutator
    Parameters
    ---------
    graph: Function
        The original graph.

    Retruns
    -------
    The graph after dowmcasting to half-precision floating-point.
    """
    filter_list = ["vision.get_valid_counts", "vision.non_max_suppression"]

    class DowncastMutator(ExprMutator):
        """Downcast to fp16 mutator"""

        def visit_call(self, call):
            dtype = "float32" if call.op.name in filter_list else "float16"
            new_fn = self.visit(call.op)
            # Collect the original dtypes
            type_list = []
            if call.op.name in filter_list:
                # For NMS
                for arg in call.args:
                    if isinstance(arg, TupleGetItem) and isinstance(
                        arg.tuple_value, Call
                    ):
                        tuple_types = arg.tuple_value.checked_type.fields
                        type_list.append(tuple_types[arg.index].dtype)
                if call.op.name == "vision.get_valid_counts":
                    tuple_types = call.checked_type.fields
                    for cur_type in tuple_types:
                        type_list.append(cur_type.dtype)

            args = [self.visit(arg) for arg in call.args]
            new_args = list()
            arg_idx = 0
            for arg in args:
                if isinstance(arg, (Var, Constant)):
                    new_args.append(cast(arg, dtype=dtype))
                else:
                    if call.op.name in filter_list:
                        if (
                            isinstance(arg, TupleGetItem)
                            and type_list[arg_idx] == "int32"
                        ):
                            new_args.append(arg)
                        else:
                            new_args.append(cast(arg, dtype=dtype))
                    else:
                        new_args.append(arg)
                arg_idx += 1
            if (
                call.op.name in filter_list
                and call.op.name != "vision.get_valid_counts"
            ):
                return cast(Call(new_fn, new_args, call.attrs), dtype="float16")
            return Call(new_fn, new_args, call.attrs)

    class UpcastMutator(ExprMutator):
        """upcast output back to fp32 mutator"""

        def visit_call(self, call):
            return cast(call, dtype="float32")

    def infer_type(node, mod=None):
        """A method to infer the type of an intermediate node in the relay graph."""
        if isinstance(mod, IRModule):
            mod["main"] = _function.Function(tvm.relay.analysis.free_vars(node), node)
            mod = _transform.InferType()(mod)
            entry = mod["main"]
            ret = entry.body
        else:
            new_mod = IRModule.from_expr(node)
            if mod is not None:
                new_mod.update(mod)
                new_mod = _transform.InferType()(new_mod)
                entry = new_mod["main"]
                ret = entry if isinstance(node, _function.Function) else entry.body

        return ret

    func = infer_type(func, module)
    downcast_pass = DowncastMutator()
    func = downcast_pass.visit(func)
    upcast_pass = UpcastMutator()
    func = upcast_pass.visit(func)
    func = infer_type(func, module)
    new_mod = IRModule.from_expr(func)
    # new_mod.update(module)
    return new_mod


def get_input_data_shape_dict(graph_def, input_shape):
    if isinstance(input_shape, list):
        input_names = {}
        shape_dict = {}
        for i in range(len(input_shape)):
            input_names[i] = graph_def.graph.input[i].name
            shape_dict[input_names[i]] = input_shape[i]
    else:
        input_names = graph_def.graph.input[0].name
        shape_dict = {input_names: input_shape}

    return input_names, shape_dict


def gluon_model(name, batch_size=None):
    if "resnet50_v1" in name or "mobilenet1.0" in name or "resnet50_v2" in name or "vgg16" in name:
        model = gluon.model_zoo.vision.get_model(name, pretrained=True)
        data_shape = (batch_size, 3, 224, 224)
    elif "inceptionv3" in name:
        model = gluon.model_zoo.vision.inception_v3(pretrained=True)
        data_shape = (batch_size, 3, 299, 299)
    else:
        raise ValueError("Input shape unknown for gluon model: " + name)
    return model, data_shape


def gluoncv_model(name, batch_size=None):
    from gluoncv import model_zoo
    if "yolo3" in name:
        model = model_zoo.get_model(name, pretrained=True)
        data_shape = (batch_size, 3, 416, 416)
    return model, data_shape

class Validator(object):
    def __init__(self, inputs):
        if isinstance(inputs, dict):
            self.inputs = inputs
        else:
            assert len(inputs) == 1
            self.inputs = {"data" : inputs[0]}
    def GetReference(self):
        return []
    def Validate(self):
        return None
    def GetInputDictionary(self):
        return self.inputs

class ImageNetValidator(Validator):
    def __init__(self, shape_dict, layout="NCHW", preproc=None):
        assert layout in ("NCHW", "NHWC"), "Requested layout is not currently supported: " + layout
        assert len(shape_dict) == 1
        from PIL import Image
        from tvm.contrib import download
        from os.path import join, isfile
        from matplotlib import pyplot as plt

        name = list(shape_dict.keys())[0]

        # Download ImageNet categories
        categ_url = "https://github.com/uwsampl/web-data/raw/main/vta/models/"
        categ_fn = "synset.txt"
        download.download(join(categ_url, categ_fn), categ_fn)
        self.synset = eval(open(categ_fn).read())

        # Download test image
        image_url = "https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg"
        image_fn = "cat.png"
        download.download(image_url, image_fn)

        # Prepare test image for inference
        #import ipdb; ipdb.set_trace()
        image = Image.open(image_fn)
        if layout == "NHWC":
            image = image.resize(shape_dict[name][1:-1])
        elif layout == "NCHW":
            image = image.resize(shape_dict[name][2:])

        #image = self.preprocess(np.array(image))
        if "mxnet" in preproc:
            image = np.array(image) - np.array([123.0, 117.0, 104.0])
            image /= np.array([58.395, 57.12, 57.375])
            image = image.transpose((2, 0, 1))
            image = image[np.newaxis, :]
        elif "keras" in preproc:
            image = np.array(image)[np.newaxis, :].astype("float32")
            from tensorflow.keras.applications.inception_v3 import preprocess_input
            image = preprocess_input(image)
        elif "keras_mobilenetv1" in preproc:
            image = np.array(image)[np.newaxis, :].astype("float32")
            from tensorflow.keras.applications.mobilenet import preprocess_input
            image = preprocess_input(image)

        self.inputs = {name : image}

    def Validate(self, m, ref_outputs=[]):
        tvm_output = m.get_output(0)
        #import ipdb; ipdb.set_trace()
        top_categories = np.argsort(tvm_output.asnumpy()[0])
        # Report top-5 classification results
        print("\nTop5 predictions: \n")
        top5 = np.flip(top_categories, axis=0)[:5]
        # print("\t#1:", self.synset[top_categories[-1]])
        # print("\t#2:", self.synset[top_categories[-2]])
        # print("\t#3:", self.synset[top_categories[-3]])
        # print("\t#4:", self.synset[top_categories[-4]])
        # print("\t#5:", self.synset[top_categories[-5]])
        print("\t#1:", self.synset[top5[1-1]])
        print("\t#2:", self.synset[top5[2-1]])
        print("\t#3:", self.synset[top5[3-1]])
        print("\t#4:", self.synset[top5[4-1]])
        print("\t#5:", self.synset[top5[5-1]])
        print("\t", top5)
        ImageNetClassifier = False
        for k in top_categories[-5:]:
            if "cat" in self.synset[k]:
                ImageNetClassifier = True
        assert ImageNetClassifier, "Failed ImageNet classifier validation check"


class VOCValidator(Validator):
    # this function is from yolo3.utils.letterbox_image
    def letterbox_image(self, image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        
        from PIL import Image
        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image

    def preprocess(self, img):
        model_image_size = (416, 416)
        boxed_image = self.letterbox_image(img, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.transpose(image_data, [2, 0, 1])
        image_data = np.expand_dims(image_data, 0)
        return image_data

    def __init__(self, shape_dict, layout="NCHW", preproc=None):
        assert layout in ("NCHW", "NHWC"), "Requested layout is not currently supported: " + layout
        assert len(shape_dict) == 1
        from PIL import Image
        from tvm.contrib import download
        from os.path import join, isfile
        from matplotlib import pyplot as plt

        name = list(shape_dict.keys())[0]

        # Download test image
        image_url = "https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg"
        image_fn = "dog.png"
        download.download(image_url, image_fn)

        # Prepare test image for inference
        #import ipdb; ipdb.set_trace()
        image = Image.open(image_fn)
        image_data = self.preprocess(image)

        self.inputs = {name : image_data}

    def Validate(self, m, ref_outputs=[]):
        # class_IDs, scores, bounding_boxs
        classid = m.get_output(0)
        scores = m.get_output(1)
        bounding_boxs = m.get_output(2)
        for a in classid:
            print(a)

class Executor(object):
    def __init__(self, use_tracker=False):
        self.benchmarks = []
        self.tuning_jobs = []
        self.tracker = None
        self.remote = None
        self.host_target = "llvm"
        self.use_tracker = use_tracker
        if use_tracker == "android":
            self.host_target = "llvm -mtriple=arm64-linux-android"
        elif use_tracker != False:

            class BackendNotImplementedForRPCBenchmarking(Exception):
                pass

            raise BackendNotImplementedForRPCBenchmarking

    def schedule(self, model, *args, **kwargs):
        importer = ModelImporter()
        self._schedule_jobs(*importer(model, *args, **kwargs))

    def run_pending_benchmarks(self):
        for bench in self.benchmarks:
            bench()

    def tune_pending_benchmarks(
        self, apply_previous_tune=False, opt=args.tuning_options
    ):
        for tune in self.tuning_jobs:
            tune(apply_previous_tune, options=args.tuning_options)

    def _connect_tracker(self):
        from tvm import rpc

        print(
            "Tracker attempting connection on {}:{}".format(
                args.rpc_tracker_host, args.rpc_tracker_port
            )
        )
        self.tracker = rpc.connect_tracker(args.rpc_tracker_host, args.rpc_tracker_port)
        self.remote = self.tracker.request(
            args.rpc_key, priority=0, session_timeout=6000
        )
        print("Tracker connected to remote RPC server")

    def _disconnect_tracker(self):
        self.remote = None
        self.tracker = None

    def advanced_time_evaluator(self, m, func_name, ctx, number=1, repeat=1, min_repeat_ms=0, time_to_work_ms=0, cooldown_interval_ms=0, f_preproc=""):
        import inspect
        import math
        def ms_to_s(ms): 
            return ms / 1000
        one_run_time = m.module.time_evaluator(func_name, ctx, number=1,repeat=1,min_repeat_ms=0)().results[0]
        repeats_to_cooldown = max(round(ms_to_s(time_to_work_ms)/one_run_time), 1)

        def _time_evaluator(func_name, m, ctx, number=1, repeat=1, min_repeat_ms=0, cooldown_interval_ms=0, repeats_to_cooldown=1, f_preproc=""):
            def evaluator():
                import time
                from tvm.runtime.module import BenchmarkResult
                results = []
                for _ in range(math.ceil(repeat / repeats_to_cooldown)):
                    time_f = m.module.time_evaluator(func_name, ctx, number=number, repeat=repeats_to_cooldown, min_repeat_ms=min_repeat_ms, f_preproc=f_preproc)
                    results.append(time_f().results)
                    time.sleep(ms_to_s(cooldown_interval_ms))
                return BenchmarkResult([np.mean(r) for r in results])
            return evaluator

        if inspect.signature(m.module.time_evaluator).parameters.get("cooldown_interval_ms"):
            time_f = m.module.time_evaluator(func_name, ctx, number=number, repeat=repeat, min_repeat_ms=min_repeat_ms, cooldown_interval_ms=cooldown_interval_ms, repeats_to_cooldown=repeats_to_cooldown, f_preproc=f_preproc)
        else:
            time_f = _time_evaluator(func_name, m, ctx, number=number, repeat=repeat, min_repeat_ms=min_repeat_ms, cooldown_interval_ms=cooldown_interval_ms, repeats_to_cooldown=repeats_to_cooldown, f_preproc=f_preproc)
            
        return time_f


    def _benchmark(
        self,
        tvm_mod,
        params,
        input_shape,
        target="llvm",
        target_host="llvm",
        dtype="float32",
        validator=None
    ):
        if args.debug:
            from tvm.contrib.debugger import debug_runtime as graph_executor
        else:
            from tvm.contrib import graph_executor

        if self.use_tracker and self.remote == None:
            self._connect_tracker()

        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(
                tvm_mod, target_host=target_host, target=target, params=params
            )
            # TODO(amalyshe): export library. conflicts with previous compilation since parameters
            # on the next line are not original parameters. Need to fix
            #lib2 = relay.build(tvm_mod, target=target, target_host=target_host, params=params)
            #lib2.export_library("_model.so", ndk.create_shared)
            # print("JSON:\n", graph)

        if self.remote:
            print("Using Android OpenCL runtime over RPC")
            temp = utils.tempdir()
            dso_binary = "dev_lib_cl.so"
            dso_binary_path = temp.relpath(dso_binary)
            if "opencl" in target:
                ctx = self.remote.cl(0)
            else:
                ctx = self.remote.cpu(0)
            lib.export_library(dso_binary_path, ndk.create_shared)
            remote_path = "/data/local/tmp/" + dso_binary
            self.remote.upload(dso_binary_path)
            print("Uploading binary...")
            rlib = self.remote.load_module(dso_binary)
            m = graph_executor.create(graph, rlib, ctx)
        else:
            print("Using local runtime")
            ctx = tvm.device(target, 0)
            m = graph_executor.create(graph, lib, ctx)

        m.set_input(**params)
        inputs = []
        if isinstance(validator, Validator):
            inputs = validator.GetInputDictionary()
            for key, data in inputs.items():
                m.set_input(key, data)
        elif isinstance(input_shape, dict):
            for key in input_shape:
                inputs.append(np.random.normal(size=input_shape[key]).astype(dtype))
                m.set_input(key, inputs[-1])
        else:
            inputs.append(np.random.normal(size=input_shape).astype(dtype))
            m.set_input("data", inputs[-1])

        print("Evaluating...", flush=True)
        number = 1
        repeat = 100
        min_repeat_ms = 0
        time_to_work_ms = 1000
        cooldown_interval_ms=1000
        if args.debug:
            m.run()
            time_f = self.advanced_time_evaluator(m, "run", ctx, number, repeat, min_repeat_ms, time_to_work_ms, cooldown_interval_ms)
        else:
            time_f = self.advanced_time_evaluator(m, "run", ctx, number, repeat, min_repeat_ms, time_to_work_ms, cooldown_interval_ms)

        benchmarkResult = time_f()
        cost = benchmarkResult.mean
        print("%g secs/iteration\n" % cost)
        print(benchmarkResult)

        if validator:
            if isinstance(validator, Validator):
                ref_outputs = validator.GetReference()
                validator.Validate(m, ref_outputs)
            else:
                ref_outputs = validator(inputs)
                for i, ref_output in enumerate(ref_outputs):
                    tvm_output = m.get_output(i)
                    output = tvm_output.asnumpy()
                    np.testing.assert_allclose(output, ref_output, rtol=1e-3, atol=1e-3)
            print("Validation done")


    def _schedule_jobs(self, mod, params, input_shape, dtype, target, validator=None):
        def bench():
            self._benchmark(
                mod,
                params,
                input_shape,
                target=target,
                target_host=self.host_target,
                dtype=dtype,
                validator=validator
            )

        benchmark_index = len(self.benchmarks)
        self.benchmarks.append(bench)

        def tune(apply_previous_tune=False, options=args.tuning_options):
            print("Extracting tasks")
            tasks = autotvm.task.extract_from_program(
                mod, target=target, target_host=self.host_target, params=params
            )
            if apply_previous_tune == False:
                print("Tuning kernels")
                Executor.tune_tasks(tasks, **options)

            def tuned_benchmark():
                print("Apply best performing tuning profiles:")

                with autotvm.apply_history_best(options["log_filename"]):
                    bench()

            self.benchmarks.pop(benchmark_index)
            self.benchmarks.append(tuned_benchmark)

        self.tuning_jobs.append(tune)

    @staticmethod
    def tune_tasks(
        tasks,
        measure_option,
        tuner="xgb",
        n_trial=333,
        early_stopping=None,
        log_filename="tuning.log",
        use_transfer_learning=False,
    ):
        from tvm.autotvm.tuner import XGBTuner
        from tvm.autotvm.tuner import GATuner

        tmp_log_file = log_filename + ".tmp"
        #if os.path.exists(tmp_log_file) and use_transfer_learning == False:
        #    os.remove(tmp_log_file)

        for i, tsk in enumerate(reversed(tasks)):
            print("Task: ", tsk)
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
            if tuner == "xgb" or tuner == "xgb-rank":
                tuner_obj = XGBTuner(tsk, loss_type="rank")
            elif tuner == "xgb_knob":
                tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
            elif tuner == "ga":
                tuner_obj = GATuner(tsk, pop_size=50)
            elif tuner == "random":
                tuner_obj = RandomTuner(tsk)
            elif tuner == "gridsearch":
                tuner_obj = GridSearchTuner(tsk)
            else:
                raise ValueError("Invalid tuner: " + tuner)

            if use_transfer_learning:
                if os.path.isfile(tmp_log_file):
                    tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

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

        autotvm.record.pick_best(tmp_log_file, log_filename)
        # os.remove(tmp_log_file)

if __name__ == "__main__":
    main()
