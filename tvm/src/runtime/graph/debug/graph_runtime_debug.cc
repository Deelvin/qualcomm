/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file graph_runtime_debug.cc
 */
#include <tvm/runtime/container.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>

#include <chrono>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <chrono>
#include <thread>

#include "../graph_runtime.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Graph runtime with debug .
 *
 *  This is the extension of GraphRuntime class used for debugging
 *  TVM runtime PackedFunc API.
 */
class GraphRuntimeDebug : public GraphRuntime {
 public:
  /*!
   * \brief Run each operation in the graph and get the time per op for all ops.
   * \param number The number of times to run this function for taking average.
   * \param repeat The number of times to repeat the measurement.
   *        In total, the function will be invoked (1 + number x repeat) times,
   *        where the first one is warmed up and will be discarded in case
   *        there is lazy initialization.
   * \param min_repeat_ms The minimum duration of one `repeat` in milliseconds.
   *        By default, one `repeat` contains `number` runs. If this parameter is set,
   *        the parameters `number` will be dynamically adjusted to meet the
   *        minimum duration requirement of one `repeat`.
   * \return Comma seperated string containing the elapsed time per op for the last
   *         iteration only, because returning a long string over rpc can be expensive.
   */
  std::string RunIndividual(int number, int repeat, int run_delay, int min_repeat_ms) {
    // warmup run
    GraphRuntime::Run();
    std::string tkey = module_->type_key();
    std::cout << "RunIndividual " << tkey << std::endl;
    std::vector<std::vector<double> > time_sec_per_op(op_execs_.size(), std::vector<double>(number, 0));
    // time_sec_per_op.reserve(op_execs_.size());
    // std::vector<double> time_sec_per_op(op_execs_.size(), 0);
    if (tkey == "rpc") { // library
      // RPC modules rely on remote timing which implements the logic from the else branch.
      for (size_t index = 0; index < op_execs_.size(); ++index) {
        std::vector<double> v = RunOpRPC(index, number, repeat, run_delay, min_repeat_ms);
        // double average = std::accumulate(v.begin(), v.end(), 0) / v.size();
        for(size_t i =0;i < number;++i) {
          time_sec_per_op[index][i] += v[i];//average;
        }
        // time_sec_per_op[index] = v;//average;
        // time_sec_per_op[index] += average;
        // ICE AVR
      }
    } else {
      for (int i = 0; i < repeat; ++i) {
        std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> tbegin, tend;
        double duration_ms = 0.0;
        // do {
          std::fill(time_sec_per_op.begin(), time_sec_per_op.end(), std::vector<double>(number, 0));
          // if (duration_ms > 0.0) {
          //   number = static_cast<int>(std::max((min_repeat_ms / (duration_ms / number) + 1), number * 1.618));  // 1.618 is chosen by random
          // }
          tbegin = std::chrono::high_resolution_clock::now();
          std::vector<std::vector<Timer>> op_timers;
          for (size_t index = 0; index < op_execs_.size(); index++) {
            op_timers.push_back({});
          }
          for (int k = 0; k < number; k++) {
            for (size_t index = 0; index < op_execs_.size(); ++index) {
              if (op_execs_[index]) {
                op_timers[index].push_back(RunOpHost(index));
                // std::this_thread::sleep_for(std::chrono::milliseconds(run_delay));
              }
            }
          }
          for (size_t index = 0; index < op_execs_.size(); ++index) {
            for (size_t j =0; j < op_timers[index].size(); ++j) {
            // for (auto t : op_timers[index]) {
              time_sec_per_op[index][j] = op_timers[index][j]->SyncAndGetElapsedNanos() / 1e9;
              // time_sec_per_op[index] += t->SyncAndGetElapsedNanos() / 1e9;
              // for(size_t i =0;i < number;++i) {
              //   time_sec_per_op[index][i] += v[i];//average;
              // }
            }
          }
          tend = std::chrono::high_resolution_clock::now();
          duration_ms = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count() * 1000;
        // } while (duration_ms < min_repeat_ms);

        LOG(INFO) << "Iteration: " << i;
        int op = 0;
        for (size_t index = 0; index < time_sec_per_op.size(); index++) {
          if (op_execs_[index]) {
            // time_sec_per_op[index] /= number;
            double average = std::accumulate(time_sec_per_op[index].begin(), time_sec_per_op[index].end(), 0) / time_sec_per_op[index].size();
            std::string s = "[";
            for(auto& i: time_sec_per_op[index]) {
                s+= std::to_string(i) + ", ";
            }
            s+="]";
            LOG(INFO) << "Op #" << op++ << " " << GetNodeName(index) << ": " << average * 1e6 << " us/iter" << " all" << s;
            // LOG(INFO) << "Op #" << op++ << " " << GetNodeName(index) << ": " << time_sec_per_op[index] * 1e6 << " us/iter";
          }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(run_delay));
      }
    }

    std::ostringstream os;
    // os<< "iceice";
    // os<< data_entry_[entry_id(0, 0)]->ctx;

    // for (size_t index = 0; index < time_sec_per_op.size(); index++) {
    //   os << time_sec_per_op[index] << ",";
    // }

    for (size_t index = 0; index < time_sec_per_op.size(); index++) {
      std::string s = "";

      // for(auto& ss: time_sec_per_op[index]) {
      // for(size_t i =0; i < number;++i) {
      //   s += std::to_string(time_sec_per_op[index][i]) + ";";
      // }

      double s_sum =  0;
      for(size_t i =0; i < number;++i) {
        s_sum += time_sec_per_op[index][i];
      }
      s_sum /= time_sec_per_op[index].size(); //average
      s = std::to_string(s_sum);

      os << s << ",";
      // os << time_sec_per_op[index] << ",";
    }
    std::cout << os.str() << std::endl;
    return os.str();
  }

  std::vector<double> RunOpRPC(int index, int number, int repeat, int run_delay, int min_repeat_ms) {
    // Right now we expect either "tvm_op" for nodes which run PackedFunc or "null" for nodes which
    // represent inputs/parameters to the graph. Other types may be supported in the future, but
    // consideration would be needed as to how to do that over RPC before we support it here.
    if (nodes_[index].op_type != "tvm_op") {
      CHECK_EQ(nodes_[index].op_type, "null")
          << "Don't know how to run op type " << nodes_[index].op_type
          << " remotely over RPC right now";

      // NOTE: GraphRuntimeDebug expects graph nodes to have an "op" attribute of "tvm_op" or "null"
      // and "null" is a placeholder node for a parameter or input.
      // return 0;
      return {};
    }

    const TVMContext& ctx = data_entry_[entry_id(index, 0)]->ctx;
    TVMOpParam param = nodes_[index].param;
    std::string name = param.func_name;
    uint32_t num_inputs = param.num_inputs;
    uint32_t num_outputs = param.num_outputs;
    std::cout  << "RunOpRPC" << std::endl; 
    PackedFunc time_eval = runtime::Registry::Get("runtime.RPCTimeEvaluator")
                               ->
                               operator()(module_, name, static_cast<int>(ctx.device_type),
                                          ctx.device_id, number, repeat, run_delay, min_repeat_ms, "");

    int num_flat_args = num_inputs + num_outputs;
    std::unique_ptr<TVMValue> values(new TVMValue[num_flat_args]);
    std::unique_ptr<int> type_codes(new int[num_flat_args]);
    TVMArgsSetter setter(values.get(), type_codes.get());
    int offs = 0;
    const auto& inode = nodes_[index];
    for (const auto& e : inode.inputs) {
      uint32_t eid = this->entry_id(e);
      DLTensor* arg = const_cast<DLTensor*>(data_entry_[eid].operator->());
      setter(offs, arg);
      offs++;
    }
    for (uint32_t i = 0; i < num_outputs; ++i) {
      uint32_t eid = this->entry_id(index, i);
      DLTensor* arg = const_cast<DLTensor*>(data_entry_[eid].operator->());
      setter(offs, arg);
      offs++;
    }
    TVMRetValue rv;//ICE
    time_eval.CallPacked(TVMArgs(values.get(), type_codes.get(), num_flat_args), &rv);
    std::string results = rv.operator std::string();
    const double* results_arr = reinterpret_cast<const double*>(results.data());
    LOG(INFO) << "Got op timing: " << results_arr[0];
    // LOG(INFO) << "Got op timing: " << results_arr[0];
    auto r = std::vector<double>(reinterpret_cast<const double*>(results.data()), reinterpret_cast<const double*>(results.data()) + number);
    std::string rr = "";
    for (auto& s: r) {
      rr += std::to_string(s) + ", ";
    }
    LOG(WARNING) << "Got op timing: " << rr;
    // LOG(DEBUG) << "Got op timing: " << rr;
    
    return r;
    // return results_arr[0];
  }

  Timer RunOpHost(int index) {
    const TVMContext& ctx = data_entry_[entry_id(index, 0)]->ctx;
    Timer t = Timer::Start(ctx);
    op_execs_[index]();
    t->Stop();
    
    return t;
  }

  /*!
   * \brief Run each operation and get the output.
   * \param index The index of op which needs to be returned.
   * \param eid The Entry id of the op.
   */
  NDArray GetOutputByLayer(int index, int eid) { return data_entry_[entry_id(index, eid)]; }

  /*!
   * \brief GetFunction Get the function based on input.
   * \param name The function which needs to be invoked.
   * \param sptr_to_self Packed function pointer.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);

  /*!
   * \brief Get the node index given the name of node.
   * \param name The name of the node.
   * \return The index of node.
   */
  int GetNodeIndex(const std::string& name) const {
    for (size_t nid = 0; nid < GetNumOfNodes(); ++nid) {
      if (GetNodeName(nid) == name) {
        return static_cast<int>(nid);
      }
    }
    LOG(FATAL) << "cannot find " << name << " among nodex";
    return -1;
  }

  /*!
   * \brief Copy index-th node to data_out.
   *
   * This method will do a partial run of the the graph
   * from begining upto the index-th node and return output of index-th node.
   * This is costly operation and suggest to use only for debug porpose.
   *
   * \param index: The  index of the node.
   * \param data_out the node data.
   */
  void DebugGetNodeOutput(int index, DLTensor* data_out) {
    ICHECK_LT(static_cast<size_t>(index), op_execs_.size());
    uint32_t eid = index;

    for (size_t i = 0; i < op_execs_.size(); ++i) {
      if (op_execs_[i]) op_execs_[i]();
      if (static_cast<int>(i) == index) break;
    }

    data_entry_[eid].CopyTo(data_out);
  }
};

/*!
 * \brief GetFunction Get the function based on input.
 * \param name The function which needs to be invoked.
 * \param sptr_to_self Packed function pointer.
 */
PackedFunc GraphRuntimeDebug::GetFunction(const std::string& name,
                                          const ObjectPtr<Object>& sptr_to_self) {
  // return member functions during query.
  if (name == "get_output_by_layer") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = this->GetOutputByLayer(args[0], args[1]);
    });
  } else if (name == "debug_get_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (String::CanConvertFrom(args[0])) {
        this->DebugGetNodeOutput(this->GetNodeIndex(args[0]), args[1]);
      } else {
        this->DebugGetNodeOutput(args[0], args[1]);
      }
    });
  } else if (name == "run_individual") {
    std::cout << "ICE run_individual" << std::endl;
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      int number = args[0];
      int repeat = args[1];
      int run_delay = args[2];
      int min_repeat_ms = args[3];
      ICHECK_GT(number, 0);
      ICHECK_GT(repeat, 0);
      ICHECK_GE(min_repeat_ms, 0);
      *rv = this->RunIndividual(number, repeat, run_delay, min_repeat_ms);
    });
  } else {
    return GraphRuntime::GetFunction(name, sptr_to_self);
  }
}

/*!
 * \brief GraphRuntimeDebugCreate Get the function based on input.
 * \param sym_json The graph symbol in json format.
 * \param m Compiled module which will be loaded.
 * \param ctxs All devices contexts.
 */
Module GraphRuntimeDebugCreate(const std::string& sym_json, const tvm::runtime::Module& m,
                               const std::vector<TVMContext>& ctxs,
                               PackedFunc lookup_linked_param_func) {
  auto exec = make_object<GraphRuntimeDebug>();
  exec->Init(sym_json, m, ctxs, lookup_linked_param_func);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.graph_runtime_debug.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  ICHECK_GE(args.num_args, 4) << "The expected number of arguments for graph_runtime.create is "
                                 "at least 4, but it has "
                              << args.num_args;
  PackedFunc lookup_linked_param_func;
  int ctx_start_arg = 2;
  if (args[2].type_code() == kTVMPackedFuncHandle) {
    lookup_linked_param_func = args[2];
    ctx_start_arg++;
  }

  *rv = GraphRuntimeDebugCreate(args[0], args[1], GetAllContext(args, ctx_start_arg),
                                lookup_linked_param_func);
});
}  // namespace runtime
}  // namespace tvm
