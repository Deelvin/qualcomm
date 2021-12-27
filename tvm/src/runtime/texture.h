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
 * \file texture.h
 * \brief Texture utilities
 */
#ifndef TVM_RUNTIME_TEXTURE_POOL_H_
#define TVM_RUNTIME_TEXTURE_POOL_H_

#include <tvm/runtime/device_api.h>
#include <tvm/ir/expr.h>

#include <memory>
#include <vector>

namespace tvm {
namespace runtime {
namespace cl {
    static const size_t OPENCL_MIN_IMAGE2D_WIDTH = 16384;
    static const size_t OPENCL_MIN_IMAGE2D_HEIGHT = 16384;
}

/*! \brief Structure to represent flattened texture shape */
template <typename T>
struct Texture2DShape {
  T width;
  T height;
  T channel;
};

/*! \brief Structure to limits on 2d texture width and height */
struct Texture2DLimits {
    int maxWidth;
    int maxHeight;
};

template<typename T, std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
inline std::vector<size_t> ShapeToIntVec(const T& shape, size_t rank) {
    std::vector<size_t> intShape(rank);
    for (size_t i = 0; i < rank; ++i) {
        intShape[i] = shape[i];
    }
    return intShape;
}
inline std::vector<size_t> ShapeToIntVec(const std::vector<ObjectRef>& shape, size_t rank) {
    std::vector<size_t> intShape(rank);
    for (size_t i = 0; i < rank; ++i) {
        if (const auto* op = shape[i].as<IntImmNode>()) {
            intShape[i] = op->value;
        } else {
            LOG(FATAL) << "Cannot get value from PrimExpr";
        }
    }
    return intShape;
}

/*!
 * \param shape_rank Rank N of the Nd-shape
 * \param convention Storage scope convention to use for flattening
 * \return The axis separator that defines the Nd shape partitioning in 2d
 */
template<typename S>
inline size_t DefaultTextureLayoutSeparator(size_t shape_rank, const S& shape, Texture2DLimits limits = {cl::OPENCL_MIN_IMAGE2D_WIDTH, cl::OPENCL_MIN_IMAGE2D_HEIGHT}) {
  ICHECK(shape_rank > 5) << "Rank cannot be bigger than 5 for 2d flattening";
  // Texture activation:
  // e.g. [N,C,H,W,c] -> Texture2d[N*C*H, W, c]
  // Texture weight:
  // e.g. [O,I,H,W,c] -> Texture2d[O, I*H*W, c]
  // Texture nhwc:
  // e.g. [N,H,W,C] -> Texture2d[N*H, W*C, c]
  auto shapeVec = ShapeToIntVec(shape, shape_rank);
  size_t separator = 0;
  if ((shapeVec[0] * shapeVec[1] * shapeVec[2] < limits.maxHeight) && (shapeVec[3] < limits.maxWidth)) {
    separator = shape_rank - 2;
  } else if ((shapeVec[0] * shapeVec[1] < limits.maxHeight) && (shapeVec[2] * shapeVec[3] < limits.maxWidth)) {
    separator = 2;
  } else if ((shapeVec[0] < limits.maxHeight) && (shapeVec[1] * shapeVec[2] * shapeVec[3] < limits.maxWidth)) {
    separator = 1;
  } else {
    std::string msg = "Cannot pack tensor with shapeVec: {";
    for (size_t i = 0; i < shape_rank; ++i) {
        if (i > 0)
            msg += ", ";
        msg += std::to_string(shapeVec[i]);
    }
    msg += "} to the 2d texture";
    LOG(FATAL) << msg;
  }
  return separator;
}

/*!
 * \param shape Nd shape
 * \param rank Number of dimensions N of the Nd shape
 * \param limits The hardware limits on the height and width of the texture
 * \return Width and height of the 2d shape
 */
template<typename T, typename S>
Texture2DShape<T> ApplyTexture2DFlattening(const S& shape, size_t rank, Texture2DLimits limits = {cl::OPENCL_MIN_IMAGE2D_WIDTH, cl::OPENCL_MIN_IMAGE2D_HEIGHT}) {
  auto axis = DefaultTextureLayoutSeparator(rank, shape, limits);
  Texture2DShape<T> texture{1, 1, shape[rank - 1]};
  for (size_t i = 0; i < rank - 1; i++) {
    if (i < axis) {
      texture.height *= shape[i];
    } else {
      texture.width *= shape[i];
    }
  }
  return texture;
}

inline bool IsTextureStorage(std::string scope) {
  return scope.find("texture") != std::string::npos;
}

class TVM_DLL TexturePool {
 public:
  /*!
   * \brief Create pool with specific device type and device.
   * \param device_type The device type.
   * \param device_api The device API.
   */
  TexturePool(DLDeviceType device_type, DeviceAPI* device_api);
  /*! \brief destructor */
  ~TexturePool();
  /*!
   * \brief Allocate temporal texture.
   * \param ctx The context of allocation.
   * \param width The width of the 2d texture to be allocated.
   * \param height The height of the 2d texture to be allocated.
   */
  void* AllocTexture(TVMContext ctx, size_t width, size_t height, DLDataType type_hint);
  /*!
   * \brief Free temporal texture in backend execution.
   *
   * \param ctx The context of allocation.
   * \param ptr The pointer to be freed.
   */
  void FreeTexture(TVMContext ctx, void* ptr);

 private:
  class Pool;
  /*! \brief pool of device local array */
  std::vector<Pool*> array_;
  /*! \brief device type this pool support */
  DLDeviceType device_type_;
  /*! \brief The device API */
  DeviceAPI* device_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_TEXTURE_POOL_H_
