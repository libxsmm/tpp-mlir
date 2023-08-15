//===- CudaTensor.h - -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Tensor.h"
#include "llvm/ADT/STLExtras.h"

#include <cassert>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

namespace {
using ListArg = std::vector<unsigned>;
using InitArg = std::initializer_list<unsigned>;
} // namespace

// CudaTensor: represents a tensor in GPU memory using CUDA.
// A wrapper around existing tensor on host.
template <typename T> struct CudaTensor {
  CudaTensor() = delete;
  CudaTensor(Tensor<T> &&tensor) : tensor(std::move(tensor)) {}

  // Move constructor
  CudaTensor(CudaTensor<T> &&other)
      : gpuData(std::exchange(other.gpuData, nullptr)),
        tensor(std::move(other.tensor)) {}

  // Copy constructor
  CudaTensor(const CudaTensor<T> &other) = delete;

  virtual ~CudaTensor() {
    if (gpuData)
      cudaFree(gpuData);
  }

  bool initGpu() {
    if (gpuData)
      cudaFree(gpuData);

    auto dataSize = tensor.getDataSize();
    cudaError_t allocErr = cudaMalloc(&gpuData, dataSize);
    if (allocErr != cudaSuccess) {
      std::cerr << "GPU allocation error\n";
      std::cerr << "cudaMalloc error : cuda code=" << allocErr << " - "
                << cudaGetErrorString(allocErr) << "\n";
      return false;
    }

    auto data = tensor.getData();
    cudaError_t cpyErr =
        cudaMemcpy(gpuData, data, dataSize, cudaMemcpyHostToDevice);
    if (cpyErr != cudaSuccess) {
      std::cerr << "GPU memcpy error\n";
      std::cerr << "cudaMemcpy error : cuda code=" << cpyErr << " - "
                << cudaGetErrorString(cpyErr) << "\n";
      return false;
    }

    return true;
  }

  T *gpuData = nullptr;
  Tensor<T> tensor;
};
