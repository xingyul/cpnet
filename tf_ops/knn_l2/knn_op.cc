/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <stdio.h>
#include <assert.h> 

using namespace tensorflow;

REGISTER_OP("KnnL2")
    .Input("input1: float32")
    .Input("input2: float32")
    .Input("input3: int32")
    .Input("input4: int32")
    .Output("output: int32")
    .Doc(R"doc(
)doc");

void KnnComputeLauncher(const float *Input, const float *input_norm, int *Output, float *tmp_data_buffer, const long long *tmp_ptr_buffer, int B, int N, int C, int K, int U);

class KnnCudaOp : public OpKernel {
public:
  explicit KnnCudaOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& I_tensor = context->input(0);
    const Tensor& I_norm_tensor = context->input(1);
    const Tensor& K_tensor = context->input(2);
    const Tensor& U_tensor = context->input(3);
    auto Input = I_tensor.flat<float>();
    auto Input_norm = I_norm_tensor.flat<float>();
    // OP_REQUIRES(context, I_tensor.dims()==3 && K_tensor.dims()==1);

    int B = I_tensor.dim_size(0);
    int N = I_tensor.dim_size(1);
    int C = I_tensor.dim_size(2);
    int K = K_tensor.dim_size(0);
    int U = U_tensor.dim_size(0);

    // Create an output tensor
    Tensor* O_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B, N, K}, &O_tensor));
    auto Output = O_tensor->template flat<int>();

	// Allocate temporary memory
    Tensor tmp_data_buffer_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape{B * N * N + B * N * K}, &tmp_data_buffer_tensor));
    auto tmp_data_buffer = tmp_data_buffer_tensor.template flat<float>();

    Tensor tmp_ptr_buffer_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape{B * 3}, &tmp_ptr_buffer_tensor));
    auto tmp_ptr_buffer = tmp_ptr_buffer_tensor.template flat<long long>();

    // Set all but the first element of the output tensor to 0.
	KnnComputeLauncher(Input.data(), Input_norm.data(), Output.data(), tmp_data_buffer.data(), tmp_ptr_buffer.data(), B, N, C, K, U); 
  }
};

REGISTER_KERNEL_BUILDER(Name("KnnL2").Device(DEVICE_GPU), KnnCudaOp);

class KnnOp : public OpKernel {
public:
  explicit KnnOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& I_tensor = context->input(0);
    const Tensor& I_norm_tensor = context->input(1);
    const Tensor& K_tensor = context->input(2);

	printf("This is CPU code. We don't need this\n");
	exit(-1);
  }
};

REGISTER_KERNEL_BUILDER(Name("KnnL2").Device(DEVICE_CPU), KnnOp);
