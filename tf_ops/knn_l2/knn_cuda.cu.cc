#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <cublas_v2.h>
#include <stdio.h>


#define NEG_INF -1e8


// from https://github.com/yuxianzhi/Top-K/blob/master/top_k_gpu.cu
__device__ inline void replace_smaller(float* val_array, int* idx_array, int k, float val_data, int idx_data) {
    if(val_data < val_array[k-1]) {
        return;
	}
    for(int j=k-2; j>=0; j--) {
        if(val_data > val_array[j]) {
            val_array[j+1] = val_array[j];
            idx_array[j+1] = idx_array[j];
		}
        else {
            val_array[j+1] = val_data;
            idx_array[j+1] = idx_data;
            return;
        }
    }
    val_array[0] = val_data;
    idx_array[0] = idx_data;
}

// from https://github.com/yuxianzhi/Top-K/blob/master/top_k_gpu.cu
__global__ void top_k_gpu_kernel(float* input_all, int B, int N, int K, float* val_array_all, int* idx_array_all) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < (N*B)) {
        int B_idx = idx / N;
        int N_idx = idx % N;
        float* val_array = val_array_all + B_idx * N * K + N_idx * K;
        int* idx_array = idx_array_all + B_idx * N * K + N_idx * K;
        // produce K data in decent order

        float* input = input_all + B_idx * N * N + N_idx * N;
        val_array[0] = input[0];
        for(int i=0; i<K; i++) {
            idx_array[i] = i;
        }
        for(int i=1; i<K; i++)
        {
            val_array[i] = NEG_INF;
            replace_smaller(val_array, idx_array, i+1, input[i], i);
        }
        
        // replace the data if bigger
        for(int i=K; i<N; i++)
        {
            replace_smaller(val_array, idx_array, K, input[i], i);
        }
    }
}

__global__ void assign_ptr(const float *Input, float* tmp_data_buffer, const float **Input_ptrs_gpu_1, const float **Input_ptrs_gpu_2, float **tmp_product_ptrs_gpu, int B, int N, int C, int K) {

    for(int tx = 0; tx < B; tx ++) {
        Input_ptrs_gpu_1[tx] = Input + tx * N * C;
        Input_ptrs_gpu_2[tx] = Input + tx * N * C;
        tmp_product_ptrs_gpu[tx] = tmp_data_buffer + tx * N * N;
    }
}


__global__ void add_l2_norm(float* tmp_data_buffer, const float* Input_norm, int B, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < (N*N*B)) {
        int B_idx = idx / (N*N);
        int row_idx = idx % N;
        int col_idx = (idx / N) % N;
        float square_1 = Input_norm[B_idx * N + row_idx];
        float square_2 = Input_norm[B_idx * N + col_idx];
        tmp_data_buffer[idx] = 2 * tmp_data_buffer[idx] - square_1 - square_2;
    }
}

__global__ void block_same_u(float* tmp_data_buffer, int B, int N, int U) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < (N*N*B)) {
        int row_idx = idx % N;
        int col_idx = (idx / N) % N;
        if ((row_idx / U) == (col_idx / U)) {
            tmp_data_buffer[idx] = NEG_INF;
        }
    }
}

// Input = (B, N, C)
// Output = (B, N, K)
void KnnComputeLauncher(const float *Input, const float *Input_norm, int *Output, float *tmp_data_buffer, const long long *tmp_ptr_buffer, int B, int N, int C, int K, int U) {

	const float** Input_ptrs_gpu_1 = (const float **)(tmp_ptr_buffer);
	const float** Input_ptrs_gpu_2 = (const float **)(tmp_ptr_buffer + B);
	float** tmp_product_ptrs_gpu = (float **)(tmp_ptr_buffer + 2 * B);
	float* tmp_top_k_val = (float *)(tmp_data_buffer + N * N * B);

    dim3 bDim1(1, 1, 1);
    dim3 gDim1(1, 1, 1);
	assign_ptr <<<gDim1, bDim1>>> (Input, tmp_data_buffer, Input_ptrs_gpu_1, Input_ptrs_gpu_2, tmp_product_ptrs_gpu, B, N, C, K);
	
	float one = 1.;
	float zero = 0.;

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        N, N, C, 
        &one,
        Input_ptrs_gpu_1, C,
        Input_ptrs_gpu_2, C,
        &zero, tmp_product_ptrs_gpu, N, B);

	cublasDestroy(handle);

    dim3 bDim2(1024, 1, 1);
    dim3 gDim2((B*N*N) / 1024 + 1, 1, 1);
	add_l2_norm <<<gDim2, bDim2>>> (tmp_data_buffer, Input_norm, B, N);
        
    dim3 bDim3(1024, 1, 1);
    dim3 gDim3((B*N*N) / 1024 + 1, 1, 1);
	block_same_u <<<gDim3, bDim3>>> (tmp_data_buffer, B, N, U);
        
    dim3 bDim4(1024, 1, 1);
    dim3 gDim4((B*N) / 1024 + 1, 1, 1);
	top_k_gpu_kernel <<<gDim4, bDim4>>> (tmp_data_buffer, B, N, K, tmp_top_k_val, Output);

// #define DEBUG
#ifdef DEBUG
    float* debug = new float [N*N*B] ();
    cudaMemcpy ( (void*) debug, (void*)tmp_data_buffer, B * N * N * sizeof(float), cudaMemcpyDeviceToHost );
    for(int i = 0; i < N; i ++) {
        for(int j = 0; j < N; j ++) {
            printf("%f ", debug[0 * N * N + i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
    delete(debug);
#endif

}

#endif

