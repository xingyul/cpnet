



CUDA_HOME := /usr/local/cuda-9.0

TF_INC = `python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())"`
TF_LIB = `python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"`

LIBFLAGS = -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart -L $(TF_LIB) -ltensorflow_framework -lcublas

arch = sm_70

NVCC = /usr/local/cuda-9.0/bin/nvcc

LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64

all: knn_op.so 

forward: knn_op.so

knn_op.so: knn_op.cc knn_cuda.cu.o
	g++ -std=c++11 -shared -fPIC -o knn_op.so knn_op.cc knn_cuda.cu.o -I$(TF_INC) -L$(LIBFLAGS) -O2 -D_GLIBCXX_USE_CXX11_ABI=0

knn_cuda.cu.o: knn_cuda.cu.cc
	$(NVCC) -std=c++11 -c -o knn_cuda.cu.o knn_cuda.cu.cc -I$(TF_INC) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch $(arch) --use_fast_math
		
clean:
	rm *.o *.so
