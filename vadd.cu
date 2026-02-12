#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void vadd(const float* a, const float* b, float* c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) c[i] = a[i] + b[i];
}

static void ck(cudaError_t e, const char* msg) {
	if (e != cudaSuccess) {
		printf("CUDA error (%s): %s\n", msg, cudaGetErrorString(e));
		std::exit(1);
	}
}

int main() {
	int n = 100000000;               // 100M
	size_t bytes = (size_t)n * sizeof(float);

	float *ha = (float*)malloc(bytes);
	float *hb = (float*)malloc(bytes);
	float *hc = (float*)malloc(bytes);
	for (int i = 0; i < n; i++) {
		ha[i] = 1.0f;
		hb[i] = 2.0f;
		hc[i] = 0.0f;
	}

	float *da, *db, *dc;
	ck(cudaMalloc(&da, bytes), "malloc da");
	ck(cudaMalloc(&db, bytes), "malloc db");
	ck(cudaMalloc(&dc, bytes), "malloc dc");

	ck(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice), "H2D a");
	ck(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice), "H2D b");
	ck(cudaMemset(dc, 0, bytes), "memset dc");

	//int block = 1024;                 // try 128/256/512/1024
	int block = 512;
	int grid  = (n + block - 1) / block;

	// warmup
	vadd<<<grid, block>>>(da, db, dc, n);
	ck(cudaGetLastError(), "launch warmup");
	ck(cudaDeviceSynchronize(), "sync warmup");

	cudaEvent_t start, stop;
	ck(cudaEventCreate(&start), "event create start");
	ck(cudaEventCreate(&stop), "event create stop");

	ck(cudaEventRecord(start), "record start");
	vadd<<<grid, block>>>(da, db, dc, n);
	ck(cudaGetLastError(), "launch timed");
	ck(cudaEventRecord(stop), "record stop");
	ck(cudaEventSynchronize(stop), "sync stop");

	float ms = 0.0f;
	ck(cudaEventElapsedTime(&ms, start, stop), "elapsed");

	ck(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost), "D2H c");

	// verify a few values
	printf("c[0]=%f c[n/2]=%f c[n-1]=%f\n", hc[0], hc[n/2], hc[n-1]);

	double bytes_moved = 3.0 * (double)bytes; // 2 reads + 1 write
	double gbps = bytes_moved / (ms * 1e-3) / 1e9;
	printf("Time: %.3f ms  Effective bandwidth: %.1f GB/s\n", ms, gbps);

	return 0;
}
