#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

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
	int n = 400000000;               // 400M for longer trace
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

	const int block_sizes[] = {128, 256, 512, 1024};
	const int num_blocks = sizeof(block_sizes) / sizeof(block_sizes[0]);
	const int profiling = getenv("NCU_PROFILE") != nullptr;
	const int iters = profiling ? 1 : 100;

	cudaEvent_t start, stop;
	ck(cudaEventCreate(&start), "event create start");
	ck(cudaEventCreate(&stop), "event create stop");

	// warmup (skip when NCU profiling - we want exactly 4 launches, one per block size)
	if (!profiling) {
		vadd<<<(n + 1023) / 1024, 1024>>>(da, db, dc, n);
		ck(cudaGetLastError(), "launch warmup");
		ck(cudaDeviceSynchronize(), "sync warmup");
	}

	for (int b = 0; b < num_blocks; b++) {
		int block = block_sizes[b];
		int grid = (n + block - 1) / block;

		char nvtx_name[64];
		snprintf(nvtx_name, sizeof(nvtx_name), "BLOCK=%d", block);
		nvtxRangePushA(nvtx_name);

		ck(cudaEventRecord(start), "record start");
		for (int i = 0; i < iters; i++) {
			vadd<<<grid, block>>>(da, db, dc, n);
		}
		ck(cudaGetLastError(), "launch");
		ck(cudaEventRecord(stop), "record stop");
		ck(cudaEventSynchronize(stop), "sync stop");

		float ms = 0.0f;
		ck(cudaEventElapsedTime(&ms, start, stop), "elapsed");
		ms /= iters;

		double bytes_moved = 3.0 * (double)bytes;
		double gbps = bytes_moved / (ms * 1e-3) / 1e9;
		printf("BLOCK=%4d  %d iters  avg=%.3f ms  BW=%.1f GB/s\n", block, iters, ms, gbps);

		nvtxRangePop();
	}

	ck(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost), "D2H c");
	printf("c[0]=%f c[n/2]=%f c[n-1]=%f\n", hc[0], hc[n/2], hc[n-1]);

	return 0;
}
