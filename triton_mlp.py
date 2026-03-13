import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]


@triton.jit
def linear_relu_kernel(
    X_ptr,
    W_ptr,
    B_ptr,
    Y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_b: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K
    k = 0
    while k < K:
        k_ids = k + offs_k
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + k_ids[None, :] * stride_xk)
        w_ptrs = W_ptr + (k_ids[:, None] * stride_wk + offs_n[None, :] * stride_wn)

        x = tl.load(
            x_ptrs, mask=(offs_m[:, None] < M) & (k_ids[None, :] < K), other=0.0
        )
        w = tl.load(
            w_ptrs, mask=(k_ids[:, None] < K) & (offs_n[None, :] < N), other=0.0
        )

        acc += tl.dot(x, w)
        k += BLOCK_K

    # add bias
    b = tl.load(B_ptr + offs_n * stride_b, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc += b[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # store
    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(
        y_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


def linear_relu_triton(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    block_m=128,
    block_n=128,
    block_k=32,
):
    assert x.is_cuda and w.is_cuda and b.is_cuda
    assert (
        x.dtype == torch.float16
        and w.dtype == torch.float16
        and b.dtype == torch.float16
    )
    M, K = x.shape
    K2, N = w.shape
    assert K == K2
    y = torch.empty((M, N), device=x.device, dtype=torch.float16)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    linear_relu_kernel[grid](
        x,
        w,
        b,
        y,
        M=M,
        N=N,
        K=K,
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_wk=w.stride(0),
        stride_wn=w.stride(1),
        stride_b=b.stride(0),
        stride_ym=y.stride(0),
        stride_yn=y.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return y


class TinyMLP(torch.nn.Module):
    def __init__(self, d_in=1024, d_hidden=2048, d_out=1024, device="cuda"):
        super().__init__()
        self.w1 = torch.nn.Parameter(
            torch.randn(d_in, d_hidden, device=device, dtype=torch.float16) * 0.02
        )
        self.b1 = torch.nn.Parameter(
            torch.zeros(d_hidden, device=device, dtype=torch.float16)
        )
        self.w2 = torch.nn.Parameter(
            torch.randn(d_hidden, d_out, device=device, dtype=torch.float16) * 0.02
        )
        self.b2 = torch.nn.Parameter(
            torch.zeros(d_out, device=device, dtype=torch.float16)
        )

    def forward(self, x):
        h = linear_relu_triton(x, self.w1, self.b1)
        y = linear_relu_triton(h, self.w2, self.b2)
        return y


def main():
	print("triton_mlp: starting", flush=True)
	torch.manual_seed(0)
	M = 4096
	d_in, d_hidden, d_out = 1024, 2048, 1024

	x = torch.randn((M, d_in), device="cuda", dtype=torch.float16)
	model = TinyMLP(d_in, d_hidden, d_out).cuda()

	# Triton forward
	y = model(x)

	# Reference (PyTorch) for correctness
	y_ref = torch.relu(x @ model.w1 + model.b1) @ model.w2 + model.b2
	y_ref = torch.relu(y_ref)

	max_err = (y - y_ref).abs().max().item()
	print("max abs error:", max_err, flush=True)

	# Timing
	torch.cuda.synchronize()
	iters = 100
	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)
	start.record()
	for _ in range(iters):
		y = model(x)
	end.record()
	torch.cuda.synchronize()
	ms = start.elapsed_time(end) / iters
	print(f"avg forward: {ms:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
