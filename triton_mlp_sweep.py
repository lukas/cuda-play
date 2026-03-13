import argparse
import math
import time
from dataclasses import dataclass
import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]

# ---------------- Kernel ----------------


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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    k = 0
    while k < K:
        offs_k = k + tl.arange(0, BLOCK_K)

        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

        x = tl.load(
            x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0
        )
        w = tl.load(
            w_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0
        )

        acc += tl.dot(x, w)
        k += BLOCK_K

    # bias + ReLU
    b = tl.load(B_ptr + offs_n * stride_b, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = tl.maximum(acc + b[None, :], 0.0)

    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(
        y_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


def linear_relu_triton(x, w, b, *, block_m, block_n, block_k, num_warps, num_stages):
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
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y


# ---------------- Metrics helpers ----------------


def gemm_flops(M, N, K):
    # GEMM: 2*M*N*K FLOPs (mul+add)
    return 2.0 * M * N * K


def bytes_estimate_fp16(M, N, K):
    # A *lower-bound-ish* estimate (single read of X, W, bias; single write of Y)
    # Real DRAM traffic depends on cache/L2 reuse and tiling; this is a useful yardstick.
    # fp16 = 2 bytes
    bytes_x = M * K * 2
    bytes_w = K * N * 2
    bytes_b = N * 2
    bytes_y = M * N * 2  # write
    return bytes_x + bytes_w + bytes_b + bytes_y


def time_ms(fn, iters=200, warmup=50):
    # warmup (also triggers JIT)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    p50 = times[len(times) // 2]
    return p50


@dataclass(frozen=True)
class Cfg:
    bm: int
    bn: int
    bk: int
    warps: int
    stages: int


def sweep(M, Din, Dhid, Dout, iters, warmup, dtype=torch.float16):
    device = "cuda"
    torch.manual_seed(0)

    x = torch.randn((M, Din), device=device, dtype=dtype)
    w1 = (torch.randn((Din, Dhid), device=device, dtype=dtype) * 0.02).contiguous()
    b1 = torch.zeros((Dhid,), device=device, dtype=dtype)
    w2 = (torch.randn((Dhid, Dout), device=device, dtype=dtype) * 0.02).contiguous()
    b2 = torch.zeros((Dout,), device=device, dtype=dtype)

    # correctness reference (use torch, fp16)
    def ref():
        y = torch.relu(x @ w1 + b1)
        y = torch.relu(y @ w2 + b2)
        return y

    y_ref = ref()
    torch.cuda.synchronize()

    # Sweep space (keep it reasonable)
    BLOCK_MS = [64, 128, 256]
    BLOCK_NS = [64, 128, 256]
    BLOCK_KS = [16, 32, 64]
    WARPS = [4, 8]
    STAGES = [2, 3, 4]

    cfgs = [
        Cfg(bm, bn, bk, w, s)
        for bm in BLOCK_MS
        for bn in BLOCK_NS
        for bk in BLOCK_KS
        for w in WARPS
        for s in STAGES
    ]

    def run_layer1(cfg: Cfg):
        return linear_relu_triton(
            x,
            w1,
            b1,
            block_m=cfg.bm,
            block_n=cfg.bn,
            block_k=cfg.bk,
            num_warps=cfg.warps,
            num_stages=cfg.stages,
        )

    def run_layer2(h, cfg: Cfg):
        return linear_relu_triton(
            h,
            w2,
            b2,
            block_m=cfg.bm,
            block_n=cfg.bn,
            block_k=cfg.bk,
            num_warps=cfg.warps,
            num_stages=cfg.stages,
        )

    print(torch.cuda.get_device_name(0))
    print(f"Shapes: X[{M},{Din}] W1[{Din},{Dhid}] W2[{Dhid},{Dout}] dtype={dtype}")

    best = None
    results = []

    for cfg in cfgs:
        # layer1 timing
        def f1():
            linear_relu_triton(
                x,
                w1,
                b1,
                block_m=cfg.bm,
                block_n=cfg.bn,
                block_k=cfg.bk,
                num_warps=cfg.warps,
                num_stages=cfg.stages,
            )

        t1 = time_ms(f1, iters=iters, warmup=warmup)

        # materialize h once for layer2 timing (keeps tuning independent)
        h = run_layer1(cfg)
        torch.cuda.synchronize()

        def f2():
            linear_relu_triton(
                h,
                w2,
                b2,
                block_m=cfg.bm,
                block_n=cfg.bn,
                block_k=cfg.bk,
                num_warps=cfg.warps,
                num_stages=cfg.stages,
            )

        t2 = time_ms(f2, iters=iters, warmup=warmup)

        # end-to-end timing (two layers back-to-back, same cfg for both)
        def fwd():
            h2 = linear_relu_triton(
                x,
                w1,
                b1,
                block_m=cfg.bm,
                block_n=cfg.bn,
                block_k=cfg.bk,
                num_warps=cfg.warps,
                num_stages=cfg.stages,
            )
            y = linear_relu_triton(
                h2,
                w2,
                b2,
                block_m=cfg.bm,
                block_n=cfg.bn,
                block_k=cfg.bk,
                num_warps=cfg.warps,
                num_stages=cfg.stages,
            )
            return y

        t_total = time_ms(fwd, iters=iters, warmup=warmup)

        # correctness spot check on this cfg
        y = fwd()
        max_err = (y - y_ref).abs().max().item()

        # Metrics
        flops_total = gemm_flops(M, Dhid, Din) + gemm_flops(M, Dout, Dhid)
        tflops = flops_total / (t_total * 1e-3) / 1e12

        bytes_total = bytes_estimate_fp16(M, Dhid, Din) + bytes_estimate_fp16(
            M, Dout, Dhid
        )
        gbps = bytes_total / (t_total * 1e-3) / 1e9

        row = (t_total, tflops, gbps, t1, t2, max_err, cfg)
        results.append(row)

        if best is None or t_total < best[0]:
            best = row

        print(
            f"cfg bm/bn/bk={cfg.bm:3d}/{cfg.bn:3d}/{cfg.bk:2d} warps={cfg.warps:2d} stages={cfg.stages} | "
            f"total p50={t_total:7.3f} ms  TFLOPs={tflops:7.2f}  estBW={gbps:7.1f} GB/s  "
            f"(L1 {t1:6.3f} ms, L2 {t2:6.3f} ms)  err={max_err:g}"
        )

    print("\nBEST:")
    t_total, tflops, gbps, t1, t2, max_err, cfg = best
    print(
        f"cfg bm/bn/bk={cfg.bm}/{cfg.bn}/{cfg.bk} warps={cfg.warps} stages={cfg.stages}"
    )
    print(
        f"total p50={t_total:.3f} ms  TFLOPs={tflops:.2f}  estBW={gbps:.1f} GB/s  err={max_err:g}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--din", type=int, default=1024)
    ap.add_argument("--dhid", type=int, default=2048)
    ap.add_argument("--dout", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()
    sweep(args.M, args.din, args.dhid, args.dout, args.iters, args.warmup)


if __name__ == "__main__":
    main()
