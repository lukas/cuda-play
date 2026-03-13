import time
import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]


@triton.jit
def vadd_kernel(a_ptr, b_ptr, c_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
	pid = tl.program_id(0)
	offsets = pid * BLOCK + tl.arange(0, BLOCK)
	mask = offsets < n_elements
	a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
	b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
	tl.store(c_ptr + offsets, a + b, mask=mask)


def bench(block: int, n: int = 100_000_000, iters: int = 200):
	a = torch.ones(n, device="cuda", dtype=torch.float32)
	b = torch.full((n,), 2.0, device="cuda", dtype=torch.float32)
	c = torch.empty_like(a)

	grid = (triton.cdiv(n, block),)

	# warmup
	for _ in range(20):
		vadd_kernel[grid](a, b, c, n_elements=n, BLOCK=block)
	torch.cuda.synchronize()

	# timing (use torch events for GPU-time)
	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)

	times_ms = []
	for _ in range(iters):
		start.record()
		vadd_kernel[grid](a, b, c, n_elements=n, BLOCK=block)
		end.record()
		torch.cuda.synchronize()
		times_ms.append(start.elapsed_time(end))

	times_ms.sort()
	p50 = times_ms[len(times_ms) // 2]

	# correctness spot-check
	c0 = c[0].item()
	cm = c[n // 2].item()
	c1 = c[-1].item()

	bytes_moved = 3.0 * n * 4  # 2 reads + 1 write, float32
	gbps = bytes_moved / (p50 * 1e-3) / 1e9

	return p50, gbps, (c0, cm, c1)


def main():
	torch.cuda.init()
	print(torch.cuda.get_device_name(0))
	blocks = [128, 256, 512, 1024]
	for b in blocks:
		ms, gbps, vals = bench(b)
		print(f"BLOCK={b:4d}  p50={ms:7.3f} ms  BW={gbps:7.1f} GB/s  c0/cm/c1={vals}")


if __name__ == "__main__":
	main()
