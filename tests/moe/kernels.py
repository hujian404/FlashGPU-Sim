"""BF16 traffic kernels. Experts are CTA groups, not pinned physical SMs."""
import triton as tr
import triton.language as tl


@tr.jit
def dispatch(X, Slots, Packed, H: tl.constexpr, TOPK: tl.constexpr,
             BLOCK: tl.constexpr):
    a = tl.program_id(0)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Slots + a)
    x = tl.load(X + (a // TOPK) * H + h, h < H, 0)
    tl.store(Packed + slot * H + h, x, h < H)


@tr.jit
def combine(Y, Slots, Scores, Out, H: tl.constexpr, TOPK: tl.constexpr,
            BLOCK: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for k in range(TOPK):
        slot = tl.load(Slots + t * TOPK + k)
        weight = tl.load(Scores + t * TOPK + k)
        acc += tl.load(Y + slot * H + h, h < H, 0).to(tl.float32) * weight
    tl.store(Out + t * H + h, acc, h < H)


@tr.jit
def expert_mm(A, W, C, Counts, CAP: tl.constexpr, N: tl.constexpr,
              K: tl.constexpr, BM: tl.constexpr = 16,
              BN: tl.constexpr = 32, BK: tl.constexpr = 32):
    e = tl.program_id(2)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    count = tl.load(Counts + e)
    acc = tl.zeros((BM, BN), tl.float32)
    # Empty CTAs exit without issuing any weight reads.
    if tl.program_id(0) * BM < count:
        for start in range(tr.cdiv(K, BK)):
            kk = start * BK + k
            a = tl.load(A + e * CAP * K + m[:, None] * K + kk[None, :],
                        (m[:, None] < count) & (kk[None, :] < K), 0)
            w = tl.load(W + e * N * K + n[None, :] * K + kk[:, None],
                        (n[None, :] < N) & (kk[:, None] < K), 0)
            acc += tl.dot(a, w)
        tl.store(C + e * CAP * N + m[:, None] * N + n[None, :], acc,
                 (m[:, None] < count) & (n[None, :] < N))


@tr.jit
def compact_mm(A, W, C, Counts, TileExperts, TileRows, TileCols, CAP: tl.constexpr,
               N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr = 16,
               BN: tl.constexpr = 32, BK: tl.constexpr = 32):
    tile = tl.program_id(0)
    e = tl.load(TileExperts + tile)
    m = tl.load(TileRows + tile) * BM + tl.arange(0, BM)
    n = tl.load(TileCols + tile) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    count = tl.load(Counts + e)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tr.cdiv(K, BK)):
        kk = start * BK + k
        a = tl.load(A + e * CAP * K + m[:, None] * K + kk[None, :],
                    (m[:, None] < count) & (kk[None, :] < K), 0)
        w = tl.load(W + e * N * K + n[None, :] * K + kk[:, None],
                    (n[None, :] < N) & (kk[:, None] < K), 0)
        acc += tl.dot(a, w)
    tl.store(C + e * CAP * N + m[:, None] * N + n[None, :], acc,
             (m[:, None] < count) & (n[None, :] < N))


@tr.jit
def swiglu(G, U, Out, Counts, CAP: tl.constexpr, I: tl.constexpr,
           BLOCK: tl.constexpr):
    e = tl.program_id(1)
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = x < tl.load(Counts + e) * I
    g = tl.load(G + e * CAP * I + x, mask, 0).to(tl.float32)
    u = tl.load(U + e * CAP * I + x, mask, 0).to(tl.float32)
    tl.store(Out + e * CAP * I + x, g / (1 + tl.exp(-g)) * u, mask)
