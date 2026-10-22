"""Deterministic synthetic routes, independent of CUDA and model weights."""
import random


def make_routes(tokens, experts=64, topk=8, pattern="balanced", seed=0):
    if tokens < 1 or not 1 <= topk <= experts:
        raise ValueError("require tokens > 0 and 1 <= topk <= experts")
    if pattern not in ("balanced", "random", "hotspot"):
        raise ValueError("unknown routing pattern")
    rng = random.Random(seed)
    routes = []
    for t in range(tokens):
        if pattern == "balanced":
            row = [(t * topk + k) % experts for k in range(topk)]
        elif pattern == "hotspot":
            row = list(range(topk))
        else:
            row = rng.sample(range(experts), topk)
        routes.append(row)
    slots, counts = pack_routes(routes, experts)
    return routes, slots, counts


def pack_routes(routes, experts):
    if not routes or not routes[0]:
        raise ValueError("empty routes")
    topk, tokens = len(routes[0]), len(routes)
    counts = [0] * experts
    slots = []
    for row in routes:
        if len(row) != topk or len(set(row)) != topk or any(not isinstance(e, int) or e < 0 or e >= experts for e in row):
            raise ValueError("routes require distinct, in-range experts per token")
        for e in row:
            slots.append(e * tokens + counts[e])
            counts[e] += 1
    return slots, counts
