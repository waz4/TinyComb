#TODO:
Finish README!

## Features

`TinyComb` is designed to efficiently calculate and navigate combinations _with repetition_, allowing users to jump directly from an index to its corresponding ordered combination without the overhead of generating all prior combinations. This makes it ideal for applications needing fast access to specific combinations in a large combinatorial space.

### Core Capabilities

- **Combinations with Repetition**: The library computes combinations based on the formula **C(n + k - 1, k)**, where:

  - `n` is the number of distinct elements (e.g., 0 to n-1).
  - `k` is the number of slots (allowing repeats).
    This represents the number of ways to choose `k` items from `n` types with repetition allowed.

- **Efficient Index-to-Combination Mapping**:

  - The `id2combo(id, n, k, combo, factMap)` function converts a given index (`id`, a big number via `struct bn`) into its corresponding combination (stored in `combo`), an array of `k` unsigned integers.
  - Unlike traditional methods that iterate through all combinations sequentially, `TinyComb` uses precomputed factorial data (`factMap`) to compute the combination at any index in O(k) time, bypassing the need to generate the entire sequence up to that point.
  - Example: For `n = 2`, `k = 2`:
    - Index 0 → `(0, 0)`
    - Index 1 → `(0, 1)`
    - Index 2 → `(1, 1)`

- **Total Combinations**:

  - `g(n, k, result, factMap)` calculates the maximum number of combinations (C(n + k - 1, k)) and stores it in `result` (a `struct bn` for big numbers).
  - This serves as the upper bound for valid indices (0 to `g(n, k) - 1`).

- **Sequential Navigation**:

  - `next_combo(ar, n, k)` takes a current combination (`ar`) and updates it to the next combination in the ordered sequence, ensuring efficient iteration when needed.
  - The sequence starts at `(0, 0, ..., 0)` (index 0) and ends at `(n-1, n-1, ..., n-1)` (index `g(n, k) - 1`).

- **Big-Number Support**:

  - `TinyComb` leverages `tiny-bignum-c`, a compact and simple big-number library, to handle large values of `n` and `k` where standard integers would overflow.
  - This lightweight dependency provides efficient arithmetic for factorials and combination counts, keeping the library’s footprint small while supporting massive combinatorial spaces.

- **Optimization with Factorial Maps**:
  - By pairing this with a factorial map array (`factMap`), the library precomputes a range of factorials ahead of time, drastically reducing computation overhead for repeated operations like calculating `g(n, k)` or mapping indices to combinations with `id2combo()`.
  - `makeFactMap(factMap, factMap_length)` precomputes a factorial lookup table up to the required size (determined by `getFactMapLength(n, k)`), speeding up combination calculations for repeated use with the same `n` and `k`.

### Why It’s Efficient

Traditional combination generation might compute all combinations up to a desired index, costing O(C(n + k - 1, k)) time and space in the worst case. `TinyComb` sidesteps this by:

- Using combinatorial number theory to map an index directly to its combination.
- Leveraging precomputed factorials to avoid redundant calculations.
- Achieving **O(k) time complexity**: The `id2combo()` function runs in O(k) time, where `k` is the number of slots, thanks to the precomputed `factMap`. This means generating any specific combination takes time linear in the output size (`k`), regardless of the index or total number of combinations (C(n + k - 1, k)). For comparison, a naive sequential approach could take O(C(n + k - 1, k) \* k) to reach a high index—orders of magnitude slower.
- Supporting both C and CUDA implementations, with CUDA offering parallel acceleration for large-scale problems.
  > > > > > > > 240a94d (Initial Commit)
