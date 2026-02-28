#[derive(Clone, Debug)]
pub(crate) struct UnionFind {
    pub(crate) parent: Vec<usize>,
    pub(crate) size: Vec<usize>,
}

impl UnionFind {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    pub(crate) fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let root = self.find(self.parent[x]);
            self.parent[x] = root;
        }
        self.parent[x]
    }

    pub(crate) fn union(&mut self, a: usize, b: usize) -> usize {
        let ra = self.find(a);
        let rb = self.find(b);
        self.union_roots(ra, rb)
    }

    pub(crate) fn union_roots(&mut self, ra: usize, rb: usize) -> usize {
        if ra == rb {
            return ra;
        }

        // Union by size.
        let (mut big, mut small) = (ra, rb);
        if self.size[big] < self.size[small] {
            std::mem::swap(&mut big, &mut small);
        }

        self.parent[small] = big;
        self.size[big] += self.size[small];
        big
    }
}

#[inline]
pub(crate) fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Compute an MST for a dense complete graph using Prim's algorithm.
///
/// `dist_fn(i, j)` returns the edge weight between points `i` and `j`.
/// Returns edges `(u, v, dist)`.
pub(crate) fn prim_mst(n: usize, dist_fn: impl Fn(usize, usize) -> f32) -> Vec<(usize, usize, f32)> {
    if n <= 1 {
        return Vec::new();
    }

    let mut in_tree = vec![false; n];
    let mut best = vec![f32::INFINITY; n];
    let mut parent = vec![usize::MAX; n];

    best[0] = 0.0;

    for _ in 0..n {
        let mut u = usize::MAX;
        let mut best_val = f32::INFINITY;
        for i in 0..n {
            if !in_tree[i] && best[i] < best_val {
                best_val = best[i];
                u = i;
            }
        }

        if u == usize::MAX {
            break;
        }
        in_tree[u] = true;

        for v in 0..n {
            if in_tree[v] {
                continue;
            }
            let d = dist_fn(u, v);
            if d < best[v] {
                best[v] = d;
                parent[v] = u;
            }
        }
    }

    let mut edges: Vec<(usize, usize, f32)> = Vec::with_capacity(n - 1);
    for v in 1..n {
        let u = parent[v];
        if u != usize::MAX {
            edges.push((u, v, best[v]));
        }
    }
    edges
}
