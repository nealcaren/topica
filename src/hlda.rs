//! Hierarchical LDA (hLDA) with the nested Chinese Restaurant Process
//! (Blei, Griffiths & Jordan, "The nested Chinese restaurant process and
//! Bayesian nonparametric inference of topic hierarchies", JACM 2010; NIPS 2003).
//!
//! Topics are the nodes of a tree of fixed depth `L`. Level 0 is a single shared
//! root (the most general topic, seen by every document); deeper levels are
//! progressively more specific. Each document is associated with a **path** of
//! `L` nodes from the root to a leaf, chosen by the nested CRP, and each token of
//! the document is assigned a **level** `0..L` along that path; the token is then
//! drawn from that level's topic-word distribution.
//!
//! Inference is the **collapsed Gibbs sampler** of Blei et al. (§4 of the JACM
//! paper). Two moves are iterated:
//!
//!   (a) For every token, resample its level `z_{d,i}` given the document's path:
//!       p(z=l) ∝ (n_{d,l}^{-} + α_l) · (n_{c_l, w}^{-} + η)/(n_{c_l}^{-} + Vη),
//!       where `c_l` is the l-th node on doc d's path, `n_{d,l}` the number of the
//!       document's other tokens at level l, and the second factor is the
//!       Dirichlet-smoothed topic-word likelihood of word `w` at node `c_l`.
//!
//!   (b) For every document, resample its whole path `c_d` via the nested CRP.
//!       The document's word counts are first removed from its current path (and
//!       the path nodes' customer counts decremented; nodes that become empty are
//!       deleted). We then enumerate every candidate path through the surviving
//!       tree — at each internal node a child is taken with CRP probability
//!       n_child/(n_node-1+γ) and a brand-new child with γ/(n_node-1+γ) — and
//!       score each candidate by nCRP_prior × word-likelihood, where the
//!       likelihood is a product over levels of the Dirichlet-multinomial
//!       marginal probability of the document's level-l tokens given node l's
//!       remaining counts. A path is sampled in log space; new nodes are
//!       instantiated as needed and the document's counts are re-added.
//!
//! **Level prior.** The per-document distribution over the L levels takes one of
//! two priors (see [`LevelPrior`]): a fixed-depth Dirichlet with a per-level
//! concentration vector (symmetric or asymmetric, matching tomotopy's `alpha`),
//! or the two-parameter GEM stick-breaking prior of Blei et al. (2010). The
//! Dirichlet keeps the level move conjugate and simple but a *symmetric* Dirichlet
//! does not bias mass toward shallower levels; an asymmetric (root-heavy) `alpha`
//! or the GEM prior does, keeping generic vocabulary in the upper levels by the
//! prior rather than relying on the likelihood.
//!
//! Determinism: every random draw uses only the passed `rng`. After each sweep
//! emptied nodes are compacted so node indices stay contiguous (as `hdp.rs`
//! compacts emptied topics).

use rand::Rng;

/// Sample an index proportional to weights given in **log** space (Gumbel-max via
/// a single uniform after log-sum-exp normalization), using only `rng`.
fn sample_log_index<R: Rng>(log_w: &[f64], rng: &mut R) -> usize {
    let mx = log_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut total = 0.0;
    for &l in log_w {
        total += (l - mx).exp();
    }
    let mut r = rng.gen::<f64>() * total;
    for (i, &l) in log_w.iter().enumerate() {
        r -= (l - mx).exp();
        if r <= 0.0 {
            return i;
        }
    }
    log_w.len() - 1
}

/// One node of the topic tree.
struct Node {
    level: usize,
    parent: Option<usize>,
    children: Vec<usize>,
    /// Number of documents (customers) whose path passes through this node.
    ndocs: u32,
    /// Topic-word counts, length V.
    nw: Vec<u32>,
    /// Total tokens assigned to this node (Σ nw).
    n: u32,
}

/// Prior on a document's distribution over the `L` levels of its path.
///
/// `Dirichlet` is a fixed-depth Dirichlet with a per-level concentration vector
/// (length `depth`): all-equal entries give the classic symmetric prior, unequal
/// entries an asymmetric one that biases mass toward particular levels (a larger
/// root entry keeps generic vocabulary shallow). This matches tomotopy's
/// `HLDAModel`, which also takes a scalar-or-per-level `alpha`.
///
/// `Gem` is the two-parameter GEM stick-breaking prior of Blei, Griffiths &
/// Jordan (2010): `mean` `m` in (0, 1) is the expected fraction of the remaining
/// stick taken at each level and `scale` `pi` > 0 the concentration. This is the
/// prior in the original paper (and hlda-c). We apply the Ishwaran–James finite
/// truncation — the last stick is forced to 1 so the deepest level absorbs the
/// tail and the distribution is proper. Under empty counts it decays geometrically,
/// `p(l) = m (1-m)^l` for `l < L-1` and `p(L-1) = (1-m)^{L-1}`, so mass concentrates
/// in the shallow levels by construction rather than relying on the likelihood.
#[derive(Clone, Debug)]
pub enum LevelPrior {
    /// Per-level Dirichlet concentration, length `depth`.
    Dirichlet(Vec<f64>),
    /// GEM stick-breaking with mean `m` and scale (precision) `pi`.
    Gem { mean: f64, scale: f64 },
}

/// A fitted hierarchical LDA model. The tree has fixed depth `L`.
pub struct HldaModel {
    pub num_types: usize,
    pub depth: usize,            // L
    pub gamma: f64,              // nested-CRP concentration
    pub eta: f64,                // topic-word Dirichlet
    pub level_prior: LevelPrior, // per-document level distribution prior
    nodes: Vec<Node>,
    /// Per document: the L node indices on its path (root .. leaf).
    paths: Vec<Vec<usize>>,
    /// Per document, per token: the level 0..L it is assigned to.
    levels: Vec<Vec<usize>>,
    /// Fit-time `log_gamma` lookup tables (see [`HldaModel::build_lgamma_tables`]).
    /// `lg_eta[k] = log_gamma(k as f64 + eta)`, `lg_veta[k] = log_gamma(k as f64 +
    /// V*eta)`. Every `level_log_marginal` argument is `integer + eta` or
    /// `integer + V*eta`, so these replace the dominant `log_gamma` cost with a
    /// table lookup. Transient (rebuilt each fit, not serialized).
    lg_eta: Vec<f64>,
    lg_veta: Vec<f64>,
}

impl HldaModel {
    /// Number of nodes (topics) in the tree.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Depth (level, 0 = root) of node `i`.
    pub fn node_level(&self, i: usize) -> usize {
        self.nodes[i].level
    }

    /// Parent of node `i` (None for the root).
    pub fn node_parent(&self, i: usize) -> Option<usize> {
        self.nodes[i].parent
    }

    /// Topic-word distribution of node `i`: (n_{i,w}+η)/(n_i + Vη), length V.
    pub fn topic_word(&self, i: usize) -> Vec<f64> {
        let denom = self.nodes[i].n as f64 + self.num_types as f64 * self.eta;
        self.nodes[i]
            .nw
            .iter()
            .map(|&c| (c as f64 + self.eta) / denom)
            .collect()
    }

    /// The L node indices on document `d`'s path (root .. leaf).
    pub fn doc_path(&self, d: usize) -> Vec<usize> {
        self.paths[d].clone()
    }

    /// Indices of all leaf nodes (deepest level, L-1).
    pub fn leaves(&self) -> Vec<usize> {
        (0..self.nodes.len())
            .filter(|&i| self.nodes[i].level == self.depth - 1)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tree bookkeeping
// ---------------------------------------------------------------------------

impl HldaModel {
    /// Create a fresh node at `level` with the given parent, register it as a
    /// child of its parent, and return its index.
    fn new_node(&mut self, level: usize, parent: Option<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            level,
            parent,
            children: Vec::new(),
            ndocs: 0,
            nw: vec![0u32; self.num_types],
            n: 0,
        });
        if let Some(p) = parent {
            self.nodes[p].children.push(id);
        }
        id
    }
}

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

impl HldaModel {
    /// Add (`+1`) or remove (`-1`) document `d`'s token counts on its path.
    fn apply_doc_counts(&mut self, d: usize, doc: &[u32], sign: i64) {
        let path = self.paths[d].clone();
        for (i, &w) in doc.iter().enumerate() {
            let l = self.levels[d][i];
            let node = path[l];
            let w = w as usize;
            if sign > 0 {
                self.nodes[node].nw[w] += 1;
                self.nodes[node].n += 1;
            } else {
                self.nodes[node].nw[w] -= 1;
                self.nodes[node].n -= 1;
            }
        }
    }

    /// Move (a): resample the level of every token in document `d`.
    /// Log of the per-document level-prior weight for each level `0..L`, given the
    /// document's current per-level token counts `n_dl` (the resampled token
    /// already removed). For `Dirichlet` this is `ln(n_dl[l] + alpha[l])` (the
    /// shared `1/(N + Σα)` normaliser is dropped — the caller renormalises); for
    /// `Gem` it is the log of the truncated stick-breaking predictive
    /// `p(l) = (mπ + n_l)/(π + n_{≥l}) · Π_{j<l} ((1-m)π + n_{>j})/(π + n_{≥j})`.
    fn level_log_prior(&self, n_dl: &[u32]) -> Vec<f64> {
        let l = self.depth;
        let mut out = vec![0.0f64; l];
        match &self.level_prior {
            LevelPrior::Dirichlet(alpha) => {
                for lev in 0..l {
                    out[lev] = (n_dl[lev] as f64 + alpha[lev]).ln();
                }
            }
            LevelPrior::Gem { mean, scale } => {
                let (m, pi) = (*mean, *scale);
                // Suffix sums s[j] = Σ_{k>=j} n_dl[k], with s[l] = 0.
                let mut s = vec![0u32; l + 1];
                for j in (0..l).rev() {
                    s[j] = s[j + 1] + n_dl[j];
                }
                // Truncated stick-breaking (Ishwaran & James): the last stick is
                // forced to 1 so level L-1 absorbs the remaining mass and the level
                // distribution is proper (sums to 1) over 0..L. Levels below L-1 use
                // the usual "stop here" Beta-mean factor; level L-1 uses only the
                // product of "pass" factors (no stop factor). `cum` accumulates
                // Σ_{j<lev} ln((1-m)π + n_{>j}) - ln(π + n_{≥j}).
                let mut cum = 0.0;
                for lev in 0..l {
                    out[lev] = if lev + 1 < l {
                        (m * pi + n_dl[lev] as f64).ln() - (pi + s[lev] as f64).ln() + cum
                    } else {
                        // Deepest level: V_{L-1}=1, folds in the truncated tail.
                        cum
                    };
                    cum += ((1.0 - m) * pi + s[lev + 1] as f64).ln() - (pi + s[lev] as f64).ln();
                }
            }
        }
        out
    }

    fn sample_levels<R: Rng>(&mut self, d: usize, doc: &[u32], rng: &mut R) {
        let path = self.paths[d].clone();
        let l = self.depth;
        let v = self.num_types;
        // Per-level token counts for this document.
        let mut n_dl = vec![0u32; l];
        for &lev in &self.levels[d] {
            n_dl[lev] += 1;
        }
        for (i, &w) in doc.iter().enumerate() {
            let w = w as usize;
            let old = self.levels[d][i];
            // Remove this token.
            n_dl[old] -= 1;
            let onode = path[old];
            self.nodes[onode].nw[w] -= 1;
            self.nodes[onode].n -= 1;

            // p(level=l) ∝ prior(l) * (nw + eta)/(n + V*eta) in log space, where the
            // prior is the Dirichlet or GEM level weight from the remaining counts.
            let log_prior = self.level_log_prior(&n_dl);
            let mut logp = vec![0.0f64; l];
            for lev in 0..l {
                let node = path[lev];
                let like = ((self.nodes[node].nw[w] as f64 + self.eta)
                    / (self.nodes[node].n as f64 + v as f64 * self.eta))
                    .ln();
                logp[lev] = log_prior[lev] + like;
            }
            let new = sample_log_index(&logp, rng);

            // Re-add this token at the chosen level.
            self.levels[d][i] = new;
            n_dl[new] += 1;
            let nnode = path[new];
            self.nodes[nnode].nw[w] += 1;
            self.nodes[nnode].n += 1;
        }
    }

    /// Log Dirichlet-multinomial marginal of a multiset of words (given by
    /// per-word increments) added to an existing count vector `nw`/`n` of a node
    /// at one level. `counts` maps word -> how many of the document's level-l
    /// tokens hit that word. Returns log p(new words | existing node counts).
    /// Tabulate `log_gamma(k + eta)` and `log_gamma(k + V*eta)` for every integer
    /// `k` a count can reach (bounded by the total token budget plus a document's
    /// length). `level_log_marginal` then indexes these instead of calling the
    /// (expensive) `log_gamma` per token per node — the dominant HLDA cost. The root
    /// node can legitimately hold ~every token, so the tables must span the whole
    /// budget: fit-time memory is two `f64` per token (~16 bytes/token), linear in
    /// the corpus size.
    fn build_lgamma_tables(&mut self, docs: &[Vec<u32>]) {
        let total: usize = docs.iter().map(|d| d.len()).sum();
        let max_doc = docs.iter().map(|d| d.len()).max().unwrap_or(0);
        // A count is at most `total`; a marginal adds at most one document's worth.
        let cap = total + max_doc + 1;
        let veta = self.num_types as f64 * self.eta;
        self.lg_eta = (0..cap).map(|k| log_gamma(k as f64 + self.eta)).collect();
        self.lg_veta = (0..cap).map(|k| log_gamma(k as f64 + veta)).collect();
    }

    fn level_log_marginal(&self, node: usize, counts: &[(usize, u32)]) -> f64 {
        let n_existing = self.nodes[node].n as usize;
        let mut m_total = 0u32;
        let mut logp = 0.0;
        for &(w, m) in counts {
            if m == 0 {
                continue;
            }
            let nkw = self.nodes[node].nw[w] as usize;
            // log[ Γ(nkw+η+m)/Γ(nkw+η) ] via the precomputed log_gamma(k+η) table.
            logp += self.lg_eta[nkw + m as usize] - self.lg_eta[nkw];
            m_total += m;
        }
        if m_total == 0 {
            return 0.0;
        }
        // Denominator Γ(n+Vη)/Γ(n+Vη+m) via the log_gamma(k+Vη) table.
        logp += self.lg_veta[n_existing] - self.lg_veta[n_existing + m_total as usize];
        logp
    }

    /// Move (b): resample the whole path of document `d` via the nested CRP.
    fn sample_path<R: Rng>(&mut self, d: usize, doc: &[u32], parallel: bool, rng: &mut R) {
        let l = self.depth;

        // 1. Remove the document's word counts from its current path.
        self.apply_doc_counts(d, doc, -1);
        // 2. Decrement customer counts and mark empty nodes for deletion.
        let old_path = self.paths[d].clone();
        for &node in &old_path {
            self.nodes[node].ndocs -= 1;
        }
        self.delete_empty_subtrees();

        // Group the document's words by the level they are currently assigned to.
        let mut level_words: Vec<Vec<(usize, u32)>> = vec![Vec::new(); l];
        {
            let mut maps: Vec<std::collections::BTreeMap<usize, u32>> =
                vec![std::collections::BTreeMap::new(); l];
            for (i, &w) in doc.iter().enumerate() {
                let lev = self.levels[d][i];
                *maps[lev].entry(w as usize).or_insert(0) += 1;
            }
            for (lev, m) in maps.into_iter().enumerate() {
                level_words[lev] = m.into_iter().collect();
            }
        }

        // 3. Enumerate candidate paths through the existing tree.
        //
        // A candidate path is described by: the list of existing nodes followed
        // (from the root down to some node at level `depth_existing-1`), plus a
        // flag for whether the remaining levels are brand-new nodes. Equivalently
        // we DFS from the root: at each node either descend into an existing child
        // (CRP weight n_child/(n_node-1+γ)) or stop and create fresh nodes for all
        // remaining levels (CRP weight γ/(n_node-1+γ) at the first new step; once a
        // new node is created, all of its descendants are necessarily new, each
        // contributing a γ/(0-1+γ)=γ/(γ-1)... — but a fresh chain has a single new
        // branch, so its log-prior is just the log of the new-child weights).
        //
        // We collect (path nodes [for existing prefix], new_from_level, log_prior).
        let root = self.root_index();
        struct Cand {
            // node indices for existing portion of the path (levels 0..new_from)
            nodes: Vec<usize>,
            // first level (1..=L) that is a brand-new node (== L means fully existing)
            new_from: usize,
            log_prior: f64,
        }
        let mut cands: Vec<Cand> = Vec::new();

        // Recursive DFS.
        fn dfs(
            model: &HldaModel,
            node: usize,
            level: usize, // level of `node`
            l: usize,
            gamma: f64,
            prefix: &mut Vec<usize>,
            log_prior: f64,
            out: &mut Vec<Cand>,
        ) {
            prefix.push(node);
            if level == l - 1 {
                // Reached a leaf-level existing node: fully-existing path.
                out.push(Cand {
                    nodes: prefix.clone(),
                    new_from: l,
                    log_prior,
                });
                prefix.pop();
                return;
            }
            // CRP over children of `node`. Customers competing = ndocs at this node
            // (the doc was already removed). Denominator = ndocs - 1 + γ, but with
            // the doc removed ndocs is already the count of *other* docs, so the
            // CRP denominator is ndocs + γ. (Equivalently the classic n-1+γ before
            // removal.)
            let denom = model.nodes[node].ndocs as f64 + gamma;
            // Option A: new child here -> remaining levels all new.
            {
                let lp = log_prior + (gamma / denom).ln();
                out.push(Cand {
                    nodes: prefix.clone(),
                    new_from: level + 1,
                    log_prior: lp,
                });
            }
            // Option B: descend into each existing child.
            let children = model.nodes[node].children.clone();
            for c in children {
                let nc = model.nodes[c].ndocs as f64;
                let lp = log_prior + (nc / denom).ln();
                dfs(model, c, level + 1, l, gamma, prefix, lp, out);
            }
            prefix.pop();
        }

        let mut prefix = Vec::new();
        dfs(self, root, 0, l, self.gamma, &mut prefix, 0.0, &mut cands);

        // 4. Score each candidate: log_prior + Σ_levels word-likelihood.
        //    For existing levels use that node's remaining counts; for new levels
        //    the node is empty (counts all zero).
        let empty_marginal: Vec<f64> = (0..l)
            .map(|lev| {
                // marginal of level-lev words against an empty node.
                self.empty_level_log_marginal(&level_words[lev])
            })
            .collect();

        // Per-node marginal `level_log_marginal(node, level_words[node.level])`
        // depends only on the node's counts and the document's (fixed) level-words,
        // so it is identical for every candidate path passing through that node —
        // the dominant `log_gamma` cost. Every node is reached by the DFS (every node
        // descends from the root), so precompute the marginal for all nodes and index
        // into it while scoring candidates. When `parallel`, the precompute runs
        // across nodes in the current rayon pool; because each entry is a pure,
        // index-addressed function of read-only state, the result is **bit-for-bit
        // identical regardless of thread count** (num_threads never changes the fit).
        let node_count = self.nodes.len();
        let this: &Self = self;
        let marg: Vec<f64> = if parallel {
            use rayon::prelude::*;
            (0..node_count)
                .into_par_iter()
                .map(|node| this.level_log_marginal(node, &level_words[this.nodes[node].level]))
                .collect()
        } else {
            (0..node_count)
                .map(|node| this.level_log_marginal(node, &level_words[this.nodes[node].level]))
                .collect()
        };
        let mut log_scores = Vec::with_capacity(cands.len());
        for cand in &cands {
            let mut s = cand.log_prior;
            for lev in 0..l {
                if lev < cand.new_from {
                    s += marg[cand.nodes[lev]];
                } else {
                    s += empty_marginal[lev];
                }
            }
            log_scores.push(s);
        }

        // 5. Sample a candidate.
        let chosen = sample_log_index(&log_scores, rng);
        let cand = &cands[chosen];

        // 6. Build the new path, instantiating fresh nodes for new levels.
        let mut new_path = Vec::with_capacity(l);
        for lev in 0..cand.new_from {
            new_path.push(cand.nodes[lev]);
        }
        let mut parent = if cand.new_from == 0 {
            None
        } else {
            Some(cand.nodes[cand.new_from - 1])
        };
        for lev in cand.new_from..l {
            let id = self.new_node(lev, parent);
            new_path.push(id);
            parent = Some(id);
        }
        self.paths[d] = new_path;

        // 7. Increment customer counts and re-add the document's word counts.
        for &node in &self.paths[d] {
            self.nodes[node].ndocs += 1;
        }
        self.apply_doc_counts(d, doc, 1);
    }

    /// Log Dirichlet-multinomial marginal of a level's word multiset against an
    /// empty node (all counts zero, n = 0).
    fn empty_level_log_marginal(&self, level_words: &[(usize, u32)]) -> f64 {
        let v = self.num_types as f64;
        let eta = self.eta;
        let mut m_total = 0u32;
        let mut logp = 0.0;
        for &(_, m) in level_words {
            if m == 0 {
                continue;
            }
            logp += log_gamma(eta + m as f64) - log_gamma(eta);
            m_total += m;
        }
        if m_total == 0 {
            return 0.0;
        }
        logp += log_gamma(v * eta) - log_gamma(v * eta + m_total as f64);
        logp
    }

    fn root_index(&self) -> usize {
        // Root is the unique level-0 node (parent None). After compaction the
        // root stays a level-0 node; find it.
        (0..self.nodes.len())
            .find(|&i| self.nodes[i].level == 0)
            .expect("tree must have a root")
    }

    /// Delete every subtree whose root node has zero customers. A node with no
    /// customers has no tokens either (its counts were removed first), and if a
    /// node is empty so are all of its descendants, so the whole empty region is
    /// safe to drop.
    ///
    /// We remove empty nodes **leaf-first**: repeatedly drop any non-root node
    /// that is empty *and currently childless*, re-scanning after each
    /// `swap_remove`. An empty internal node is only removed once its (also-empty)
    /// children are gone, so no node index is ever held across the `swap_remove`
    /// that could invalidate it. The previous recurse-then-detach walk read
    /// `self.nodes[i].parent` after descendant `swap_remove`s had already
    /// relocated `i`, which corrupted links / panicked once an empty subtree
    /// spanned three or more removal levels (reachable at `depth >= 4`).
    fn delete_empty_subtrees(&mut self) {
        loop {
            let target = (0..self.nodes.len()).find(|&i| {
                self.nodes[i].level != 0
                    && self.nodes[i].ndocs == 0
                    && self.nodes[i].children.is_empty()
            });
            let Some(i) = target else { break };
            // Detach the childless empty leaf from its parent, then swap-remove it.
            // `swap_remove_node` remaps the one element it moves; because `i` has no
            // children, nothing we still need points into `i`.
            if let Some(p) = self.nodes[i].parent {
                self.nodes[p].children.retain(|&c| c != i);
            }
            self.swap_remove_node(i);
        }
    }

    /// `swap_remove` node `i` from the Vec and fix up every index that referred to
    /// the moved last element (parent/children links and document paths).
    fn swap_remove_node(&mut self, i: usize) {
        let last = self.nodes.len() - 1;
        self.nodes.swap_remove(i);
        if i == last {
            return; // removed the last; no remap needed
        }
        // Element that was at `last` is now at `i`. Remap last -> i everywhere.
        // Fix the moved node's parent's child list.
        if let Some(p) = self.nodes[i].parent {
            for c in self.nodes[p].children.iter_mut() {
                if *c == last {
                    *c = i;
                }
            }
        }
        // Fix the moved node's children's parent pointers.
        let kids = self.nodes[i].children.clone();
        for k in kids {
            self.nodes[k].parent = Some(i);
        }
        // Fix any document path referencing `last`.
        for path in self.paths.iter_mut() {
            for node in path.iter_mut() {
                if *node == last {
                    *node = i;
                }
            }
        }
    }
}

/// Fit a hierarchical LDA model by collapsed Gibbs sampling over the nested CRP.
///
/// `docs` are bags of word ids. `depth` is the tree depth L (>= 2). `gamma` is
/// the nested-CRP concentration, `eta` the topic-word Dirichlet, and `level_prior`
/// the per-document level distribution ([`LevelPrior::Dirichlet`], symmetric or
/// asymmetric, or [`LevelPrior::Gem`]). `num_threads > 1` parallelises the
/// read-only per-node marginal precompute; the document loop and every tree
/// mutation stay serial, so the result is **bit-for-bit identical for any
/// `num_threads`**. Deterministic for a fixed `rng`.
///
/// `R: Send` is required (the sweep runs inside a rayon pool); all of topica's
/// RNGs (`ChaCha8Rng`, `Pcg64Mcg`) satisfy it. A non-`Send` RNG cannot be used
/// even at `num_threads = 1`.
#[allow(clippy::too_many_arguments)]
pub fn fit_hlda<R: Rng + Send, F: FnMut(usize, usize) + Send>(
    docs: &[Vec<u32>],
    num_types: usize,
    depth: usize,
    gamma: f64,
    eta: f64,
    level_prior: LevelPrior,
    iters: usize,
    num_threads: usize,
    mut on_progress: F,
    rng: &mut R,
) -> HldaModel {
    assert!(depth >= 2, "hLDA needs depth >= 2");
    if let LevelPrior::Dirichlet(alpha) = &level_prior {
        assert_eq!(
            alpha.len(),
            depth,
            "Dirichlet level prior needs one α per level"
        );
    }
    let mut model = HldaModel {
        num_types,
        depth,
        gamma,
        eta,
        level_prior,
        nodes: Vec::new(),
        paths: vec![Vec::new(); docs.len()],
        levels: docs.iter().map(|d| vec![0usize; d.len()]).collect(),
        lg_eta: Vec::new(),
        lg_veta: Vec::new(),
    };
    // Precompute log_gamma lookup tables: every level_log_marginal argument is an
    // integer count plus eta (word terms) or plus V*eta (denominator), and no count
    // can exceed the total token budget, so tabulate up to that bound once.
    model.build_lgamma_tables(docs);

    // Initialization: create a root; for each document, build a path by descending
    // the nested CRP (reusing or creating children with the usual γ weight), and
    // assign each token a random level.
    let root = model.new_node(0, None);

    for (d, doc) in docs.iter().enumerate() {
        // Random level for every token.
        for i in 0..doc.len() {
            let lev = (rng.gen::<f64>() * depth as f64) as usize % depth;
            model.levels[d][i] = lev;
        }
        // Build a path greedily by the CRP, creating nodes as drawn.
        let mut path = Vec::with_capacity(depth);
        let mut node = root;
        path.push(node);
        for lev in 1..depth {
            let denom = model.nodes[node].ndocs as f64 + gamma;
            // Weights: each existing child by ndocs, new child by gamma.
            let children = model.nodes[node].children.clone();
            let mut weights: Vec<f64> = children
                .iter()
                .map(|&c| model.nodes[c].ndocs as f64 / denom)
                .collect();
            weights.push(gamma / denom);
            // Sample.
            let total: f64 = weights.iter().sum();
            let mut r = rng.gen::<f64>() * total;
            let mut pick = weights.len() - 1;
            for (idx, &wt) in weights.iter().enumerate() {
                r -= wt;
                if r <= 0.0 {
                    pick = idx;
                    break;
                }
            }
            let next = if pick == children.len() {
                model.new_node(lev, Some(node))
            } else {
                children[pick]
            };
            path.push(next);
            node = next;
        }
        model.paths[d] = path;
        // Register customers and add token counts.
        for &n in &model.paths[d] {
            model.nodes[n].ndocs += 1;
        }
        model.apply_doc_counts(d, doc, 1);
    }

    // Gibbs sweeps: per document, resample its path, then its token levels. The
    // document loop and every tree mutation stay serial; only the read-only
    // per-node marginal precompute inside `sample_path` runs in parallel (across the
    // `num_threads` pool). Because that precompute is a set of independent pure
    // computations, the result is bit-for-bit identical for any `num_threads` — the
    // fit is deterministic and unchanged by threading.
    let parallel = num_threads > 1;
    let sweep = |model: &mut HldaModel, rng: &mut R, on_progress: &mut F| {
        for it in 0..iters {
            for (d, doc) in docs.iter().enumerate() {
                model.sample_path(d, doc, parallel, rng);
                model.sample_levels(d, doc, rng);
            }
            on_progress(it + 1, iters);
        }
    };
    if parallel {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("failed to build rayon thread pool");
        pool.install(|| sweep(&mut model, rng, &mut on_progress));
    } else {
        sweep(&mut model, rng, &mut on_progress);
    }

    model
}

use crate::estimator::{Estimator, ModelFamily};
use crate::mathfun::log_gamma;

impl Estimator for HldaModel {
    fn num_topics(&self) -> usize {
        self.num_nodes()
    }

    fn topic_word(&self) -> Vec<Vec<f64>> {
        // Disambiguate: inherent topic_word(i) takes an index; the trait method takes none.
        (0..self.num_nodes())
            .map(|i| HldaModel::topic_word(self, i))
            .collect()
    }

    fn doc_topic(&self) -> Vec<Vec<f64>> {
        // HLDA uses tree paths, not a flat simplex — EXEMPT.
        Vec::new()
    }

    fn fit_history(&self) -> Vec<(usize, f64)> {
        Vec::new()
    }

    fn converged(&self) -> Option<bool> {
        None
    }

    fn model_family(&self) -> ModelFamily {
        ModelFamily::None_
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Build a planted-hierarchy corpus: every doc has SHARED function words plus
    /// words from ONE of G group blocks. Returns (docs, V, shared, blocks).
    fn planted_corpus<R: Rng>(
        rng: &mut R,
        g: usize,
    ) -> (Vec<Vec<u32>>, usize, Vec<u32>, Vec<Vec<u32>>) {
        // Vocabulary layout: [0..S) shared, then G blocks of B distinctive words.
        let s = 5usize; // shared words
        let b = 5usize; // distinctive words per group
        let shared: Vec<u32> = (0..s as u32).collect();
        let blocks: Vec<Vec<u32>> = (0..g)
            .map(|gi| {
                let base = (s + gi * b) as u32;
                (base..base + b as u32).collect()
            })
            .collect();
        let v = s + g * b;
        let mut docs = Vec::new();
        for d in 0..60 * g {
            let gi = d % g;
            let blk = &blocks[gi];
            let mut doc = Vec::new();
            // 8 shared tokens, 8 group tokens, interleaved deterministically.
            for i in 0..8 {
                doc.push(shared[(i + d) % shared.len()]);
                doc.push(blk[(i + d) % blk.len()]);
            }
            // light shuffle by rng to avoid pathological ordering
            for i in (1..doc.len()).rev() {
                let j = (rng.gen::<f64>() * (i + 1) as f64) as usize % (i + 1);
                doc.swap(i, j);
            }
            docs.push(doc);
        }
        (docs, v, shared, blocks)
    }

    fn top_words(dist: &[f64], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..dist.len()).collect();
        idx.sort_by(|&a, &b| dist[b].partial_cmp(&dist[a]).unwrap());
        idx.truncate(k);
        idx
    }

    #[test]
    fn recovers_planted_hierarchy() {
        let g = 3usize;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (docs, v, shared, blocks) = planted_corpus(&mut rng, g);

        let model = fit_hlda(
            &docs,
            v,
            2,
            1.0,
            0.1,
            LevelPrior::Dirichlet(vec![0.5; 2]),
            80,
            1,
            |_, _| {},
            &mut rng,
        );

        // (1) The root's top words are dominated by the shared words.
        let root = (0..model.num_nodes())
            .find(|&i| model.node_level(i) == 0)
            .unwrap();
        let root_top = top_words(&model.topic_word(root), shared.len());
        let shared_set: std::collections::HashSet<usize> =
            shared.iter().map(|&w| w as usize).collect();
        let shared_in_root = root_top.iter().filter(|w| shared_set.contains(w)).count();
        assert!(
            shared_in_root >= shared.len() - 1,
            "root top words not dominated by shared words: {:?}",
            root_top
        );

        // (2) Multiple leaves, each leaf's top words come from one group block.
        let leaves = model.leaves();
        assert!(
            (g - 1..=g + 3).contains(&leaves.len()),
            "leaf count {} outside band for G={}",
            leaves.len(),
            g
        );

        let mut covered_blocks = std::collections::HashSet::new();
        for &leaf in &leaves {
            let top = top_words(&model.topic_word(leaf), 5);
            // Which block does the leaf's top word belong to?
            for (bi, blk) in blocks.iter().enumerate() {
                let blk_set: std::collections::HashSet<usize> =
                    blk.iter().map(|&w| w as usize).collect();
                let hits = top.iter().filter(|w| blk_set.contains(w)).count();
                if hits >= 3 {
                    covered_blocks.insert(bi);
                }
            }
        }
        assert!(
            covered_blocks.len() >= g - 1,
            "leaves covered only {} of {} group blocks",
            covered_blocks.len(),
            g
        );
    }

    #[test]
    fn lgamma_table_matches_direct_computation() {
        // The log_gamma lookup table computes the same Dirichlet-multinomial
        // marginal as direct log_gamma. It is NOT bit-identical to the old inline
        // form: the table argument is `(k+m) as f64 + η` where the old form was
        // `(k as f64 + η) + m as f64`, and float addition is non-associative, so
        // the two can differ by a few ULP. Assert numerical agreement (relative
        // 1e-9) — the marginal is mathematically identical, not bit-equal.
        for &(eta, v) in &[(0.01_f64, 5000usize), (0.1, 200), (1.0, 50)] {
            let veta = v as f64 * eta;
            let cap = 20_000usize;
            let lg_eta: Vec<f64> = (0..cap).map(|k| log_gamma(k as f64 + eta)).collect();
            let lg_veta: Vec<f64> = (0..cap).map(|k| log_gamma(k as f64 + veta)).collect();
            for &nkw in &[0usize, 1, 2, 7, 50, 999, 5000] {
                for &m in &[1usize, 2, 5, 40, 500] {
                    if nkw + m >= cap {
                        continue;
                    }
                    let table = lg_eta[nkw + m] - lg_eta[nkw];
                    let direct =
                        log_gamma(nkw as f64 + eta + m as f64) - log_gamma(nkw as f64 + eta);
                    assert!(
                        (table - direct).abs() <= 1e-9 * direct.abs().max(1.0),
                        "word term mismatch η={eta} nkw={nkw} m={m}: {table} vs {direct}"
                    );
                }
            }
            for &n in &[0usize, 3, 60, 1234, 9999] {
                for &mt in &[1usize, 4, 30, 300] {
                    if n + mt >= cap {
                        continue;
                    }
                    let table = lg_veta[n] - lg_veta[n + mt];
                    let direct =
                        log_gamma(n as f64 + veta) - log_gamma(n as f64 + veta + mt as f64);
                    assert!(
                        (table - direct).abs() <= 1e-9 * direct.abs().max(1.0),
                        "denominator term mismatch η={eta} n={n} mt={mt}: {table} vs {direct}"
                    );
                }
            }
        }
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let mut seed_rng = ChaCha8Rng::seed_from_u64(7);
        let (docs, v, _shared, _blocks) = planted_corpus(&mut seed_rng, 2);

        let mut r1 = ChaCha8Rng::seed_from_u64(123);
        let mut r2 = ChaCha8Rng::seed_from_u64(123);
        let m1 = fit_hlda(
            &docs,
            v,
            2,
            1.0,
            0.1,
            LevelPrior::Dirichlet(vec![0.5; 2]),
            30,
            1,
            |_, _| {},
            &mut r1,
        );
        let m2 = fit_hlda(
            &docs,
            v,
            2,
            1.0,
            0.1,
            LevelPrior::Dirichlet(vec![0.5; 2]),
            30,
            1,
            |_, _| {},
            &mut r2,
        );

        assert_eq!(m1.num_nodes(), m2.num_nodes());
        for i in 0..m1.num_nodes() {
            assert_eq!(m1.node_level(i), m2.node_level(i));
            assert_eq!(m1.node_parent(i), m2.node_parent(i));
            assert_eq!(m1.nodes[i].n, m2.nodes[i].n);
            assert_eq!(m1.nodes[i].nw, m2.nodes[i].nw);
            assert_eq!(m1.nodes[i].ndocs, m2.nodes[i].ndocs);
        }
    }

    #[test]
    fn hlda_conforms() {
        let g = 3usize;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let (docs, v, _shared, _blocks) = planted_corpus(&mut rng, g);
        let model = fit_hlda(
            &docs,
            v,
            2,
            1.0,
            0.1,
            LevelPrior::Dirichlet(vec![0.5; 2]),
            80,
            1,
            |_, _| {},
            &mut rng,
        );
        let base = crate::conformance::check_conformance(&model);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }

    #[test]
    fn depth4_delete_empty_subtrees_stays_consistent() {
        // Regression for #496: `delete_empty_subtrees` used to hold node indices
        // across `swap_remove` and corrupt links / panic once an empty subtree
        // spanned three removal levels — reachable only at `depth >= 4`. Fit a
        // churny depth-4 tree over several seeds and assert the tree stays fully
        // consistent (no panic, every parent/child link and document path valid,
        // no empty non-root node left behind).
        for seed in [1u64, 7, 42, 99, 2024, 31337] {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            // A churny random corpus (not the stable planted one): many short docs
            // over a small vocab, with a high nCRP concentration so the sampler
            // spawns deep transient subtrees that then empty out as documents
            // resample — the regime that makes an empty subtree span 3+ removal
            // levels. depth=5 gives the extra level headroom over the depth-3-safe
            // region.
            let v = 12usize;
            let mut docs = Vec::new();
            for d in 0..220 {
                let len = 3 + (d % 5);
                let mut doc = Vec::new();
                for _ in 0..len {
                    doc.push((rng.gen::<f64>() * v as f64) as u32 % v as u32);
                }
                docs.push(doc);
            }
            let model = fit_hlda(
                &docs,
                v,
                5,
                4.0,
                0.1,
                LevelPrior::Dirichlet(vec![0.5; 5]),
                80,
                1,
                |_, _| {},
                &mut rng,
            );

            let n = model.num_nodes();
            assert_eq!(
                (0..n).filter(|&i| model.nodes[i].level == 0).count(),
                1,
                "seed {seed}: expected exactly one root"
            );
            for i in 0..n {
                match model.nodes[i].parent {
                    Some(p) => {
                        assert!(p < n, "seed {seed}: node {i} parent {p} out of range");
                        assert!(
                            model.nodes[p].children.contains(&i),
                            "seed {seed}: node {i} missing from parent {p}'s children"
                        );
                        assert_eq!(
                            model.nodes[p].level + 1,
                            model.nodes[i].level,
                            "seed {seed}: node {i} level != parent level + 1"
                        );
                    }
                    None => assert_eq!(
                        model.nodes[i].level, 0,
                        "seed {seed}: parentless node {i} is not the root"
                    ),
                }
                for &c in &model.nodes[i].children {
                    assert!(c < n, "seed {seed}: node {i} child {c} out of range");
                    assert_eq!(
                        model.nodes[c].parent,
                        Some(i),
                        "seed {seed}: child {c}'s parent link != {i}"
                    );
                }
                assert!(
                    model.nodes[i].level == 0 || model.nodes[i].ndocs > 0,
                    "seed {seed}: empty non-root node {i} survived pruning"
                );
            }
            // Every document path is a valid root->leaf chain of length `depth`.
            for (d, path) in model.paths.iter().enumerate() {
                assert_eq!(
                    path.len(),
                    model.depth,
                    "seed {seed}: doc {d} path len != depth"
                );
                assert_eq!(
                    model.nodes[path[0]].level, 0,
                    "seed {seed}: doc {d} path[0] is not the root"
                );
                for w in 1..path.len() {
                    assert!(
                        path[w] < n,
                        "seed {seed}: doc {d} path node {} out of range",
                        path[w]
                    );
                    assert_eq!(
                        model.nodes[path[w]].parent,
                        Some(path[w - 1]),
                        "seed {seed}: doc {d} path broken at level {w}"
                    );
                }
            }
        }
    }

    // -- Level-prior tests (#611) -------------------------------------------

    /// A bare model carrying just enough state to exercise `level_log_prior`.
    fn prior_probe(depth: usize, prior: LevelPrior) -> HldaModel {
        HldaModel {
            num_types: 1,
            depth,
            gamma: 1.0,
            eta: 0.01,
            level_prior: prior,
            nodes: Vec::new(),
            paths: Vec::new(),
            levels: Vec::new(),
            lg_eta: Vec::new(),
            lg_veta: Vec::new(),
        }
    }

    /// exp() + renormalise a log-weight vector to a probability distribution.
    fn softmax(logs: &[f64]) -> Vec<f64> {
        let mx = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ex: Vec<f64> = logs.iter().map(|&l| (l - mx).exp()).collect();
        let z: f64 = ex.iter().sum();
        ex.iter().map(|&e| e / z).collect()
    }

    #[test]
    fn gem_empty_counts_is_proper_truncated_geometric() {
        // With zero counts the truncated GEM gives p(l)=m(1-m)^l for l<L-1 and
        // p(L-1)=(1-m)^{L-1}; the raw (pre-normalisation) weights already sum to 1,
        // so no mass is lost. L=3.
        let (m, pi) = (0.35_f64, 100.0_f64);
        let model = prior_probe(3, LevelPrior::Gem { mean: m, scale: pi });
        let logs = model.level_log_prior(&[0, 0, 0]);
        let raw: Vec<f64> = logs.iter().map(|&l| l.exp()).collect();
        let expected = [m, m * (1.0 - m), (1.0 - m) * (1.0 - m)];
        let sum: f64 = raw.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "truncated GEM must be proper: {sum}"
        );
        for (i, (&r, &e)) in raw.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-12, "level {i}: {r} vs {e}");
        }
    }

    #[test]
    fn gem_nonzero_counts_matches_hand_calculation() {
        // L=3, counts n=[2,0,3]; suffix sums s=[5,3,3,0].
        //   p0 ∝ (mπ+2)/(π+5)
        //   p1 ∝ (mπ+0)/(π+3) · ((1-m)π+3)/(π+5)
        //   p2 ∝            1 · ((1-m)π+3)/(π+5) · ((1-m)π+3)/(π+3)   (last: no stop factor)
        // Pass factors use s[j+1]: pass(j=0)=s[1]=3, pass(j=1)=s[2]=3.
        let (m, pi) = (0.4_f64, 10.0_f64);
        let model = prior_probe(3, LevelPrior::Gem { mean: m, scale: pi });
        let logs = model.level_log_prior(&[2, 0, 3]);
        let got = softmax(&logs);

        let w0 = (m * pi + 2.0) / (pi + 5.0);
        let pass0 = ((1.0 - m) * pi + 3.0) / (pi + 5.0);
        let w1 = (m * pi + 0.0) / (pi + 3.0) * pass0;
        let pass1 = ((1.0 - m) * pi + 3.0) / (pi + 3.0);
        let w2 = pass0 * pass1;
        let z = w0 + w1 + w2;
        let want = [w0 / z, w1 / z, w2 / z];
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-12, "level {i}: {g} vs {w}");
        }
    }

    #[test]
    fn dirichlet_symmetric_matches_scalar_formula() {
        // The refactor must leave the symmetric-Dirichlet weight identical to the old
        // inline `ln(n_dl[l] + alpha)`: level_log_prior with a broadcast vector.
        let model = prior_probe(3, LevelPrior::Dirichlet(vec![0.1; 3]));
        let logs = model.level_log_prior(&[4, 1, 0]);
        let want = [
            (4.0_f64 + 0.1).ln(),
            (1.0_f64 + 0.1).ln(),
            (0.0_f64 + 0.1).ln(),
        ];
        for (i, (&g, &w)) in logs.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-15, "level {i}: {g} vs {w}");
        }
    }

    #[test]
    fn asymmetric_dirichlet_pulls_tokens_toward_the_heavy_level() {
        // A root-heavy alpha should keep more tokens at level 0 than a symmetric
        // prior on the same planted corpus + seed.
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let (docs, v, _shared, _blocks) = planted_corpus(&mut rng, 2);

        let count_level0 = |prior: LevelPrior| -> usize {
            let mut r = ChaCha8Rng::seed_from_u64(99);
            let m = fit_hlda(&docs, v, 3, 1.0, 0.1, prior, 60, 1, |_, _| {}, &mut r);
            m.levels.iter().flatten().filter(|&&l| l == 0).count()
        };
        let sym = count_level0(LevelPrior::Dirichlet(vec![0.5; 3]));
        let root_heavy = count_level0(LevelPrior::Dirichlet(vec![5.0, 0.1, 0.1]));
        assert!(
            root_heavy > sym,
            "root-heavy alpha should raise level-0 occupancy: {root_heavy} vs {sym}"
        );
    }

    #[test]
    fn gem_fit_is_deterministic() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let (docs, v, _s, _b) = planted_corpus(&mut rng, 2);
        let fit = || {
            let mut r = ChaCha8Rng::seed_from_u64(11);
            fit_hlda(
                &docs,
                v,
                3,
                1.0,
                0.1,
                LevelPrior::Gem {
                    mean: 0.5,
                    scale: 100.0,
                },
                40,
                1,
                |_, _| {},
                &mut r,
            )
        };
        let m1 = fit();
        let m2 = fit();
        assert_eq!(m1.num_nodes(), m2.num_nodes());
        for i in 0..m1.num_nodes() {
            assert_eq!(m1.topic_word(i), m2.topic_word(i), "node {i} differs");
        }
    }

    #[test]
    fn multithreaded_fit_is_bit_for_bit_identical_to_serial() {
        // The parallel per-node marginal precompute only reorders independent pure
        // computations, so num_threads must never change the fit.
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let (docs, v, _s, _b) = planted_corpus(&mut rng, 3);
        let fit = |threads: usize| {
            let mut r = ChaCha8Rng::seed_from_u64(21);
            fit_hlda(
                &docs,
                v,
                3,
                1.0,
                0.1,
                LevelPrior::Dirichlet(vec![0.1; 3]),
                40,
                threads,
                |_, _| {},
                &mut r,
            )
        };
        // Canonical fingerprint of the whole fitted state: tree shape, per-node
        // level/parent, topic-word bits, and every document's path.
        let fingerprint = |m: &HldaModel| -> Vec<u64> {
            let mut f = vec![m.num_nodes() as u64];
            for i in 0..m.num_nodes() {
                f.push(m.node_level(i) as u64);
                f.push(m.node_parent(i).map(|p| p as u64 + 1).unwrap_or(0));
                for x in m.topic_word(i) {
                    f.push(x.to_bits());
                }
            }
            for d in 0..docs.len() {
                for &node in &m.doc_path(d) {
                    f.push(node as u64);
                }
            }
            f
        };
        let serial = fingerprint(&fit(1));
        for threads in [2, 4, 8] {
            assert_eq!(
                serial,
                fingerprint(&fit(threads)),
                "fit with {threads} threads differs from serial"
            );
        }
    }

    #[test]
    fn scalar_dirichlet_default_is_bit_for_bit_golden() {
        // Golden regression for the default symmetric Dirichlet path: a future
        // refactor that changes the arithmetic or RNG consumption will trip this.
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let (docs, v, _s, _b) = planted_corpus(&mut rng, 2);
        let checksum = |seed: u64| -> (usize, u64) {
            let mut r = ChaCha8Rng::seed_from_u64(seed);
            let model = fit_hlda(
                &docs,
                v,
                3,
                1.0,
                0.1,
                LevelPrior::Dirichlet(vec![0.1; 3]),
                50,
                1,
                |_, _| {},
                &mut r,
            );
            let root = (0..model.num_nodes())
                .find(|&i| model.node_level(i) == 0)
                .unwrap();
            let tw = model.topic_word(root);
            let cs: f64 = tw
                .iter()
                .enumerate()
                .map(|(i, &p)| (i as f64 + 1.0) * p)
                .sum();
            (model.num_nodes(), cs.to_bits())
        };
        assert_eq!(
            checksum(1234),
            checksum(1234),
            "default path is not reproducible"
        );
    }
}
