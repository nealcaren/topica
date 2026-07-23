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
//! **Level prior.** We use a *symmetric* Dirichlet smoothing `alpha` over the L
//! levels (the per-document level-count Dirichlet of the prompt) rather than the
//! GEM stick-breaking prior. This keeps the level move conjugate and simple; the
//! trade-off is that it does not bias mass toward shallower levels the way GEM's
//! `(α_π, α_m)` stick does, but for a fixed shallow depth the symmetric prior
//! recovers the planted root/leaf split cleanly.
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

/// A fitted hierarchical LDA model. The tree has fixed depth `L`.
pub struct HldaModel {
    pub num_types: usize,
    pub depth: usize, // L
    pub gamma: f64,   // nested-CRP concentration
    pub eta: f64,     // topic-word Dirichlet
    pub alpha: f64,   // symmetric level (GEM-replacement) Dirichlet
    nodes: Vec<Node>,
    /// Per document: the L node indices on its path (root .. leaf).
    paths: Vec<Vec<usize>>,
    /// Per document, per token: the level 0..L it is assigned to.
    levels: Vec<Vec<usize>>,
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

            // p(level=l) ∝ (n_dl + alpha) * (nw + eta)/(n + V*eta) in log space.
            let mut logp = vec![0.0f64; l];
            for lev in 0..l {
                let node = path[lev];
                let prior = (n_dl[lev] as f64 + self.alpha).ln();
                let like = ((self.nodes[node].nw[w] as f64 + self.eta)
                    / (self.nodes[node].n as f64 + v as f64 * self.eta))
                    .ln();
                logp[lev] = prior + like;
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
    fn level_log_marginal(&self, node: usize, counts: &[(usize, u32)]) -> f64 {
        let v = self.num_types as f64;
        let eta = self.eta;
        let n_existing = self.nodes[node].n as f64;
        let mut m_total = 0u32;
        let mut logp = 0.0;
        for &(w, m) in counts {
            if m == 0 {
                continue;
            }
            let a = self.nodes[node].nw[w] as f64 + eta;
            // Π_{j=0}^{m-1} (a + j) = Γ(a+m)/Γ(a).
            logp += log_gamma(a + m as f64) - log_gamma(a);
            m_total += m;
        }
        if m_total == 0 {
            return 0.0;
        }
        // Denominator Γ(n+Vη)/Γ(n+Vη+m).
        logp += log_gamma(n_existing + v * eta) - log_gamma(n_existing + v * eta + m_total as f64);
        logp
    }

    /// Move (b): resample the whole path of document `d` via the nested CRP.
    fn sample_path<R: Rng>(&mut self, d: usize, doc: &[u32], rng: &mut R) {
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

        let mut log_scores = Vec::with_capacity(cands.len());
        for cand in &cands {
            let mut s = cand.log_prior;
            for lev in 0..l {
                if lev < cand.new_from {
                    let node = cand.nodes[lev];
                    s += self.level_log_marginal(node, &level_words[lev]);
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

    /// Delete subtrees whose root node has zero customers. A node with no
    /// customers has no tokens either (its counts were removed first), so it is
    /// safe to drop along with its (necessarily also-empty) descendants.
    fn delete_empty_subtrees(&mut self) {
        // Mark reachable-and-nonempty nodes; collect deletions of any node with
        // ndocs == 0 that is not the root.
        loop {
            let mut to_delete = None;
            for i in 0..self.nodes.len() {
                if self.nodes[i].level != 0 && self.nodes[i].ndocs == 0 {
                    to_delete = Some(i);
                    break;
                }
            }
            match to_delete {
                None => break,
                Some(i) => self.remove_node(i),
            }
        }
    }

    /// Remove a single (childless or empty-subtree) node `i`, detaching it from
    /// its parent. Assumes `i` has ndocs == 0; if it has children they must also
    /// be empty and will be removed on subsequent passes (we only remove leaves
    /// of the empty region here by requiring no children).
    fn remove_node(&mut self, i: usize) {
        // If it still has children, remove them first (they are empty too).
        let children = self.nodes[i].children.clone();
        for c in children {
            self.remove_node_subtree_member(c);
        }
        // Detach from parent.
        if let Some(p) = self.nodes[i].parent {
            self.nodes[p].children.retain(|&c| c != i);
        }
        self.swap_remove_node(i);
    }

    /// Recursively remove a node and its descendants (all empty) without touching
    /// the parent link cleanup of the top call.
    fn remove_node_subtree_member(&mut self, i: usize) {
        let children = self.nodes[i].children.clone();
        for c in children {
            self.remove_node_subtree_member(c);
        }
        if let Some(p) = self.nodes[i].parent {
            self.nodes[p].children.retain(|&c| c != i);
        }
        self.swap_remove_node(i);
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
/// the nested-CRP concentration, `eta` the topic-word Dirichlet, `alpha` the
/// symmetric per-document level Dirichlet. Deterministic for a fixed `rng`.
#[allow(clippy::too_many_arguments)]
pub fn fit_hlda<R: Rng>(
    docs: &[Vec<u32>],
    num_types: usize,
    depth: usize,
    gamma: f64,
    eta: f64,
    alpha: f64,
    iters: usize,
    rng: &mut R,
) -> HldaModel {
    assert!(depth >= 2, "hLDA needs depth >= 2");
    let mut model = HldaModel {
        num_types,
        depth,
        gamma,
        eta,
        alpha,
        nodes: Vec::new(),
        paths: vec![Vec::new(); docs.len()],
        levels: docs.iter().map(|d| vec![0usize; d.len()]).collect(),
    };

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

    // Gibbs sweeps: per document, resample its path, then its token levels.
    for _ in 0..iters {
        for (d, doc) in docs.iter().enumerate() {
            model.sample_path(d, doc, rng);
            model.sample_levels(d, doc, rng);
        }
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

        let model = fit_hlda(&docs, v, 2, 1.0, 0.1, 0.5, 80, &mut rng);

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
    fn deterministic_for_fixed_seed() {
        let mut seed_rng = ChaCha8Rng::seed_from_u64(7);
        let (docs, v, _shared, _blocks) = planted_corpus(&mut seed_rng, 2);

        let mut r1 = ChaCha8Rng::seed_from_u64(123);
        let mut r2 = ChaCha8Rng::seed_from_u64(123);
        let m1 = fit_hlda(&docs, v, 2, 1.0, 0.1, 0.5, 30, &mut r1);
        let m2 = fit_hlda(&docs, v, 2, 1.0, 0.1, 0.5, 30, &mut r2);

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
        let model = fit_hlda(&docs, v, 2, 1.0, 0.1, 0.5, 80, &mut rng);
        let base = crate::conformance::check_conformance(&model);
        assert!(base.is_empty(), "check_conformance: {:?}", base);
    }
}
