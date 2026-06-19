//! Rust-side estimator conformance — the analog of `python/topica/conformance.py`.
//! Catches contract gaps in `cargo test --lib`, at the source, before the Python
//! layer or a release. The full multi-model registry and EXEMPT set are populated
//! in a later step; this file establishes the check.

use crate::estimator::{DirichletModel, Estimator, ModelFamily};
use crate::variational::LogisticNormalModel;

/// Check a fitted model against the Tier-0 contract and its family's Tier-2
/// requirements. Returns a list of violation messages (empty = conforming).
///
/// Note: logistic-normal Tier-2 (`eta_mean`/`eta_cov`) is checked through the
/// `LogisticNormalModel` trait by the typed helper [`check_logistic_normal`],
/// since `&dyn Estimator` cannot be downcast to it; a registered logistic-normal
/// model is wired through that helper in the conformance test.
#[cfg_attr(not(test), allow(dead_code))] // test-only contract checker; live under `cargo test`
pub fn check_conformance(m: &dyn Estimator) -> Vec<String> {
    let mut v = Vec::new();
    let k = m.num_topics();
    let tw = m.topic_word();
    if tw.len() != k {
        v.push(format!(
            "topic_word has {} rows but num_topics is {}",
            tw.len(),
            k
        ));
    }
    if matches!(
        m.model_family(),
        ModelFamily::Dirichlet | ModelFamily::LogisticNormal
    ) {
        let theta = m.doc_topic();
        for (d, row) in theta.iter().enumerate() {
            let s: f64 = row.iter().sum();
            if (s - 1.0).abs() > 1e-6 {
                v.push(format!("doc_topic row {d} sums to {s}, expected 1"));
                break;
            }
        }
    }
    v
}

/// Tier-2 shape check for Dirichlet (collapsed-Gibbs) models: alpha length,
/// doc_lengths length, and theta_draws nesting.
#[cfg_attr(not(test), allow(dead_code))] // test-only contract checker; live under `cargo test`
pub fn check_dirichlet(m: &dyn DirichletModel) -> Vec<String> {
    let mut v = Vec::new();
    let k = m.num_topics();
    let d = m.doc_topic().len();
    if m.alpha().len() != k {
        v.push(format!(
            "alpha has len {} but num_topics is {}",
            m.alpha().len(),
            k
        ));
    }
    if m.doc_lengths().len() != d {
        v.push(format!(
            "doc_lengths has len {} but doc_topic has {} rows",
            m.doc_lengths().len(),
            d
        ));
    }
    let draws = m.theta_draws();
    if !draws.is_empty() {
        for (s, draw) in draws.iter().enumerate() {
            if draw.len() != d {
                v.push(format!(
                    "theta_draws[{s}] has {} rows but doc_topic has {d}",
                    draw.len()
                ));
                break;
            }
            for (di, row) in draw.iter().enumerate() {
                if row.len() != k {
                    v.push(format!(
                        "theta_draws[{s}][{di}] has len {} but num_topics is {k}",
                        row.len()
                    ));
                    break;
                }
            }
        }
    }
    v
}

/// Tier-2 shape check for logistic-normal models: eta_mean is (D, eta_dim) and
/// eta_cov is (D, eta_dim²) with a consistent eta_dim.
#[cfg_attr(not(test), allow(dead_code))] // test-only contract checker; live under `cargo test`
pub fn check_logistic_normal(m: &dyn LogisticNormalModel) -> Vec<String> {
    let mut v = Vec::new();
    let dim = m.eta_dim();
    for (d, row) in m.eta_mean().iter().enumerate() {
        if row.len() != dim {
            v.push(format!(
                "eta_mean row {d} has len {} != eta_dim {dim}",
                row.len()
            ));
            break;
        }
    }
    for (d, row) in m.eta_cov().iter().enumerate() {
        if row.len() != dim * dim {
            v.push(format!(
                "eta_cov row {d} has len {} != eta_dim² {}",
                row.len(),
                dim * dim
            ));
            break;
        }
    }
    v
}

// ---------------------------------------------------------------------------
// Registry — the Rust-side mirror of python/topica/conformance.py
// ---------------------------------------------------------------------------

/// One row of the Rust estimator registry: a model's user-facing name (matching
/// the Python `REGISTRY` in `python/topica/conformance.py`), the inference family
/// its fitted struct reports from `Estimator::model_family`, and any Tier-0
/// Estimator method it structurally cannot provide — returned empty by design,
/// the Rust analog of `conformance.py`'s `EXEMPT`.
#[cfg_attr(not(test), allow(dead_code))] // test-only contract checker; live under `cargo test`
pub struct RegistryEntry {
    pub name: &'static str,
    pub family: ModelFamily,
    /// Estimator methods this model legitimately returns empty for (structural,
    /// not a gap): e.g. a time-sliced or tree model has no flat `doc_topic`.
    pub exempt: &'static [&'static str],
}

/// Every fitted struct that implements [`Estimator`], with its family. Mirrors
/// the Python `REGISTRY`/`EXEMPT` so the two stay in lockstep. Enforcement is
/// per-struct: each model's own `#[cfg(test)]` module has a `*_conforms` test
/// that fits a small instance and asserts `check_conformance` (and, by family,
/// `check_dirichlet`/`check_logistic_normal`) returns no violations. This table
/// is the single place that documents which struct belongs to which family and
/// which Tier-0 surfaces are structurally exempt.
///
/// `STM`/`CTM` and the `LDA`/`DMR`/`LabeledLDA` group share one backing struct
/// (`CtmModel`, `TopicModel`) but are listed under their user-facing names to
/// match the Python registry. The embedding-cluster models (`BERTopic`,
/// `Top2Vec`) carry a `topic_word`/`doc_topic` from their Rust struct and so
/// participate here.
#[cfg_attr(not(test), allow(dead_code))] // test-only contract checker; live under `cargo test`
pub const RUST_ESTIMATORS: &[RegistryEntry] = &[
    // Logistic-normal variational (eta posterior).
    RegistryEntry {
        name: "STM",
        family: ModelFamily::LogisticNormal,
        exempt: &[],
    },
    RegistryEntry {
        name: "CTM",
        family: ModelFamily::LogisticNormal,
        exempt: &[],
    },
    RegistryEntry {
        name: "STS",
        family: ModelFamily::LogisticNormal,
        exempt: &[],
    },
    RegistryEntry {
        name: "ECTM",
        family: ModelFamily::LogisticNormal,
        exempt: &[],
    },
    // Collapsed-Gibbs / Dirichlet doc-topic posterior.
    RegistryEntry {
        name: "LDA",
        family: ModelFamily::Dirichlet,
        exempt: &[],
    },
    RegistryEntry {
        name: "DMR",
        family: ModelFamily::Dirichlet,
        exempt: &[],
    },
    // GDMR shares the `TopicModel` struct with LDA/DMR, so it is covered by the
    // same `topicmodel_conforms` test; listed here under its user-facing name.
    RegistryEntry {
        name: "GDMR",
        family: ModelFamily::Dirichlet,
        exempt: &[],
    },
    RegistryEntry {
        name: "LabeledLDA",
        family: ModelFamily::Dirichlet,
        exempt: &[],
    },
    RegistryEntry {
        name: "SeededLDA",
        family: ModelFamily::Dirichlet,
        exempt: &[],
    },
    RegistryEntry {
        name: "KeyATM",
        family: ModelFamily::Dirichlet,
        exempt: &[],
    },
    RegistryEntry {
        name: "PA",
        family: ModelFamily::Dirichlet,
        exempt: &[],
    },
    RegistryEntry {
        name: "PT",
        family: ModelFamily::Dirichlet,
        exempt: &[],
    },
    RegistryEntry {
        name: "SAGE",
        family: ModelFamily::Dirichlet,
        exempt: &[],
    },
    RegistryEntry {
        name: "HDP",
        family: ModelFamily::Dirichlet,
        exempt: &[],
    },
    RegistryEntry {
        name: "SupervisedLDA",
        family: ModelFamily::Dirichlet,
        exempt: &[],
    },
    // Neural / embedding / nonparametric — no theta posterior.
    RegistryEntry {
        name: "ProdLDA",
        family: ModelFamily::None_,
        exempt: &[],
    },
    // Matrix factorization — signed latent coordinates, no theta posterior.
    RegistryEntry {
        name: "NMF",
        family: ModelFamily::None_,
        exempt: &[],
    },
    // LSA/LSI: doc_topic is signed coordinates (U Sigma), not a (D,K) simplex;
    // the SVD is a direct solve, so there is no fit_history trajectory.
    RegistryEntry {
        name: "LSA",
        family: ModelFamily::None_,
        exempt: &["fit_history"],
    },
    RegistryEntry {
        name: "ETM",
        family: ModelFamily::None_,
        exempt: &[],
    },
    RegistryEntry {
        name: "DETM",
        family: ModelFamily::None_,
        exempt: &[],
    },
    RegistryEntry {
        name: "FASTopic",
        family: ModelFamily::None_,
        exempt: &[],
    },
    RegistryEntry {
        name: "GSDMM",
        family: ModelFamily::None_,
        exempt: &[],
    },
    RegistryEntry {
        name: "BERTopic",
        family: ModelFamily::None_,
        exempt: &[],
    },
    RegistryEntry {
        name: "Top2Vec",
        family: ModelFamily::None_,
        exempt: &[],
    },
    // Time-sliced and tree-structured — flat doc_topic is structurally undefined.
    RegistryEntry {
        name: "DTM",
        family: ModelFamily::None_,
        exempt: &["doc_topic", "fit_history"],
    },
    RegistryEntry {
        name: "HLDA",
        family: ModelFamily::None_,
        exempt: &["doc_topic"],
    },
];

#[cfg(test)]
mod registry_tests {
    use super::*;

    /// The registry is well-formed: unique names, and every exempt entry names a
    /// real Estimator method. Catches a typo or a duplicate when a model is added.
    #[test]
    fn registry_is_well_formed() {
        const METHODS: &[&str] = &[
            "num_topics",
            "topic_word",
            "doc_topic",
            "fit_history",
            "converged",
        ];
        let mut seen = std::collections::HashSet::new();
        for e in RUST_ESTIMATORS {
            assert!(seen.insert(e.name), "duplicate registry entry: {}", e.name);
            for &req in e.exempt {
                assert!(
                    METHODS.contains(&req),
                    "{}: unknown exempt method {req:?}",
                    e.name
                );
            }
        }
        // Mirror of the Python REGISTRY size (user-facing models with an
        // Estimator-backed Rust struct).
        assert_eq!(
            RUST_ESTIMATORS.len(),
            26,
            "registry size drifted from the Python REGISTRY"
        );
    }
}

#[cfg(test)]
mod ctm_conformance_tests {
    // Relocated from ctm.rs when CTM moved to the `topica-core` member: this test
    // exercises `topica`'s conformance checks against a `topica_core` CTM fit.
    use crate::conformance::{check_conformance, check_logistic_normal};
    use crate::ctm::{fit_ctm, GammaPrior};
    use crate::variational::LogisticNormalModel;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn ctm_conforms() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let docs: Vec<Vec<u32>> = vec![
            vec![0, 1, 0, 1, 2],
            vec![1, 2, 1, 2, 0],
            vec![2, 0, 2, 0, 1],
            vec![0, 0, 1, 2, 0],
            vec![1, 1, 2, 0, 1],
            vec![2, 2, 0, 1, 2],
        ];
        let model = fit_ctm(
            &docs,
            3,
            3,
            5,
            0.0,
            0.0,
            None,
            None,
            false,
            None,
            GammaPrior::Pooled,
            true,
            false,
            &mut rng,
        );

        let base_violations = check_conformance(&model);
        assert!(
            base_violations.is_empty(),
            "check_conformance violations: {:?}",
            base_violations
        );

        let ln_violations = check_logistic_normal(&model);
        assert!(
            ln_violations.is_empty(),
            "check_logistic_normal violations: {:?}",
            ln_violations
        );

        assert_eq!(
            model.eta_dim(),
            model.num_topics - 1,
            "eta_dim should be num_topics - 1"
        );
        assert_eq!(
            model.eta_mean().len(),
            model.eta_cov().len(),
            "eta_mean and eta_cov should have the same number of rows (one per document)"
        );
    }
}
