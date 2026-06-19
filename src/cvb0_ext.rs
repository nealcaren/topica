//! `Cvb0::to_topic_model`, re-added as an extension trait.
//!
//! The CVB0 sampler ([`crate::cvb0::Cvb0`]) lives in `topica-core`, but
//! [`TopicModel`] stays in `topica` (it pulls in the rest of the sampler/model
//! machinery), so packing a CVB0 fit into a `TopicModel` is implemented here.
//! `topica-core` exposes the only piece that needs CVB0 internals,
//! [`Cvb0::map_topic_assignments`]; this trait builds the `TopicModel` from it so
//! the rest of the codebase (coherence, save/load, held-out inference) is reused
//! unchanged. Bring it into scope with `use crate::cvb0_ext::Cvb0ToModel;`.

use crate::corpus::Corpus;
use crate::cvb0::Cvb0;
use crate::model::TopicModel;

/// Pack a fitted CVB0 model into a [`TopicModel`] via the MAP hard assignment.
pub trait Cvb0ToModel {
    fn to_topic_model(&self, corpus: &Corpus) -> TopicModel;
}

impl Cvb0ToModel for Cvb0 {
    fn to_topic_model(&self, corpus: &Corpus) -> TopicModel {
        let mut model =
            TopicModel::new(self.num_topics, self.alpha_sum, self.beta, self.num_types);
        model.alpha.copy_from_slice(&self.alpha);
        model.alpha_sum = self.alpha_sum;
        model.beta = self.beta;
        model.beta_sum = self.beta_sum;
        model.initialize_from_assignments(corpus, self.map_topic_assignments(corpus));
        model
    }
}
