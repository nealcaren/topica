# Replications

A model is only as trustworthy as the published results it can reproduce. This
section takes a peer-reviewed paper, refits its analysis with topica on the
authors' own data, and reports how closely the output matches. The goal is not a
new finding but a verifiable one: if topica recovers the same topics, the same
classification performance, and the same substantive trends as the reference
implementation, you can use it in their place and expect their results.

Each replication states the source, the data, the exact model call, and a
side-by-side comparison with the paper's published tables and figures. Where we
diverge, we say so.

- [Structural Topic Model: the `stm` vignette](stm.md) (Roberts, Stewart & Tingley)
- [Dynamic keyATM: U.S. Supreme Court opinions](keyatm-dynamic.md) (Eshima, Imai & Sasaki 2024)
- [TensorLDA: implementation validation](tlda.md) (Kangaslahti et al. 2026)
