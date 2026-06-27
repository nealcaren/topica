# Citing

## Citing topica

If topica contributes to published work, please cite it:

```bibtex
@software{caren_topica,
  author = {Caren, Neal},
  title  = {topica: fast, all-purpose topic modeling for Python},
  year   = {2026},
  url    = {https://github.com/nealcaren/topica}
}
```

topica reimplements published methods and is validated against their reference
implementations. It does not replace the original work: please also cite the
model you use, and, where you rely on a feature ported from a specific package
(for example `estimateEffect` from `stm`), that package too.

## Cite the model you used

| Model | Primary citation |
|-------|------------------|
| `LDA` | Blei, Ng & Jordan (2003); fast sampler: Yao, Mimno & McCallum (2009) |
| `DMR` | Mimno & McCallum (2008) |
| `LabeledLDA` | Ramage, Hall, Nallapati & Manning (2009) |
| `CTM` | Blei & Lafferty (2007) |
| `STM` | Roberts, Stewart & Tingley (2014, 2019) |
| `SAGE` | Eisenstein, Ahmed & Xing (2011) |
| `HDP` | Teh, Jordan, Beal & Blei (2006) |
| `DTM` | Blei & Lafferty (2006) |
| `SupervisedLDA` | Blei & McAuliffe (2008) |
| `PT` | Zuo et al. (2016) |
| `GSDMM` | Yin & Wang (2014) |
| `SeededLDA` | method: Jagarlamudi, Daumé III & Udupa (2012); package: Watanabe (2023) |
| `KeyATM` | Eshima, Imai & Sasaki (2024) |
| `PA` | Li & McCallum (2006) |
| `HLDA` | Griffiths, Jordan, Tenenbaum & Blei (2004) |
| `LightLDA` (alias sampler) | Yuan et al. (2015) |
| `BERTopic` | Grootendorst (2022) |
| `Top2Vec` | Angelov (2020) |
| `ETM` | Dieng, Ruiz & Blei (2020) |
| `ProdLDA` | Srivastava & Sutton (2017) |
| `FASTopic` | Wu et al. (2024) |
| `Wordfish` | Slapin & Proksch (2008) |
| `TBIP` | Vafa, Naidu & Blei (2020) |

## Full references

### Models

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993–1022.
- Yao, L., Mimno, D., & McCallum, A. (2009). Efficient methods for topic model inference on streaming document collections. *KDD 2009*, 937–946. [doi:10.1145/1557019.1557121](https://doi.org/10.1145/1557019.1557121) (SparseLDA)
- Mimno, D., & McCallum, A. (2008). Topic models conditioned on arbitrary features with Dirichlet-multinomial regression. *UAI 2008*.
- Ramage, D., Hall, D., Nallapati, R., & Manning, C. D. (2009). Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora. *EMNLP 2009*, 248–256. [doi:10.3115/1699510.1699543](https://doi.org/10.3115/1699510.1699543)
- Blei, D. M., & Lafferty, J. D. (2007). A correlated topic model of *Science*. *The Annals of Applied Statistics*, 1(1), 17–35. [doi:10.1214/07-AOAS114](https://doi.org/10.1214/07-AOAS114)
- Roberts, M. E., Stewart, B. M., & Airoldi, E. M. (2016). A model of text for experimentation in the social sciences. *Journal of the American Statistical Association*, 111(515), 988–1003. [doi:10.1080/01621459.2016.1141684](https://doi.org/10.1080/01621459.2016.1141684)
- Roberts, M. E., Stewart, B. M., & Tingley, D. (2019). stm: An R package for structural topic models. *Journal of Statistical Software*, 91(2), 1–40. [doi:10.18637/jss.v091.i02](https://doi.org/10.18637/jss.v091.i02)
- Eisenstein, J., Ahmed, A., & Xing, E. P. (2011). Sparse additive generative models of text. *ICML 2011*, 1041–1048.
- Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006). Hierarchical Dirichlet processes. *Journal of the American Statistical Association*, 101(476), 1566–1581. [doi:10.1198/016214506000000302](https://doi.org/10.1198/016214506000000302)
- Blei, D. M., & Lafferty, J. D. (2006). Dynamic topic models. *ICML 2006*, 113–120. [doi:10.1145/1143844.1143859](https://doi.org/10.1145/1143844.1143859)
- Blei, D. M., & McAuliffe, J. D. (2008). Supervised topic models. *NIPS 2007 (Advances in Neural Information Processing Systems 20)*, 121–128.
- Zuo, Y., Wu, J., Zhang, H., Lin, H., Wang, F., Xu, K., & Xiong, H. (2016). Topic modeling of short texts: A pseudo-document view. *KDD 2016*, 2105–2114. [doi:10.1145/2939672.2939880](https://doi.org/10.1145/2939672.2939880)
- Yin, J., & Wang, J. (2014). A Dirichlet multinomial mixture model-based approach for short text clustering. *KDD 2014*, 233–242. [doi:10.1145/2623330.2623715](https://doi.org/10.1145/2623330.2623715) (GSDMM)
- Jagarlamudi, J., Daumé III, H., & Udupa, R. (2012). Incorporating lexical priors into topic models. *EACL 2012*, 204–213. (seeded-LDA method)
- Watanabe, K. (2023). seededlda: Seeded sequential LDA for topic modeling. R package.
- Eshima, S., Imai, K., & Sasaki, T. (2024). Keyword-assisted topic models. *American Journal of Political Science*, 68(2), 730–750. [doi:10.1111/ajps.12779](https://doi.org/10.1111/ajps.12779)
- Li, W., & McCallum, A. (2006). Pachinko allocation: DAG-structured mixture models of topic correlations. *ICML 2006*, 577–584.
- Griffiths, T. L., Jordan, M. I., Tenenbaum, J. B., & Blei, D. M. (2004). Hierarchical topic models and the nested Chinese restaurant process. *NIPS 2003 (Advances in Neural Information Processing Systems 16)*, 17–24.
- Yuan, J., Gao, F., Ho, Q., Dai, W., Wei, J., Zheng, X., Xing, E. P., Liu, T.-Y., & Ma, W.-Y. (2015). LightLDA: Big topic models on modest computer clusters. *WWW 2015*, 1351–1361. [doi:10.1145/2736277.2741115](https://doi.org/10.1145/2736277.2741115)
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. [arXiv:2203.05794](https://arxiv.org/abs/2203.05794).
- Angelov, D. (2020). Top2Vec: Distributed representations of topics. [arXiv:2008.09470](https://arxiv.org/abs/2008.09470).
- Dieng, A. B., Ruiz, F. J. R., & Blei, D. M. (2020). Topic modeling in embedding spaces. *Transactions of the Association for Computational Linguistics*, 8, 439–453. [doi:10.1162/tacl_a_00325](https://doi.org/10.1162/tacl_a_00325)
- Srivastava, A., & Sutton, C. (2017). Autoencoding variational inference for topic models. *ICLR 2017*. [arXiv:1703.01488](https://arxiv.org/abs/1703.01488) (ProdLDA / AVITM)
- Wu, X., Nguyen, T., Zhang, D. C., Wang, W. Y., & Luu, A. T. (2024). FASTopic: A fast, adaptive, stable, and transferable topic modeling paradigm. *NeurIPS 2024*. [arXiv:2405.17978](https://arxiv.org/abs/2405.17978)
- Slapin, J. B., & Proksch, S.-O. (2008). A scaling model for estimating time-series party positions from texts. *American Journal of Political Science*, 52(3), 705–722. [doi:10.1111/j.1540-5907.2008.00338.x](https://doi.org/10.1111/j.1540-5907.2008.00338.x) (Wordfish)
- Vafa, K., Naidu, S., & Blei, D. M. (2020). Text-based ideal points. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)*, 5345–5357. [doi:10.18653/v1/2020.acl-main.475](https://doi.org/10.18653/v1/2020.acl-main.475) (TBIP)

### Methods used in the diagnostics and effects tools

- Chib, S. (1998). Estimation and comparison of multiple change-point models. *Journal of Econometrics*, 86(2), 221–241. [doi:10.1016/S0304-4076(97)00115-2](https://doi.org/10.1016/S0304-4076(97)00115-2) (dynamic keyATM HMM)
- Lau, J. H., Newman, D., & Baldwin, T. (2014). Machine reading tea leaves: Automatically evaluating topic coherence and topic model quality. *EACL 2014*, 530–539. [doi:10.3115/v1/E14-1056](https://doi.org/10.3115/v1/E14-1056) (coherence)
- Sievert, C., & Shirley, K. E. (2014). LDAvis: A method for visualizing and interpreting topics. *Workshop on Interactive Language Learning, Visualization, and Interfaces (ACL 2014)*, 63–70. [doi:10.3115/v1/W14-3110](https://doi.org/10.3115/v1/W14-3110) (relevance)
- Chang, J., Gerrish, S., Wang, C., Boyd-Graber, J., & Blei, D. M. (2009). Reading tea leaves: How humans interpret topic models. *NIPS 2009 (Advances in Neural Information Processing Systems 22)*, 288–296. (intrusion tests)
- Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' words: Lexical feature selection and evaluation for identifying the content of political conflict. *Political Analysis*, 16(4), 372–403. [doi:10.1093/pan/mpn018](https://doi.org/10.1093/pan/mpn018) (fighting words)
- Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. *PAKDD 2013*, 160–172. [doi:10.1007/978-3-642-37456-2_14](https://doi.org/10.1007/978-3-642-37456-2_14) (HDBSCAN)
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. [arXiv:1802.03426](https://arxiv.org/abs/1802.03426).

## Reference implementations and libraries

topica is validated against, and in places ports features from, these projects:

- [MALLET](https://github.com/mimno/Mallet) (McCallum, 2002) and [RustMallet](https://github.com/mimno/RustMallet) (Mimno)
- [stm](https://github.com/bstewart/stm) (Roberts, Stewart & Tingley)
- [keyATM](https://github.com/keyATM/keyATM) (Eshima, Imai & Sasaki)
- [seededlda](https://github.com/koheiw/seededlda) (Watanabe)
- [lda-c / ctm-c / dtm / hdp](https://github.com/blei-lab) (Blei lab)
- [gensim](https://github.com/piskvorky/gensim) (Řehůřek & Sojka, 2010)
- [tomotopy](https://github.com/bab2min/tomotopy) (bab2min)
- [BERTopic](https://github.com/MaartenGr/BERTopic) (Grootendorst) and [Top2Vec](https://github.com/ddangelov/Top2Vec) (Angelov)
- [ETM](https://github.com/adjidieng/ETM) (Dieng, Ruiz & Blei) and [FASTopic](https://github.com/BobXWu/FASTopic) (Wu et al.)
- [quanteda.textmodels](https://github.com/quanteda/quanteda.textmodels) (Benoit et al.) — Wordfish (`textmodel_wordfish`)
- [tbip](https://github.com/keyonvafa/tbip) (Vafa, Naidu & Blei) — TBIP (TensorFlow reference)
- [petal-clustering](https://github.com/petabi/petal-clustering) (HDBSCAN) and [umap-rs](https://github.com/wilsonzlin/umap-rs) (UMAP), both pure-Rust and BLAS-free
