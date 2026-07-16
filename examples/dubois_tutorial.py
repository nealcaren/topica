"""Topics in W.E.B. Du Bois's *Crisis* writing — an end-to-end topica tour.

This tutorial works the topica topic-modeling library through a real
corpus: 704 articles from *The Crisis*, the NAACP magazine Du Bois edited from
its founding in 1910 until he left in 1934. The articles span 1910-1934 (the
great bulk are Du Bois's own editorials), so the corpus lets us ask not only
*what* Du Bois wrote about but *how those topics shifted across his editorship*
— from the agitation and WWI of the 1910s, through the New Negro / Harlem
Renaissance 1920s, into the economic-program early 1930s.

The data file `examples/dubois_crisis.csv` is produced by the corpus builder
that ships alongside this script; columns are
``title,year,decade,volume,issue,author,subjects,text``.

We move through topica's models in the order you would in real research:

    1. Preprocess  : tokenize + stoplist, build a pruned ``Corpus``.
    2. Phrases     : learn collocations ("jim crow", "colored people") first.
    3. LDA         : a plain topic model, scored with coherence + diversity.
    4. STM         : prevalence on decade — which topics rise/fall over time.
    5. DTM         : dynamic topics — trace a word's trajectory across decades.
    6. HDP         : let the model infer how many topics there are.
    7. Guided      : seed named topics with KeyATM when you know the themes.
    8. Utilities   : summary(), save/load, prepare_pyldavis().

Everything here is tuned to run in a couple of minutes. Where a knob trades
speed for quality (K, iterations, EM sweeps) we use a modest value and note in
a comment how to scale it up for a real analysis.

Run:  .venv-dev/bin/python examples/dubois_tutorial.py
"""

import csv
import os
import tempfile

import numpy as np

import topica
from topica.models import DTM, HDP, LDA, STM
from topica import Corpus, stm, tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "dubois_crisis.csv")
STOP_PATH = os.path.join(HERE, "english-stoplist.txt")

# Corpus-specific stopwords. "crisis"/"negro"/"colored" are ubiquitous in this
# corpus (it IS *The Crisis*, about "the Negro") and would otherwise dominate
# every topic; the rest are high-frequency function-ish words that survive the
# English stoplist. Pruning these sharpens every model below.
EXTRA_STOPS = [
    "crisis", "negro", "negroes", "colored", "mr", "mrs", "dr",
    "shall", "men", "man", "upon", "us", "one", "two", "every",
    "may", "must", "said", "yet", "thing", "things", "make", "made",
]


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_corpus_rows():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["year"] = int(r["year"])
        r["decade"] = int(r["decade"])
    return rows


def main():
    banner("Du Bois in *The Crisis* — a topica walkthrough")

    # ------------------------------------------------------------------ #
    # 0. Load the corpus.
    # ------------------------------------------------------------------ #
    rows = load_corpus_rows()
    years = [r["year"] for r in rows]
    dubois = sum(1 for r in rows if "Du Bois" in r["author"])
    print(f"Loaded {len(rows)} articles, {min(years)}-{max(years)}.")
    print(f"  Du Bois-authored: {dubois}  ({100 * dubois / len(rows):.0f}%)")
    by_decade = {}
    for r in rows:
        by_decade[r["decade"]] = by_decade.get(r["decade"], 0) + 1
    print("  Articles per decade: "
          + ", ".join(f"{d}s={n}" for d, n in sorted(by_decade.items())))

    # ------------------------------------------------------------------ #
    # 1. PREPROCESS.  topica.tokenize does the lowercase / regex /
    #    stoplist / min-length work the corpus loader uses internally. We
    #    keep the per-document token lists around (`docs`) because several
    #    helpers (coherence, phrases, pyLDAvis) want the tokenized texts.
    #
    #    NOTE: `stopwords` must be a *sequence* (list), not a set.
    # ------------------------------------------------------------------ #
    banner("[1] Preprocess: tokenize + build a pruned Corpus")
    english_stops = open(STOP_PATH, encoding="utf-8").read().split()
    stopwords = sorted(set(english_stops) | set(EXTRA_STOPS))
    print(f"Stoplist: {len(english_stops)} English + "
          f"{len(EXTRA_STOPS)} corpus-specific = {len(stopwords)} words.")

    # min_length=3 drops 1-2 char OCR noise.
    docs = [tokenize(r["text"], stopwords=stopwords, min_length=3) for r in rows]
    raw_vocab = len({w for d in docs for w in d})
    print(f"Tokenized {len(docs)} documents -> {raw_vocab} raw word types.")

    # Build a Corpus, pruning the vocabulary tomotopy-style:
    #   min_doc_freq=10  -> a word must appear in >=10 documents (kills hapaxes)
    #   rm_top=20        -> drop the 20 most frequent words (residual stopwords)
    corpus = Corpus.from_documents(docs, min_doc_freq=10, rm_top=20)
    print(f"Pruned Corpus: {corpus.num_docs} docs, vocab={corpus.num_words}, "
          f"{corpus.total_tokens} tokens after pruning.")

    # ------------------------------------------------------------------ #
    # 2. PHRASES.  Du Bois's prose is full of fixed expressions ("jim crow",
    #    "colored people", "souls of black folk"). Detecting them BEFORE
    #    modeling means a topic can be about *the phrase* rather than its
    #    parts scattered across topics. learn_phrases scores adjacent word
    #    pairs; apply_phrases rewrites the token lists, joining survivors
    #    with "_". We then rebuild the Corpus from the phrased documents.
    # ------------------------------------------------------------------ #
    banner("[2] Phrases: detect collocations before modeling")
    phrase_model = topica.learn_phrases(docs, min_count=8, threshold=12.0)
    phrased_docs = topica.apply_phrases(docs, phrase_model)
    detected = sorted({w for d in phrased_docs for w in d if "_" in w})
    print(f"Detected {len(detected)} multiword phrases. A sample:")
    # surface a few historically resonant ones if present, else the first 12
    wanted = ["jim_crow", "colored_people", "booker_washington",
              "civil_rights", "abraham_lincoln", "atlanta_university"]
    show = [p for p in wanted if p in detected] or detected[:6]
    show += [p for p in detected if p not in show][:8]
    for p in show[:12]:
        print(f"    {p.replace('_', ' ')}")

    # All later models train on the *phrased* corpus.
    pcorpus = Corpus.from_documents(phrased_docs, min_doc_freq=10, rm_top=20)
    print(f"Phrased Corpus vocab: {pcorpus.num_words}.")

    # ------------------------------------------------------------------ #
    # 3. LDA.  The workhorse. We fit K=15 topics with collapsed Gibbs
    #    sampling, then read each topic off as its top words, and score the
    #    model two ways:
    #      coherence(c_v)  -> do a topic's top words actually co-occur?
    #      topic_diversity -> do topics avoid recycling the same words?
    #    (Scale up: K up to ~40 and iterations to ~1000 for a real run.)
    # ------------------------------------------------------------------ #
    banner("[3] LDA: K=15 topics over the whole corpus")
    K = 15
    lda = LDA(num_topics=K, seed=1)
    # progress callback prints likelihood as it samples.
    lda.fit(pcorpus, iters=400, num_samples=4, sample_interval=25)
    print(f"Fit LDA(K={K}). Top words per topic:")
    for t in range(K):
        words = [w for w, _ in lda.top_words(8, topic=t)]
        print(f"  T{t:2d}: {', '.join(w.replace('_', ' ') for w in words)}")

    # coherence/topic_diversity accept the fitted model directly; coherence
    # needs the reference texts (our phrased documents).
    coh = topica.coherence(lda, phrased_docs, coherence_type="c_v", topn=10)
    div = topica.topic_diversity(lda, topn=15)
    print(f"\nMean c_v coherence: {coh.mean():.3f}  "
          f"(best topic {coh.argmax()} = {coh.max():.3f}, "
          f"worst topic {coh.argmin()} = {coh.min():.3f})")
    print(f"Topic diversity (top-15): {div:.3f}  "
          f"(1.0 = every top word unique to its topic)")

    # ------------------------------------------------------------------ #
    # 4. STM.  The Structural Topic Model lets topic *prevalence* depend on
    #    document metadata. We make prevalence a function of DECADE (one-hot
    #    dummies via topica.one_hot), so the model can express that, say,
    #    a "Harlem Renaissance" topic is more common in the 1920s.
    #
    #    To ask "which topics rose or fell over time" with proper uncertainty,
    #    we use the method of composition (Treier & Jackman 2008), exactly as
    #    the R `stm` package does:
    #      posterior_theta_samples -> draws of the doc-topic matrix theta
    #      estimate_effect(draws, X=year) -> regress each topic on year,
    #          pooling the draws by Rubin's rules so the SEs include
    #          topic-estimation uncertainty.
    #    A positive, significant `year` slope = the topic grew over time.
    # ------------------------------------------------------------------ #
    banner("[4] STM: prevalence ~ decade, then topic trends over time")
    Ks = 12
    X_decade, decade_names = topica.one_hot([r["decade"] for r in rows], prefix="dec")
    print(f"Prevalence design: {X_decade.shape[1]} decade dummies "
          f"({', '.join(decade_names)}); reference decade dropped.")
    stm_model = STM(num_topics=Ks, seed=1)
    # em_iters modest for runtime; ~75-100 for a real fit.
    stm_model.fit(phrased_docs, X_decade,
                  prevalence_names=decade_names, iters=30)
    print(f"Fit STM(K={Ks}) with decade prevalence.")

    # Interpret topics with stm-style word lists: prob (frequent) vs FREX
    # (frequent AND exclusive to the topic — usually the most evocative).
    labels = stm.label_topics(stm_model.topic_word, stm_model.vocabulary, n=7)
    print("\nTopic labels (Highest-probability vs FREX words):")
    for t in range(Ks):
        prob = ", ".join(w.replace("_", " ") for w, _ in labels[t]["prob"])
        frex = ", ".join(w.replace("_", " ") for w, _ in labels[t]["frex"])
        print(f"  T{t:2d}  prob: {prob}")
        print(f"       frex: {frex}")

    # Method of composition: regress each topic's proportion on YEAR.
    year = np.array([r["year"] for r in rows], dtype=float).reshape(-1, 1)
    theta_draws = stm.posterior_theta_samples(stm_model, nsims=20, seed=0)
    effects = stm.estimate_effect(theta_draws, year, feature_names=["year"])
    trends = []
    for t in range(Ks):
        d = effects[t].as_dict()
        trends.append((t, d["year"]["coef"], d["year"]["z"]))
    print("\nTopic trends over time (per-year change in prevalence):")
    risers = sorted(trends, key=lambda x: x[1], reverse=True)
    short = lambda t: ", ".join(
        w.replace("_", " ") for w, _ in labels[t]["frex"][:3])
    print("  Rising fastest:")
    for t, coef, z in risers[:3]:
        star = "*" if abs(z) > 1.96 else " "
        print(f"    T{t:2d} {coef:+.5f}/yr (z={z:+.2f}){star}  [{short(t)}]")
    print("  Falling fastest:")
    for t, coef, z in risers[-3:][::-1]:
        star = "*" if abs(z) > 1.96 else " "
        print(f"    T{t:2d} {coef:+.5f}/yr (z={z:+.2f}){star}  [{short(t)}]")
    print("    (* = |z| > 1.96; year coef is the OLS slope of theta on year.)")

    # ------------------------------------------------------------------ #
    # 5. DTM.  The Dynamic Topic Model fixes the number of topics but lets
    #    each topic's word distribution DRIFT across ordered time slices.
    #    We slice by decade and trace a single salient word's probability
    #    through time within the topic where it lives — Du Bois's vocabulary
    #    of "war", "labor", "africa" waxes and wanes across his editorship.
    #
    #    chain_variance controls how freely a topic may drift between slices.
    #    The default (0.005) is deliberately stiff and barely moves on a
    #    3-slice corpus; 0.05 lets real trends show while staying smooth.
    #    (Push it higher for more movement, at the cost of noisier estimates.)
    # ------------------------------------------------------------------ #
    banner("[5] DTM: word trajectories across decades")
    decades = sorted(by_decade)            # contiguous, 0-based time indices
    didx = {d: i for i, d in enumerate(decades)}
    times = [didx[r["decade"]] for r in rows]
    dtm = DTM(num_topics=8, chain_variance=0.05, seed=1)
    dtm.fit(pcorpus, times, iters=20)   # ~30+ EM iters for a real run
    print(f"Fit DTM(K=8) over {dtm.num_times} decade slices "
          f"({decades[0]}s ... {decades[-1]}s).")

    def best_topic_for(word):
        """The DTM topic in which `word` is most probable (averaged over time)."""
        if word not in dtm.vocabulary:
            return None
        per_time = np.stack([dtm.topic_word(s) for s in range(dtm.num_times)])
        wid = list(dtm.vocabulary).index(word)
        return int(per_time[:, :, wid].mean(axis=0).argmax())

    print("\nWord probability trajectories (x1000) by decade:")
    header = "    " + "  ".join(f"{d}s" for d in decades)
    print(header)
    # Words chosen to survive phrasing + pruning (e.g. "schools" not "school").
    # "war" and "labor" tell the clearest story: the WWI vocabulary recedes
    # after the 1910s while Du Bois's economic/labor language climbs.
    for word in ["war", "labor", "africa", "lynching", "schools"]:
        topic = best_topic_for(word)
        if topic is None:
            print(f"    {word:10s} (not in pruned vocab)")
            continue
        evo = dtm.word_evolution(topic, word)
        cells = "  ".join(f"{1000 * float(p):5.2f}" for p in evo)
        print(f"  {word:10s} (T{topic}): {cells}")

    # ------------------------------------------------------------------ #
    # 6. HDP.  The Hierarchical Dirichlet Process is nonparametric: instead
    #    of you picking K, it infers how many topics the data support. Handy
    #    as a sanity check on the K you chose for LDA/STM above.
    #
    #    `eta` is the topic-word smoothing: the default (0.01) is sharp and
    #    over-segments this OCR-noisy corpus into dozens of tiny topics; 0.3
    #    is smoother and lands on a sensible count. HDP topics aren't ordered
    #    by size, so we rank them by prevalence (total theta mass) and show
    #    the largest — the small tail topics are mostly noise.
    # ------------------------------------------------------------------ #
    banner("[6] HDP: infer the number of topics")
    hdp = HDP(eta=0.3, seed=1)
    hdp.fit(pcorpus, iters=150)            # ~300+ iters for a stable estimate
    print(f"HDP inferred K = {hdp.num_topics} topics from the data "
          f"(reassuringly close to the K=15 we picked for LDA, K=12 for STM).")
    mass = hdp.doc_topic.sum(axis=0)       # total prevalence of each topic
    order = np.argsort(mass)[::-1]
    print("The most prevalent inferred topics:")
    for t in order[:6]:
        words = [w.replace("_", " ") for w, _ in hdp.top_words(8, topic=int(t))]
        print(f"  ({mass[t]:5.1f} docs) {', '.join(words)}")

    # ------------------------------------------------------------------ #
    # 7. GUIDED TOPICS.  Everything above *discovers* topics, which you then
    #    label. When you already know the themes you want to measure, a guided
    #    model seeds them by name so each topic maps to a construct by
    #    construction — better validity, and reproducible across runs. KeyATM
    #    takes a {name: [seed words]} dict; here we name four themes from Du
    #    Bois's program and let KeyATM learn four more freely (num_topics=8).
    #    `keyword_rate` reports how much each topic leans on its seed words.
    # ------------------------------------------------------------------ #
    banner("[7] Guided topics: seed named themes with KeyATM")
    seeds = {
        "education": ["school", "schools", "education", "college", "children"],
        "labor":     ["labor", "wages", "industrial", "economic", "workers"],
        "voting":    ["vote", "votes", "ballot", "suffrage", "franchise"],
        "africa":    ["africa", "african", "congo", "liberia", "empire"],
    }
    ka = topica.models.KeyATM(seeds, num_topics=8, seed=1)
    ka.fit(phrased_docs, iters=800)         # ~1500+ iters for a real run
    print("Seeded topics (and how much each leans on its keywords):")
    for t in range(len(seeds)):
        words = [w.replace("_", " ") for w, _ in ka.top_words(7, topic=t)]
        print(f"  {ka.topic_names[t]:10s} (kw {ka.keyword_rate[t]:.2f}): "
              f"{', '.join(words)}")
    print("education, voting, and africa land cleanly on their seeds; labor is "
          "diffuse here because Du Bois\n  ties labor to race throughout. The "
          "four unseeded topics (4-7) are discovered as in plain LDA.")

    # ------------------------------------------------------------------ #
    # 8. UTILITIES.  Three things you'll reach for constantly.
    # ------------------------------------------------------------------ #
    banner("[8] Utilities: summary, save/load, pyLDAvis")

    # summary() — a tomotopy-style one-shot overview of a fitted model.
    print("topica.summary(lda):\n")
    print(topica.summary(lda, topn=6))

    # save/load — persist a fitted model and read it straight back.
    model_path = os.path.join(tempfile.gettempdir(), "dubois_lda.bin")
    lda.save(model_path)
    reloaded = LDA.load(model_path)
    print(f"\nSaved LDA to {model_path} and reloaded it "
          f"(K={reloaded.num_topics}, vocab={len(reloaded.vocabulary)}).")
    os.remove(model_path)

    # prepare_pyldavis — builds the intertopic-distance visualization. If
    # pyLDAvis is installed this returns its PreparedData (pass to
    # pyLDAvis.save_html); otherwise you get a PyLDAvisInputs you can feed to
    # pyLDAvis.prepare later. Either way it costs nothing to assemble here.
    vis = stm.prepare_pyldavis(lda, phrased_docs)
    print(f"prepare_pyldavis -> {type(vis).__name__} "
          "(install pyLDAvis to render it as an interactive HTML chart).")

    # ------------------------------------------------------------------ #
    # 9. TRANSFORM.  Every model exposes a sklearn-style `transform` that
    #    infers topic proportions theta for NEW, unseen documents against the
    #    fitted topics (the variational E-step for CTM/STM, collapsed Gibbs
    #    for LDA/DMR/HDP/...). OOV tokens are dropped. Hand it two snippets and
    #    watch them load on the right topics.
    # ------------------------------------------------------------------ #
    banner("[9] transform: held-out inference on new documents")
    new_docs = [
        "the mob lynched a man in the southern state and the murder went "
        "unpunished".split(),
        "children in the public schools need education and college "
        "training".split(),
    ]
    theta = lda.transform(new_docs)
    print("Inferred topic proportions for 2 held-out snippets:")
    for i, row in enumerate(theta):
        top = int(row.argmax())
        words = ", ".join(w for w, _ in lda.top_words(6, topic=top))
        print(f"  snippet {i}: top topic T{top} (p={row[top]:.2f}) -> {words}")

    banner("Done — that's the full topica workflow on Du Bois's Crisis.")
    print("Scale up K, iterations, and EM sweeps (see the inline comments) for "
          "a publication-grade analysis.")


if __name__ == "__main__":
    main()
