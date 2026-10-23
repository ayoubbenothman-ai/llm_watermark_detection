# Statistical Detection of Watermarked LLM Text

Likelihood-ratio tests, tournament sampling (SynthID-style) and a key-based binomial detector, studied with two small open language models (SmolLM2-135M and SmolLM2-360M).

## Context and research question

LLM-generated text is increasingly hard to tell apart from human writing. Text watermarking addresses this problem: generation is slightly biased with a secret key, so that the key holder can later detect the watermark with a statistical test.

This project addresses two questions from a hypothesis-testing perspective:

1. **Model attribution.** Given a continuation of a prompt, which of two known language models generated it?
2. **Watermark detection.** Was the continuation generated with a specific watermark and key?

For each question, the project also asks what guarantees (level, power) the corresponding tests offer.

## Objectives and methodology

| # | Objective | Method | Status |
|---|---|---|---|
| 1 | Explore next-token distributions | Top-10 distributions and sampled continuations for both models | executed |
| 2 | Attribute a text to model A or B | Simple-vs-simple likelihood-ratio (Neyman–Pearson) test on $T_{AB}=\sum_t \log p_{t,B}(x_t)/p_{t,A}(x_t)$; Monte Carlo calibration; empirical level and power for $n\in\{10,20,30\}$ | executed |
| 3 | Implement a watermark | Tournament sampling (SynthID-Text): exact law of the winner, effect of depth $d$, distortion over 100 keys | executed |
| 4 | Detect the watermark with full model access | LR test between top-50 sampling and tournament sampling ($d=2$) | executed |
| 5 | Detect the watermark with the key only | One-sided binomial test on the bits $G_{t,\ell}=g_{t,\ell}(X_t)$, $K_{n,d}\sim\mathrm{Bin}(nd,1/2)$ under an idealised null | derived and implemented, not yet evaluated |
| 6 | Robustness to paraphrasing | Round-trip translation EN → FR → EN | planned |

Working hypotheses:
- **H-A.** Two models of different sizes are separable by an LR test on about 30 tokens.
- **H-B.** Tournament sampling distorts the distribution for a fixed key but is unbiased on average over keys.
- **H-C.** For text produced independently of the key, the $g$ bits behave like i.i.d. Bernoulli(1/2) variables.

## Main results

All numbers come from a single CPU run of the notebook.

**Model attribution (A vs B, prompt "In statistics, a confidence interval is")**
- On single realisations, the cumulative log-likelihood ratio reaches about **−22.5** for a continuation from A and **+23** for one from B after 30 tokens.
- Monte Carlo evaluation used 20 calibration texts and 20 test texts per hypothesis, with $\alpha = 0.05$:

  | n | threshold (log scale) | empirical level | empirical power |
  |---|---|---|---|
  | 10 | −0.715 | 0.10 | 0.95 |
  | 20 | −5.789 | 0.10 | 1.00 |
  | 30 | −9.148 | 0.35 | 1.00 |

  Power is high, which supports H-A. The level, however, is **not controlled**: with only 20 calibration samples, the empirical quantile used as threshold is very unstable.
- On a blind file of 12 continuations, 8 are attributed to B and 4 to A. Two of the B decisions have a negative log-likelihood ratio and are the least reliable. The true labels are not available.

**Tournament watermark**
- Exact distribution of one match:
  - $q(v)=p(v)(1+m_0)$ if $g(v)=1$;
  - $q(v)=p(v)\,m_0$ otherwise, where $m_0$ is the probability mass of tokens with bit 0.
- With the default key, the most likely token " a" goes from probability ≈0.48 to ≈0.58, 0.76 and 0.92 for $d=1,2,3$.
- At $d=2$, across 100 keys:
  - for a single key, $d_{\mathrm{TV}}(p,q)$ ranges from ≈0.07 to ≈0.44 (mean ≈0.24);
  - for the key-averaged distribution, $d_{\mathrm{TV}}(p,\bar q)$ is ≈0.018.

  Large distortion per key with no systematic bias on average supports H-B.

**Watermarked vs unwatermarked generation** ($k=50$, $d=2$)
- The cumulative log-likelihood ratio separates the two continuations from the first tokens, ending at ≈ **+6.7** (watermarked) and ≈ **−4.3** (unwatermarked) after 30 tokens.

## Limitations

- **Scale.** The models are small, a single prompt is used, and the Monte Carlo samples are small. Several figures show single realisations.
- **Calibration.** The Section 2 test is miscalibrated (empirical level up to 0.35 for a nominal 0.05).
- **Model knowledge.** The LR tests require full knowledge of both generation mechanisms.
- **Idealised null.** The binomial detector relies on an idealised null hypothesis and has not been evaluated on data, so H-C remains untested.
- **Truncation.** The distortion analysis uses a top-10 support, whereas generation uses top-50.
- **Position-dependent seed.** The hash seed includes the absolute token position. Detection therefore needs the exact prompt length, and a single inserted or deleted token shifts all subsequent bits.
- **Implementation caveats**, kept as in the original run:
  - `total_variation` returns $\sum|p-q| = 2\,d_{\mathrm{TV}}$ (the values are converted in the analysis);
  - `g_bits_statistic` computes $P(K > k)$ instead of the $p$-value $P(K \ge k)$.

## Conclusions and next steps

- **Model attribution.** The LR test is a powerful attribution tool even on 10–30 tokens, but its level requires a much larger calibration sample.
- **Tournament sampling.** It embeds a strong per-token signal while remaining unbiased on average over keys.
- **Next steps:**
  - evaluate the key-only binomial detector on the labelled file `phrases_filigrane.csv`;
  - recalibrate the Section 2 test;
  - run the round-trip translation attack;
  - extend the study to image watermarks.

## Repository structure

```
.
├── llm_watermark_detection.ipynb   # full study: derivations, code, results, discussion
├── data/
│   ├── phrases_AB.csv              # blind continuations from model A or B (Section 2)
│   └── phrases_filigrane.csv       # watermarked / unwatermarked continuations (Section 5)
└── README.md
```

## Installation and usage

Requirements: Python 3 (the notebook was run with Python 3.14), `numpy`, `pandas`, `matplotlib`, `scipy`, `torch`, `transformers`.

```bash
pip install numpy pandas matplotlib scipy torch transformers jupyter
jupyter notebook llm_watermark_detection.ipynb
```

- The models `HuggingFaceTB/SmolLM2-135M` and `HuggingFaceTB/SmolLM2-360M` are downloaded from the Hugging Face Hub on first run.
- Everything runs on CPU; CUDA or Apple MPS are used automatically if available.
- Generation is seeded. Results may still differ slightly across hardware and library versions.
- Sections 1–4 run end to end, provided `data/phrases_AB.csv` is present. The Section 6 cells are templates that only print instructions until their placeholders are filled.

## References

- S. Dathathri et al., "Scalable watermarking for identifying large language model outputs", *Nature* 634, 818–823 (2024). https://doi.org/10.1038/s41586-024-08025-4
- Google DeepMind, `synthid-text`: https://github.com/google-deepmind/synthid-text

## Acknowledgements

This project is adapted from a problem set of the statistics course APM41033 (École polytechnique, A. Dieuleveut), whose statement is freely reusable.
