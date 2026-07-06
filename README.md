# One Kernel, Two Readouts: Unifying Representer Point Selection and TRAK for Training-Data Attribution via NTK–JL Sketching

**[📄 Read the paper (PDF)](paper.pdf)** &nbsp;·&nbsp; *Preprint, 2026*

This repository accompanies our paper on **NTK-JL**, a retrain-free framework for
training-data attribution. It shows that two independently developed attribution
methods — **Representer Point Selection (RPS)** and **TRAK** — are two *readouts*
of a single sketched empirical-Neural-Tangent-Kernel, and characterises which
question each one answers.

---

## TL;DR

Given a trained network, we linearise it around its trained parameters, forming
an empirical-NTK kernel, and compress the per-example gradients with a
Johnson–Lindenstrauss (JL) sketch (reducing gradient storage from
`O(nP)` to `O(ns)`, `s ≪ P`). One kernel then admits **two readouts**:

| Readout | Recovers | Answers | Counterfactual fidelity (LDS) |
|---|---|---|---|
| **Decomposition** | Representer Point Selection | *What explains the current prediction?* | ≈ 0 |
| **Datamodel** | single-model TRAK | *What predicts the effect of retraining?* | **0.63 / 0.75** (SST-2 / QNLI) |

- The **decomposition readout** reduces to RPS in the frozen-backbone regime
  (Spearman ρ = 0.97 on frozen ViT-B/16).
- The **datamodel readout** is, under TRAK's projected-gradient representation
  and Gauss–Newton curvature approximation, *algebraically identical* to
  single-model TRAK.
- Using the **Linear Datamodeling Score (LDS)**, we show the two readouts answer
  genuinely different questions: only the datamodel readout has counterfactual
  fidelity, which explains why deletion-based validation of representer-style
  influence is confounded.

## Contributions

1. **A unified framework that strictly generalizes representer-style
   attribution.** A single sketched empirical-NTK kernel yields a decomposition
   readout (recovering RPS under a frozen backbone) and a datamodel readout
   (algebraically identical to single-model TRAK under TRAK's own assumptions),
   unifying RPS and TRAK as two readouts of one object.
2. **Task-specific attribution beyond frozen features, plus a diagnostic.**
   On DistilBERT/SST-2 and RoBERTa-base/QNLI, NTK-JL's fine-tuned gradients
   identify proponents whose removal drops confidence far more than
   frozen-feature RPS (AUC-DEL⁺ gaps of 20 and 14 pp). The Linear Datamodeling
   Score separates the two attribution goals.
3. **A scalable algorithm.** Exact Jacobian computation + parameter subsampling +
   JL sketching, scaling to architectures with millions of parameters.
4. **A data-quality application.** Self-influence scores surface probable
   annotation errors in QNLI and SST-2 at 2.8–3.2× enrichment over random.

## Experiments

The paper evaluates NTK-JL across three architecture families with one algorithm:

- **CNN** — ResNet-50 (frozen-backbone validation; Oxford-IIIT Pet qualitative)
- **Encoder Transformer** — DistilBERT/SST-2, RoBERTa-base/QNLI (deletion, LDS)
- **Vision Transformer** — ViT-B/16 (frozen-backbone RPS recovery)

## Repository structure

```
.
├── paper.pdf              # the manuscript
├── README.md             # this file
├── LICENSE               # MIT
├── requirements.txt      # dependencies
├── code/                 # experiment scripts (as run for the paper)
│   ├── lds_sst2_qnli.py       # Linear Datamodeling Score harness (SST-2, QNLI)
│   ├── deletion.py            # proponent / opponent deletion experiments
│   ├── frozen_backbone.py     # RPS-recovery validation (ViT-B/16, ResNet-50)
│   └── qualitative.py         # Oxford-IIIT Pet, SST-2 qualitative attribution
└── figures/              # plots that appear in the paper
```

*(Adjust the tree above to match what you actually upload.)*

## Contact

For questions about the work, please reach out to **Indranil Paul**
(*paulindra009@gmail.com*).

---

*This is a public version of a manuscript currently under peer review.*
