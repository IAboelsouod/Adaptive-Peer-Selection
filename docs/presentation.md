---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: "Segoe UI", "Source Sans 3", sans-serif;
    font-size: 26px;
    color: #1a1a1a;
  }
  h1, h2 {
    font-family: Georgia, "Source Serif 4", serif;
    color: #002147;
  }
  h2 {
    border-bottom: 2px solid #002147;
    padding-bottom: 0.15em;
    font-size: 1.25em;
  }
  table { font-size: 0.72em; }
  th { background: #002147; color: white; }
  tr:nth-child(even) { background: #f4f6f8; }
  img { max-height: 380px; display: block; margin: 0 auto; }
  .caption { font-size: 0.55em; color: #5c677d; font-style: italic; }
  .footer { font-size: 0.45em; color: #5c677d; position: absolute; bottom: 24px; left: 48px; }
  .title-meta { font-size: 0.6em; color: #5c677d; line-height: 1.7; }
  .dept { font-size: 0.5em; letter-spacing: 0.06em; text-transform: uppercase; color: #5c677d; }
---

<!-- _class: lead -->

<p class="dept">[Faculty / Department] · [University Name]</p>

# Adaptive Peer Selection for Hierarchical Federated Learning

A data-similarity approach to client–edge assignment under non-IID data

<p class="title-meta">
<strong>[Course / Module Title]</strong><br/>
Presented by: <strong>[Your Name]</strong> · Supervisor: <strong>[Supervisor Name]</strong><br/>
[Date — June 2026]
</p>

---

## Outline

1. Introduction & motivation
2. Problem statement & research objectives
3. Related work
4. Proposed methodology
5. Experimental evaluation
6. Results & discussion
7. Conclusion & future work

---

## 1. Introduction & motivation

- **Federated Learning (FL)** trains models collaboratively without centralising raw data (McMahan et al., 2017)
- Client data are typically **non-IID**, degrading global model quality
- **Hierarchical FL (HFL)** introduces edge aggregators with periodic global sync (Abad et al., 2020)
- Static client-to-aggregator assignment may not reflect underlying data similarity

---

## 2. Problem statement & objectives

**Problem:** How should clients be assigned to edge aggregators under heterogeneous, non-IID data?

**Objectives:**
1. Design **adaptive assignment** from model-update similarity
2. Compare **global accuracy** vs flat FedAvg and static HFL
3. Measure **cluster recovery** (ARI, NMI) against ground-truth groups
4. Analyse **accuracy–communication trade-off** as sync period *H* varies

---

## 3. Related work

- **FedAvg** — flat federated averaging (McMahan et al., 2017)
- **Abad et al. (2020)** — HFL with *H*-periodic sync; static clustering
- **Rana et al.** — dynamic UE/aggregator placement (HierFedML)
- **Clustered FL** (Sattler et al., 2019) — cosine similarity of updates

**Contribution:** Adaptive, similarity-driven assignment within Abad et al.'s hierarchical loop under non-IID data.

---

## 4. Proposed methodology

1. Compute client delta: \(\Delta_i = w_i^{(t)} - w_{\text{global}}^{(t-1)}\)
2. Pairwise **cosine similarity** → **peer graph** (NetworkX)
3. **Louvain** community detection → edge-aggregator groups
4. **Hysteresis** stabilises group labels across rounds
5. Intra-cluster FedAvg every round; global merge every *H* rounds

---

## 4. System architecture & baselines

**Logical hierarchy** (Flower simulation): Clients ⟷ Edge aggregators ⟷ Global server

| System | Assignment |
|--------|------------|
| **A — Flat FedAvg** | N/A |
| **B — Static HFL** | Fixed random groups |
| **C — Adaptive HFL** (proposed) | Similarity-driven, re-clustered at sync |

---

## 5. Experimental setup

| Parameter | Value |
|-----------|-------|
| Dataset | Fashion-MNIST, CNN |
| Clients | 50 |
| Non-IID | Dirichlet α = 0.1, 3 latent groups |
| Sync period *H* | 1, 2 |
| Metrics | Accuracy, ARI, NMI, comm-cost proxy |

---

## 6. Results — quantitative summary

*Reduced sweep: α = 0.1, 10 rounds, 2 seeds (mean ± std)*

| Strategy | *H*=1 acc. | *H*=2 acc. | ARI | vs flat comm. |
|----------|------------|------------|-----|---------------|
| Flat (A) | 0.74 ± 0.09 | 0.78 ± 0.05 | — | 1.0× |
| Static (B) | 0.79 ± 0.04 | 0.77 | ≈ 0 | ~4.8× |
| **Adaptive (C)** | **0.79 ± 0.06** | 0.74 | **0.36** | **~5.0×** |

---

## 6. Results — global accuracy

![w:900](../results/accuracy_abc_overlay.png)

<p class="caption">Figure 1. Global test accuracy over training rounds (α = 0.1).</p>

---

## 6. Results — cluster recovery

![w:900](../results/cluster_quality_adaptive.png)

<p class="caption">Figure 2. ARI and NMI for the adaptive system; stabilisation after warm-up.</p>

---

## 7. Discussion

- Adaptive HFL matches or exceeds flat accuracy under strong non-IID skew
- Only the proposed system recovers latent structure (ARI ≈ 0.36)
- ~5–7× communication savings vs flat FedAvg depending on *H*
- Higher *H* reduces global sync cost but increases inter-cluster drift

---

## 7. Limitations & future work

**Limitations:** simulation only; single dataset; moderate scale

**Future work:** full sweep; covariate shift (rotated-MNIST); client dropout; decentralised peer graphs

---

## 8. Conclusion

1. Adaptive peer selection via community detection on update-similarity graphs
2. Competitive accuracy **and** meaningful cluster recovery vs static HFL
3. Favourable accuracy–communication trade-off under non-IID data
4. Reproducible open-source Flower simulation

---

## References

1. Abad *et al.*, ICASSP 2020 — Hierarchical Federated Learning
2. Rana *et al.* — Hierarchical and Decentralised Federated Learning
3. McMahan *et al.*, AISTATS 2017 — FedAvg
4. Sattler *et al.*, 2019 — Clustered FL
5. Xu *et al.*, IEEE TPDS — HierFedML

---

<!-- _class: lead -->

# Thank you

**Questions & discussion**

[Your Name] · [your.email@university.ac.uk]
