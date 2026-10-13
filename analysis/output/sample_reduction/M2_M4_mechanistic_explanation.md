# Person 4 — Why M2 and M4 are difficult to distinguish, and why the matched-A campaign helps

**Scope:** Mechanistic interpretation of *the six supplied candidate ODE networks* in System B. This is not a structural identification of the real molecules A, B, I, P or Q: the brief does not establish their detailed chemical structures. The interpretation is therefore kinetic and stoichiometric, not a claim about a specific transition state or an external reagent.

## 1. The two surviving reaction networks

| | M2: intermediate branches directly | M4: impurity branch requires additional A |
|---|---|---|
| Formation of I | `A + B -> I` (k1) | `A + B -> I` (k1) |
| Formation of P | `I -> P` (k2) | `I -> P` (k2) |
| Formation of Q | `I -> Q` (k3) | `I + A -> Q` (k3) |
| Impurity rate | `d[Q]/dt = k3[I]` | `d[Q]/dt = k3[I][A]` |
| Product rate | `d[P]/dt = k2[I]` | `d[P]/dt = k2[I]` |

The shared first two reaction steps mean that the observable major species A, B and P can look very similar, especially when Q remains a trace species. In both models, disappearance of B need not immediately match P formation: some consumed B can be stored transiently as the **unmeasured** intermediate I.

**Stoichiometric distinction:** M4 consumes an extra equivalent of A on the impurity branch; M2 does not. The size of this difference is limited by the low amount of impurity Q, so the A profile alone may not be the most sensitive discriminator.

## 2. What is structurally testable?

At a fixed temperature, with initially zero P and Q, M2 implies:

\[
\frac{dQ}{dP}=\frac{k_3[I]}{k_2[I]}=\frac{k_3}{k_2}
\quad\Longrightarrow\quad
\boxed{Q(t)/P(t)=k_3/k_2}\,.
\]

The ratio holds wherever P is nonzero and is independent of the starting charges A0 and B0 **at the same temperature**, under the ideal supplied M2 equations. This does **not** say that the absolute amount of Q is independent of A: A affects the formation of I and therefore affects both P and Q.

M4 implies:

\[
\frac{dQ}{dP}=\frac{k_3}{k_2}[A](t),\qquad
\frac{Q(t)}{P(t)}=\frac{k_3}{k_2}\,
\frac{\int_0^t [A](\tau)[I](\tau)d\tau}{\int_0^t [I](\tau)d\tau}\,.
\]

The rightmost term is an **I-weighted, time-averaged A concentration**, not simply initial A0 or instantaneous A(t). Thus, M4 can show a charge-dependent Q/P signature at the same temperature. Its signature is not a universal claim that Q/P is perfectly proportional to A0 across every reaction regime.

## 3. Why the historical data leave an ambiguity

All five System B historical experiments used **A0 = 1.0 M, B0 = 0.25 M**; temperatures varied from **65 to 85 C**. There is no concentration perturbation designed to isolate the additional A on the M4 side branch. Over those trajectories, M4's factor `k3*[A](t)` can partly behave like an *effective* unimolecular branching rate and be compensated by fitting k3 and activation energies. The match is approximate, not an exact algebraic equivalence because A(t) changes over time.

Q is a trace assay with **15% relative noise, 2.2e-5 M absolute noise and 3e-5 M LOQ**; some observations are censored. That reduces the resolving power for differences in the Q pathway. Both models explain the major-species trajectories and can earn virtually identical AICs (previous comparison M4 -1375.80 vs M2 -1375.71). **AIC weights near 50/50 are relative ranking weights, not the probability that either model is chemically true.**

Temperature changes rate constants for both models through their Arrhenius parameters, and may alter the reaction *time course*. Similar simulated final Q levels at some long times do **not** establish temperature-independent Q formation, nor do they rule out either M2 or M4.

## 4. Why the official matched-A design is a good confirmatory test

| Run | T | A0 | B0 | Role |
|---|---:|---:|---:|---|
| E1 | 95 C | 0.1 M | 1.5 M | Temperature perturbation versus E2 at low A |
| E2 | 75 C | 0.1 M | 1.5 M | Low-A member of direct charge comparison |
| E3 | 75 C | 1.0 M | 1.5 M | High-A member; **T, B0 and sampling times match E2** |

E2 vs E3 is the clearest mechanistic contrast: M2 predicts essentially the same Q/P ratio in both runs at 75 C, whereas M4 allows it to change because the I -> Q path requires a second molecule of A. Holding B0 and temperature fixed reduces competing explanations. It **does not** hold [I], reaction conversion or product formation rate constant; compare Q to P rather than only comparing Q concentrations.

**Numerical predictions for E2 and E3 at 180 min**, using medians of the saved 60 bootstrap parameter draws for each model (simulated, not measured):

| Run | Model | Median P (M) | Median Q (M) | Median of Q/P | Simulated chance Q result is below LOQ* |
|---|---|---:|---:|---:|---:|
| E2 (A0=0.1) | M2 | 0.09497 | 9.51e-5 | 1.004e-3 | 0.6% |
| E2 (A0=0.1) | M4 | 0.09507 | 1.98e-6 | 2.08e-5 | 89.9% |
| E3 (A0=1.0) | M2 | 0.90476 | 9.05e-4 | 1.004e-3 | <0.01% |
| E3 (A0=1.0) | M4 | 0.90474 | 2.67e-4 | 2.95e-4 | <0.01% |

\*This probability averages each parameter draw's Gaussian assay outcome at the Q LOQ, **not** a measured frequency; Q/P itself is not precisely quantifiable if Q is reported as `<LOQ`. When censored, the likelihood must treat Q as an upper-bound event, never replace it by Q=0 or Q=LOQ.

**Critical subtlety:** The official E2 low-A condition is **very** low in predicted Q under M4, so one cannot necessarily draw an empirical low-A numerical Q/P point. One can still discriminate using a censored-Q likelihood against the M2 expectation. If the science goal shifts to estimating a continuous charge-dependence curve rather than merely classifying M2/M4, include a more moderate-A pair with Q quantifiable under both models (requires a separate DOE decision and budget check).

## 5. Major limitations to say out loud

- Strong EIG or simulated model-classification is **conditional on M2 and M4** being the only possibilities and on the saved bootstrap uncertainty being representative. It is not real-world model-identification confidence.
- The charge ratios are outside the historically studied ratio A0:B0 = 4:1; extrapolative kinetic-model error, batch effects, unexpected side pathways and systematics are not covered.
- The chosen three runs all use B0=1.5 M. This is efficient for M2/M4 discrimination but cannot test sensitivity across B levels. Nor does it independently assess the effect of a new model mechanism.
- Comparing fixed fitted-parameter curves exaggerates discrimination if each model can re-fit after seeing the new data. This review does **not** provide a full re-fitting / adversarial model-identifiability guarantee.
- All measurements have finite precision. Q LOQ censoring is a data event (likelihood CDF), **not** the 15-ppm isolated-product impurity specification.
- Reducing samples can preserve model-classification while weakening full kinetic-profile checking and opportunities to identify model misspecification.

## 6. Slide-ready interpretation (English)

> **Why the ambiguity exists:** Both M2 and M4 form the same unobserved intermediate from A+B and subsequently form P from I. The difference is the impurity branch: M2 forms Q directly from I, while M4 needs I+A. Historical experiments varied temperature but not initial A:B ratio. With trace, noisy and partially censored Q observations, the two pathways can produce similarly good fits.
>
> **Why the new experiments help:** At a fixed temperature, M2 predicts a charge-independent Q/P formation ratio; M4 predicts an A-dependent ratio. Our matched E2/E3 pair varies A0 from 0.1 to 1.0 M while holding B0, temperature and sampling times constant. We analyse quantified and below-LOQ Q values using the appropriate likelihood, not Q=0. The campaign tests a kinetic mechanism, not just a difference between two fitted curves.
>
> **What we cannot claim:** This is a conditional two-model discrimination exercise, with unvalidated out-of-distribution predictions. A high simulated discrimination score is not proof that either mechanism is chemically correct.
