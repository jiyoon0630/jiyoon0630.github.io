---
layout: paper
lang: en
ref: act-2-generalizing-reliability
kind: paper-review
title: "ACT-2 Preview: Generalizing Reliability (ACT-2)"
date: 2026-07-18 12:00:00 -0700
paper_date: 2026-07-17
venue: "Sunday Robotics Blog (technical preview) · not on arXiv"
tags: [Robot-Foundation-Model, Home-Robotics, Imitation-Learning, Scaling, Evaluation, Paper-Review]
authors: "Sunday Team"
affiliations: "Sunday Robotics"
summary: "Scale pretraining on human data from wearable sensors, and reliability raised in-house with a little post-training transfers to unseen homes — laundry folding at 99.1% (785 trials, zero per-home adaptation), backed by the Solve standard of declared scope and adaptation cost."
paper_url: "https://www.sunday.ai/blog/act-2-preview"
---

> **Core claim** — Scale up pretraining on human data collected with wearable sensors, and the reliability raised on in-house robots through a small amount of post-training transfers to homes the robot has never seen. And such a claim is interpretable only when it comes with **a declared scope and an adaptation cost**, not a success rate alone. Laundry folding at 99.1% (778 of 785 trials, zero adaptation per home) is the first instance.

---

## Introduction

In 1903 the Wright brothers' first flight lasted 12 seconds. It showed that flying was possible, but flight became part of everyday life only 11 years later, when a scheduled route began crossing Tampa Bay twice a day. Sunday opens the ACT-2 preview with this analogy. Robotics is now pouring out demos that amount to a "first flight," but what has to be shown next is reliability — that it works **every time**, even as conditions change.

ACT-2 is the robot foundation model running on Memo, Sunday's home mobile manipulator, and the successor to ACT-1, released in November 2025. This is a technical blog preview, not a paper, so the architecture, parameter count and training recipe are not disclosed. This review therefore follows **what it claims, and in what form it proves it**. For readers with a thin background in robot learning, I build up the necessary concepts where they appear.

---

## 1. The problem — general robots are not reliable, and reliable robots are not general

### 1.1 An old tension

Robot learning has an old tension. A policy trained on diverse data is general, but its behavior cannot be trusted. A policy trained on narrow, curated data performs, but that performance is tied to the environment where the data was collected.

In home robots this tension takes its most extreme form. Homes vary without end, and no prior setup can be expected. This is exactly why Sunday argues that home robots have "research-market fit": the conditions for being useful at home are the conditions for general intelligence.

The problem is that the only development loop a company can actually run is "find failures on in-house robots → fix them by adding a bit of data → deploy." This loop means something **only if performance raised in-house carries over to homes never seen**.

### 1.2 The measurement problem

The second problem is measurement. Every demo has a different environment, objects, starting state, prior preparation and intervention rules, and there is no way to describe these consistently. So results cannot be compared or built upon. Even at the same 99.1%, the two claims below are entirely different.

| | Claim A | Claim B |
|---|---|---|
| Environment | Many unseen homes | One familiar room |
| Objects | 9 types of clothing | One type of clothing |
| Initial state | Randomly crumpled | Prepared work surface |
| Adaptation per deployment | None | Not stated |

### 1.3 So the question this piece asks

> Can reliability raised in-house with little data be made to hold even in a home the robot has never been to? And can that be proven in a form others can interpret?

Three walls stand in front of this question. The whole piece reads as the process of dissecting and breaking through them.

**⛔ Wall 1 — In-house improvements do not transfer to the wild.** Narrow post-training raises reliability only in the environment where the data was collected (overfitting).

**⛔ Wall 2 — The improvement signal is expensive.** Every change to the data composition requires a physical evaluation with real robots, real homes and human grading.

**⛔ Wall 3 — Success-rate numbers are not comparable.** A success rate with undescribed conditions carries almost no information.

---

## 2. Background — the minimum needed to read the three walls

### 2.1 Where does the pretraining data come from — the pipeline ACT-1 laid down

The ACT-2 post describes its pretraining data only as high-quality, high-diversity "sensorized human data" gathered with in-house collection hardware, a curation system and a processing pipeline. What that hardware is comes from its predecessor, ACT-1.

ACT-1's starting point is **embodiment mismatch**. If a human hand and a robot hand differ, human data does not carry over to the robot. Sunday matched the two at the hardware level.

| Component | Role |
|---|---|
| **Skill Capture Glove** | A glove with the same geometry and sensor placement as Memo's hand. A motion made by a person wearing the glove becomes the robot hand's motion |
| **Skill Transform** | Aligns human-specific traces, such as differences in height and arm length or a human arm visible in the video, to look like robot observations, on both the kinematic and visual sides. Glove data → robot data conversion success rate 90% |

ACT-1 claimed to have trained with this pipeline without teleoperation trajectories. It is natural to read ACT-2's "sensorized human data" in the same lineage. But the ACT-2 post does not name the glove, and it does not say whether robot data was mixed into pretraining.

Mapped onto LLMs, pretraining is "human behavior collected to fit the robot's body" instead of web text, and post-training is SFT on a small amount of data collected on in-house robots. What differs from LLMs is that **a transformation of body (human → robot)** sits in the pretraining data.

### 2.2 What Wall 1 really is — the generalization gap

The generalization gap is what turns Wall 1 into a measurable quantity.

$$G(D) = \mathrm{SR}_{\text{ID}}(D) - \mathrm{SR}_{\text{OOD}}(D)$$

- $D$ — pretraining data scale (% of the total)
- $\mathrm{SR}_{\text{ID}}$ — success rate in environments included in the post-training data (in-domain)
- $\mathrm{SR}_{\text{OOD}}$ — success rate in held-out environments, objects and configurations (out-of-domain)
- Both success rates are measured **after the same post-training procedure**

Unlike the gap usually meant in ML (train vs test), "train" here is not pretraining but **the environment the post-training data came from**. Without pretraining ($D=0$), the post-training data is everything, so the model memorizes that environment. It is the same picture as the difference, in LLMs, between a model trained from scratch on a little domain data and a pretrained model SFT'd on the same data.

### 2.3 What Wall 2 really is — offline loss is not, by nature, a good proxy for success rate

If physical evaluation is expensive, why not use a cheap proxy metric? Why that does not work well in robot imitation learning is what Wall 2 really is.

Validation loss is a teacher-forced metric that predicts the next action from states in the demonstration data.

$$\mathcal{L}_{\text{val}}(\theta) = \mathbb{E}_{(o,a)\sim\mathcal{D}_{\text{val}}}\big[\ell(\pi_\theta(o),\ a)\big]$$

- $o$ — observation, $a$ — the demonstrator's action
- $\ell$ — loss depending on the action head (MSE, NLL, flow-matching, etc.)
- $\mathcal{D}_{\text{val}}$ — held-out demonstrations. **The states come from the distribution the demonstrator produced**

Success rate, by contrast, is the outcome of the policy rolling out closed-loop over **the state distribution it produces itself**. An error at one step pushes the next state out of the demonstration distribution, and errors accumulate. The classic behavior cloning bound (Ross & Bagnell, 2010) shows this.

$$J(\hat\pi) \le J(\pi^*) + T^2\epsilon$$

- $J$ — expected cost over the whole horizon
- $\pi^*$ — the demonstrator, $\hat\pi$ — the learned policy
- $T$ — horizon length
- $\epsilon$ — per-step error rate under the demonstrator's state distribution

Even with the same $\epsilon$ (≈ the same loss), the success rate can differ greatly depending on which states the errors occur in. The problem grows when the action distribution is multimodal. Folding a shirt starting from the left or from the right are both successes, but the loss penalizes a model that picks the side the demonstration did not.

For readers with an LLM or diffusion background, this is a familiar phenomenon.

| Domain | Cheap metric | Real metric | Why they diverge |
|---|---|---|---|
| LLM | perplexity (teacher forcing) | generation quality | exposure bias |
| Diffusion | denoising loss | FID, human evaluation | loss scales cannot be compared across models and schedules |
| Robot BC | validation loss | closed-loop success rate | error accumulation, multimodality |

Yet Sunday reports an $R^2$ of 0.98 between validation loss and success rate. How this is possible is the subject of §3.2.

---

## 3. Method — the ACT-2 recipe

Sunday presents the recipe as three observations and one loop. Because the model architecture is undisclosed, the "method" here is not an algorithm but **an empirical claim about the relationship between data scale, data quality and post-training**.

```
  PRETRAIN                    POST-TRAIN (in-house Memos)         DEPLOY
  -----------------------     -----------------------------       ----------------------
  sensorized human data  -->  narrow, curated robot data    -->   unseen homes
  (glove, diverse, large)     SFT on few demos                    same checkpoint
                                  ^               |               zero per-home data
                                  |               v
                              recovery data <-- failures surface
                              (hill-climbing loop, details TBD)
```

- **PRETRAIN** — builds a broad prior from human data in the glove lineage
- **POST-TRAIN** — SFT on narrow, curated data on in-house Memos; when failures surface, recovery data is collected and post-training runs again
- **DEPLOY** — the same checkpoint is deployed to unseen homes with no per-home data

### 3.1 Closing the gap with pretraining scale — Wall 1

The first observation is that scaling pretraining closes the gap. The same post-training procedure was applied at each scale, and ID and OOD success rates were each evaluated over 50 trials (Figure 1).

| Pretraining scale | ID SR | OOD SR | Gap (pp) |
|---|---|---|---|
| 0% | 96% | 14% | 82 |
| 12% | 100% | 90% | 10 |
| 25% | 100% | 92% | 8 |
| 50% | 100% | 96% | 4 |
| 100% | 100% | 100% | 0 |

The gap, 82pp at 0%, drops to 10pp with just 12% of pretraining and reaches 0 at 100%. Sunday's reading is this: as the gap shrinks, in-house performance becomes a predictor of performance in the wild, so hill-climbing in-house can be done with confidence.

But look at the table again and the ID column is pinned at 100% from 12% onward. The gap shrank entirely because OOD went up. Then what exactly does "gap 0" show?

> ### ⚠️ A gap of 0 is agreement in *level*, not direct evidence that *improvements* transfer
>
> **⓵ Ceiling effect.** Because ID is saturated at 100%, in this range $G = 100 - \mathrm{SR}_{\text{OOD}}$. The gap is not an independent metric; it is the OOD success rate turned upside down.
>
> **⓶ What the hill-climbing argument needs is the change.** To show that "what goes up in-house goes up in the wild by the same amount," the following must hold across post-training iterations $k$.
>
> $$\Delta\mathrm{SR}_{\text{OOD}}^{(k)} \approx \Delta\mathrm{SR}_{\text{ID}}^{(k)}$$
>
> Figure 1 measures only the endpoint of a single post-training run at each scale.
>
> **⓷ Precision of the endpoint.** The Wilson 95% lower bound for 50 successes out of 50 is $50/(50+1.96^2) \approx 92.9\%$. "Gap 0" is also consistent with a true gap of around 7pp.
>
> The direction itself is convincing. The difference between 0% and 12% (OOD 14% → 90%) is overwhelming even with 50-trial samples. What is weak is the endpoint "0" and the direct evidence for "transfer."

### 3.2 There is more than one scaling curve — data quality and Wall 2

The second observation is that scale alone is not enough. Subsets of 12.5/25/50% were drawn from the full pretraining data, with the sampling split two ways.

- **uniform** — random sampling. A scaled-down copy of the original distribution
- **high-quality** — selected by Sunday's curation criteria

Because data volume and compute were matched, the performance difference is attributable to the selection method alone (Figures 2–4).

| Scale | HQ SR | Uniform SR | HQ val loss | Uniform val loss |
|---|---|---|---|---|
| 12.5% | 75.6% | 43.8% | 0.0700 | 0.0738 |
| 25% | 87.9% | 53.6% | 0.0682 | 0.0713 |
| 50% | 92.3% | 64.1% | 0.0668 | 0.0682 |
| 100% (flagship, shared by both series) | 99.1% | 99.1% | 0.0653 | 0.0653 |

At the same volume, success rates differ by 28–34pp. Validation loss fits log-linearly for both series.

$$L_u = 84 - 9.5\log D_u\ \ (R^2=0.998),\qquad L_h = 76 - 5.1\log D_h\ \ (R^2=0.997)$$

- $L$ — validation loss in units of $10^{-3}$ (0.0738 → 73.8)
- $D$ — pretraining data fraction (%)
- subscripts $u$, $h$ — uniform, high-quality
- The base of the log is not stated, but base 10 matches the table values (my check)

```
  L (val loss x 1e-3)
  74 |  u
  73 |
  72 |
  71 |         u
  70 |  h
  69 |
  68 |         h      u
  67 |                h
  66 |
  65 |                       *
     +--+------+------+------+----> D (log scale)
      12.5     25     50     100
```

- `u` — uniform, `h` — high-quality, `*` — the 100% flagship shared by both series
- Keep in mind that `h` at 25% and `u` at 50% sit at the same height (0.0682). It comes up again shortly

Setting the two equations equal gives the amount of uniform data needed to reach the same loss as $D_h$ of HQ data.

$$\log D_u = \frac{8 + 5.1\log D_h}{9.5}$$

- $D_h$ — high-quality data fraction
- $D_u$ — uniform data fraction needed for the same validation loss

| HQ scale $D_h$ | Uniform scale $D_u$ at the same loss | Ratio |
|---|---|---|
| 12.5% | about 27% | about 2.2× |
| 25% | about 39% | about 1.6× |
| 50% | about 57% | about 1.1× |

The benefit of curation is larger the less data there is, and it vanishes at 100%, where both series are the same dataset (my calculation). What curation changes, then, is not the destination but **the cost of getting there**. The scaling curve is not one curve but a family of curves, one per curation policy.

Sunday's answer to Wall 2 is also here. Because loss and success rate are strongly correlated, the data mixture can be improved on loss first, before expensive physical evaluation.

$$\mathrm{SR}_h = 420 - 4.9\,L_h\ \ (R^2=0.98),\qquad \mathrm{SR}_u = 500 - 6.2\,L_u\ \ (R^2=0.90)$$

- $\mathrm{SR}$ — success rate (%)
- Lines fit to the two series **separately** (per-series linear fit)

As §2.3 showed, loss in BC is not by nature a good proxy for success rate. So how does $R^2$ reach 0.98?

> ### 💡 Validation loss is a proxy for success rate only *within the same curation series*
>
> **⓵ Why it works within a series.** Within one series, the model architecture, val set and post-training are fixed and only the data volume changes. What is being compared is "the same model family trained on more," so the loss ordering follows the quality ordering. It is the same structure as in diffusion, where within one training run FID improves as loss falls, but loss must not be compared across different models.
>
> **⓶ It breaks across series.** The two points at the same height in the figure above, HQ 25% and uniform 50%, both have a val loss of 0.0682, yet their success rates are 87.9% vs 64.1%. A 24pp difference at the same loss. That Sunday fit separate lines to the two series itself reflects this.
>
> **⓷ Loss underestimates the effect of curation.** By loss, the conversion ratio at 12.5% is about 2.2×. But by success rate, HQ 12.5% (75.6%) is higher than uniform 50% (64.1%), which looks like an efficiency of more than 4×. That said, the difference between these two points (11.5pp) is only about 1.2 times their combined standard error, so it suggests a direction only.
>
> **⓸ The weight of the fits themselves.** Each series has only four points: its own three plus the one shared flagship. The uniform line leans heavily on that shared point; fitting only the three points without it gives a slope of about −3.6, unlike the reported −6.2 (my check). Nor is it disclosed what distribution the val set came from.
>
> In short, the loss proxy can be used for "how much more data to add under this curation policy," but telling "which curation policy is better" still needs physical evaluation. **Wall 2 is only half broken.** On top of that, the most important variable, the curation criteria themselves, is not disclosed.

Put this table next to the one in §3.1 and one more thing catches. In §3.1, just 12% of pretraining gave an OOD success rate of 90%, but here 12.5% pretraining gives success rates of 75.6% and 43.8%.

> ### ⚠️ The success rates in Figure 1 and Figure 2 are not on the same scale
>
> | Scale | Fig 1 OOD SR | Fig 2 HQ SR | Fig 2 Uniform SR |
> |---|---|---|---|
> | 12% / 12.5% | 90% | 75.6% | 43.8% |
> | 50% | 96% | 92.3% | 64.1% |
>
> This means Figure 1's "OOD" is an easier distribution than Figure 2's evaluation, or that the post-training and evaluation protocols differ. The post does not say which subsampling Figure 1 used, or what Figure 2 evaluated and over how many trials. Back-calculating from the ±1 SE widths, the trials per point in Figure 2 range unevenly from roughly 35 to 260. Numbers from the two figures must not be quoted as if they connect.

### 3.3 Learning a new behavior from one demonstration

The narrower the gap, the more leverage each post-training example has. Sunday pushes this to the extreme. Four independent copies of the same pretrained model were each given **just one** demonstration of a different folding technique and trained with plain SFT.

$$\theta_k = \arg\min_{\theta}\ \sum_{t=1}^{T_k} \ell\big(\pi_\theta(o_t^{(k)}),\ a_t^{(k)}\big),\qquad k=1,\dots,4$$

- $\tau_k = \lbrace(o_t^{(k)}, a_t^{(k)})\rbrace_{t=1}^{T_k}$ — the single demonstration of technique $k$, $T_k$ — its length
- Optimization starts from the shared pretrained weights $\theta_{\text{pre}}$
- $\ell$ — loss (its form is unknown because the action head is undisclosed)

Evaluated on held-out garments not used in SFT, all four models executed the newly learned technique. Sunday claims this is the first case of an end-to-end model learning a new behavior from a single demonstration and generalizing to unseen environments. But does the scope of that claim match the experiment described?

> ### ⚠️ The claim is broader than the experiment described
>
> The claim is "generalizing to unseen **environments**," but the evaluation described uses one held-out **garment**. Only "all four succeeded" is reported, with no trial counts or success rates.

Beyond the scope issue, a more fundamental question remains. Is what was learned from one demonstration really a "new" behavior?

> ### 💡 One demonstration selects a mode inside the prior more than it teaches a skill — the same structure as DreamBooth
>
> This is familiar from diffusion. DreamBooth puts a subject into scenes it has never seen from 3–5 photos of it. The few photos do not teach "how to draw an object into a scene." That ability is already in the prior; the photos only **pin down which subject**. LIMA on the LLM side has the same structure. A small SFT does not add a new capability; it selects an output style among the capabilities already there.
>
> Mapped onto ACT-2, it reads like this (my interpretation). People's everyday data very likely already contains the many folding techniques people actually use. If so, the role of one demonstration is not to teach "folding" but **to pick one of several techniques**, and generalization to held-out garments is supplied by the prior.
>
> This experiment alone cannot distinguish "acquiring a new skill" from "selecting a mode of the prior." To distinguish them, one would have to teach a technique that is absent from the pretraining data. Either way, the practical implication is the same: post-training becomes a cheap, fast **means of steering**.

### 3.4 The hill-climbing loop — where Wall 1 closes

Pretraining provides the ability to generalize, but one-off demonstrations alone fall short of deployment level. The remaining gap comes from edge cases and failures that surface only when the robot is actually run over and over. Sunday says the same generalization ability that lets it learn a behavior from one demonstration applies equally to **learning from recovery**. Because it owns the robot, the model, the fleet infrastructure and the data operations, this loop turns very fast. The details are deferred to a separate technical post.

This loop is the last piece of the narrative. Its value hinges on the two earlier observations.

| Condition for the loop to hold | Basis |
|---|---|
| What is fixed in-house must also be fixed in the wild | §3.1 — gap reduction |
| One failure must be fixable with little data | §3.3 — 1-shot SFT |

If both conditions hold, "find a failure in-house → a little recovery data → post-training → transfer to unseen homes" works, and **Wall 1 closes.**

Structurally it looks like a DAgger-style loop that collects corrective data in the failure states the policy falls into by itself. But it is not disclosed whether the recovery data is human correction or adopted autonomous recoveries. From the LLM side, it has the same shape as a data flywheel that feeds production failure logs back as targeted SFT data.

---

## 4. Why it works

Sunday's own explanation is one sentence. The stronger the pretrained model, the more the improvements learned from a small amount of in-house data transfer instead of being tied to the environment where that data was collected. It also likens this to the trajectory of LLMs: GPT-1 needed task-specific fine-tuning, but for GPT-3 a prompt and a few examples were enough.

Tying the three observations into one mechanism reads like this (my interpretation).

| Prior | What the post-training data does | Result |
|---|---|---|
| Weak ($D=0$) | Must teach **both** "what to do" and "how to see this environment" | Memorizes the environment → gap 82pp |
| Strong ($D$ = 100%) | Handling of environments is already general, so only "what to do" needs specifying | Only behavior changes, environmental generality is kept → gap ≈ 0 |

The reading is that the role of post-training changes from "teaching" to "selecting," and it is consistent with the 1-shot result of §3.3. The curation result of §3.2 is a question of **how cheaply** such a prior can be built.

---

## 5. The form of proof — the Solve standard (Wall 3)

### 5.1 The definition of Solve

If the recipe is the answer to Walls 1 and 2, the answer to Wall 3 is an evaluation format. Instead of demos, Sunday proposes the **Solve** as the unit of progress. A Solve is reliable performance shown within a declared scope, under a stated adaptation cost.

| Element | Question it answers |
|---|---|
| **Performance** | How well does it do within the boundary (success, quality, speed) |
| **Scope** | What is the distribution of environments, objects and configurations over which that performance is claimed to hold |
| **Adaptation cost** | What extra data, demonstrations, fine-tuning, interventions or system changes each new deployment requires |

The three elements are not equal and parallel; they form a hierarchy. Sunday itself fixes the order: success, quality and speed mean something only after scope and adaptation cost are set. Written as an equation, performance is an expectation over the scope, and the adaptation cost attaches to the policy inside that expectation (my formalization).

$$\mathcal{P} = \mathbb{E}_{e\sim\mathcal{S}}\Big[\ \mathrm{success}\big(\pi_{\theta+\Delta_{\mathcal{C}}(e)},\ e\big)\ \Big]$$

- $e$ — one deployment (a combination of home, garments and starting state)
- $\mathcal{S}$ — the declared scope distribution
- $\Delta_{\mathcal{C}}(e)$ — the adaptation allowed per deployment $e$ (for ACT-2, $\Delta_{\mathcal{C}} = 0$)
- Even at the same $\mathcal{P}$, a narrow $\mathcal{S}$ or a large $\Delta_{\mathcal{C}}$ makes it a different claim. This is exactly the difference in the table of §1.2

From an LLM background, this is less a new idea than **LLM evaluation hygiene transplanted into robotics**.

| Solve | LLM evaluation counterpart |
|---|---|
| Performance | accuracy, pass@k |
| Scope | benchmark (task distribution) |
| Adaptation cost | zero-/one-/few-shot, whether fine-tuned |
| Weights frozen, and data from evaluation homes not used for post-training or model selection | preventing test-set contamination, not choosing checkpoints on the test set |

The GPT-3 paper reporting results separately for zero-/one-/few-shot was exactly making the adaptation-cost axis explicit. In robotics results reporting, performance and scope have been handled, if only implicitly. Then what does raising adaptation cost to the same rank as performance change?

> ### 📌 Adaptation cost is both a technical metric and a business metric
>
> As adaptation cost converges to 0, the marginal cost of a new deployment converges to 0 too, and only then do fleet-level scaling and compounding improvement hold. Conversely, if every home requires data collection, cost grows in proportion to the number of deployments. The real novelty of Solve is raising this quantity to a first-class metric beside performance.

### 5.2 The boundary ACT-2 declared

| Item | Declaration |
|---|---|
| Scope — garments | T-shirts, long sleeves (thick/thin), polos, sleeveless, blouses, pants, leggings, shorts / XXS–8XL / varied colors, materials, thicknesses and textures. Excludes socks, bras, underwear and accessories (usually not folded but paired, sorted or hung) |
| Scope — scenes | Unseen rooms, varied beds, work surfaces and lighting, from the left, right or foot of the bed |
| Scope — initial configuration | Starting from a basket, a pile, on the bed or on the floor, in arbitrary orientation, naturally crumpled |
| Adaptation cost | Zero per home. No home- or garment-specific data at deployment, no expert demonstrations in the target home, no post-training, the same checkpoint and the same system configuration for every evaluation |

Sunday states that it declared the scope and adaptation cost before evaluation, fixed the rubric in advance and did not revise it, and documented the grading procedure. Data from the evaluation homes and the actual garments were used neither for post-training nor for model selection, and the weights were frozen throughout evaluation. It is the same structure as pre-registration in clinical trials, because adjusting the boundary after seeing the results lets any result be reconstructed into whatever narrative one wants.

Then does this declaration fully satisfy Solve's own definition?

> ### ⚠️ The gap between Solve's own definition and ACT-2's declaration
>
> Solve's definition includes **intervention** in adaptation cost. But ACT-2's declaration covers only data, demonstrations, post-training, checkpoint and system configuration; it does not report resets between trials, who arranged the garments, or whether there was remote intervention. The accurate phrasing is not "adaptation cost 0" but "0 on the declared learning-related dimensions." The pre-declaration, too, is the company's own statement, not an external registry.

With this, half of Wall 3, **making a single claim interpretable**, is solved. That the other half, **comparison between claims**, still remains is confirmed in §7.

---

## 6. Experiments — Performance within the declared boundary

### 6.1 Success rate

Success is defined as autonomously folding the garment through to stacking it. Over 785 autonomous trials in unseen homes, across 9 garment types, the overall result is **99.1% (±0.3% SE), 778 successes** (Figure 5).

| Garment type | Trials | SR | Failures (back-calculated) |
|---|---|---|---|
| T-shirts | 312 | 99.0% | 3 |
| Pants | 85 | 98.8% | 1 |
| Leggings | 54 | 96.3% | 2 |
| Blouses | 19 | 94.7% | 1 |
| Shorts / long sleeve (thick) / long sleeve (thin) / Polos / sleeveless | 98 / 85 / 79 / 46 / 7 | 100% | 0 |
| **Total** | **785** | **99.1%** | **7** |

It is also broken down by environmental variable (Figure 6).

| Variable | SR by category (trials) |
|---|---|
| Starting configuration | pile on bed 98.8% (514) · basket on bed 100% (73) · basket on floor 99.5% (198) |
| Robot position | left of bed 98.7% (315) · right 100% (271) · foot 98.5% (199) |
| Sheet color | light 99.2% (527) · dark 98.0% (149) · colored 99.6% (273) — groups overlap |

Back-calculating, 6 of the 7 failures came from the "pile on bed" starting configuration.

If the error on 99.1% is ±0.3%, that is a very narrow interval. How far can this error bar be trusted?

> ### ⚠️ ±0.3% is the error when trials are independent of each other
>
> The reported standard error is the binomial formula as is.
>
> $$\mathrm{SE} = \sqrt{\frac{\hat p(1-\hat p)}{n}} = \sqrt{\frac{0.991\times0.009}{785}} \approx 0.34\%$$
>
> - $\hat p$ — observed success rate, $n$ — number of trials
>
> But the trials are clustered within the same home, the same session, the same pile of clothes. Accounting for within-cluster correlation makes the error larger.
>
> $$\mathrm{SE}_{\text{eff}} = \mathrm{SE}\cdot\sqrt{1+(m-1)\rho}$$
>
> - $m$ — average trials per cluster (e.g., per home)
> - $\rho$ — within-cluster correlation
>
> With 50 trials per home, even $\rho=0.02$ makes the SE about 1.4 times larger (hypothetical numbers). The number of homes is not reported, so the actual size is unknown. The differences between categories also rest on 1–3 failures. Blouses, "the hardest garment," had 1 failure in 19 trials. That Sunday offered the cause of the blouse underperformance (light and highly deformable, so few geometric cues for grasping and alignment) only as a hypothesis is the right stance for this sample size.

### 6.2 Quality

Separately from success, all 778 completed folds were graded on a 5-star rubric. Each starts at 5 and loses 1 point per defect category. The 2-inch threshold is used because beyond it a folded garment shows visible unevenness and becomes hard to stack.

| Defect (−1★) | Criterion |
|---|---|
| Overfold | More than 2 inches of fabric folded inward |
| Misalignment | Corresponding edges misaligned by more than 2 inches |
| Unfolded element | A sleeve, pant leg or hood sticking out by more than 2 inches |
| Stacking error | Protruding more than 1/3 of the stack width, or the fold disturbed |

The mean is **4.72/5**, with 98.3% at 4★ or above and 73.8% at full marks (Figure 7). Among types with enough samples, means range from 4.63 for polos to 4.88 for leggings.

Grading ran in two rounds. An annotator grades first, another annotator reviews independently, and a review lead resolves disagreements. Annotators were trained beforehand on reference folds. But inter-rater agreement is not reported, and the side-by-side comparison with humans is an illustration, not a controlled comparison.

### 6.3 Speed

The measured interval runs from the moment the robot starts picking up the garment until it places the folded garment on the stack, including autonomous retries and recovery time. Over the 778 successes, the **median is 2 min 13 s and the mean 2 min 19 s**, with per-type medians ranging from about 1 min 14 s for shorts to about 2 min 44 s for thin long sleeves (Figure 8).

Because this is the distribution of successful trials only, the time taken by the 7 failures and the setup time between trials are left out. It cannot be converted into "how many minutes to process a basket of laundry."

### 6.4 Emergent behavior

Sunday shows behaviors that appeared under long-tail conditions without being explicitly taught, in three categories.

| Category | Examples |
|---|---|
| **Edge-case recovery** | Retrieving a garment dropped on the floor, reorienting a garment, resuming folding after the state changes, fine adjustments for quality |
| **Replanning under disturbance** | Replanning instead of following a fixed sequence under a child's interference, adversarial disturbance and extreme lighting contrast |
| **Extended workspace** | Moving, adjusting height and leaning the body to handle everything from baby clothes (16″×8″) to 8XL shirts (38″×42″) and large towels (53″×28″) |

The last category is presented as grounds for the full-stack approach. The argument is that because the model runs on a mobile body rather than a fixed tabletop, its workspace is physically wider. All are qualitative examples, and their frequency is not reported.

---

## 7. Positioning — among neighboring work

**Lineage.** ACT-1 showed, separately, pretraining without robot data, zero-shot generalization to unseen homes, and high-difficulty dexterity. ACT-2 moves the question to **how reliably those abilities hold as environments change**.

**Where Sunday places itself.** In a footnote, Sunday places itself on the line of work showing that scaling pretraining reduces downstream adaptation cost.

| Cited work | The context Sunday points to |
|---|---|
| GPT-1 → GPT-3 | From task-specific fine-tuning to a few examples — scaling pretraining reduced adaptation cost |
| π0 | Rapid acquisition of new skills through fine-tuning on top of broad pretraining |
| GEN-0 / GEN-1 | Scaling physical pretraining cut task-specific robot data to about 1 hour |

Of these, GEN-1 even shares the same pretraining data strategy. Setting two results with the same hypothesis side by side shows what the Solve format actually changes and what it cannot.

> ### 🔗 GEN-1 (Generalist AI, 2026-04) — the same hypothesis, a different vocabulary of measurement
>
> The two companies make the same bet: pretrain without robot data on human data collected with wearable devices, and reach 99%-level reliability with a small amount of robot data. What differs is **what they offer as evidence**.
>
> | | **ACT-2** | **GEN-1** |
> |---|---|---|
> | Pretraining data | Sensorized human data (scale undisclosed) | Human data from wearable devices, zero robot data, 500K+ hours |
> | Task-specific robot data | In-house Memo fleet (amount undisclosed) | About 1 hour per task |
> | Post-training | SFT + recovery data loop (details undisclosed) | Post-training techniques + RL (learning from experience) + multimodal human guidance + inference-time techniques |
> | Evaluation vocabulary | **Solve** = Performance · Scope · Adaptation cost | **Mastery** = Reliability · Speed · Improvisation |
> | Evidence of reliability | 785 trials in unseen homes, broken down by condition | N consecutive runs without intervention (e.g., 86 consecutive T-shirts) |
> | Unit of adaptation | Zero per home | About 1 hour per task |
>
> The two results overlap on T-shirt folding. ACT-2 is 309 of 312 in unseen homes; GEN-1 is 86 in a row without intervention. Both are high, but **which is better cannot be said.** GEN-1 did not declare a scope, and that is exactly the situation Solve takes issue with.
>
> At the same time, Solve's limit shows too. Even if GEN-1 had declared a scope, Solve contains no measure for comparing the breadth of two scopes. That is the remaining half of Wall 3.

---

## 8. Limitations

**What Sunday acknowledges**

- **Post-training details** — deferred to a separate technical post.
- **Other capabilities** — vacuuming, tidying toys, zipping, turning pants inside out and making coffee are being learned by the same model but are unverified by the Solve standard.
- **Blouses** — the cause of the lower performance is a hypothesis.
- **Deployment** — home deployment is planned as a beta this fall; it is not a completed deployment.

**Further points to note**

- **Not reproducible** — architecture, parameters, data scale and curation criteria are all undisclosed. In particular, the variable §3.2 showed to matter most, what counts as high-quality, is not disclosed.
- **First-party evaluation** — both the pre-declaration and the grading are the company's own statements, with no external reproduction.
- **Uneven statistical weight** — 99.1% (785 trials) is solid. Figure 1 (50 trials per point, ceiling effect), Figures 2–4 (trials per point undisclosed, 3+1 points per series) and the 1-shot experiment (no numbers) are much lighter. The three must not be read with equal weight.
- **No comparison across scopes** — as §7 showed, Solve makes a single claim interpretable but defines no measure of scope breadth to rank two claims.
- **Excluded axes** — sock pairing (combinatorial matching, instance re-identification) and hanging (tool use) are a different kind of difficulty from folding. Socks in particular were a task ACT-1 showed as a dexterity demo, finding and balling pairs, yet they are left out of this Solve. A scope declaration should be read as two parts: "coverage within the declared axes" and "axes left out of the declaration."

---

## 9. Closing — what this piece suggests

ACT-2's contribution is not a new algorithm. It is that **it argues with data for the economics of a development loop in which "in-house hill-climbing is deployment improvement," and presents that argument in an interpretable form**.

And this post reads unusually well to someone with an LLM or diffusion background.

- **Solve is the robotics version of LLM evaluation hygiene.** It attaches benchmarks, k-shot and contamination prevention to robot success rates. It is the part of this post easiest to transplant into another domain.
- **The family of scaling curves by curation** has the same shape as the observation in LLM pretraining that quality filtering shifts the curve at a given token count. Here it goes further, showing that the benefit is large when data is scarce and vanishes on the full data.
- **The conditional validity of the val loss proxy** is a constraint familiar to anyone who knows the relationship between loss and FID in diffusion. The ordering can be trusted only within the same model family.
- Read as "a little steering on top of a strong prior," like DreamBooth and LIMA, **1-shot SFT** is a signal that robot post-training is entering the same phase as LLM post-training.

What was needed after the Wright brothers' 12 seconds was not a longer flight but a timetable. Solve proposes the form of a timetable for robotics. But there is not yet a table for comparing routes.

---

## Appendix — glossary

| Term | Definition |
|---|---|
| **generalization gap** | In-domain success rate − out-of-domain success rate after the same post-training. $G(D) = \mathrm{SR}\_{\text{ID}} - \mathrm{SR}\_{\text{OOD}}$ |
| **in-domain / out-of-domain** | The environments the post-training data came from / held-out environments, objects and configurations |
| **uniform vs high-quality subsampling** | Drawing the same amount at random vs selecting by curation criteria |
| **validation loss proxy** | A cheap signal for choosing the data mixture before physical evaluation. Valid only within the same curation series |
| **Skill Capture Glove / Skill Transform** | A collection glove with the same structure as Memo's hand / a transform that aligns human traces to robot observations |
| **hill-climbing loop** | Find in-house failures → recovery data → post-training → transfer to deployment |
| **Solve** | Reliable performance shown within a declared scope, under a stated adaptation cost |
| **adaptation cost** | The extra data, demonstrations, fine-tuning, interventions or system changes each new deployment requires |

**Original** — [ACT-2 Preview: Generalizing Reliability](https://www.sunday.ai/blog/act-2-preview) · **Full evaluation videos** — [YouTube](https://www.youtube.com/watch?v=a2HZyURUE_o) · **Predecessor** — [ACT-1](https://www.sunday.ai/journal/no-robot-data) · **Comparison** — [GEN-1](https://generalistai.com/blog/gen-1)
