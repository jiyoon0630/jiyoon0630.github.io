---
layout: paper
lang: en
ref: dyna-2-scaling-law
kind: paper-review
title: "Dyna-2: A 1-Million-Hour Scaling Law for World-Action Models"
date: 2026-08-11 12:00:00 -0700
paper_date: 2026-08-01
venue: "Dyna Robotics technical report (2026-08)"
tags: [WAM, World-Model, Scaling-Law, Robot-Foundation-Model, Cross-Embodiment, Paper-Review]
authors: "Dyna Robotics (institutional authorship; no individual authors listed)"
affiliations: "Dyna Robotics"
summary: "A transfer scaling law measured out to one million hours of human egocentric video — robot metrics improve without a single robot trajectory, and the driver is the video loss, not the action loss."
paper_url: "https://www.dyna.co/dyna-2"
---

> **Core claim** — The right source of robot pretraining data is human egocentric video, and scaling it to one million hours improves robot metrics predictably **without seeing a single robot trajectory**. And what creates that transfer is not the action loss but the **video-prediction loss**.

---

## Introduction

The field does not yet have even the most basic agreement on how to scale robot foundation models. The Dyna-2 report opens by organizing that vacuum into three questions.

> ⓵ What is the **right source of pretraining data** for robot learning?
> ⓶ If that source is scaled, does a **scaling law for robot performance** emerge?
> ⓷ What **modeling and objective choices** are required for that law to hold?

These three questions are the report's skeleton. This post follows the same order, building up the concepts along the way so that a reader new to scaling laws and world models can see the **reason** behind each design decision.

One thing to state up front. This is not an arXiv paper but a **company technical report**. There is no peer review, and the data corpus, model size, and training compute are undisclosed. It is still worth covering because the **measurement design** it proposes takes direct aim at one of the field's open questions.

---

## 1. The Problem — Where Does Robot Pretraining Data Come From

### 1.1 The ceiling on action labels

The report points to two existing data sources: **teleoperation** and **dedicated capture rigs** (the UMI family). Both are precious action-labeled data, and Dyna collects and trains on them too. There is one problem.

> **Every hour of that data has to be deliberately produced.** That sets the ceiling on how far pretraining can be pushed.

So the report derives the answer from the goal itself. If the goal of a general-purpose robot is "every economically valuable task humans do today," then the right source of pretraining data must be **sensor recordings — video — of humans doing exactly those tasks**. That data already exists at effectively unlimited scale, and it contains exactly what a manipulation policy must learn: how a scene unfolds, how objects respond to contact, how hands interact with objects.

There is of course an immediate objection: **a human body is not a robot body.** So the report's thesis is stated conditionally.

> If a **transfer scaling law** can be established — that is, if a scaling law over human data implies one over robot data — then the technical path afterwards becomes relatively clear: collect more sensorized human data, and move robot form factors closer to the human one.

### 1.2 What prior work has not yet established

The report distinguishes itself from related work on three points. These are the **walls** this post will collect on later.

**⛔ Wall 1 — the ceiling on the measured scale axis.** Existing scaling studies in robotics measured environment and object diversity, compute, and model size. The closest work to this one, EgoScale, measured action-prediction loss on human egocentric data up to roughly **20,000 hours**. Above that is unmapped territory.

**⛔ Wall 2 — no scaling law that crosses the embodiment gap.** EgoScale's law was measured on held-out **human** loss, and robot results were obtained **only after aligned human–robot mid-training**. Meanwhile recent foundation models such as RDT2 and LAP showed zero-shot cross-embodiment transfer, but **only at a single data scale**. They showed that transfer is possible, not **how it scales** — and their pretraining still contains **robot-form data** such as UMI.

**⛔ Wall 3 — we do not know which objective produces the law.** Even when a scaling law is observed, it has never been separated into a function of data quantity versus a function of modeling choices.

The three walls are collected in §4.2, §4.3–4.4, and §4.5 respectively.

---

## 2. Background — The Minimum Needed to Read the Three Walls

### 2.1 Scaling laws, and the "transfer" scaling law

An ordinary scaling law states that performance follows a power law in data quantity.

$$L(D)\ =\ A\cdot D^{-\alpha}$$

| Symbol | Meaning |
|---|---|
| $D$ | pretraining data quantity — here, **hours of human video** |
| $L$ | the evaluation metric: action-prediction MSE / L1 / accuracy@$\tau$ |
| $A$ | the prefactor; the intercept on a log-log plot |
| $\alpha$ | the **scaling exponent** — the log-log slope, i.e. "how much better does 10× the data make it" |

The transfer scaling law is one step beyond this.

```
  ordinary scaling law
     D(human) increases  --->  L(human held-out) decreases

  transfer scaling law
     D(human) increases  --->  L(robot  held-out) decreases
                                ^^^^^^^^^^^^^^^^^
```

- The underlined part is the point. The arrow's **destination is a different body**, and not one trajectory from that body appears during training.
- What the report claims as a first is precisely this second statement.

### 2.2 WAM — what a world-action model is

This is the report's central concept. The fastest way in is a contrast with VLA.

| Axis | **VLA** (RT-2, OpenVLA, π₀, GR00T N1) | **WAM** (Dyna-2) |
|---|---|---|
| initialized from | a pretrained **VLM** | a **video-diffusion** backbone |
| prior it brings | language and visual semantics | how the physical world unfolds |
| what it predicts | actions only | **future video + future actions** |
| training signal | action labels | action labels + **unlabeled video** |

In one sentence — a WAM is **a single generative model that denoises future video and future actions, jointly or separately.**

### 2.3 pseudo-action — how to put action labels on human video

There is an immediate blocker. Human video has no robot action labels, yet the action stream has to be trained. Where does the action supervision come from?

> ### 💡 Wrist pose and thumb–index aperture stand in for action labels
>
> The report's data pipeline is cleaning → **hand-pose extraction** → validation → filtering. Episodes that pass the hand-pose quality bar are annotated with a **3D hand-pose track**, from which two pseudo-actions are derived.
>
> | pseudo-action | How it is derived | Robot counterpart |
> |---|---|---|
> | end-effector trajectory | **wrist pose** | the 6-DoF pose of the gripper or hand |
> | grasp signal | a continuous value derived from the **thumb–index aperture** | gripper open/close |
>
> This is the boldest compression in the approach: the twenty-plus degrees of freedom of a human hand are folded into two channels — **"where is the wrist" and "how far is it open."**
>
> The report also states that it deliberately applied **no visual or embodiment-specific preprocessing at all.** It performed no manipulation to narrow the visual or kinematic gap between human and robot data, and the reason is "so that the effect of scale alone can be seen."
>
> This decision keeps the argument clean: with alignment in place, you could not tell whether transfer came from alignment or from scale. Absolute performance pays for it — and the report does concede that alignment and co-training would do better.

### 2.4 The metrics, and a defense of the metrics themselves

Scaling claims have a trap of their own: Schaeffer et al.'s point that **nonlinear or discontinuous metrics can manufacture fake emergence out of a smoothly improving model.** The report cites this directly and, so that the conclusion does not hinge on metric choice, reports **two continuous errors plus two discrete threshold accuracies.**

For a predicted action chunk $\hat a$ against ground truth $a$, MSE and L1 are averaged over action dimensions and the chunk horizon, and threshold accuracy is:

$$\text{acc@}\tau\ =\ \frac{\#\{\,|\hat a_i - a_i|\le\tau\,\}}{\#\{\text{all action dimensions}\}}$$

| Symbol | Meaning |
|---|---|
| $\tau$ | tolerance, in normalized action units |
| $\tau = 0.5$ | by the report's internal experience, a measure of overall **motion intent** → suited to human→robot transfer studies |
| $\tau = 0.1$ | a measure of **motion precision** → suited to within-embodiment (human→human) trends |

This split reading of $\tau$ is not decoration. It becomes the decisive lens when comparing two slopes in §4.3.

### 2.5 Flow matching

Dyna-2 trains with flow matching, corrupting each modality along a straight path toward noise.

$$z_t\ =\ t\,z\ +\ (1-t)\,\varepsilon_z,\qquad a_t\ =\ t\,a\ +\ (1-t)\,\varepsilon_a,\qquad \varepsilon_z,\ \varepsilon_a\sim\mathcal{N}(0,I)$$

| Symbol | Meaning |
|---|---|
| $c$ | conditioning context — past frames, proprioception, language instruction |
| $z$ | latent future **video** |
| $a$ | future **action chunk** |
| $t$ | flow time; $t=0$ is noise, $t=1$ is data |
| $u_\theta$ | the **velocity-field** network that denoises the corrupted sample |

---

## 3. Method — Architecture and Training Objective

### 3.1 Mixture of transformers

Dyna-2 is a **mixture of transformers** built on a video-diffusion backbone. Each input modality (video, action) is tokenized separately, has **its own set of DiT layers**, and references the other through attention. Proprioception is tokenized and fed **directly into the action transformer.**

```
   Instruction (text)
         |
   [ Text encoder ]
         |
         |  cross-attn  -->  VIDEO tokens only
         v
   +--------------------------------+
   |  VIDEO DiT stack   (deep)      |   causal mask
   +--------------------------------+
         ^              |
         |              |  attention at EARLY layers only
         |              v
   +--------------------------------+
   |  ACTION DiT stack  (shallow)   |   bidirectional, no causal mask
   +--------------------------------+

   in   : context frames + noised z_t  ->  VIDEO stack
          proprio + noised a_t         ->  ACTION stack
   out  : u_vid (video velocity)   ... training only
          u_act (action velocity)  ... the deployed policy
```

The masking design is asymmetric.

| Stream | Attention | Why |
|---|---|---|
| video tokens | **causal mask** | the future must be generated in temporal order |
| action tokens | **bidirectional** (no causal mask), attending to video tokens in the observed context | an action chunk is predicted all at once |
| text | video tokens cross-attend to text. **Text never influences action tokens directly** | — |

That last row looks odd. If the language instruction never reaches the actions directly, how does this model follow instructions? The question is collected in §5.3.

**Depth design.** Early architecture exploration found that DiT-style video-diffusion architectures **hold most of their temporal reasoning in the early layers**, so the action transformer was made **deliberately shallow** and **joins the video stream only at early layers.** This is reported to have greatly improved real-time inference latency at no cost in performance.

### 3.2 The co-training loss — two separate velocity fields

The variant used in the scaling-law study is co-trained on video prediction and action prediction. It shares one trunk but is fit as **two distinct marginal velocity fields.**

$$\mathcal{L}_{\mathrm{co}}(\theta)\ =\ \mathbb{E}\big\lVert u_\theta^{\mathrm{vid}}(z_t;\,t,\,c)\ -\ (z-\varepsilon_z)\big\rVert^2\ +\ \lambda\,\mathbb{E}\big\lVert u_\theta^{\mathrm{act}}(a_t;\,t,\,c)\ -\ (a-\varepsilon_a)\big\rVert^2$$

| Symbol | Meaning |
|---|---|
| $u_\theta^{\mathrm{vid}}$ | the video velocity-field head |
| $u_\theta^{\mathrm{act}}$ | the action velocity-field head |
| $(z-\varepsilon_z)$, $(a-\varepsilon_a)$ | the flow-matching regression target for each modality (the velocity of the straight path) |
| $\lambda$ | the action-loss weight. **Its value is not disclosed in the report** |

And then comes the most important line in the report.

> Because $u_\theta^{\mathrm{act}}$ **never takes $z_t$ as an argument**, the video loss can shape the shared representation while the model stays **reactive at inference.** The policy neither generates future video nor attends to predicted future video at inference time.

It carries the name "world model," yet it does not imagine the world in the actual control loop. In what sense, then, is this a world model?

> ### 💡 Here, world modeling is a representation learner, not a planning tool
>
> In robotics, "world model" usually evokes Dreamer-style model-based RL — rolling the future out in latent space and planning or learning inside that imagination. **Dyna-2 is not in that family.**
>
> | Axis | Model-based-RL-style world model | **Dyna-2's world modeling** |
> |---|---|---|
> | when future prediction is used | **at inference** — roll out in imagination and plan | **only during training** — solely to shape the representation |
> | form of the policy | a planner, or imagination-based learning | a **reactive** feedforward policy |
> | inference cost | pays for rollouts | zero video-generation cost |
>
> So the video loss is an **auxiliary objective.** It forces the shared trunk to represent "how a scene unfolds," and the action head sits on top of the representation shaped that way. At deployment only the action head runs.
>
> This setup is familiar from generative modeling: using a generative objective as a means of representation learning and then discarding the generation itself — exactly the idea behind using diffusion as a representation learner. It is also why the one-step video generation of §5.4 sits in its own section: the generative capability is not part of the policy but **a separate artifact for planning and evaluation.**

---

## 4. The Scaling Law — Collecting on the Three Walls

In this report the method is short and the measurement is the body of the work. It answers four questions in order.

### 4.1 Experimental design — nested subsets

From a corpus of more than one million hours of human manipulation video, it builds **nested subsets of exactly 1,000 / 10,000 / 100,000 / 1,000,000 hours.**

```
  1k     [##]
  10k    [####]
  100k   [########]
  1M     [################]

  each rung : identical proportion from every data source
              a larger budget only ADDS data, never EXCHANGES
```

- Most of the data is **head-mounted egocentric recording** of everyday manipulation — cooking, tidying, folding, assembly — collected through data partners and in-house operations.
- The reason for the nested design is clear: a larger budget **adds** data without **exchanging** it, so differences between points on the curve **cannot be explained by distribution shift between subsets.**
- Human evaluation is fixed on a **100-hour validation set** disjoint from every subset. Robot evaluation covers **39 tasks** — 12 internal stationary YAM bimanual plus 27 external xdof ABC. External data was deliberately mixed in so the evaluation would not be biased toward self-designed tasks.
- Training and evaluation settings are identical at every point, and the only variable is hours of experience. Compute and model-size scaling are left as future work.

### 4.2 Collecting on Wall 1 — does the law hold out to one million hours?

This is the held-out human evaluation. To remove checkpoint bias, **10 checkpoints from a late-step window are evaluated and mean and standard deviation** are reported.

**Figure 5 — held-out human data**

| Metric | Fit | $R^2$ | 1k → 10k → 100k → 1M |
|---|---|---|---|
| MSE ↓ | $0.0691\cdot D^{-0.0184}$ | 0.919 | 0.062 → 0.057 → 0.056 → 0.054 |
| L1 ↓ | $0.151\cdot D^{-0.0132}$ | 0.879 | 0.140 → 0.131 → 0.129 → 0.127 |
| acc@0.1 ↑ | $0.0116\cdot D^{+0.0606}$ | 0.926 | 0.017 → 0.021 → 0.024 → 0.026 |
| acc@0.5 ↑ | $0.357\cdot D^{+0.0203}$ | 0.865 | 0.40 → 0.44 → 0.45 → 0.47 |

All four metrics improve monotonically and each is well described by a power law. The observation the report emphasizes is that **the threshold metrics improve fastest**: across the whole ladder acc@0.1 rises 51% while MSE improves 12%. Read together with §2.4, that means **precision** is the main beneficiary of scale within the same embodiment.

**Wall 1 is collected.** The measured axis extends more than two orders of magnitude beyond EgoScale's ~20,000 hours, and the curve does not bend all the way to the end.

> ### ⚠️ Fact-check — attaching $R^2$ to a power law fit on four points
>
> Each curve is a two-parameter fit over **four data points.** With only two degrees of freedom left, $R^2 = 0.865$–$0.926$ does not carry the evidentiary weight it would in an ordinary regression. An $R^2$ obtained by drawing a line through four points says little more than "it is monotone and does not bend badly on log axes."
>
> More striking is the **absolute size of the change**: 1000× the data takes held-out MSE from 0.062 to 0.054. The exponent $-0.0184$ is very shallow even compared with language-model scaling exponents.
>
> So the real message of this section is not "human data yields dramatic improvement" but **"the architecture absorbs a million hours without breaking, and monotonicity holds."** The report itself concludes the section with "architecture is sound and can absorb millions of hours," so its own phrasing is not an overstatement.

### 4.3 Collecting on Wall 2 — does the human law cross to robots?

This is the report's central claim. The same checkpoints — which **saw not one robot trajectory in pretraining** — are evaluated as-is on a held-out robot dataset of 39 tasks, with no adaptation and no fine-tuning.

**Figure 7 — zero-shot robot data**

| Metric | Fit | $R^2$ | 1k → 10k → 100k → 1M |
|---|---|---|---|
| zero-shot action MSE ↓ | $0.306\cdot D^{-0.0713}$ | 0.884 | 0.180 → 0.174 → 0.124 → 0.117 |
| zero-shot acc@0.5 ↑ | $0.0241\cdot D^{+0.139}$ | 0.918 | 0.067 → 0.074 → 0.136 → 0.159 |

Every metric aligns monotonically with the scale of human pretraining data. The report notes an empirically observed **inflection between 10k and 100k** and reads it as suggesting that "once sufficient coverage is secured, cross-embodiment knowledge transfer can emerge **from scale alone.**" Indeed the 1k→10k stretch is nearly flat (0.180→0.174, 0.067→0.074) and it moves sharply from 10k→100k (0.174→0.124, 0.074→0.136).

Set beside the table in §4.2, something odd appears: the slope is steeper on the robot side — a **different body** — than on the human side, which shares the training distribution.

> ### 💡 The slope is steeper where there is more room to improve
>
> | | MSE exponent $\alpha$ | prefactor $A$ | value range |
> |---|---|---|---|
> | held-out **human** | $-0.0184$ | 0.0691 | 0.054 ~ 0.062 |
> | zero-shot **robot** | $\mathbf{-0.0713}$ | 0.306 | 0.117 ~ 0.180 |
>
> The robot slope is about 4× steeper. Intuition says it should be the other way around. But read together with the prefactor and the value range, the picture fits: the human side is already crawling **near saturation** at low error, while the robot side still has **large room to improve from high error.** A steeper slope where there is more room is natural.
>
> And the $\tau$ reading from §2.4 overlays here. The metric climbing steeply on the robot side is acc@**0.5** — **overall motion intent.** On the human side the fastest riser was the precision metric acc@**0.1** (51%).
>
> | | Fastest-improving metric | Reading |
> |---|---|---|
> | same body (human) | acc@0.1 | **precision** lives within one body |
> | different body (robot) | acc@0.5 | **motion intent** is what transfers across bodies |
>
> What scale buys first is not "exactly how to move" but "roughly where, and with what intent" — and that is the component that crosses the embodiment gap.

**Wall 2 is collected.** EgoScale produced robot results only after aligned mid-training, and RDT2 and LAP showed transfer only at a single scale. Here robot metrics fall monotonically with human data quantity **with no alignment, no fine-tuning, and no robot-form data.**

### 4.4 Does the offline law carry to real robots?

Offline metrics improving and a robot actually getting the job done are different things. So equal-step checkpoints are taken from each of the four rungs and post-trained on **the same 14-task dataset.** Robot data per task is **at most 10 hours**, and the only variable is hours of human-video pretraining.

Here too, **human–robot alignment and co-training are deliberately omitted.** The report concedes they would help, but the interest is in the improvement attributable to pretraining scale.

**Evaluation setup**

| Embodiment | Tasks |
|---|---|
| 6-DOF YAM arms + in-house parallel-jaw gripper (stationary bimanual) | 11 |
| the same arms + a **WUJI-2 20-DOF multi-finger hand** | 2 |
| an early semi-humanoid robot prototype (language-following task) | 1 |

Because each task has a different native metric, scores are normalized as a **fraction of each task's maximum attainable value** and averaged across the 14 tasks into a single score. Ten trials per task (12 for the language task), a **blind test** run by evaluators not involved in model development.

**Figure 8 — mean normalized score across 14 tasks**

| Pretraining | 1k hr | 10k hr | 100k hr | 1M hr |
|---|---|---|---|---|
| mean normalized score | 20% | 28% | 45% | **53%** |

The 1M-hour model gave the strongest real-world performance overall and was **best on 9 of the 14 tasks.** Two cases the report highlights are especially striking.

**⓵ A task with a critical mass.** On **Lockbox Key Turning**, no checkpoint up to 100,000 hours could turn the key (0% / 0% / 0%). At one million hours: **90%.** Evidence that some capabilities require a threshold amount of pretraining to unlock.

**⓶ The opposite extreme of data efficiency.** **Bottle Cap Untwisting** was post-trained on only **about 10 minutes** of robot demonstration and still climbs 10% / 10% → 40% → 50% — on a 20-DOF multi-finger hand, no less.

> ### ⚠️ Fact-check — monotonicity holds only at the aggregate level
>
> The report states precisely: "aggregated across the 14 tasks, mean normalized performance increases monotonically." **It attaches the aggregate qualifier itself, so the statement is honest.** But unfolding the per-task bars gives a rather different picture, and citations need to keep the distinction.
>
> | Task | 1k | 10k | 100k | 1M | Shape |
> |---|---|---|---|---|---|
> | Rope Tie | 0% | 40% | **90%** | 40% | peaks at 100k, halves at 1M |
> | Food Scooping | 10% | 30% | **80%** | 50% | same |
> | Unsort | **6.4** | 5.5 | 3.6 | 5.8 | **1k is best** |
> | Tote Construction | 20% | 10% | **40%** | 30% | peaks at 100k |
> | Pick & Place | 1.8 | 2.1 | **4.2** | 3.9 | peaks at 100k |
> | First Aid Kitting | 2.0 | 0.2 | 2.9 | **4.8** | collapses at 10k, then recovers |
>
> "1M is best on 9 of 14" inverts to **not best on 5**, and several of those are large regressions against 100k. With 10 trials per task the **resolution is 10 percentage points**, and no confidence intervals are reported. The gap between 40% and 90% looks meaningful; the gap between 10% and 20% is one trial.
>
> The aggregate curve 20 → 28 → 45 → 53% is substantive and the most valuable number in the report. But reading it as "scale monotonically raises every capability" goes beyond the data; what the data supports is **"scale raises the expected value across a portfolio of tasks."** Factoring in that 53% in absolute terms is "a bit over half of each task's ceiling," this is not a performance report but a **trend report.**

### 4.5 Collecting on Wall 3 — what creates the transfer?

Now the most important controlled experiment. The report's question is blunt: **"does the modeling choice of predicting video and action together actually matter?"**

Holding action-data quantity fixed, it runs a three-way comparison varying only the training objective and data composition. Human video with hand-pose annotation is fixed at 5k, 50k, and 100k hours, and the same Dyna-2 architecture is trained under three recipes.

| Recipe | What it does |
|---|---|
| **⓵ action-only** | action loss only. No world modeling |
| **⓶ joint** | predicts both action chunks and future video on the same dataset |
| **⓷ video co-training** | joint, plus video prediction on **an equal amount of additional human video with no action labels** |

All three conditions are evaluated zero-shot on the 39-task robot suite at the same training step (Figure 10).

- **Future prediction in any form beats action-only by a wide margin.** The joint recipe beats action-only on **all 39 tasks (39/39) at every action scale.**
- action-only shows **severe and unpredictable overfitting** as data grows.
- joint overfits less but **also does not scale with data.**
- This trend **reverses only** when a large volume of action-label-free human video is added for video prediction. That gain is absent at the small scale (5k), and **the gap widens as data scale grows.**

A natural next question follows: what happens if action data is held fixed and **only video** is scaled?

This is not merely academic curiosity. The report gives a practical reason: human video is abundant, but **accurate hand-pose extraction is hard**, not every capture setup yields poses that clear the internal quality bar, and annotation at the scale of millions of hours is itself a huge infrastructure problem — so **there is always a lag between total data volume and action-labeled data volume.** In a regime that pretrains on human data, it is safe to assume **a large reservoir of unannotated video always remains.**

**Figure 11 — action fixed, video scaled alone**

| Action data (fixed) | Video-only hours | zero-shot robot MSE ↓ |
|---|---|---|
| 50k hr | 0 → 1k → 10k → 50k | **0.34** → … → **0.120** |
| 250k hr | 0 → 250k → 750k | **0.10** → … → **0.084** |

At two scales an order of magnitude apart, training without video (action-only) is consistently worse, and **generalization improves monotonically from scaling video alone.**

But evaluating the same checkpoints on **human** data produces a result the report itself calls surprising "at least to us."

> ### 📌 The gain from video scaling appears only cross-embodiment
>
> Figure 12 shows relative improvement against each domain's own "zero video" condition.
>
> | Evaluation domain | Relative to zero video |
> |---|---|
> | **human** (same embodiment) | **104%** — no change, if anything slightly worse |
> | **robot** (different embodiment) | **34%** — dramatic improvement |
>
> Scaling human video has no effect, or a marginally negative one, on human-data evaluation. The report's hypothesis is that **video training dilutes the action-learning gradient, and same-embodiment evaluation only needs sufficient action data.**
>
> This is the report's sharpest research contribution. Video co-training is **not a device for raising in-domain performance but a device for buying generalization across the embodiment gap.** Hence its conclusion: "video co-training is the primary driver for establishing cross-embodiment transfer scaling law."
>
> The report defends this from first principles too: the promise of world modeling was always to grant the model **a general understanding of the physical world**, and such an ability ought to help generalize to a completely unseen embodiment.

**Wall 3 is collected.** The scaling law is not a function of data quantity alone but **a function of the objective and the data composition**, and a controlled experiment isolates the video loss, not the action loss, as the component that creates transfer.

---

## 5. Additional Capabilities — Results Outside the Scaling Law

The experiments in this section were run with the **"production" Dyna-2 model** under a different training recipe, as the report states. The ladder experiments of §4 and the results of §5 are not the same checkpoints.

### 5.1 WAM vs VLA — an apples-to-apples comparison

It starts from the observation that the community has no consensus on whether WAM or VLA is the better architecture, and that existing comparisons differed in training data, compute budget, evaluation protocol, and degree of tuning.

An early version of Dyna-2 (WAM) is compared against Dyna Robotics' previous production VLA, **Dyna-1.** Dyna-1 uses a similar mixture-of-transformers architecture for action prediction but is **initialized from Qwen3-VL-4B.** The same pretraining and post-training datasets and hyperparameters are used, and post-training starts from three different pretraining checkpoints per architecture to remove checkpoint-selection bias.

**Figure 13 — 7 benchmark tasks, 21 task×checkpoint cells per architecture**

| Metric | WAM / VLA |
|---|---|
| success rate | **1.55×** |
| quality grade | **1.12×** |
| head-to-head win rate | WAM 65% / VLA 29% / ties 6% |

VLA's wins came mostly from the **earliest pretraining checkpoints** — before WAM's pretraining advantage had accumulated.

> ### ⚠️ Fact-check — the WAM in this comparison is not the WAM of this report
>
> The report discloses two things itself. The early Dyna-2 used here predates most of the report's content, so ⓐ it **lacks the 1M-hour pretraining** and ⓑ **the whole model was supervised with an action-only loss.** The entire experimental pipeline was also tuned for VLA: the dataset was collected and curated under a VLA recipe and the hyperparameters were inherited from VLA tuning. The report therefore asks that the result be read as a **lower bound.**
>
> The disclosure is conscientious, but there is one more implication. The WAM in this comparison **lacks the very video loss that §4.5 identified as the cause of transfer** — it is action-only. So what the 1.55× measures is less "the superiority of world modeling" and closer to **"video-diffusion-backbone initialization vs. VLM initialization."**
>
> Both are interesting results, but they are not the same claim. When citing, §4.5 (the video loss creates transfer) and §5.1 (WAM beats VLA) should be treated as **separate pieces of evidence.**

**Qualitative case.** On a vegetable-cutting task, Dyna-2 cut celery more thinly and uniformly than Dyna-1, closely matching an expert demonstration, and kept going under changed lighting, near-total removal of lighting, partial removal of visual input, and a person blocking the front while repeatedly putting cut pieces back. The description of that last case is interesting: the policy did not stop after a fixed number of cycles but **stopped when the cutting board was empty.** For the removed-visual-input case the report adds its own caveat, reporting it as **robustness to sensor loss** rather than an ability to predict unobserved scene state.

### 5.2 Zero-shot real-world deployment

The section opens by noting that in production, "task completion" is not the bar — **quality, throughput, and reliability** are evaluated together. A policy can satisfy the completion criterion and still fail all three.

At a customer site where both models were deployed, operators not involved in model development scored them against customer acceptance criteria (Figure 14).

| | in-house evaluation | **zero-shot customer site** |
|---|---|---|
| Dyna-1 (VLA) | ~100% | **46%** |
| Dyna-2 (WAM) | ~100% | **87%** |

A **41-point gap** at the same post-training budget. Neither model had seen data from the deployment site.

The structure of this table is the point: indistinguishable at 100% in-house, separating only in the field. It is valuable as data showing that **in-distribution benchmarks do not predict deployment performance** — demonstrated by one company across two generations of its own product. That said, the number of sites, the number of trials, and the task composition are undisclosed, so the statistical weight is limited.

### 5.3 Language following — collecting on the §3.1 setup

Back to the question deferred in §3.1: **if text never influences action tokens directly, how does the model follow language instructions?**

First the problem as the report diagnoses it. End-to-end robot policies struggle to follow language instructions, because images carry far more information and because **the continuous action loss can destroy pretrained representations.** **Counterfactual** cases — a scene resembling the training data but a different instruction — are especially hard. On the VLA side, **multi-stage training and backbone freezing** have been marshalled to preserve the backbone's semantic knowledge, producing a fragile and slow pipeline.

> ### 💡 Language enters through the video head, not the action head
>
> Looking again at Dyna-2's wiring gives the answer.
>
> ```
>   text --cross-attn--> VIDEO tokens --early-layer attn--> ACTION tokens
>          (direct)                                          (indirect)
>
>   text ------------------- X ------------------> ACTION tokens
>          no direct path
> ```
>
> Language reaches actions **only by way of the video-prediction objective.** And this is design intent, not a detour. Through video prediction the world-action model gains **a new data source for learning language** — learning what objects and motions mean in the physical world, and putting abundant egocentric data to work.
>
> So "stack the red piece on the yellow one" is not a command injected into the action head but **a condition that the model must be able to render a future video in which that instruction holds.** Language understanding becomes the video loss's burden rather than the action loss's, so the semantic knowledge that VLA tried to protect by freezing the backbone is never placed where the action loss could destroy it in the first place.

**The benchmark** is a counterfactual design that fixes the scene and varies only the instruction.

| Task | What it asks |
|---|---|
| Push/pull jenga block | push or pull a Jenga block in a specified direction |
| Object kitting | pick the specified one of five objects and place it in a bin |
| Piece stacking | red on yellow, or the reverse — **language–space grounding** |
| Napkin manipulation | pick, place, pull, rotate, flip, fold left-right, fold top-bottom, unfold |

Scoring has three levels: achieving the instructed outcome = 1, **attempting the correct language primitive but failing to complete = 0.5**, a different primitive (wrong verb, referent, or direction) = 0.

**Figure 15**

| | action-only<br>(initial corpus) | video co-train<br>(initial corpus) | video co-train<br>(**full corpus**) | $n$ |
|---|---|---|---|---|
| **all four tasks** | 0.35 | 0.67 | **0.96** | 36 |
| Push/pull jenga | 0.44 | 1.00 | 1.00 | 8 |
| Object kitting | 0.10 | 0.35 | **0.95** | 10 |
| Piece stacking | 0.60 | 0.95 | 1.00 | 10 |
| Napkin manipulation | 0.25 | 0.38 | **0.88** | 8 |

Both axes contribute. Changing the **objective** from action-only to video co-training gives 0.35 → 0.67, and raising **data scale** to the full corpus gives 0.96. That the largest gains land on object grounding (kitting: 0.10 → 0.95) and fine motion primitives (napkin: 0.25 → 0.88) is also consistent with the picture in §4.5.

> ### ⚠️ Fact-check — the sample sizes are very small
>
> The overall $n=36$ is the sum of trials across four tasks, and individual tasks have $n = 8$–$10$. With the 0.5 partial-credit rule, **one trial moves the overall score by about 0.03.** The 0.35 → 0.67 → 0.96 trend is wide enough to be hard to overturn, but the "1.00" on jenga and stacking should not be read as saturation — it is a clean sweep over 8–10 trials.

### 5.4 One-step video generation

A separate contribution, independent of the policy. Dyna-2's video generator is **distilled into a one-step student**, cutting latent generation time by roughly **90×**, and the report claims this is **the first pipeline to produce instruction-conditioned manipulation video in a single step** at a quality usable for planning and evaluation. It states plainly that it still falls short of the full-precision teacher.

But few-step distillation is already standard. Why can video not get down to one step?

> ### 💡 Three distinct barriers block the single step
>
> The report separates the failure into three causes.
>
> **⓵ Regression losses collapse to the mean.** Trajectory-regression losses are minimized at the **conditional mean** $\mathbb{E}[x_0\mid z]$. Taking one step from noise returns **the average of all possible futures**, and the result is blurry.
>
> **⓶ Distribution matching leaves the teacher score's domain.** DMD-style methods preserve detail, but their gradient has the form
>
> $$\mathbb{E}_{x\sim p_\theta}\big[(s_\theta - s_q)\,\partial x/\partial\theta\big]$$
>
> which **evaluates the teacher score $s_q$ where it was never trained.** The probability-flow path is locally straight but globally curved, and the value measured on Dyna's own teacher is **about 4 degrees per step, about 58 degrees overall.** A single step leaves the support of $s_q$.
>
> **⓷ Dimensionality makes both worse.** Under the manifold hypothesis $p_\theta$ and $q$ occupy low-dimensional sets inside $\mathbb{R}^D$, and
>
> $$\dim M_p + \dim M_q\ <\ D\ \Longrightarrow\ \chi^2(p\,\Vert\,q) = \infty$$
>
> so the divergence saturates and gradients vanish — unless both are smoothed with noise. Multi-step samplers dodge this by reprojecting onto the data at every step, but **one-step has to carry all of that smoothing alone.**
>
> | Symbol | Meaning |
> |---|---|
> | $s_\theta,\ s_q$ | the score of the student and of the teacher (= the data distribution) |
> | $M_p,\ M_q$ | the low-dimensional manifolds on which $p_\theta$ and $q$ lie |
> | $D$ | the ambient dimension. Far larger for video than for images |
>
> This is why one-step methods work for images but not directly for video: $D$ is far larger, and distribution matching can **collapse to a static clip.**

**The fix is a moving target.** The report reframes one-step generation as a **pursuit game — a control problem.** Instead of fixing the teacher as a static target, it lays down a **continuous path of targets** running from one that is **reachable at initialization** to the data, and advances that path according to an online measurement of the student. The target retreats toward the data only as fast as the student can follow.

$$x = G_\theta(\varepsilon),\quad \varepsilon\sim\mathcal{D}(0,I);\qquad \{q_r\},\ r\in[0,1],\quad q_0 = \text{reachable},\ q_1 = \text{data}$$

It is a coupled update with two separated timescales.

$$\text{(fast, student)}\qquad \mathrm{d}\theta/\mathrm{d}t\ \propto\ -\,w(\hat m)\,\nabla_\theta\,\mathbb{D}\!\left(p_\theta * \mathcal{D}_\sigma\ \big\Vert\ q_r * \mathcal{D}_\sigma\right)$$

$$\text{(slow, target)}\qquad \mathrm{d}r/\mathrm{d}t\ =\ f(\hat m)$$

| Symbol | Meaning |
|---|---|
| $G_\theta$ | the one-step generator (student) |
| $q_r$ | a continuous path of target measures; $r$ is the progress parameter |
| $\mathcal{D}_\sigma$ | smoothing noise to make the two measures overlap (the answer to ⓷) |
| $\hat m$ | an **online readout** on the student's own samples |
| $w(\hat m)$ | the gain |
| $\mathbb{D}$ | the divergence. **Left unspecified** — mode-seeking, adversarial, or mixed are all permitted |
| $f$ | the **control law.** It advances $r$ only when $\hat m$ says the student has closed the gap to the current target, and holds otherwise |

The form of $f$ is the key that makes the target something the student can **follow** rather than something that runs ahead of it.

**Speed vs. quality** (one H100, a 3-second 3-view manipulation clip)

| Sampler | NFE | Time (ms) | Speedup | FVD ↓ | Motion ↑ | Flicker |
|---|---|---|---|---|---|---|
| real recorded future | — | — | — | — | 100% | 2.37 |
| Teacher, default schedule | 100 | 10,203 | 1× | **80** | 94% | 2.69 |
| Teacher, truncated to 1 step | 2 | 210 | 48.6× | 1039 | 27% | 15.81 |
| DMD2, 2 steps | 2 | 211 | 48.4× | 115 | 79% | 2.95 |
| DMD2, 1 step | 1 | 109 | 93.6× | 599 | 56% | 5.81 |
| **Ours, 1 step** | **1** | **110** | **93×** | **121** | 75% | **1.94** |

Truncating the teacher to one step collapses completely (FVD 1039, flicker 15.81), and DMD2 also breaks down at one step (FVD 599). The proposed method reaches FVD 121 at one step, close to DMD2's **two-step** quality (115).

> ### ⚠️ Fact-check — motion 75% and flicker 1.94 have to be read together
>
> Flicker being **lower** than the real recording (2.37) means **over-smoothed** rather than stable, and points the same way as motion reaching only 75% of the real value. It still moves less and shakes less. In other words, ⓵ above — the averaging bias of regression losses — has not been fully removed. The report honestly writes that motion "still trails the full teacher."
>
> This section is also **not a reproducible description.** The divergence $\mathbb{D}$ is left open, and the concrete forms of the control law $f$, the gain $w$, and the readout $\hat m$ are undisclosed. The skeleton is clear but the implementation is not recoverable, so it is safest to read it as a result report and withhold citation of the method.

---

## 6. Positioning — Among Neighboring Work

Dyna-2 sits at the intersection of two lineages.

**⓵ The scaling-law lineage** — Kaplan's and Hoffmann's power laws are the organizing principle, with transfer-learning variants in "effective data transferred" (Hernandez et al.) and alignment-dependent downstream scaling (Isik et al.). On the robotics side, scaling studies have measured environment and object diversity (Lin et al., ACT-2), compute and model size (GEN-0), and action-prediction loss on human egocentric data (EgoScale).

**⓶ The robot-foundation-model lineage** — the line that builds generalist policies from VLMs (RT-2, OpenVLA, π₀, GR00T N1), and the line that builds policies on top of video generative models. The latter splits again into using video generation as a **planner**, making a policy **directly by fine-tuning** (Cosmos Policy), and the most directly related **joint world-action modeling** (DreamZero, Unified World Models, mimic-video).

What distinguishes it from its closest cousins is the measurement design.

| | **EgoScale** | **RDT2 / LAP** | **Dyna-2** |
|---|---|---|---|
| scale axis measured | ~20,000 hours | **a single scale** | **1,000,000 hours** |
| where the law was measured | held-out **human** loss | n/a | human **+ robot** |
| how robot results were obtained | **only after aligned human–robot mid-training** | zero-shot | **zero-shot, no alignment** |
| robot-form data in pretraining | — | **remains (UMI, etc.)** | **none (human video only)** |
| what it showed | scaling of human loss | that transfer is **possible** | **how** transfer scales |

RDT2 and LAP showed "transfer happens"; EgoScale showed "human data scales." What Dyna-2 adds is **the functional relation joining the two** — shown without robot-form data and without alignment.

But the report's real antagonist is not inside its lineage. It is the VLA line that currently dominates robot foundation models.

> ### 🔗 Axis by axis against the VLA line — the real difference is the bottleneck, not the architecture
>
> | Axis | **VLA** (π₀, OpenVLA, GR00T N1, Dyna-1) | **WAM** (Dyna-2) |
> |---|---|---|
> | what pretraining borrows | **semantics** from web text-image | the **physical unfolding** of video |
> | initialization | a VLM (Qwen3-VL, etc.) | a video-diffusion backbone |
> | path from language to action | **directly** through the backbone | **via** the video stream |
> | strategy for preserving language ability | **multi-stage training + backbone freezing** | put language on the video loss so the action loss never touches it |
> | use of unlabeled data | hard (needs action labels) | **absorbed by the video objective** |
> | bottleneck on scaling | **the production rate of action-labeled data** | **hand-pose annotation quality** (the video itself is unlimited) |
>
> The last row is the point of the table. The real difference between the two approaches is not architecture but **what the bottleneck is.** The VLA line is bound to the production rate of action labels (= Wall 1 in §1); the WAM line partially unbinds that with the video objective.
>
> WAM is not entirely free either. The action stream still requires hand-pose annotation, and the report itself states in §4.5 that extraction quality and annotation infrastructure are real constraints. Wall 1 was lowered, not removed.

---

## 7. Limitations

**What the report acknowledges**

- **No compute or model-size scaling experiments.** Explicitly future work. So this is a **data scaling law**, not a comprehensive Kaplan/Hoffmann-style one.
- **No human–robot alignment or co-training.** Deliberately omitted, while conceding it would improve post-training performance.
- **The one-step student falls short of the teacher.**
- **Video scaling slightly degrades human-data evaluation** (Figure 12).

**Additional points to flag**

- **The format itself is a company technical report.** There is no peer review; the data corpus, model size, $\lambda$, and training compute are all undisclosed; and 12 of the 39-task evaluation suite are internal benchmarks. It is not reproducible. Mixing in the 27 external xdof ABC tasks only partially mitigates this.
- **"Quantity" and "diversity" are not separated.** The nested-subset design controls distribution shift across sources beautifully, but it does not control for the fact that a larger subset contains more scenes, objects, and tasks — indeed, being nested, that is inevitable. So whether this is **a law of hours or a law of coverage** remains open. Given that the report itself reads the 10k→100k inflection as "sufficient coverage," the authors appear to see coverage as the operative variable too.
- **The extreme compression of pseudo-actions is unexplained.** There is no mechanistic account of how a representation trained on wrist pose and thumb–index aperture transfers to a **WUJI-2 20-DOF multi-finger hand.** Bottle Cap Untwisting reaching 50% on 10 minutes of data is impressive, but whether that transfer comes from the human-hand video seen by the video stream or from the action stream is not separated. Repeating the §4.5 ablation on multi-finger-hand tasks would answer it.
- **The defense against metric artifacts is partial.** Reporting four metrics against Schaeffer's point is good, but **all four are offline action-prediction metrics.** The correlation between the real quantity of interest — on-robot success rate — and these offline metrics is confirmed only at the aggregate level of §4.4, and only over four points.
- **The gap between aggregate and per-task results, and the absolute performance level.** See the fact-checks in §4.4 and §5.1.

---

## 8. Closing — What This Report Suggests

Dyna-2's real contribution is not a particular architecture. It is **having measured — zero-shot, without alignment, and out to one million hours — that an effectively unlimited resource, human video, becomes a function of robot performance.** If that relation is real, the technical roadmap for robot foundation models simplifies considerably: collect more sensorized human data, and move robots closer to the human form factor.

And this report reads especially well for someone with an LLM/diffusion background.

- The data economics of **"abundant unlabeled data + scarce labels"** carries over directly. If LLMs were "pretrain on unlimited text + SFT on scarce labels," Dyna-2 is **"a video objective on unlimited video + an action objective on scarce hand-pose labels."** The flow of the report — a long explanation of the hand-pose infrastructure bottleneck, then asking and testing "can unlabeled video alone do it?" (§4.5) — shows the parallel is deliberate design, not coincidence.
- **Use the generative objective as a representation learner and discard it at inference.** The single line that $u^{\mathrm{act}}_\theta$ never takes $z_t$ as an argument is the most compressed design declaration in the report. It belongs with the line of work using diffusion for representation learning, not with model-based RL's world models.
- **The profit-and-loss structure of an auxiliary loss is familiar.** Figure 12's "human 104%, robot 34%" is the pattern observed many times in multi-task learning: an auxiliary objective shaves a little in-domain performance to buy OOD generalization. The report's hypothesis — that video training dilutes the action gradient — speaks the same language.
- **A design that isolates language from the action loss.** What the VLA side tried to protect with backbone freezing and multi-stage training, Dyna-2 solves **with wiring.** Anyone who has dealt with which loss destroys which representation will read the logic of that choice immediately.

Above all, the observation confirmed in §4.5 — **video prediction is not a device for raising in-domain performance but a device for buying generalization across the embodiment gap** — looks like concrete guidance on where any attempt to bring world models into robotics should set its expectations. A world model does not make it do better; it makes it **do it on a body it has never seen.**

---

## Appendix — Glossary

| Term | Definition |
|---|---|
| **WAM (world-action model)** | a single generative model that denoises future video and future actions, jointly or separately, built on a video-diffusion backbone |
| **transfer scaling law** | the relation by which increasing human data quantity predictably improves metrics on **robot data never seen in pretraining** |
| **pseudo-action** | action supervision derived from 3D hand pose in human video: wrist pose (EE trajectory) + thumb–index aperture (grasp) |
| **nested subset** | a data ladder built so that a larger budget adds rather than exchanges data, so differences between points are not explained by distribution shift |
| **video co-training** | on top of joint training, performing video prediction on additional **action-label-free** video. The primary driver of cross-embodiment transfer |
| **acc@$\tau$** | the fraction of action dimensions within $\tau$ of ground truth. $\tau=0.5$ measures motion intent, $\tau=0.1$ motion precision |
| **reactive policy** | a policy that neither generates the future nor attends to it at inference. Dyna-2's deployed form |

**Original report** — [dyna.co/dyna-2](https://www.dyna.co/dyna-2) (Dyna Robotics technical report, August 2026)
