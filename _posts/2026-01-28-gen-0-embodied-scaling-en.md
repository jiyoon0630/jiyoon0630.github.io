---
layout: paper
lang: en
ref: gen-0-embodied-scaling
kind: tech-review
title: "GEN-0: Embodied Foundation Models That Scale with Physical Interaction (GEN-0)"
date: 2026-01-28 12:00:00 -0800
paper_date: 2025-11-04
venue: "Generalist AI Blog · research blog (not on arXiv)"
tags: [Robot-Foundation-Model, Scaling-Law, Pretraining, Embodied-AI, Tech-Review]
authors: "Generalist Team"
affiliations: "Generalist AI"
summary: "The claim that, for a large enough model pretrained on large-scale physical-interaction data, a power law holds between pretraining data and post-training performance — read for what was measured and what the measurements actually say."
paper_url: "https://generalistai.com/blog/gen-0"
---

> **Core claim**: Robot foundation models follow scaling laws just as LLMs do. Pretrain on large-scale real-world physical-interaction data with a large enough model (7B or more, per the blog), and a power law holds between the amount of pretraining data and downstream post-training performance. With this curve, the data needed for a target performance can be estimated in advance.

---

## Introduction

The standard recipe for robot foundation models is to put action output on top of a VLM. The semantic understanding gained from web-scale vision-language pretraining is carried over to robot control, and the target task is fine-tuned with teleoperation demonstrations. PaLM-E, RT-2 and π₀ all belong to this picture.

GEN-0 points out what this recipe is missing. What put LLMs where they are today was less any particular architecture than the scaling law: "grow compute and data, and performance rises predictably." Yet no such relationship has ever been established in the robot domain itself. GEN-0 sets out to show that relationship in robots for the first time. How far that "first" holds is examined again in Section 6.

One premise up front. GEN-0 is a research blog, not a paper, and its architecture and training procedure are not disclosed. So this piece reads it with a focus on **what was measured, and what those measurements tell us**. Reading the measurements properly takes the tools of the LLM scaling literature, so the needed concepts are explained as they come up.

---

## 1. The problem: why robots had no scaling law

### 1.1 What VLM transfer does not give you

With vision-language pretraining as a stepping stone, a robot gains semantic generalization. Given "pick up the red cup," it knows which thing is the red cup and what picking up means. But **sensorimotor knowledge** — how hard to grip the cup, along what finger trajectory, how to regrip when it slips — is almost absent from web images and text.

GEN-0 tries to get this sensorimotor knowledge through scaling. The blog says this needs an architecture, a training procedure and a data engine, all together. Restated in terms of obstacles, that becomes three walls.

### 1.2 Three walls

**⛔ Wall 1: There is no data regime** LLM scaling laws are observed at the scale of hundreds of billions of tokens. Robot teleoperation data only comes into being as people drive robots one at a time, so it falls short in scale by several orders of magnitude. There is no x-axis on which to draw a scaling curve at all.

**⛔ Wall 2: Bigger models make control slower** An LLM may think as long as it likes before answering. The physical world does not wait for inference to finish. The larger the model, the longer the inference latency, and latency turns directly into worse control. Model size, the very thing to be scaled, becomes a factor that hampers control.

**⛔ Wall 3: What do you measure "predictable" with** LLMs have pretraining loss, a cheap and smooth metric. A robot's real goal, success rate, needs physical rollouts, so it is expensive and noisy. To establish a scaling law you need a proxy that is cheap and smooth yet moves in the same direction as success rate.

Walls 1 and 2 are obstacles the blog names directly. Wall 3 is not given a name by the blog, but the whole experimental design can be read as an answer to it.

### 1.3 So the question the blog asks

> If you pretrain directly on physical-interaction data, does "grow data and compute, and downstream performance rises predictably" hold as it does for LLMs? If it does, under what conditions?

Section 2 prepares the tools needed to read this question and looks at what exactly Wall 2 is. Section 3 covers GEN-0's answers to the three walls, and Section 4 the condition that newly surfaced after getting over Wall 1.

---

## 2. Background: the grammar of scaling laws and the nature of Wall 2

### 2.1 Kaplan-style scaling laws: every model has a floor

GEN-0's experiments all transplant the frame of the LLM scaling literature onto robots. The starting point is the power law of Kaplan et al. (2020).

$$L(N)=\left(\frac{N_c}{N}\right)^{\alpha_N},\qquad L(D)=\left(\frac{D_c}{D}\right)^{\alpha_D}$$

| Symbol | Meaning |
|---|---|
| $L$ | test loss (cross-entropy for LLMs) |
| $N$ | number of model parameters |
| $D$ | amount of training data (number of tokens) |
| $N_c,\ D_c$ | scale constants obtained by fitting |
| $\alpha_N,\ \alpha_D$ | exponents |

Taking the log of both sides gives:

$$\log L=\alpha_D\log D_c-\alpha_D\log D$$

- On a log-log plot this is a straight line with slope $-\alpha_D$
- "A scaling law holds" means this line persists across several orders of magnitude

The formula with both variables together is:

$$L(N,D)=\left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D}+\frac{D_c}{D}\right]^{\alpha_D}$$

- The $(N_c/N)^{\alpha_N/\alpha_D}$ term: error from insufficient model size
- The $D_c/D$ term: error from insufficient data

What matters in this formula is the limit. Fix $N$ and send $D\to\infty$, and you get:

$$L(N,D)\ \xrightarrow{\ D\to\infty\ }\ \left(\frac{N_c}{N}\right)^{\alpha_N}$$

That is, the loss hits **a floor set by model size**. A small model cannot go below this floor however much more data it is given. So when loss curves for each size are overlaid on a compute axis, the usual shape is this:

```
 loss (log)
   ^
   | *
   |  *   +
   |   *    +     o
   |    * * * * *   +      o                <- small N: hits floor early
   |                  + + + + +    o        <- mid N: floor later
   |                                o  o    <- large N: still falling
   +------------------------------------------> compute (log)
```

- small N: hits the floor early and flattens
- large N: keeps falling over the same compute range

This picture is used again in Section 4 when reading GEN-0's "ossification" claim.

### 2.2 Transfer scaling and ossification: the pretrain → finetune setting

The LLM literature has one more kind of scaling, distinct from the scaling of pretraining itself: scaling of how far the benefit of pretraining carries past fine-tuning. Hernandez et al. (2021) converted that benefit into an amount of data.

$$D_T = D_E - D_F$$

- $D_F$: the amount of fine-tuning data actually used
- $D_E$: the amount of data needed to reach the same performance without pretraining (from scratch)
- $D_T$: effective data transferred. The value of pretraining converted into an equivalent amount of fine-tuning data

The same paper also named a phenomenon, **ossification**: when the model is small and fine-tuning data is plentiful, the pretrained model ends up worse than a from-scratch model. The sense is that pretraining sets the weights so they can no longer take in a new distribution. Springer et al. (2025)'s observation, that the more an LLM is overtrained in pretraining the worse it does after fine-tuning, belongs to the same line.

Two things to keep in mind here.

- **⓵** Ossification is a phenomenon of the **pretrain → finetune setting** (revisited in Section 4)
- **⓶** "How much fine-tuning data does pretraining replace" is a question about $D_T$ (revisited in Section 5.2)

### 2.3 The nature of Wall 2: action chunking and inference latency

Most current VLAs use **action chunking**: one inference pass produces $H$ steps' worth of actions at once, and that chunk is executed open-loop.

$$a_{t:t+H}\ \sim\ \pi_\theta(\cdot \mid o_t,\ g)$$

- $a_{t:t+H}$: the action chunk covering $H$ steps from time $t$
- $o_t$: the observation at time $t$ (images, proprioception)
- $g$: the goal given in language
- $\pi_\theta$: the policy with parameters $\theta$

The problem is the time $\delta$ that inference takes. By the time inference started on observation $o_t$ produces actions, the world has already moved on by $\delta$. The bigger the model, the bigger $\delta$. Run synchronously, it looks like this:

```
(a) synchronous chunking
  infer  [===]           [===]           [===]
  act         >>>>>>>>>>      >>>>>>>>>>      >>>>
```

- While inferring, the robot either stops or runs the previous chunk open-loop to the end
- Actions break at chunk boundaries

It is not that the robotics field had no solutions to this. There are two representative lines.

```
(b) System1-System2 (e.g. Helix)
  S2 big   [==========]  [==========]  [==========]
  S1 small  > > > > > > > > > > > > > > > > > > > >

(c) inference-time guidance (e.g. RTC)
  act    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  infer       [=== next chunk, inpainted ===]
```

Then why does GEN-0 say it needs a new approach rather than leaning on these solutions?

> ### 💡 Existing solutions fix latency but are in tension with scaling
>
> | | How it works | How it handles latency | Tension with scaling |
> |---|---|---|---|
> | **(b) System1-System2** (Figure Helix) | A large VLM (S2) slowly emits latents; a small fast policy (S1) emits the actual actions | The small S1 produces actions quickly | The part that produces actions stays small. Growing the large model does not carry its gains down to reflexes and dexterity |
> | **(c) inference-time guidance** (RTC, Black et al., 2025) | Generates the next chunk while executing the current one. Actions to be executed during the latency window are frozen, and the rest is inpainted to follow on naturally | Stitches chunk boundaries smoothly | The longer the latency $\delta$, the longer the frozen segment, and responsiveness drops. The structure still penalizes larger models |
>
> Both solutions accept the premise "big models are slow" and choose to manage its consequences. What GEN-0 wants is a structure where the large model itself sits inside the control loop without being held back by latency. Harmonic Reasoning in Section 3.2 aims at exactly this point.

---

## 3. Method: GEN-0's answers to the three walls

First, a summary of GEN-0's answer to each wall and how much of it is disclosed.

| Wall | GEN-0's answer | Disclosure level |
|---|---|---|
| ⛔ Wall 1: data regime | An in-house data engine at the scale of 270K hours | Scale and infrastructure disclosed. Collection devices and action representation not disclosed |
| ⛔ Wall 2: real time | Harmonic Reasoning | Only the name and direction disclosed |
| ⛔ Wall 3: measurement | Next-action validation error as a proxy, validated with physical A/B tests | Figure level. Coefficients not disclosed |

### 3.1 The data engine (Wall 1)

GEN-0 built the x-axis with its own data. The scale the blog discloses is as follows.

| Item | Scale |
|---|---|
| Cumulative data | Over 270K hours of real-world manipulation trajectories |
| Sources | Thousands of homes, warehouses and workplaces around the world |
| Growth rate | Over 10K hours per week, still accelerating |
| Comparison | Several orders of magnitude larger than the large robot datasets public as of November 2025 (Figure 5) |

Moving and processing data is as much a problem as collecting it. According to the blog, they laid new dedicated internet lines to secure upload bandwidth at collection sites, signed multi-cloud contracts and built custom upload hardware, and continuously processed multimodal data at O(10K) cores, compressing tens of PB. As a result, a single day of training can absorb 6.85 years' worth of real-world experience.

6.85 years is about 60K hours. So one pass over all 270K hours takes about 4.5 days. Measured by throughput as well as data volume, it is an LLM-class pipeline.

The blog calls this data an in-house robotics dataset, and describes the means of collection as "a global hardware network and thousands of data collection devices and robots." What devices recorded what, and in what representation the actions were stored, is not disclosed.

### 3.2 Harmonic Reasoning (Wall 2)

The two solutions in Section 2.3 solve the latency problem through **structure** (S1/S2) or **correction at inference time** (RTC). GEN-0 says it will solve it through the training method itself. The blog's explanation, summarized:

- **Definition**: a new training method that creates harmonious interaction between asynchronous, continuous-time streams of sensing tokens and acting tokens. The model is trained to think and act at the same time
- **Claim**: thanks to this method, it can scale to very large model sizes without relying on System1-System2 architectures or inference-time guidance
- **Example**: camera kit assembly. A long-horizon task: put a cloth into a box, fold a cardboard tray, take the camera out of its plastic bag and put it in, close the box down to tucking in the small flap, then throw away the plastic bag. It is carried out within a single stream, with no explicit subtask boundaries

Appended to the picture in Section 2.3, it looks like this:

```
(d) Harmonic Reasoning (claimed)
  sense  s  s s   s  s s  s   s  s
  act     a a  a   a a  a  a a   a
```

- (a)–(c) all assume a structure where "observe → infer → action chunk" alternate
- (d) claims that sensing and acting flow overlapping on the time axis, without a fixed period

The implementation is not disclosed, but two interpretations can be drawn directly from the phrase "asynchronous, continuous-time streams."

- **⓵ Event-based placement**: instead of an (observation, action) pair at every fixed $\Delta t$, sensing tokens and acting tokens are placed as timestamped events
- **⓶ Continuous-time encoding**: continuous time is used as the position encoding instead of the tokens' sequence index

The blog contrasts this method with LLM test-time reasoning, that is, thinking longer before answering. Then is there anything on the LLM side corresponding to a structure that "thinks and acts at the same time"?

> ### 💡 The closest LLM-side counterpart is a full-duplex speech model
>
> An ordinary chatbot answers after the user finishes speaking. It is a turn-taking structure, the same shape as the synchronous chunking of Section 2.3 (a). A full-duplex spoken dialogue model (e.g., Moshi), by contrast, models the user's speech stream and its own speech stream in parallel. So it can backchannel or cut in while the other party is still talking.
>
> | Spoken dialogue | Robot control |
> |---|---|
> | Listening stream | Sensing-token stream |
> | Speaking stream | Acting-token stream |
> | Turn-taking chatbot | Synchronous action chunking |
> | Full-duplex model | Harmonic Reasoning (claimed) |
>
> This correspondence is only an analogy drawn from the blog's wording, not evidence that GEN-0 is actually implemented this way.

That is as far as the disclosure goes. Tokenization, time encoding, loss and inference period are all undisclosed, and there is no experiment comparing against (b) or (c). **For Wall 2, only the name and direction of a solution are given.**

### 3.3 Measurement design (Wall 3)

GEN-0 uses **next-action validation prediction error** as the y-axis of its scaling curves. Given observations from held-out data, the model predicts the next action, and the error against the ground-truth action is measured. Table 1 defines this error as MSE.

$$\text{MSE}_{\text{val}}=\lVert a^\star-\hat a\rVert_2^2$$

- $a^\star$: the ground-truth action in the held-out data
- $\hat a$: the action predicted by the model

Like an LLM's validation loss, it is a smooth proxy computable without physical rollouts. The blog's experiments use this proxy in two layers.

```
 pretrain (size N, compute C) -----------------------> zero-shot val error  (Fig 1)
 pretrain (data subset D) ---> post-train (fixed) ---> val error            (Fig 2, 4)
                                                  \--> robot success        (Fig 3)
```

- **Scaling of the proxy**: Figures 1, 2 and 4 all use validation error as the y-axis
- **Validity of the proxy**: Figure 3 checks with blind A/B evaluation whether the same trend appears in physical success rates

One thing should be made clear here. The "zero-shot" of Figure 1 is **offline prediction error** on fully withheld long-horizon task data. It is not the result of the robot actually performing that task.

---

## 4. Why it works: the condition that surfaced after getting over Wall 1

The blog explains why scaling had not been observed in robots with two reasons: there was no high-data regime, and there was no model large enough for that regime. Once over Wall 1 and into the high-data regime, the second condition surfaced.

| Model size | Observation (Figure 1) |
|---|---|
| 1B | Cannot absorb complex, diverse sensorimotor data. As training proceeds it stops taking in new information |
| 6B | The benefit of pretraining starts to appear, and it shows strong multi-task ability |
| 7B+ | Internalizes large-scale pretraining data and transfers to downstream tasks with only a few thousand steps of post-training |

The axes of Figure 1 are as follows.

- **y-axis**: next-action validation prediction error on fully withheld long-horizon tasks (lower is better)
- **x-axis**: pretraining compute, normalized so GEN-0 7B's value is 1.0
- **Color**: model size

The blog says it later scaled to 10B+ and observed the post-training needed to adapt to new tasks steadily shrinking. It gives no numbers, however.

The blog links this result to Moravec's paradox. Ossification in the LLM literature was observed at the O(10M)-parameter scale, but in robots it appeared at O(1B). The interpretation is that, just as perception and dexterity that are easy for people actually demand more computation than abstract reasoning, intelligence about the physical world (physical commonsense) may have a higher activation threshold in terms of compute.

But recall the ossification seen in Section 2.2 and a question arises. Is what GEN-0 observed the same phenomenon as that ossification in the LLM literature?

> ### ⚠️ Same name, different measurement
>
> As the blog's own footnote 8 admits, ossification in the LLM literature is a phenomenon of the **pretrain → finetune setting** (⓵ of Section 2.2). What GEN-0 observed in Figure 1 is "ossification-type behavior" in which zero-shot generalization stalls **during pure pretraining**.
>
> Going a step further, a small model's curve stopping despite more compute during pure pretraining cannot be distinguished, from the text description alone, from the $L(N)$ floor of Section 2.1. When curves for each size are overlaid, the small model flattening first is a shape commonly seen in Kaplan compute plots. The key to telling the two apart is whether the 1B curve **stalls** or **rises** (whether held-out error grows as training continues). The caption and body alone cannot settle this.

There is also an alternative to the Moravec interpretation. For high-frequency multimodal data carrying far more information per hour than text, the reason a larger $N$ is needed may not be a domain-specific "intelligence threshold" but simply the entropy of the data. Separating the two interpretations would need a comparison normalized by token count or information content, and there is no such experiment.

Meanwhile, the blog's headline is "a phase transition at 7B." But can the disclosed model sizes alone pin the transition at 7B?

> ### ⚠️ Fact check: the resolution of the "7B phase transition"
>
> The disclosed model sizes are three: 1B, 6B and 7B. With no measurement point between 1B and 6B, all that can be said about the transition's location is "somewhere between 1B and 6B." The body itself also writes that the benefit of pretraining appears from 6B. The headline's "7B" is more precise than the figure's resolution. The only thing that can be said clearly is that 1B stalls early.

---

## 5. Experiments

### 5.1 Pretraining data volume and post-training performance (Figures 2, 3)

**Design**: prepare checkpoints trained on different subsets of the pretraining data, and SFT each checkpoint on the same multi-task, language-conditioned data. 16 task sets are trained simultaneously, covering the following three kinds.

| Kind | Example |
|---|---|
| dexterity | Lego assembly |
| industrial workflow | Fast-food packing |
| generalization | "_ anything"-style tasks |

**Results**

- **Figure 2**: the more pretraining data, the better the post-training validation loss and the next-action prediction error on all 16 task sets
- **Figure 3**: evaluated on physical robots with blind A/B, closed-loop rollouts. Even post-trained on just 5.6 hours of task data (1% of the total), average success rate rises with more pretraining data. The best performance comes from using the full pretraining data together with 550+ hours of task data, reaching up to 99% in some cases
- **Data separation**: pretraining data and post-training data were collected by different people in entirely different environments, so they do not overlap

Figure 3 is the final link in getting over Wall 3, because it is evidence that the trend seen in the offline proxy carries over to physical success rates. Then what level is GEN-0's actual success rate? Can "up to 99%" be read as a representative value?

> ### ⚠️ Fact check: 99% is a peak
>
> The 99% in the body is a peak for particular cases; no average success rate is given. What Figure 3 shows is the trend that success rate rises with more pretraining data, not GEN-0's typical success rate.

The GEN-0 post does not say how the post-training data (5.6 hours, 550+ hours) was collected.

### 5.2 Power law (Figure 4)

Fix the downstream task's data and fine-tuning budget and vary only the size of the pretraining data, and downstream error is predicted by the following power law.

$$L(D)=\left(\frac{D_c}{D}\right)^{\alpha_D}$$

- $L$: next-action validation error on the downstream task after post-training
- $D$: pretraining data size (number of action trajectories)
- $D_c$: scale constant obtained by fitting
- $\alpha_D$: exponent. On a log-log plot it becomes a straight line with slope $-\alpha_D$

The example the blog discloses is Clothes Handling (sorting clothes, tidying tangled clothes, buttoning, hanging on hangers), predicting performance at 1 billion pretraining trajectories.

The form is identical to Kaplan's $L(D)$ in Section 2.1. So is it measuring the same thing?

> ### 📌 Figure 4 is transfer scaling, not pretraining scaling
>
> | | Kaplan $L(D)$ | GEN-0 Figure 4 |
> |---|---|---|
> | $L$ | Pretraining test loss | Downstream error **after post-training** |
> | $D$ | Number of pretraining tokens | Number of pretraining trajectories |
> | Held fixed | Model size large enough | Downstream data and fine-tuning budget |
> | Nature | pretraining scaling | **transfer scaling** |
>
> The blog puts forward two uses. One is "how much post-training data can be saved by adding pretraining data," which is exactly a question about the effective data transferred $D_T$ of Section 2.2 (⓶ of Section 2.2). The other is "how much pretraining data is needed to reach a target error." The blog claims that combining this with a model-size scaling law can predict the optimal allocation of pretraining compute and data per task. It is a Chinchilla-style analysis, though the blog does not mention Chinchilla directly.

The blog states that it uses this curve in partner task discussions and data estimates. Then is the disclosed evidence sufficient for use as an estimation tool?

> ### ⚠️ The basis for extrapolation is not disclosed
>
> **⓵ Coefficients undisclosed**: $D_c$, $\alpha_D$, fit error and the number of data points are not disclosed. The only disclosed fit is Clothes Handling, and that it applies to every measured task is only stated in prose.
>
> **⓶ No floor term**: being a pure power law, $L\to 0$ as $D\to\infty$. But for demonstration data in which different people act differently in the same situation, it is natural for next-action error to have an irreducible floor. That is also why Chinchilla (Hoffmann et al., 2022) includes a constant term.
>
> $$L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$$
>
> - $E$: irreducible loss
> - $A,\ B,\ \alpha,\ \beta$: fitted constants
>
> The extrapolation to 1 billion trajectories depends heavily on whether this floor term is included.
>
> **⓷ Units cannot be converted**: the unit of $D$ is number of trajectories, but the average trajectory length needed to convert against 270K "hours" is not disclosed. So how many hours 1 billion trajectories corresponds to is unknown.
>
> **⓸ No success-rate fit**: the power law is for validation error. For success rate there is only the trend in Figure 3.

### 5.3 The science of data mixtures (Table 1)

**Design**: models each trained on one of 8 pretraining datasets are fine-tuned on 10 long-horizon task sets and compared. The task sets fall into three groups: dexterity, applications (real-world applications) and generalization. The datasets are distinguished by the combination of external data-collection partner (data foundry) A, B, C and collection method (Class).

| Class | Collection method |
|---|---|
| Class 1 | Task-specific data |
| Class 2 | Between Class 1 and Class 3 |
| Class 3 | Do-anything-style data |

There are two metrics. One is $\text{MSE}_{\text{val}}$ from Section 3.3, and the other is reverse KL. From an LLM background a question naturally arises: why have reverse KL separately when there is already prediction error?

> ### 💡 SFT reduces forward KL; reverse KL looks at the policy's own samples
>
> With $p$ the data (ground-truth) distribution and $q$ the policy distribution, KL behaves differently depending on direction.
>
> $$D_{KL}(p\,\|\,q)=\mathbb{E}_{a\sim p}\!\left[\log\frac{p(a)}{q(a)}\right],\qquad D_{KL}(q\,\|\,p)=\mathbb{E}_{a\sim q}\!\left[\log\frac{q(a)}{p(a)}\right]$$
>
> - The left is forward KL. It is evaluated on samples drawn from the data
> - The right is reverse KL. It is evaluated on samples drawn by the policy
>
> | | forward | reverse |
> |---|---|---|
> | Samples the expectation is taken over | Data $a\sim p$ | Policy $a\sim q$ |
> | When the penalty is large | When $q$ gives almost no probability to actions present in $p$ | When samples produced by $q$ have low probability under $p$ |
> | Property | mode-covering | mode-seeking |
> | LLM-side counterpart | MLE / SFT | The KL penalty in RLHF, on-policy distillation |
>
> In the f-divergence view of imitation learning that the blog cites, BC (=SFT) also corresponds to forward KL minimization. Since the SFT loss is computed only on data samples, it cannot directly see **the quality of the samples the policy actually emits**. Reverse KL is the metric that fills that gap.

The blog estimates reverse KL by Monte Carlo. It uses $q$, a mixture placing a unit-variance Gaussian on each of $M$ policy samples, and $p$, a unit-variance Gaussian on the ground-truth action.

$$q(a)=\frac{1}{M}\sum_{j=1}^{M}\mathcal{N}(a;\ \hat a_j,\ I),\qquad p(a)=\mathcal{N}(a;\ a^\star,\ I)$$

$$\widehat{D}_{KL}(q\,\|\,p)\approx\frac{1}{M}\sum_{m=1}^{M}\Big[\log q(\hat a_m)-\log p(\hat a_m)\Big]$$

- $\hat a_m$: the $m$-th action sample drawn by the policy
- $a^\star$: the ground-truth action in the data
- $M$: number of policy samples (undisclosed)
- $I$: identity covariance matrix

The blog's decision rule is this: a model with both low prediction error and low reverse KL is favorable for SFT post-training; a model with high prediction error but low reverse KL is distributionally multimodal and favorable for RL post-training. But why does the rule take this shape? What does this estimator actually measure?

> ### 💡 The reverse KL estimator is "½·MSE + a sample-clustering term"
>
> Writing out the logs of the two Gaussian densities gives:
>
> $$\log p(\hat a_m)=-\tfrac{1}{2}\lVert\hat a_m-a^\star\rVert^2-\tfrac{d}{2}\log 2\pi,\qquad \log q(\hat a_m)=\log\Big[\tfrac{1}{M}\textstyle\sum_{j}e^{-\frac{1}{2}\lVert\hat a_m-\hat a_j\rVert^2}\Big]-\tfrac{d}{2}\log 2\pi$$
>
> - $d$: the dimension of the action vector. The $-\tfrac{d}{2}\log 2\pi$ in the two expressions cancels on subtraction
>
> Substituting into the estimator gives exactly:
>
> $$\widehat{D}_{KL}=\underbrace{\frac{1}{2M}\sum_{m=1}^{M}\lVert\hat a_m-a^\star\rVert^2}_{\text{(i)}}\;+\;\underbrace{\frac{1}{M}\sum_{m=1}^{M}\log\Big[\frac{1}{M}\sum_{j=1}^{M}e^{-\frac{1}{2}\lVert\hat a_m-\hat a_j\rVert^2}\Big]}_{\text{(ii)}\ \in\ [-\log M,\ 0]}$$
>
> - **(i)**: how far the samples are from the ground truth. Half the sample MSE
> - **(ii)**: how clustered the samples are. It is 0 if all samples collapse to one point, and approaches $-\log M$ the farther apart they spread
>
> So low reverse KL means one of two things: **accurate** (small i) or **samples spread out** (small ii). Paired with MSE, the two cases separate.
>
> | MSE | Rev KL | Reading via the decomposition | Blog's verdict |
> |---|---|---|---|
> | Low | Low | An accurate, consistent policy | Favorable for SFT post-training |
> | High | Low | Off on average, but samples widely spread → multimodal | Favorable for RL post-training |
>
> Why RL comes out favorable also becomes clear here. RL post-training reinforces the good ones among the policy's samples, so the more diverse the samples, the wider the exploration. It is the same intuition as in LLMs, where a base model's sample diversity (pass@k) sets the room RL can lift it into.
>
> Two caveats apply, however.
>
> - **⓵ It is not textbook mode-seeking**: since $p$ is a Gaussian with a single peak per sample, this estimator does not measure mode-seeking directly. What it actually measures is a combination of "distance from the ground truth" and "sample spread"
> - **⓶ Absolute values cannot be back-computed**: whether MSE was computed per sample or on the mean, and the value of $M$, are not disclosed

Now the table can be read. Values are in units of ×10⁻³, and the reading column is the interpretation per the decomposition above.

| Dataset | Pred Err (Dex / App / Gen) | Rev KL (Dex / App / Gen) | Reading |
|---|---|---|---|
| A · Class 1 | 3.077 / 3.342 / 3.090 | 2.006 / 2.589 / 1.981 | Highest KL in all three groups |
| A · Class 2 | 3.062 / 3.333 / 3.065 | 1.887 / 2.446 / 1.939 | Middle |
| A · Class 3 | 3.057 / 3.313 / 3.059 | 1.983 / 2.461 / 1.902 | Middle |
| A · Class 2+3 | 3.160 / 3.419 / 3.157 | 1.841 / 2.286 / 1.855 | High MSE, low KL → RL-type |
| B · Class 1 | 3.027 / 3.304 / 3.046 | 1.893 / 2.461 / 1.923 | Low MSE, middle KL |
| B · Class 2 Objs | 3.144 / 3.411 / 3.160 | 1.847 / 2.332 / 1.867 | High MSE, low KL → RL-type |
| B · Class 2 Skills | 3.020 / 3.292 / 3.053 | 1.826 / 2.423 / 1.903 | Among lowest MSE, lowest Dex KL → SFT-type |
| C · Class 3 | 3.062 / 3.321 / 3.079 | 1.921 / 2.369 / 1.910 | Middle |

Two signals in the table are worth reading.

- **The mixture effect is the most consistent signal**: Partner A's Class 2+3 has higher prediction error and lower reverse KL than Class 2 or Class 3 alone in **all** three groups. Read through the decomposition, mixing the two collection methods made the policy's samples spread more widely
- **Even the same partner and same Class can diverge in character**: splitting Partner B's Class 2 into object diversification (Objs) and skill diversification (Skills), Objs lands in the RL-type quadrant and Skills in the SFT-type quadrant. "What was diversified" divides the model's character more than the Class label

The blog draws three conclusions.

- **⓵** Data quality and diversity matter more than quantity
- **⓶** By designing the data mixture, pretrained models of different character can be obtained
- **⓷** Run multiple collection strategies at scale, keep A/B testing which data improves pretraining most, and on that basis feed back to partners what to collect and how

Then how reliable are the differences shown in this table?

> ### ⚠️ Fact check: the differences are small and there is no statistical information
>
> Within a group, the gap between the maximum and minimum is 3.7–4.6% for prediction error and 6.8–13.3% for reverse KL. There is no variance or confidence interval, and whether dataset sizes were matched is not disclosed, so it is impossible to judge how far the ranking can be trusted. The verdicts of favoring SFT or RL are also only given in prose; the table contains no results of actually doing SFT or RL post-training.
>
> Minor, but the footnotes also have errors. Minka's divergence report is given as 1988 but is actually 2005, footnotes 12 and 13 point to the same link, and Kaplan et al. is given as 2021 though it appeared on arXiv in 2020.

---

## 6. Positioning: among neighboring work

The blog treats related work only at the level of footnotes. GEN-0's position becomes visible only by placing the cited work alongside uncited work that sits close by.

| Lineage | Representative work | Relation to GEN-0 |
|---|---|---|
| VLM-transfer RFMs | PaLM-E, RT-2 | Opposing stance. The recipe GEN-0 says is missing something |
| LLM scaling laws | Kaplan, Hernandez, Springer | Analysis frame carried over as is |
| Real-time execution | Figure Helix (S1/S2), RTC | What Harmonic Reasoning aims to differentiate from |
| Hierarchical long-horizon tasks | Hi Robot (Shi et al., 2025) | A design where a high-level VLM explicitly hands down language subtasks. GEN-0 conversely puts forward a single stream with no explicit subtasks |

The introduction said GEN-0 sets out to show a robot scaling law "for the first time." Is it really the first?

> ### 🔗 Lin et al. (2024): an earlier robot data scaling study
>
> *Data Scaling Laws in Imitation Learning for Robotic Manipulation* (ICLR 2025 oral) systematically investigated data scaling in robot imitation learning about a year before GEN-0. Collecting over 40K demonstrations and evaluating with over 15K physical rollouts, it reported that a policy's generalization performance roughly follows a power law in the number of training environments and objects.
>
> | | Lin et al. (2024) | GEN-0 (2025) |
> |---|---|---|
> | Subject | Single-task imitation learning policies | Large-scale pretrained foundation model |
> | x-axis | Number of training environments, number of objects | Pretraining data volume (trajectories), model size, compute |
> | y-axis | Generalization performance in new environments and on new objects (physical rollouts) | Validation error after post-training (+ physical success-rate trend) |
> | Question | How must data be diversified to generalize | How far does pretraining transfer downstream |
>
> So GEN-0's "first" is most accurately read as **the first demonstration of scaling in large-scale pretraining → post-training transfer**. The GEN-0 blog does not cite Lin et al.

---

## 7. Limitations

Limitations the blog itself acknowledges, together with points worth flagging while reading.

**Limitations the blog acknowledges**

- Practically none. Just footnote 8's note that the term's definition differs, and a promise that more details will come in future posts.

**Further points to note**

- **Implementation undisclosed**: the blog itself names architecture, training procedure and data engine as requirements for scaling, but only the data engine's scale is disclosed. Harmonic Reasoning has only a name and a direction (Section 3.2).
- **Data composition undisclosed**: the collection devices, action representation and embodiment alignment method of the pretraining data are not disclosed (Section 3.1).
- **Thin evidence on success rates**: the power law is on validation error; for success rate there is only a trend and a peak of 99% (Sections 5.1, 5.2).
- **Cross-embodiment**: only a mention of testing on 6DoF, 7DoF and 16+DoF semi-humanoid robots, with no numbers. It is a property many RFMs claim in common, so it is hard to see as a differentiator unique to GEN-0.
- **Statistics in Table 1**: the differences are small and there are no confidence intervals (Section 5.3).

---

## 8. Closing: what this post suggests

GEN-0's real contribution does not lie in any particular architecture. It lies in having **built a data regime** in which scaling curves can be drawn in the robot domain, and in showing on top of it that the benefit of pretraining transfers predictably downstream.

For someone with an LLM or diffusion background, this post reads unusually familiarly.

- **The analysis tools all came from LLMs**: Kaplan's power law, Hernandez's transfer scaling, forward/reverse KL. What is new is not the tools but the x-axis that made them usable.
- **Table 1 is the robot version of data-mixture research**: except that the ingredients of the mixture are operational units — "which partner collected it, and how." Data-collection operations themselves become an experimental variable.
- **Harmonic Reasoning is the physical version of full-duplex**: a direction toward simultaneously flowing streams rather than turn-taking. If the implementation is disclosed, it is the first thing to check.

Finally, if you are **on the receiving end of this power law offered as a data-estimation tool**, there are three things to check.

| What to check | Why |
|---|---|
| ⓵ $\alpha_D$, $D_c$ and fit error for the relevant task family | The only disclosed fit is Clothes Handling |
| ⓶ Comparison with a fit that includes a floor term $E$ | A pure power law can give optimistic results when extrapolated |
| ⓷ The mapping between target validation error and success rate | The power law is about error, not success rate |

---

## Appendix: Glossary

| Term | Definition |
|---|---|
| **scaling law** | A relationship in which performance (loss) is predicted as a power function of model size, data volume and compute. It is a straight line on a log-log plot |
| **transfer scaling** | Scaling between the amount of pretraining and downstream performance after fine-tuning. GEN-0 Figure 4 takes this form |
| **effective data transferred** | $D_T=D_E-D_F$. The value of pretraining converted into an amount of fine-tuning data |
| **ossification** | In the LLM literature, the phenomenon where pretraining actually hinders fine-tuning in small models. GEN-0 borrows the name for zero-shot generalization stalling during pure pretraining |
| **action chunking** | Producing several steps of actions from one inference pass and executing them open-loop |
| **Harmonic Reasoning** | GEN-0's method of training to handle asynchronous, continuous-time streams of sensing and acting tokens simultaneously. Implementation undisclosed |
| **forward / reverse KL** | $D_{KL}(p\,\Vert\,q)$ / $D_{KL}(q\,\Vert\,p)$. The former is mode-covering (corresponds to SFT), the latter mode-seeking |
| **data foundry** | An external data-collection partner |
| **Class 1 / 2 / 3** | Task-specific / intermediate / do-anything-style collection methods, respectively |

**Original**: [GEN-0 (Generalist AI Blog, 2025-11-04)](https://generalistai.com/blog/gen-0) · **Related**: [Lin et al., arXiv:2410.18647](https://arxiv.org/abs/2410.18647)
