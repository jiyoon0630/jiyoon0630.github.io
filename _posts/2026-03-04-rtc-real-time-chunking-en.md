---
layout: paper
lang: en
ref: rtc-real-time-chunking
kind: paper-review
title: "Real-Time Execution of Action Chunking Flow Policies (RTC)"
date: 2026-03-04 12:00:00 -0800
paper_date: 2025-06-09
venue: "NeurIPS 2025 · arXiv:2506.07339"
tags: [VLA, Action-Chunking, Flow-Matching, Real-Time-Control, Inpainting, Paper-Review]
authors: "Kevin Black, Manuel Y. Galliker, Sergey Levine"
affiliations: "Physical Intelligence · UC Berkeley"
summary: "Generate the next action chunk in parallel with execution, freeze the actions that will already run during inference, and fill in the rest with training-free inpainting (ΠGDM guidance + a soft mask). Without retraining, it keeps both continuity at chunk boundaries and reactivity, and the gain grows with latency."
paper_url: "https://arxiv.org/abs/2506.07339"
code_url: "https://github.com/Physical-Intelligence/real-time-chunking-kinetix"
---

> **Core claim** — Inference latency cannot be eliminated by acceleration alone. But if you generate the next action chunk in parallel with execution, freeze the actions that will already be executed during inference, and fill in the rest by inpainting, you can get continuity at chunk boundaries and reactivity to new observations together, **without retraining**.

---

## Introduction

The VLA inference pipeline inherited a great deal from LLMs: a huge VLM backbone, KV cache prefill, and iterative generation over many steps. But where inference speed was a user-experience problem for LLMs, for robots it is **a problem of physics**. A slow chatbot only frustrates its user; a slow robot policy spills the hot coffee it was holding (the analogy from the paper's intro).

RTC (Real-Time Chunking) tackles this gap head-on. It proposes neither a new model nor a new training recipe. It changes only **how an already-trained flow/diffusion VLA is executed**. And its core tool is one very familiar to anyone who knows diffusion — **inpainting**. This piece follows the paper's argument, building up the concepts where they are needed, so that a reader with a thin robot-control background can see the reason behind each design decision.

---

## 1. The problem — the world keeps moving while the robot thinks

### 1.1 Latency is a structural by-product of large VLAs

Cyber-physical systems always run in real time. The world keeps flowing by the laws of physics while the model infers, and the delay between input and output shows up directly in performance. VLAs, however, are slow because they have billions of parameters, and they demand GPUs that are hard to put on a mobile robot, adding remote-inference overhead on top. The paper gives these numbers.

| Model | Latency | Reference point |
|---|---|---|
| π0 (3B) | 46ms for KV cache prefill alone on an RTX 4090 (before denoising) | target 50Hz → 20ms control period |
| π0 remote inference | 13ms for the network alone (wired, ideal conditions) | — |
| OpenVLA-OFT (7B) | 321ms on an A100 even after optimizing inference speed | — |

Prefill alone is more than twice the control period.

### 1.2 Action chunking is half an answer

The current standard solution is **action chunking**: output several steps of actions in one inference and execute them in succession. Fewer inferences spread out the burden of latency. But the latency itself does not go away. When a chunk runs out, you still have to wait for the next one, and the longer the chunk you use, the later the response to new observations.

So this paper has two walls to cross.

**⛔ Wall 1 — Inference latency can be reduced but not eliminated.** There is plenty of acceleration research: distillation, parallel decoding, quantization. But as the paper's related work points out, none of it can **get below a single forward pass.** If that one pass is longer than the control period, acceleration alone cannot meet the real-time constraint. Moreover, as robot data grows, the best-performing VLAs will grow too, so the gap will not shrink on its own. What is ultimately needed is **asynchronous execution** that overlaps inference and execution.

**⛔ Wall 2 — The moment you go asynchronous, chunk boundaries break.** If you compute the next chunk ahead of time, that chunk is made seeing only the observation at the start of inference. It does not know the actions that will be executed during inference. On top of that, the learned action distribution is multimodal, so two adjacent chunks can pick different strategies (modes). The result is an OOD jerk at the boundary. The common smoothing that averages several predictions does not guarantee valid actions.

Two constraints are attached to this.

| Constraint | Meaning |
|---|---|
| **Without retraining** | do not touch the existing VLA's weights or training recipe |
| **Without losing reactivity** | caring only about continuity makes it open-loop, ignoring new observations |

### 1.3 So the question the paper asks

> Can an already-trained flow VLA, without retraining, overlap inference with execution and still preserve both **continuity at chunk boundaries** and **reactivity to new observations**?

Wall 1 is gathered back in Section 3.4, and Wall 2 in the design of Sections 3.1–3.3 and the analysis of Section 4.

---

## 2. Background — putting the problem on the time axis

### 2.1 Action chunking and two horizons

$$\pi(A_t\mid o_t),\qquad A_t=[a_t,\ a_{t+1},\ \dots,\ a_{t+H-1}]$$

| Symbol | Meaning |
|---|---|
| $o_t$ | observation at control step $t$ |
| $a_t$ | the action for one control step (e.g., joint target positions) |
| $A_t$ | the action chunk generated from $o_t$ |
| $H$ | **prediction horizon** — the number of actions predicted at once |
| $s$ | **execution horizon** — how many of them are actually executed. $s\le H$, usually $s\approx H/2$ |

$s$ is a double-edged sword.

- **Long $s$** — new observations are reflected late, so reactivity drops.
- **Short $s$** — switching chunks often makes the motion jump through mode-jumping.

**Continuity vs reactivity.** This tension runs through the whole paper. How RTC resolves this tradeoff is revisited in Section 4.

### 2.2 Sampling from a flow policy

$$A_t^{\tau+\frac1n}=A_t^\tau+\frac1n\,v_\pi(A_t^\tau,\ o_t,\ \tau),\qquad A_t^0\sim\mathcal N(0,I)$$

| Symbol | Meaning |
|---|---|
| $\tau\in[0,1)$ | flow time. **τ=0 is noise, τ=1 is data** (the opposite direction to diffusion's $t$) |
| $v_\pi$ | the learned velocity field |
| $n$ | number of denoising steps. Every experiment in this paper uses $n=5$ |

The paper notes that a diffusion policy can be handled the same way by converting it to a flow at inference time. This piece explains in terms of flow.

### 2.3 Synchronous vs asynchronous — where Wall 2 shows itself

| Symbol | Meaning |
|---|---|
| $\Delta t$ | control period (20ms at 50Hz) |
| $\delta$ | time taken to generate one chunk |
| $d=\lfloor\delta/\Delta t\rfloor$ | **inference delay** — the number of control steps that pass between receiving the observation and the chunk coming out |

Real time means the guarantee "after receiving observation $o_t$, output $a_t$ within $\Delta t$." If $\delta\le\Delta t$ this is trivial, but by the numbers of Section 1.1 it is nearly impossible with modern VLAs. So execution strategies split two ways.

```
t :   0                        s-d               s
      |-------------------------|----------------|------------------>
SYNC  [=== A_0 : run s actions =================][ pause ~d ][ A_s ...
                                                 ^ observe o_s, then infer
ASYNC [=== A_0 : run s actions =================][ A_{s-d} ...........
                                ^ observe o_{s-d}, infer in parallel
                                                 ^ A_{s-d} ready -> swap
```

- **Synchronous** — execute $s$ actions, then stop and infer. This is the default in much prior work, and it produces a visible pause between chunks.
- **Asynchronous** — start inference at time $s-d$ and run it in parallel with execution. If $d\le H-s$, actions never stop. But the new chunk is made seeing only $o_{s-d}$ and does not know what happened between $s-d$ and $s$, so the boundary $a_{s-1\mid 0}\to a_{s\mid s-d}$ can be arbitrarily discontinuous. It gets worse as latency grows.

Here $a_{t'\mid t}$ is the $(t'-t)$-th action of the chunk $A_t$ made from observation $o_t$.

The problem with asynchronous execution is clear. Then isn't the synchronous approach merely slow, with nothing wrong in terms of accuracy?

> ### 💡 The pause in synchronous inference is not just a loss of speed — it changes the dynamics
>
> The paper points out that the pause changes the robot's dynamics itself, creating a distribution shift between training and evaluation. The demonstration data contains no robot that hesitates every second. While paused, the contact and inertial state with the object changes, and the next chunk starts on a velocity profile (stop → re-accelerate) never seen in training.
>
> There are even more extreme cases. Torque/force control has no notion of "holding in place," because gravity and inertia keep acting even while stopped. So the paper's simulation (Kinetix), which uses force-based control, has no synchronous baseline at all and compares only asynchronous variants against each other. The real robot, on the other hand, uses position control, so synchronous is a reasonable default and becomes the main comparison in the real-robot experiments (Section 5.2).

If both synchronous and asynchronous have problems, can the boundary discontinuity of the asynchronous approach be patched another way? The easiest idea is to average the predictions over the overlapping span to join them smoothly. **Temporal ensembling (TE)**, proposed by ACT, is exactly this. The paper's Figure 2 shows why this idea fails.

```
               old chunk {a}  : ------------------>   plan: go OVER
              /
  robot  *---+                  [  OBSTACLE  ]
              \
               new chunk {a'} : ------------------>   plan: go UNDER

  naive async : a_10 (over) -> a'_11 (under)   => huge OOD acceleration
  TE (average): mean(over, under)              => straight into obstacle
```

- The old chunk planned to go **over** the obstacle, the new chunk **under** it.
- Naive async jumps from the upper path to the lower path in a single step.
- TE heads for the average of the two paths — straight into the obstacle.

> ### 💡 On a multimodal distribution, the average is not a valid action
>
> Going over the obstacle and going under it are both valid modes. But their average is a head-on collision. From a diffusion background this is a familiar story: pixel-average two different face samples and you get a blurry ghost face.
>
> The reason for using a diffusion/flow policy instead of MSE regression in the first place was **to avoid mode averaging**. TE brings that averaging back in at execution time. In fact, in the paper's simulation TE did poorly even at zero latency, which the paper interprets as evidence that the benchmark is multimodal.

In short, to cross Wall 2 the new chunk must be conditioned from the generation stage onward **to stay in the same mode as the previous chunk**. After-the-fact averaging will not do.

---

## 3. Method — rewriting asynchronous chunking as inpainting

### 3.1 Key insight — actions whose execution is settled are already "known pixels"

Consider the moment asynchronous inference starts. The first $d$ actions of the new chunk are useless: that time will already have passed during inference, and those slots will be filled by actions from the previous chunk. In other words, **the first $d$ actions are values whose execution is already settled**.

Then you can fix those values and fill in the rest consistently with them. It is exactly the same structure as **inpainting**, which fills a masked region of an image so that it fits its surroundings. Three regions arise, indexed by the new chunk.

```
index      | 0  ...  d-1 | d  .......  H-s-1 | H-s  ...  H-1
-----------+-------------+-------------------+---------------
prev chunk | a  a  a  a  | a  a  a  a  a  a  |
new chunk  | a' a' a' a' | a' a' a' a' a' a' | a' a' a' a' a'
weight W   | 1  1  1  1  | 1 -> exp -> 0     | 0  0  0  0  0
region     | frozen      | intermediate      | fresh
length     | d           | H - d - s         | s
```

- **frozen** ($d$ actions) — actions that will already be executed during inference. They cannot physically be changed, so weight 1
- **intermediate** ($H-d-s$ actions) — the previous chunk has values here, but they execute **after** the new chunk arrives, so they can be revised. A gradually decreasing weight
- **fresh** ($s$ actions) — beyond the end of the previous chunk. Newly generated, so weight 0

The example in the paper's Figure 3 uses $H=16,\ d=4,\ s=5$, giving 7 intermediate actions.

One point is worth making. **Frozen is not overwriting.** The first $d$ actions of the new chunk are only guided to match the previous chunk; they are never actually executed. When the chunk is swapped, the execution index has already advanced by $d$, so execution starts from index $d$ of the new chunk (Algorithm 1, Section 3.4). The frozen span is an **anchor** that pulls on the rest of the actions.

### 3.2 Inpainting without retraining — ΠGDM guidance

Inpainting is something diffusion and flow models are naturally good at. RTC borrows the training-free flow inpainting of Pokle et al., which in turn is based on pseudoinverse guidance (ΠGDM). At every denoising step, one guidance term is added to the learned velocity field.

$$v_{\Pi\text{GDM}}(A_t^\tau,o_t,\tau)=v(A_t^\tau,o_t,\tau)+\min\!\Big(\beta,\ \frac{1-\tau}{\tau\,r_\tau^2}\Big)\big(Y-\hat A_t^{1}\big)^{\top}\mathrm{diag}(W)\,\frac{\partial \hat A_t^{1}}{\partial A_t^\tau}$$

$$\hat A_t^1=A_t^\tau+(1-\tau)\,v(A_t^\tau,o_t,\tau),\qquad r_\tau^2=\frac{(1-\tau)^2}{\tau^2+(1-\tau)^2}$$

| Symbol | Meaning |
|---|---|
| $Y$ | the target = the remaining actions of the previous chunk (right-padded to length $H$) |
| $W$ | per-position weight mask (frozen / intermediate / fresh from Section 3.1) |
| $\hat A_t^1$ | the "finished estimate," jumping from the current $A_t^\tau$ straight to τ=1 |
| $r_\tau^2$ | the uncertainty (variance) of that estimate |
| $\beta$ | upper bound on the guidance weight — the part the paper added |
| $\partial\hat A^1/\partial A^\tau$ | the Jacobian. Multiplied by the row vector in front, it becomes a vector-Jacobian product (VJP), computed with a single backprop |

$Y,A,W$ are treated as vectors of length $HM$ ($M$ is the action dimension). The mechanism reads in three steps.

- **⓵ Finished estimate** — estimate the final chunk once from the current noisy state. It plays the same role as diffusion's $\hat x_0$ prediction (DDIM's predicted $x_0$).
- **⓶ Weighted error** — measure how much the estimated finished chunk differs from the previous chunk at the positions $W$ points to.
- **⓷ Backpropagation** — pull that error back into $A^\tau$ space through the Jacobian and add it to the velocity field. The integration trajectory bends so that "once finished, it matches the prefix."

Think of it as reconstruction guidance, with "agreement with the prefix" in the place of the classifier in classifier guidance.

But the weight $\frac{1-\tau}{\tau r_\tau^2}$ becomes infinite at τ=0. The paper says only that it is clipped by β — so where does this expression come from, and what exactly does β clip?

> ### 💡 The guidance weight is (score→velocity conversion) × (confidence in the estimate)
>
> The paper does not derive this weight, so I work it out here. On the linear path $A^\tau=\tau A^1+(1-\tau)\epsilon$, the velocity field and the score are related as follows.
>
> $$v(x,\tau)=\frac{x}{\tau}+\frac{1-\tau}{\tau}\,\nabla_x\log p_\tau(x)$$
>
> - $\nabla_x\log p_\tau(x)$ — the score of the distribution at time τ
> - $\frac{1-\tau}{\tau}$ — the rate at which a change in score converts into a change in the velocity field
>
> Add a likelihood term $\nabla\log p(Y\mid A^\tau)$ to the score and the velocity field changes by $\frac{1-\tau}{\tau}$ times that. ΠGDM approximates $p(Y\mid A^\tau)$ as a Gaussian with mean $\hat A^1$ and variance $r_\tau^2$, so the score correction becomes $\frac{1}{r_\tau^2}(Y-\hat A^1)^\top\mathrm{diag}(W)\frac{\partial\hat A^1}{\partial A^\tau}$, and multiplying the two gives the weight in the equation above. $r_\tau^2$ is the posterior variance left in $A^1$ after seeing $A^\tau$, assuming the data is standard Gaussian.
>
> Put together, the weight takes a clean form.
>
> $$w(\tau)=\frac{1-\tau}{\tau\,r_\tau^2}=\frac{\tau}{1-\tau}+\frac{1-\tau}{\tau}$$
>
> It is a U-shaped curve with a minimum of 2 at τ=0.5, diverging at both ends. This explains β's role exactly. With $n=5$, τ∈{0, 0.2, 0.4, 0.6, 0.8}; the weight is infinite only at τ=0, and the maximum over the rest is 4.25, at τ=0.2 and 0.8. This matches the 4.25 in the paper's Figure 7 caption. So the β=5 the paper uses is in effect a setting that **fixes only the weight of the first step (τ=0) at 5**.
>
> Why is such a device needed? Image inpainting runs at around n=100, so each step's correction is small. Control runs at n=5, so one step's guidance is large, and if it is too much the chunk diverges (App. A.2). Indeed, raising β increased the maximum acceleration of the real-robot chunks, and in simulation there was no further gain at β=5 and above. β is a stabilizer that became necessary in moving an image technique into the **few-step regime**.

**Cost.** A VJP is computed at every step, so backpropagation is added. But backpropagation passes **only through the action expert**, since the KV cache of the image and language prefix does not depend on $A^\tau$. π0.5's profiling (Table 3, RTX 4090, bfloat16) shows this.

| Component | vanilla π0.5 | + RTC |
|---|---|---|
| Image encoder (SigLIP) | 18ms | 18ms |
| LLM prefill (Gemma 2B) | 44ms | 44ms |
| Denoising ×5 | 14ms | 35ms (2.5× per step) |
| **Total** | **76ms** | **97ms** |

Most of the latency is still the backbone; the 21ms RTC adds arises only in the action expert of about 300M parameters (based on the π0-family architecture, App. A.3).

Still, 2.5× per step is not light. Is backpropagation really necessary? Wouldn't it be much cheaper to overwrite the prefix positions with the desired values at every step?

> ### 💡 Overwriting the prefix alone does not bring the rest along
>
> In App. A.4, the paper applied Diffuser-style inpainting — overwriting the known span with the desired values at every denoising step — in the same setting and compared. It beats naive async but trails the guidance approach.
>
> The reason lies in the path along which information travels. Overwriting changes only the values at known positions, and relies on the network to adjust the remaining positions accordingly at the next step — **indirect transmission**. Guidance backpropagates the "prefix mismatch of the finished estimate" **directly** to every position through the Jacobian, so even the intermediate and fresh regions are adjusted to be consistent with the prefix. It is the same family of problem as, on the image side, the replacement approach needing supplements such as RePaint's resampling because of mismatches at the boundary.

### 3.3 Soft masking — use the whole overlap as an anchor

Hard masking, which gives weight 1 only to the $d$ frozen actions, was not enough. The ΠGDM correction is an approximation, and especially when $d$ is small there are few constrained positions, so the guidance signal is weak. The new chunk can then still switch strategies. In the paper's Figure 4, hard masking fails to match even the frozen span well and changes direction abruptly.

The fix is to use not just the $d$ frozen actions but **all $H-s$ overlapping ones**, decaying the weight further into the future because it is less certain.

$$W_i=\begin{cases}1 & i<d\\ c_i\,\dfrac{e^{c_i}-1}{e-1} & d\le i<H-s\\ 0 & i\ge H-s\end{cases},\qquad c_i=\frac{H-s-i}{H-s-d+1}$$

| Symbol | Meaning |
|---|---|
| $i$ | index within the new chunk, $0\le i\le H-1$ |
| $c_i$ | progress that decreases linearly from near 1 to near 0 over the intermediate span |
| $W_i$ | the "attention" weight given to the $i$-th action of the previous chunk |

Computed for the real-robot setting ($H=50,\ s=25,\ d=6$), the decay is quite fast.

| $i$ | 6 | 10 | 15 | 20 | 24 |
|---|---|---|---|---|---|
| $W_i$ | 0.88 | 0.49 | 0.19 | 0.04 | 0.001 |

This mask is the very knob that handles the tension of Section 2.1.

| Region | Attitude toward the previous chunk |
|---|---|
| frozen | physically enforced — must follow it |
| intermediate | follow the previous plan where possible, but may depart if the new observation is different enough |
| fresh | free — generated from the new observation alone |

In the comparison of decay schedules (App. A.4), exponential was best, with linear a close second. Giving 1 to the whole overlap with no decay, and hard masking, both trailed.

### 3.4 The whole system — two threads that hide latency

The whole system (Algorithm 1) is two threads sharing state under a mutex.

| Component | Role |
|---|---|
| `GETACTION` | called by the controller every $\Delta t$. Hands out one action and stores the latest observation |
| `INFERENCELOOP` | a background thread. Starts inference when the execution counter reaches $s_{\min}$, and swaps in the chunk as soon as it finishes |
| `GUIDEDINFERENCE` | computes $W$, pads the previous chunk, and runs $n$-step guided integration |

There are three operational details.

- **⓵ Latency is predicted conservatively** — the **maximum** of the last $b$ measured latencies (10 on the real robot) is used as the next $d$.
- **⓶ $s$ varies by chunk** — the actual $s=\max(d,\ s_{\min})$, with $s_{\min}=25$ on the real robot.
- **⓷ Keep $d\le s\le H-d$** — the lower bound is because the next inference can start only after the current one finishes. The upper bound is because one chunk must supply actions from index $d$, counted from the start of its own inference, up to $s+d-1$, when the next chunk arrives ($s+d\le H$).

**Wall 1 is gathered back here.** RTC does not reduce latency. It actually **increases** it, from 76ms to 97ms. Instead, as long as $d\le H-s$, it **hides** the latency. A π0.5 chunk is $H=50$ × 20ms = 1 second, so even injecting +200ms to make $d\approx16$ (32% of the chunk) stays comfortably within the constraint. Rather than trying to break through the wall of a single forward pass with acceleration, it hides that wall behind execution.

> ### 📌 What sets RTC apart — hide latency instead of reducing it, and pay the price of boundary discontinuity with inpainting
>
> - **Inference-time only** — uses existing flow/diffusion VLAs' weights and training recipes as they are.
> - **Redefining the problem** — sees asynchronous chunking as "generation conditioned on a prefix whose execution is settled," i.e., inpainting. The paper claims it is the first application of inpainting or guidance to real-time control.
> - **Porting an image technique to few-step** — it did not work as is, and two modifications were needed: soft masking (reinforcing a weak signal) and β clipping (preventing divergence).

---

## 4. Why it works — gathering back Wall 2 and the tradeoff

The paper's own explanation falls into three strands.

| Axis | Mechanism responsible | Effect |
|---|---|---|
| **⓵ Continuity** | guidance on the frozen span | the OOD jerk at the boundary disappears |
| **⓶ Strategy consistency** | soft mask | widens the guidance signal to the whole overlap, suppressing mode switches |
| **⓷ Reactivity** | conditioning on the latest observation + decaying weights | leaves room for new information in the intermediate and fresh spans |

**⓵ and ⓶ gather Wall 2 back.** Instead of smoothing the chunk boundary after the fact, the new chunk is made to be generated on the same mode as the previous chunk from the start.

The most convincing evidence is **the scene where the continuity vs reactivity tradeoff previewed in Section 2.1 is resolved**. In an experiment with latency fixed at $d=1$ and $s$ varied from 1 to 7 (Figure 5, bottom left), only RTC and BID improved **monotonically** as $s$ got shorter. Normally, shortening $s$ hurts because mode jumps increase. But with continuity guaranteed, shortening $s$ becomes a pure gain of "reflecting new observations more often." **Once continuity is secured, reactivity comes for free.**

The same trend shows on the real robot. Even counting only pure control steps, excluding pause time, RTC progressed through tasks faster. The paper interprets this as the result of fewer mistakes and retries. This, however, is an interpretation tied to the paper's inference that "pauses create distribution shift"; the distribution shift was not measured directly.

From an LLM background, though, one thing looks odd. LLMs also generate the next sentence while streaming, but they have no special mechanism to avoid contradicting what they have already said. Why does this become a whole paper's worth of problem for VLAs?

> ### 💡 For an AR model, continuing a prefix is free; a chunk-level flow model has no entry point for it
>
> **An LLM** is trained on $p(x_k\mid x_{<k})$, so it is automatically conditioned on tokens already output and irreversible. That what follows agrees with what came before is guaranteed by the model's structure.
>
> **A flow VLA** samples a chunk of $H$ actions at once as $\pi(A_t\mid o_t)$, conditioned only on the observation. During training, "the actions from the previous chunk whose execution is settled" are not in the input. **There is no entry point at all** through which to condition on them.
>
> What RTC does is approximately sample, at inference time via guidance, a conditional distribution that was never trained.
>
> $$A_t\ \sim\ \pi\big(A_t\ \mid\ o_t,\ A_{\text{prefix}}\big)$$
>
> - $A_{\text{prefix}}$ — the remaining actions of the previous chunk (reflected with weight $W$)
> - This distribution was never trained — RTC **attaches the condition after the fact** with guidance on top of an unconditional model
>
> By analogy with a voice agent, the frozen span is "words already spoken aloud," and the intermediate span is "words you meant to say but can still change." The natural next move, then, is to put this condition in **at training time** — Section 6 looks at follow-up work in that direction.

---

## 5. Experiments

### 5.1 Simulation — a Kinetix benchmark of dynamic tasks only

Most existing simulated imitation-learning benchmarks are quasi-static, so they nearly saturate even with essentially open-loop execution over a long $s$. They are no stage for revealing the effect of latency. So the paper built a new benchmark with Kinetix, a simulator with force-based control: 12 environments, 10 existing and 2 new, including dynamic manipulation such as throwing, catching and balancing, and locomotion (half-cheetah, walker and others). Gaussian noise is added to the actions so that closed-loop correction becomes essential.

| Item | Setting |
|---|---|
| Experts | RPO + binary success reward, 6 seeds per environment |
| Data | 1M transitions. A different expert per episode makes the data multimodal |
| Policy | flow policy, $H=8$, 4-layer MLP-Mixer, 32 epochs |
| Evaluation | 2048 rollouts per data point, 95% Wilson interval |
| Latency | $d\in\{0,\dots,4\}$ (4 is the maximum possible at $H=8$) |

There are three baselines.

| Baseline | Behavior |
|---|---|
| Naive async | ignores the previous chunk entirely and swaps as soon as the new chunk is ready |
| BID | keeps continuity between chunks by rejection sampling. N=32, K=3, with the 8-epoch checkpoint as the weak model |
| TE | averages the predictions of several chunks for the same time step |

The results are presented only as graphs, so I summarize them qualitatively (Figures 5, 7, 8).

- **TE** — does poorly even at zero latency. It directly confirms Section 2.3's "the average is not a valid action."
- **RTC** — the most robust to increasing latency, and its gap over BID widens as latency grows.
- **Hard vs soft masking** — hard trails, with the difference especially large when $d$ is small. This supports the claim of Section 3.3.
- **Ablations** — β gives no further gain at 5 and above; for decay schedules, exponential ≳ linear; Diffuser-style inpainting falls short of RTC.

The paper stresses that RTC beats BID while using far less compute. This comparison has to be read with its conditions.

> ### ⚠️ Fact check — "better than BID with far less compute" is not a compute-matched comparison
>
> - The simulation's BID samples 64 chunks per step (32 strong + 32 weak). RTC winning is a fact, but there is no comparison at the same compute budget.
> - The "2.3× latency" given as grounds for excluding BID on the real robot is based on full BID (223ms). A lightweight BID without the forward contrast takes 115ms, about 1.2× RTC's 97ms (Table 1). The performance of this lightweight variant is not reported, so there is no basis to confirm whether dropping BID on the real robot was justified.

### 5.2 Real robot — π0.5, bimanual 6-DoF

**Setup.** With π0.5 ($H=50$, $\Delta t$=20ms, $n=5$) as the base, evaluation runs on a bimanual system of two 6-DoF arms with parallel grippers. Model latency is 76ms for the baseline and 97ms for RTC, and LAN remote inference adds 10–20ms, for a default $d\approx6$. Measured end-to-end latency was 109ms stationary and 139ms mobile (Table 2). On top of this, **+100ms ($d\approx11$) and +200ms ($d\approx16$)** were injected artificially, assuming a larger model or a distant cloud server.

| Task | Stages | Time limit | Notes |
|---|---|---|---|
| Light candle | 5 | 40s | strike a match and light a candle. The most precise, and the only one where retries are impossible |
| Plug ethernet | 6 | 120s | reorient the cable end and insert it into a server rack, repeated for both ends |
| Make bed (mobile) | 3 | 200s | move a blanket corner and two pillows. The hardest of all |
| Shirt folding | 1 | 300s | fold a laid-out shirt |
| Batch folding | 4 | 300s | take out crumpled clothes, flatten, fold and stack them |
| Dishes in sink (mobile) | 8 | 300s | move 4 items into the sink |

| Baseline | Behavior |
|---|---|
| Synchronous | execute 25, then stop and infer (the default in prior work) |
| TE, sparse | $s=25$, parallel inference + TE applied over the overlap |
| TE, dense | infer as often as possible ($s=d$). Closest to ACT's original TE |

Evaluation is 10 runs per task × method × latency combination, 480 episodes in total (28 hours of pure execution time). 480 = 6 tasks × 8 combinations × 10, because the two TE variants could run only at +0ms (RTC 3 + Synchronous 3 + TE 2). The metric is **throughput**: the mean over episodes of (fraction of stages completed) ÷ (time taken). It reflects speed and success together.

**Results (Figure 6)**

| Observation | Details |
|---|---|
| Throughput | RTC is highest under every latency condition |
| Robustness to latency | RTC shows no degradation with injected latency. Synchronous degrades linearly. Both TE variants oscillate so badly at +100/+200ms that the robot's protective stop triggers, making execution impossible |
| Speed excluding pauses | even counted in control steps with pause time removed, RTC progresses faster |
| Final scores | RTC leads by a wide margin on light candle, where retries are impossible, and on bed making, the hardest |

Figure 1 shows this result in its most dramatic scene: even with more than 300ms of latency (over 30% of the prediction horizon), the robot strikes a match and lights a candle. Summing up, the paper says it "significantly improved task throughput." The scope of this summary needs to be pinned down precisely.

> ### ⚠️ Fact check — statistical significance is confirmed only in the injected-latency range
>
> - As the text states, statistical significance holds **only at +100/+200ms**. Under actual operating conditions (+0ms, $d\approx6$), RTC scores highest but not significantly (10 runs per combination, error bars ±1 SEM).
> - In the +100/+200ms range TE cannot run, so it is effectively an **RTC vs Synchronous** comparison.
> - The "20% faster than synchronous" in Figure 1 is an example figure from **the first 10 seconds of a single** match-lighting rollout, not an aggregate metric.
> - One labeling error — the text's "per-task results (Figure 5, top)" should point to Figure 6 top. Figure 5 is the simulation results.
>
> The most accurate summary is **"the larger the latency, the greater RTC's value."**

---

## 6. Positioning — among neighboring work

RTC sits at the intersection of four streams.

**⓵ Action chunking → VLA.** Since ACT and Diffusion Policy, action chunking has become the de facto standard in visuomotor imitation learning, and it has scaled up into VLAs such as π0/π0.5, OpenVLA, Octo and GR00T. The paper mentions that a chunking policy running on top of low-level PID can be seen as the outer loop of cascade control, but leaves that intersection for future work.

**⓶ Research that reduces latency itself.** Consistency policy (distillation), streaming diffusion policy, OpenVLA-OFT (parallel decoding), and LLM serving optimizations in general such as PagedAttention and quantization. That none of them can get below a single forward pass was the grounds for Wall 1. They are complementary to RTC, not competing with it.

**⓷ Inpainting and guidance.** Image inverse-problem solvers such as Pokle et al., ΠGDM and RePaint, and Diffuser, which used replacement-style inpainting for planning. RTC brought the guidance approach among these into real-time control.

**⓸ Real-time execution.** Structurally, the closest thing is **MPC**. They share parallelizing execution and computation, and warm-starting the next plan from the previous one. But MPC requires an explicit dynamics model and cost function. RTC can be read as "a receding-horizon warm-start for model-free imitation-learning policies." Meanwhile, hierarchical VLAs that split into System 2 (slow planning) and System 1 (fast action) are an orthogonal approach, with the tradeoff that System 1's size is limited and it needs a separate training recipe.

Setting the methods for securing continuity at chunk boundaries side by side makes RTC's coordinates clear.

| Method | How it secures continuity | Added cost | Result in the paper | LLM-side counterpart |
|---|---|---|---|---|
| TE (ACT) | averages predictions for the same time step | almost none | poor even at sim $d=0$; cannot run at high latency on the real robot | averaging several sampled outputs |
| Diffuser-style inpainting | overwrites the known span at every step | almost none | some gain, but below RTC | forced prefix insertion |
| BID | batch sampling, then rejection sampling | batch samples + two models | below RTC, gap widening with latency | best-of-N reranking |
| **RTC** | ΠGDM guidance + soft mask | a VJP per step (denoising 2.5×) | best across the board | guided decoding |

The closest prior work is BID. The paper, too, acknowledges that although BID did not account for latency, it is a method that can produce the same effect. The difference between the two sums up in one line of the table — **BID samples many and picks; RTC samples one and bends it.**

Finally, back to the cost question of Section 3.2. All of RTC's overhead comes from "approximating, with inference-time guidance, a condition that was never trained." As previewed in Section 4, what if that condition were put into training from the start?

> ### 🔗 Training-Time RTC (Black et al., 2025-12) — moving the same condition to training time
>
> A follow-up by the same group (arXiv:2512.05964) simulates inference latency during training and conditions directly on the action prefix, eliminating the overhead of inference-time inpainting. It is implemented in a few lines of code, with no changes to the model architecture or the robot runtime.
>
> | | **RTC (this paper)** | **Training-time RTC** |
> |---|---|---|
> | When the prefix condition is applied | inference (guidance) | training (latency simulation) |
> | Inference overhead | a VJP per step | none |
> | Retraining | not needed | needed |
> | Reported results (per the abstract) | — | beats RTC at high latency in simulation; on par in performance and speed on a π0.6 real robot |
>
> What comes next is the interesting part. A follow-up study in 2026 (Soft RTC, arXiv:2605.25537) points out as a limitation that training-time RTC uses a binary mask that leaves everything outside the prefix completely free, and reintroduces a soft window over the overlap. The soft masking of this paper's Section 3.3 turned out to be needed again in the training-time version. It is evidence that RTC's observation — that the intermediate span must be treated as "a region neither fixed nor free" — survives even when the method changes.

---

## 7. Limitations

**Limitations the paper acknowledges**

| Limitation | Details |
|---|---|
| Compute overhead | considerably heavier than sampling directly from the base policy (76 → 97ms) |
| Scope | for diffusion/flow policies only. Cannot be used as is on AR token-based VLAs |
| Dynamic range in the real world | more dynamic settings such as locomotion exist only in simulation |
| Simplifying assumptions | latency and synchronization issues below the level of a control step were not considered |

**Further points to raise**

- **The tradeoff was relocated, not removed.** The intermediate weights pull samples toward the previous plan. So the speed of response to sudden disturbances (an object being pushed, say) is tied to the decay schedule of $W$. No experiment directly measured disturbance response on the real robot; the action noise in simulation is only indirect evidence.
- **ΠGDM is an approximation.** It relies on a Gaussian posterior variance and the one-step estimate $\hat A^1$, so at $n=5$ the prefix match is approximate. Since the frozen span is never executed anyway, the approximation error shows up as a small discontinuity at the swap point (index $d$), and the soft mask reads as the structure that softens it.
- **The gain concentrates where latency is large.** Under current hardware and LAN conditions, no significant difference was confirmed (Section 5.2).
- **Scope of validation for network conditions.** Validation covers only LAN and artificially injected fixed latency. Since latency prediction is the maximum of the last 10, behavior under high-jitter WAN or packet loss needs to be checked separately.

---

## 8. Closing — what this paper suggests

RTC's contribution is not a new model but **a redefinition of how execution works**. Instead of trying to eliminate inference latency by acceleration, it hides it behind execution, and turns the boundary discontinuity that hiding creates into a conditional generation problem. Since latency will grow as models grow, the value of this view will increase over time.

And this paper reads unusually well for someone with an LLM and diffusion background.

- **A direct port of image inverse-problem solvers** — an action chunk is an $H\times M$ "1D image," and training-free solvers such as ΠGDM carry over almost as is. But image techniques assume n≈100, so the few-step regime needs stabilizers such as β. It is a rare point where experience from image-generation research transfers directly to robot control.
- **A division of labor with LLM serving** — of π0.5's 76ms latency, 62ms is SigLIP and the Gemma prefill. Serving optimizations such as quantization and caching reduce $\delta$, and RTC hides the $\delta$ that remains. It is the same way of thinking as latency hiding on GPUs or prefetch pipelining.
- **Generalization to agentic AI** — the pattern "treat the actions being executed as a frozen prefix and plan what comes next on top of it" is a design principle for any agent whose slow planner must think in parallel with execution. It has the same structure as a voice agent continuing its next sentence consistently with the sentence it has already spoken.

Above all, the result of no performance degradation up to +200ms (32% of the chunk length) means that running a large VLA through remote or cloud inference, without putting a GPU on the robot, could become a realistic option. As noted in Section 7, validation under real WAN conditions remains, but it clearly shakes the premise that "the VLA must sit next to the robot."

---

## Appendix — Glossary

| Term | Definition |
|---|---|
| **action chunking** | outputting an action chunk of $H$ steps in one inference and executing $s$ of them |
| **prediction / execution horizon** | chunk length $H$ / actual execution length $s$ |
| **inference delay** | the number of control steps that pass between receiving the observation and the chunk coming out, $d=\lfloor\delta/\Delta t\rfloor$ |
| **synchronous / asynchronous inference** | stop and infer / infer in parallel with execution |
| **temporal ensembling (TE)** | smoothing that averages several chunks' predictions for the same time step |
| **frozen / intermediate / fresh** | the three regions of the new chunk — execution settled / revisable overlap / newly generated |
| **ΠGDM guidance** | training-free inpainting that backpropagates the mismatch between the finished estimate and the target through the Jacobian and adds it to the velocity field |
| **soft masking** | a mask that gives exponentially decaying weights across the whole overlap |
| **β (guidance weight clipping)** | the upper bound on the guidance weight, which diverges at τ=0. A few-step stabilizer |
| **throughput** | the mean over episodes of (fraction of stages completed) ÷ (time taken) |

**Original** — [arXiv:2506.07339](https://arxiv.org/abs/2506.07339) · **Project page** — [pi.website/research/real_time_chunking](https://pi.website/research/real_time_chunking) · **Code (simulation)** — [Physical-Intelligence/real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)
