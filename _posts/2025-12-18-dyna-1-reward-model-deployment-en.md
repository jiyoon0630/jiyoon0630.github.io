---
layout: paper
lang: en
ref: dyna-1-reward-model-deployment
kind: tech-review
title: "Dynamism v1 (DYNA-1) Model: A Breakthrough in Performance and Production-Ready Embodied AI (DYNA-1)"
date: 2025-12-18 12:00:00 -0800
paper_date: 2025-04-29
venue: "Dyna Robotics technical blog · no arXiv"
tags: [VLA, Reward-Model, Robot-Foundation-Model, Self-Improvement, Real-World-Deployment, Tech-Review]
authors: "Dyna Team (no individual authors listed)"
affiliations: "Dyna Robotics"
summary: "What breaks a policy in continuous deployment is not a low success rate but unrecoverable failure — reading, through the disclosed pieces and inference, the claim that a task-progress reward model scoring, segmenting and selecting the deployment stream reached 24 hours of intervention-free napkin folding."
paper_url: "https://www.dyna.co/research/dyna-1"
---

> **Core claim** — What brings down a robot policy in continuous deployment is not a low episode success rate but **unrecoverable failure**. Instead of adding human demonstrations, DYNA-1 iteratively improves the policy by using **a reward model that estimates task progress** to score, segment and select the continuous experience it accumulates itself during deployment, and reports that it reached 24 hours of intervention-free continuous operation on napkin folding.

---

## Introduction

Robot foundation model performance is usually reported as "episode success rate." A person resets the scene, the policy tries once, and you count whether it succeeded. But what a robot folding napkins at the back of a restaurant needs is not a single success but a full shift. There is no one to reset it, and if it botches one napkin it has to carry on to the next from that tangled state.

DYNA-1 is Dyna Robotics' first foundation model, aimed squarely at this gap. It was announced on April 29, 2025. Dyna Robotics was co-founded by Lindon Gao and York Yang, who sold Caper AI, and Jason Ma, a former DeepMind research scientist.

One thing to state up front. The source this piece covers is not a paper but **a technical blog for a product announcement**. The policy architecture, how the reward model is trained and the policy-update algorithm are barely disclosed. So this review proceeds while distinguishing three levels.

| Marker | Meaning |
|---|---|
| **[Blog]** | What the source says directly |
| **[Inference]** | Inferred from the source's description and the underlying research |
| **[Assessment]** | The author's judgment |

Little is disclosed, but the problem setting the blog poses is itself sharp. This review follows that problem setting and builds up, with the concepts needed, the mechanisms by which the disclosed pieces could fit together.

---

## 1. The problem — doing it well once versus working all day

### 1.1 The blog's diagnosis

**[Blog]** The usual pipeline of adding broad data to a big model plateaus at around 80% single-episode success on hard dexterous tasks. Most models drift into unrecoverable states when run continuously for more than 30 minutes. Even Dyna's best internal VLA baseline looked fine at first, then lost context after an hour or two and could not correct itself.

Two kinds of numbers are mixed in this diagnosis. One is "80%," a per-episode number; the other is "30 minutes," a per-time number. How the two connect is the starting point of this whole piece.

### 1.2 Three walls

**⛔ Wall 1 — Unrecoverable states** Continuous operation has no reset. A single dead-end state ends the whole run. Why 80% is fatal in continuous operation is calculated in Section 2.1.

**⛔ Wall 2 — Deployment data with no judge** When the robot runs on its own, data piles up. But ⓐ there are **no labels** for which segments went well, and ⓑ being a continuous stream, there are **no episode boundaries** either. The blog itself directly flags, as a distinctive challenge, that continuous deployment data does not naturally come with episode boundaries.

**⛔ Wall 3 — Success ≠ commercial viability** A restaurant's standard is not "did it fold" but "how fast, and how neatly, did it fold." Throughput and quality are required separately from success rate.

### 1.3 So the question this piece asks

> Without adding human demonstrations, can a policy be lifted to **24 hours intervention-free + commercial speed and quality** using only the experience the robot accumulates itself during deployment?

The blog's answer is one sentence — **you need an accurate reward model.** To see how this answer gets over each of the three walls, two pieces of background come first: how continuous operation changes the metric, and what a reward model in robotics is actually estimating.

---

## 2. Background

### 2.1 Continuous operation changes the metric — the unrecoverable failure rate

Treat one napkin as one episode, and split its outcome three ways.

$$q_s + q_r + q_u = 1$$

- $q_s$ — probability of success
- $q_r$ — a failure after which the robot can move on to the next napkin by itself (recoverable)
- $q_u$ — a dead-end failure that ends only with human intervention (unrecoverable)

The only event that ends continuous operation is $q_u$. Simplifying by assuming each napkin's outcome is independent, the number of napkins handled without intervention follows a geometric distribution.

$$\Pr\big[N\text{ napkins in a row without intervention}\big] = (1-q_u)^N,\qquad \mathbb{E}[\text{run length}] = \frac{1}{q_u}$$

- $N$ — number of napkins to handle in a row. About 850 napkins in 24 hours is the reference (Section 5.1)

Filling in the table by $q_u$:

| $q_u$ | Mean run length $1/q_u$ | Probability of 850 napkins in a row without intervention |
|---|---|---|
| 0.2 (every failure of an 80% model is a dead end) | 5 napkins | $\approx 10^{-82}$ |
| 0.01 | 100 napkins | $\approx 0.02\%$ |
| 0.001 | 1,000 napkins | $\approx 43\%$ |
| 0.0001 | 10,000 napkins | $\approx 92\%$ |

Three things can be read from the table.

- **$q_s$ determines yield; $q_u$ determines uptime.** Success rate and the ability to run continuously are different metrics.
- If every failure of an 80% model is a dead end, it stops after 5 napkins on average. This explains the "fine at first, then quickly falls apart" pattern of Section 1.1.
- To last 24 hours with even a one-in-two chance requires $(1-q_u)^{850}\ge 0.5$, that is, $q_u \lesssim 0.08\%$. Driving the episode failure rate itself down to that level with imitation learning alone is unrealistic.

So the design goal changes. **Not eliminating failures, but moving failures from $q_u$ to $q_r$.** What Wall 1 really is, in the end, is the absence of recovery ability.

Then why not have people demonstrate more recovery scenes? Here the classic problem of imitation learning gets in the way.

$$\min_\theta\ \mathbb{E}_{s\sim p_{\text{demo}}}\big[\ell(\pi_\theta(s),\,a^*)\big]\qquad\text{vs.}\qquad \text{deployment performance is determined over } s\sim d^{\pi_\theta}$$

- $\pi_\theta$ — the policy being trained, $\theta$ — policy parameters
- $p_{\text{demo}}$ — the state distribution visited by human demonstrations
- $d^{\pi_\theta}$ — the state distribution the policy visits in actual deployment
- $\ell$ — imitation loss, $a^*$ — demonstrated action

A crumpled state where several napkins get pulled out at once is **a state the policy itself produces**. People cannot imagine all such variations in advance and put them in demonstrations. Closing the mismatch between the two distributions (covariate shift) needs data from states the policy visited itself. And at that moment Wall 2 appears — who scores that data?

### 2.2 A robot reward model is a task-progress estimator

The reward model (RM) that is the blog's answer estimates **task progress**.

$$p_\phi(o_t,\,g)\in[0,1]$$

- $o_t$ — observation at time $t$ (multi-camera images, etc.)
- $g$ — the task (e.g., folding one napkin)
- $\phi$ — RM parameters
- Meaning of the value — 0 is the start, 1 is completion

Why progress is a "reward model" becomes clear through its relation to value. A common training scheme labels each frame of a successful demonstration of length $T$ with $t/T$. Meanwhile, if reward 1 is given only at completion with discount $\gamma$, the value of a policy that finishes in the remaining $T-t$ steps is:

$$V(s_t) = \gamma^{\,T-t}$$

- $V(s_t)$ — expected discounted cumulative reward starting from state $s_t$
- $\gamma\in(0,1]$ — discount factor

This value is monotonically increasing in $t/T$. That is, progress is **value with its shape changed**. GVL (Section 3.2), which looks like the underlying research, also states explicitly that progress estimation is equivalent to general value learning under a particular choice of reward.

Using progress as a potential yields a dense reward (potential-based shaping, Ng et al., 1999).

$$r_t = \gamma\,p_\phi(o_{t+1}) - p_\phi(o_t)$$

$$\sum_{t=0}^{T-1}\gamma^{t}\,r_t \;=\; \gamma^{T}\,p_\phi(o_T)\;-\;p_\phi(o_0)$$

- $r_t$ — shaped reward at step $t$
- The second equation — all intermediate terms cancel (telescoping), leaving only the start and end states
- Potential-based shaping is known not to change the optimal policy

This reward has two properties.

**⓵ A signed signal at every step** — when the napkin tangles, $p$ drops and immediately $r_t<0$. Unlike a success reward that fires once at the end, it tells you **where things went wrong**.

**⓶ A preference for speed is built into the objective** — with $\gamma<1$, for the same completion, the smaller $T$ is, the larger $\gamma^{T}$. This property returns in Section 3.7 together with Wall 3.

A reader with an LLM background will have a question here. An RLHF reward model assigns one score to a whole response; is this the same thing?

> ### 💡 A progress RM is closer to a PRM than an ORM
>
> | | LLM ORM | LLM PRM | Robot progress RM |
> |---|---|---|---|
> | Unit of evaluation | Whole response | Each reasoning step | Each time point (frame) |
> | Signal density | Once at the end | Every step | Every time point, continuous value |
> | What it tells you | Was it right | Which step went wrong | Where progress stalled or went backward |
>
> There is one decisive difference. A PRM's "steps" are already given in the text. A robot's continuous stream comes with neither steps nor episodes. So the progress curve, on top of scoring, takes on the role of **creating the very structure of the data** (Section 3.4).

---

## 3. Method — RM-in-the-loop

The motto the blog puts up is this:

> "Don't practice until you get it right. Practice until you can't get it wrong."

In the language of Section 2.1, it is **"practice until $q_u\to 0$."** It declares that the goal is not succeeding once ($q_s$) but making dead-end failures disappear ($q_u$).

### 3.1 [Blog] The disclosed components

The blog places the core of the recipe for building a robust, autonomous robot foundation model on an accurate RM, and calls it robotics' first scalable foundation reward model. Gathering the disclosed pieces:

| Component | The blog's description | Disclosed detail |
|---|---|---|
| Foundation RM | Estimates task progress from diverse robot experience; claimed to far outperform existing approaches | No architecture, training data or quantitative metrics |
| RM use ⓵ | Autonomous exploration — explores the action space to discover effective strategies | No method |
| RM use ⓶ | Intentional error recovery — recognizes mistakes and recovers on its own | No method |
| RM use ⓷ | Generating and curating high-quality datasets | No method |
| Stream processing | Automatically segments boundary-less streams, labels progress and subtasks | No method |
| RM-in-the-loop training | Scaled this training and improved over 6 weeks | Update operator undisclosed |
| On-site training | A small amount of extra training at a new site | No amount or method |

Sections 3.2–3.7 below are **[Inference]** about how these pieces could fit together. The inference rests on two things: the goals and uses the blog itself states, and the applications proposed by GVL, which looks like the underlying research.

### 3.2 [Inference] What the RM is — GVL, the likely underlying research

The blog says only that the RM was built on the company's earlier research, without a citation. The most likely candidate is **GVL (Generative Value Learning, ICLR 2025)**, first-authored by co-founder Jason Ma.

GVL is a general value estimator that predicts task progress using the world knowledge a VLM has. Its key trick is to ask with the frames **shuffled**.

$$\big(\hat v_{\sigma(1)},\dots,\hat v_{\sigma(T)}\big)=\text{VLM}\big(o_{\sigma(1)},\dots,o_{\sigma(T)};\ l,\ o_1\big)$$

- $\sigma$ — a random permutation (shuffling the frames)
- $o_{\sigma(i)}$ — the $i$-th frame in shuffled order
- $l$ — the task's language description
- $o_1$ — the first frame, provided as the 0% reference point
- $\hat v$ — the progress prediction for each frame. Undoing with $\sigma^{-1}$ gives the chronological curve

Why not show them in order? Consecutive frames are very strongly correlated in time. Given in order, a VLM can take the shortcut of "making the numbers bigger toward the end" without looking at frame content. Shuffled, it can only answer by actually judging each frame's content. It is the same idea as shuffling multiple-choice option order in LLM evaluation to remove position bias.

GVL, however, works in-context without training. Whether DYNA's foundation RM is a model trained on the company's robot data, and what its architecture is, is not disclosed.

### 3.3 [Inference] The overall structure of the loop

The loop that most naturally connects the blog's descriptions is this:

```
 +-------------+   stream (o_t, a_t)   +---------------------+
 |  policy     | --------------------> |  reward model       |
 |  pi_k       |                       |  p(o_t) per frame   |
 +-------------+                       +---------------------+
        ^                                        |
        |                                        v
 +-------------+   weighted / filtered +---------------------+
 |  update     | <-------------------- |  segment + subtask  |
 |  pi_{k+1}   |   data D_k            |  + advantage A_t    |
 +-------------+                       +---------------------+
```

- **policy $\pi_k$** — the round-$k$ policy is deployed and produces a continuous stream
- **reward model** — assigns progress to every frame
- **segment + subtask + advantage** — splits the stream into episodes and subtasks and computes how good or bad each segment is (Sections 3.4, 3.5)
- **update** — builds the next round's policy from the selected or weighted data

The blog's "6 weeks" reads as a record of running this loop several times on a weekly basis (Section 5.2). Now let's see, step by step, which wall each stage gets over.

### 3.4 Segmentation — Wall 2ⓑ, recovering episode boundaries from the progress curve

Plot progress over a continuous stream and you get a curve like this:

```
 p(o_t)
  1.0 |       /|               /|           /|
      |      / |              / |          / |
      |     /  |     /\      /  |         /  |
      |    /   |    /  \____/   |   _____/   |
  0.0 |___/    |___/            |__/         |___
      +-------------------------------------------> t
               B1     (a)       B2   (b)     B3
```

- **B1, B2, B3** — points where $p$ drops sharply from near 1 to near 0. One napkin has finished and the next has started, so these become **episode boundaries**
- **(a)** — a segment where $p$ drops sharply mid-progress and then recovers. An error (e.g., several napkins pulled out) and the **recovery** from it
- **(b)** — a segment where $p$ stays flat for a long time. Stuck, or a **slow strategy**
- **Subtask labels** — mapped onto segments of the curve (pull → spread → first fold → …)

Then is scoring all the RM does in this loop? The curve says otherwise. Before scoring, you first have to decide "what to score as one unit," and the RM does that too.

> ### 📌 The RM is both judge and segmenter
>
> A reset-free continuous stream has no episodes. Without episodes, neither return, nor success labels, nor units of curation are defined. DYNA-1's RM, before rating good and bad, first **turns the stream into an episodic dataset.** This is where it structurally parts ways with most robot-learning setups that assume episode resets. A single model fills both defects of Wall 2 (no labels, no boundaries) at once.

### 3.5 Judging and improving — Wall 2ⓐ, selecting and weighting data by RM score

For each segmented span, a progress-based advantage can be computed.

$$\hat A_t = p_\phi(o_{t+k}) - p_\phi(o_t) - b(o_t)$$

- $k$ — length of the evaluation window (steps)
- $b(o_t)$ — baseline. The average progress gain from similar states
- $\hat A_t>0$ — a span that made more progress than average, $\hat A_t<0$ — a span that cut into progress

The two standard improvement operators that best fit the description "RM-in-the-loop training" are:

**⓵ Filtered BC** — train only on samples whose advantage exceeds a threshold.

$$\mathcal{L}_{\text{filt}}(\theta)=\mathbb{E}_{(o_t,a_t)\sim\mathcal{D}_k}\Big[\mathbf{1}\big[\hat A_t>\tau\big]\cdot\ell_\theta(o_t,a_t)\Big]$$

**⓶ AWR (advantage-weighted regression)** — use all samples, but weight them exponentially by advantage.

$$\mathcal{L}_{\text{AWR}}(\theta)=\mathbb{E}_{(o_t,a_t)\sim\mathcal{D}_k}\Big[\exp\big(\hat A_t/\beta\big)\cdot\ell_\theta(o_t,a_t)\Big]$$

- $\mathcal{D}_k$ — data collected and segmented in the $k$-th deployment round
- $\ell_\theta$ — the policy's standard SFT loss. For an autoregressive head, $-\log\pi_\theta(a_t\mid o_t)$
- $\mathbf{1}[\cdot]$ — indicator function, $\tau$ — filter threshold
- $\beta$ — temperature. The smaller it is, the more it concentrates on the top samples

For a flow-matching action head, the standard flow-matching loss goes straight into the place of $\ell_\theta$.

$$\ell^{\text{FM}}_\theta(o,a)=\mathbb{E}_{s\sim U[0,1],\ \epsilon\sim\mathcal{N}(0,I)}\Big\lVert v_\theta(a^{s},o,s)-(a-\epsilon)\Big\rVert^2,\qquad a^{s}=s\,a+(1-s)\,\epsilon$$

- $s$ — flow time (0 is noise, 1 is the action)
- $\epsilon$ — Gaussian noise
- $a^{s}$ — the intermediate state interpolated between noise and action
- $v_\theta$ — the velocity-field network. The target velocity is $\frac{d a^{s}}{ds} = a-\epsilon$

The point is that **the weight simply multiplies the per-sample loss**. There is no need to backprop through a multi-step sampling chain, nor to compute $\log\pi$. So this operator plugs straight into any VLA's SFT pipeline. Which action head and which operator DYNA-1 actually uses, however, is not disclosed.

Laid out this way, its character becomes ambiguous. It improves on its own data according to a learned reward, so it seems like RL, but the update is a supervised loss. Is this RL, or SFT?

> ### 💡 RM-in-the-loop is the robot version of rejection-sampling fine-tuning
>
> In that its own policy produces the data and a learned reward does the judging, it is RL; in that the update uses a supervised loss, it is SFT. On the LLM side, exactly this combination already exists.
>
> | | LLM RFT / ReST | RM-in-the-loop (inferred) | Online RL (PPO, SAC) |
> |---|---|---|---|
> | Data | Own samples | Own deployment stream | Own rollouts |
> | Judge | RM / verifier | progress RM | Reward + critic |
> | Update | Filtered / weighted SFT | Filtered / weighted SFT | Policy gradient / Q-learning |
> | Unit of iteration | Round | Weekly round | Step |
>
> And "RL or SFT" is not really an either-or. AWR comes out of the closed-form solution of a KL-constrained policy improvement problem.
>
> $$\pi^*(a\mid o)\ \propto\ \pi_k(a\mid o)\,\exp\!\big(\hat A(o,a)/\beta\big)$$
>
> - $\pi_k$ — the current policy, $\pi^*$ — the policy improved under the KL constraint
>
> It is the same formula as the optimal policy of KL-regularized RLHF in LLM alignment, $\pi^*\propto\pi_{\text{ref}}\exp(r/\beta)$, the starting point of the DPO derivation. Weighted SFT is the process of projecting this $\pi^*$ into parameter space. The investor's description of DYNA-1 as a structure where deployment feeds a continual reinforcement-learning loop is not wrong in this sense. The blog itself does not name the algorithm.

### 3.6 Exploration and recovery — how it gets over Wall 1

The blog's three uses of the RM can all be explained on top of the operators of Section 3.5.

| The blog's use | Mechanism on top of Section 3.5 [Inference] |
|---|---|
| Autonomous exploration | Flow/diffusion policies are stochastic, so they act slightly differently even in the same situation. The RM picks the better variants and weighted SFT amplifies them. Iterating selection and amplification is itself policy improvement |
| Intentional error recovery | Actions in the falling segment of curve (a) are suppressed with $\hat A<0$, and actions that climb out of low $p$ are reinforced with $\hat A>0$ |
| Data curation | Filter by $\hat A$ and whether the task completed |

The key is recovery. In this loop, **recovery actions on off-nominal states the policy itself created** enter the training data. It is data that exactly fills the gap between $p_{\text{demo}}$ and $d^{\pi}$ seen in Section 2.1, and the mechanism that moves failures from $q_u$ to $q_r$. In this process, people do not need to choose which states to demonstrate. The states the policy falls into by itself become the practice problems.

Where this effect shows up in the actual improvement trajectory is checked in Section 5.2.

### 3.7 Throughput — half of Wall 3

Back to property ⓶ previewed in Section 2.2. With $\gamma<1$ the discounted cumulative reward is $\gamma^{T}p_\phi(o_T)-p_\phi(o_0)$, so for the same completion, a trajectory that finishes sooner has a larger return. Long-stalled spans like (b) in the segmentation curve get low $\hat A$ and their weight shrinks. **Speed comes under selection pressure inside the progress reward, without a separate objective.**

The blog also stresses that every second spent on an edge case eats into throughput, so the policy must find the fastest solution. Recovery itself has to be fast too.

But the other half of Wall 3, **quality**, is a different problem. Can a progress RM distinguish a 1/3-inch difference in the first fold? This question comes back with the numbers of Section 5.1 and in Section 7.

---

## 4. Why it works

**[Blog]** The blog's own explanation is **behavioral generalization**. Simple skills like pick-and-place transfer with data diversified over environments and objects, but long-horizon dexterous tasks need behavioral generalization, since perceptual generalization alone is not enough. DYNA-1, guided by the RM, went through a very wide variety of behaviors and their outcomes during deployment, and this is what made transfer to untrained customer environments possible.

| | Perceptual generalization | Behavioral generalization |
|---|---|---|
| What changes | Background, lighting, object appearance | The resulting states behaviors produce (creases, overlaps) |
| How it is obtained | Data diversified over environments and objects | Experience of diverse behaviors and their outcomes |
| Tasks it suffices for | pick-and-place | Long-horizon dexterous |

**[Assessment]** This explanation can be unpacked on two levels.

**⓵ State coverage** — the training data comes from $d^{\pi_k}$, not $p_{\text{demo}}$. The training distribution is aligned with the deployment distribution.

**⓶ Signed labels** — human demonstrations carry only "do it like this." Data weighted by the RM also carries, in the form of weights, the information "in this state, this action cuts into progress."

A new restaurant may have different tables and lighting, but napkins crumple and overlap the same way. So the explanation that coverage on the behavior side is the main driver of environment transfer is persuasive.

---

## 5. Experiments

The blog provides no paper-style tables. All the numbers below come from the blog's body (the Overview, Continual Improvement, Quality, Generalization and Additional Skills sections), and the converted values were calculated by the author.

### 5.1 24-hour continuous operation

**[Blog]** Over 24 hours it folded more than 850 napkins without intervention, recording a 99.4% success rate while holding about 60% of human speed.

| Metric | Value | Converted |
|---|---|---|
| Throughput | 850+ / 24h | $\approx$ 35 napkins/hour (about 1.7 minutes per napkin) |
| Speed relative to humans | ~60% | Implied human speed $\approx$ 59 napkins/hour |
| Success rate | 99.4% | Failures $\approx$ 5 napkins |
| Interventions | 0 | The roughly 5 failures were all on the $q_r$ side **[Inference]** |

Through the frame of Section 2.1, this result is consistent. There were failures, but the run did not end. It means failures moved from $q_u$ to $q_r$.

The quality standard is separate. **[Blog]** On a 5-point scale, scores of 4–5 count as commercial quality. 98% score 3 or higher, but the share above the commercial bar is 75%, and what separates a 5 from a 3 is precision of under 1/3 inch in the first fold.

But put the success rate and the quality distribution side by side and something odd shows. The success rate (99.4%) is higher than the share scoring 3 or above (98%). How can a fold that does not even reach a 3 be counted as a success?

> ### ⚠️ Fact check — the "success" in 99.4% is not a quality standard
>
> Since the success rate is higher than the share scoring 3 or above, "success" includes folds below 3. So success should be read as **completion**, independent of quality. The blog gives no definition of success. The metric card shows commercial quality next to the speed figure, but by the body's standard the rate of reaching commercial quality is 75%.
>
> In short, 99.4% is a metric of **uptime (Wall 1)**, and the precise figure for **commercial viability (Wall 3)** is 75%. The quality problem left open in Section 3.7 is exactly this 25%.

### 5.2 The six-week improvement trajectory

**[Blog]** Scaling RM-in-the-loop training, it improved as follows within a few weeks.

| Week | Intervention-free duration | Throughput | Notes |
|---|---|---|---|
| 1 | About 5 minutes | 1 success | Fell apart afterward |
| 2 | 1 hour | — | Compounding errors, unrecoverable |
| 3 | 8 hours | 6–7 napkins/hour | About 10 minutes per napkin |
| 4 | 24 hours | ~200 napkins ($\approx$ 8/hour) | Low quality and speed |
| 5 | 24 hours+ | ~350 napkins ($\approx$ 15/hour) | Decent commercial quality |
| 6 | 24 hours+ | ~800 napkins ($\approx$ 33/hour) | High commercial quality |

Drawing throughput alone as bars:

```
  napkins per hour   (1 '#' = 2 napkins)
  --------+----------------------+------
  Week 3  | ###                  |  6.5
  Week 4  | ####                 |  8.3
  Week 5  | #######              | 14.6
  Week 6  | #################    | 33.3
  24h run | ##################   | 35.4
```

The order is interesting.

- **Weeks 1–4** — intervention-free duration grows from 5 minutes to 24 hours. Wall 1 falls first
- **Weeks 4–6** — with duration fixed at 24 hours, throughput jumps 4×. The speed side of Wall 3 follows

It is an order of lowering $q_u$ first and optimizing speed later. In particular, the step from Week 2's "compounding errors, unrecoverable" to Week 3's 8-hour run reads as where the effect of the recovery data described in Section 3.6 first shows. The later rise in throughput is consistent with the speed selection pressure of Section 3.7 **[Inference]**.

Then which row of this table is the headline's 850 napkins? Week 6 is about 800.

> ### ⚠️ Fact check — the same 24-hour run has different numbers depending on the source
>
> The blog headline says 850+, Week 6 in the same post says about 800, and the press release says 800+. Which run the 99.4% was measured on is not specified. A later investor post mentions 900+ at 99%, but adds that throughput and robustness improved in the meantime, so it should be taken as a separate, later run. The throughput graph is labeled towels rather than napkins and its values are not in the body, so it cannot be verified. The date on the blog page (June 2025) also differs from the actual announcement date (2025-04-29).

### 5.3 Environment generalization

**[Blog]** DYNA-1 folded napkins right away even in real customer environments it had not trained on. It admits, however, that zero-shot quality and throughput often dropped, and that it became proficient quickly after a small amount of extra training on site. The evidence is video only; there are no numbers.

Read together with the behavioral-generalization explanation of Section 4, the accurate summary is "the job gets done, but an extra loop tuned to the site's distribution is needed."

### 5.4 Additional skills

**[Blog]** It also reports positive transfer from napkin folding to other commercial tasks. It folds shirts of different sizes and materials in succession, and also performs cup filling, which began at a customer's request. Cup filling chains delicate grasping, precise placement, handover and tool use, and is a no-reset task where a single mistake ends the run. It is not yet perfect, but while the internal baseline could not reliably get past even the first stage, DYNA-1 passed every stage, and going from task discovery to validation took under two weeks with just 0.7% of the total training data.

The reference for all these comparisons is the "internal baseline." Then how were that baseline and the RM's own performance measured?

> ### ⚠️ Fact check — the comparison baselines are not disclosed
>
> The internal VLA baseline's architecture is not disclosed, and the "80% plateau" of Section 1.1 has no source. The claim that the RM far outperforms existing approaches has no quantitative comparison. This contrasts with GVL, which was evaluated with metrics like VOC (rank correlation between predicted value and actual temporal order). The 0.7% is also a relative value, so the absolute data volume is unknown.

---

## 6. Positioning

The blog has no related-work section. So the coordinates are set by the blog's own claims and the lineage of the underlying research.

### 6.1 Lineage — from progress/value learning to a deployment loop

The DYNA RM appears to sit on Jason Ma's research line, which runs through the VIP and LIV value-representation work to GVL **[Inference]**. The most interesting thing is how the application lists correspond. GVL proposed dataset filtering, success detection and advantage-weighted regression as its applications.

| Applications GVL proposed | DYNA-1's RM uses |
|---|---|
| dataset filtering | Dataset generation and curation |
| success detection | Error recognition and recovery, stream segmentation |
| advantage-weighted regression | Policy improvement through autonomous exploration |

**[Assessment]** DYNA-1 reads as though it **moved the list of applications GVL showed on offline video into a real-deployment continuous loop**. What is newly added is segmentation that handles streams with no episode boundaries (Section 3.4).

### 6.2 Comparison with self-improving VLAs that came after

After DYNA-1, work appeared that tackled the same goal — "a robot policy improves itself from its own experience" — with different designs. Where did they get their judging signal for what counts as good behavior?

> ### 🔗 Comparison with PLD and π*₀.₆ / RECAP — where the judging signal comes from
>
> | | **DYNA-1** | **[PLD](/notes/pld-self-improving-vla-en/)** | **π*₀.₆ / RECAP** |
> |---|---|---|---|
> | Judging signal | Dense progress RM | Sparse binary success reward | Learned V (distributional MC) |
> | Data structure | Segmented reset-free continuous stream | Episodic | Episodic |
> | Improvement operator | Undisclosed (inferred as filtered/weighted SFT) | Residual off-policy RL, then SFT distillation | advantage conditioning |
> | Human intervention | None (claimed) | None | Includes teleoperated corrections |
> | Disclosure level | Product blog (2025-04) | Paper (arXiv, 2025-10) | Technical report (2025-11) |
>
> The three approaches form a spectrum in the density of the judging signal. [PLD](/notes/pld-self-improving-vla-en/) keeps the sparse reward as is and makes learning possible with the base policy's successful trajectories and symmetric replay. RECAP estimates advantage with a value function and uses it as a conditioning input. DYNA-1 builds a dense progress signal from the start, and uses that signal even for stream segmentation. The denser the judging signal, the closer one can get to reset-free continuous deployment, but the more the whole loop hangs on the RM's accuracy (Section 7).

---

## 7. Limitations

**What the blog acknowledges**

- **Quality drops in zero-shot environment transfer** — on-site extra training is needed
- **Cup filling is still imperfect**
- **75% rate of reaching commercial quality** — the precision between a 5 and a 3 remains an open task

**Further points to note**

**⓵ Method undisclosed** — the policy and RM architectures, the update operator and the data scale are all undisclosed, and there is no ablation against a loop without the RM. The core claims cannot be verified from outside. That is why most of Section 3 has to be inference.

**⓶ Risk of RM exploitation (Goodhart)** — train on data selected by RM score, and the policy moves toward states the RM overrates. The question left open in Section 3.7 closes here. The 1/3-inch difference separating a 5 from a 3 is a region a vision-based progress model is easily insensitive to, and looks like a candidate cause for stalling at 75% **[Inference]**. Who recalibrates, and how, the self-reinforcing loop in which the RM's misjudgments accumulate in the data through filtering is also unclear.

**⓷ A per-task recipe** — the press release also says the robot masters one task at a time. Each task needs a reliable RM and weeks of the loop. It is closer to **a per-task deployment-learning recipe** than the name foundation model suggests **[Assessment]**.

**⓸ Evaluation design** — a single 24-hour run. No repeated experiments, no confidence intervals, no definition of success.

**⓹ What level "production-ready" means** — throughput is about 60% of human speed (35 napkins per hour), and the rate of reaching commercial quality is 75% (Section 5.1). "Production-ready" as of 2025 is most accurately read as the level of a customer-site pilot **[Assessment]**.

---

## 8. Closing — what this post suggests

DYNA-1's contribution is not any particular model architecture. It is the problem setting itself: **switching the unit of evaluation from "episode success rate" to "continuous operation," and solving the judge that setting requires with a single reward model**. As the calculation in Section 2.1 shows, in continuous operation the unrecoverable failure rate becomes the metric; recovery behavior can only be learned from states the policy visits itself; and using that data needs a judge that scores and segments at the same time. This chain of logic holds even with the method undisclosed.

And this post reads unusually familiarly to someone with an LLM and agentic AI background.

- **The bottleneck moves from the policy to the judge** — the same picture as in LLM post-training, where the quality of the verifier or RM sets the ceiling on RFT and RL performance. In robotics, checking against a ground truth is not free, so the RM itself becomes the core asset.
- **The LLM alignment toolbox transplants as is** — weighted SFT is a projection of $\pi_k\exp(\hat A/\beta)$, the same formula as RLHF/DPO. Flow/diffusion VLAs need only multiply the per-sample loss by a weight, so the cost of transplanting is low.
- **Long-running operation with no episodes given** — where to cut a long-running agent's trace and how to score it is the same problem as trajectory evaluation in agentic AI. The idea of segmenting by a progress curve is worth trying directly on agent logs too.

Finally, this post also lays out what to look at when reading claims about deployed robots.

| Metric | What it tells you | DYNA-1's value |
|---|---|---|
| Intervention-free operating time ($q_u$) | Uptime | 24 hours |
| Throughput relative to humans | Economics | ~60% (35 napkins/hour) |
| Quality-grade distribution | Commercial viability | 75% at the commercial bar |
| On-site adaptation cost | Scalability | A small amount of on-site training (no numbers) |

In that it disclosed all four of these (even if only qualitatively) rather than a single demo video, DYNA-1 is an announcement that at least knows exactly what it has to prove.

---

## Appendix — Glossary

| Term | Definition |
|---|---|
| **Unrecoverable failure rate ($q_u$)** | The per-episode probability of a failure that ends the run without human intervention. The expected continuous run length is $1/q_u$ |
| **task progress** | A value from 0 to 1 expressing how far an observation has come toward task completion. Equivalent to value under a particular choice of reward |
| **potential-based shaping** | A dense reward of the form $r_t=\gamma\Phi(s_{t+1})-\Phi(s_t)$. Does not change the optimal policy |
| **GVL** | An in-context value estimator that estimates progress by having a VLM recover the order of shuffled frames (ICLR 2025) |
| **Stream segmentation** | Splitting boundary-less continuous deployment data into episode and subtask units using the progress curve |
| **Filtered BC / AWR** | Policy improvement operators that SFT by selecting samples by advantage (filter) or weighting them exponentially |
| **RM-in-the-loop** | The iterative loop of deploy → RM scoring and segmentation → filtered/weighted training → redeploy |

**Original** — [DYNA-1 technical blog](https://www.dyna.co/research/dyna-1) · **Press release** — [PR Newswire (2025-04-29)](https://www.prnewswire.com/news-releases/dyna-robotics-unveils-dyna-1-the-first-commercial-ready-robot-foundation-model-offering-fully-autonomous-round-the-clock-dexterity-302441437.html) · **Underlying research** — [GVL, arXiv:2411.04549](https://arxiv.org/abs/2411.04549) · **Later coverage** — [Salesforce Ventures](https://salesforceventures.com/perspectives/welcome-dyna-robotics/)
