---
layout: paper
lang: en
ref: pi05-open-world-generalization
kind: paper-review
title: "π₀.₅: a Vision-Language-Action Model with Open-World Generalization"
date: 2026-01-09 12:00:00 -0800
paper_date: 2025-04-22
venue: "CoRL 2025 · arXiv:2504.16054"
tags: [VLA, Robot-Foundation-Model, Co-Training, Hierarchical-Policy, Flow-Matching, Paper-Review]
authors: "Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, Ury Zhilinsky"
affiliations: "Physical Intelligence"
summary: "Target robot data is only 2.4% of the pretraining examples — a training recipe that co-trains other robots, the web, subtask labels and verbal instructions at the level each fits, and has the model 'say the subtask first, then move', to tidy homes it has never seen."
paper_url: "https://arxiv.org/abs/2504.16054"
---

> **Core claim** — Building a robot that works in a home it has never seen does not require endlessly growing target robot data. What it needs is **a training recipe that feeds knowledge sources of different kinds to the level each one fits**. π₀.₅ co-trains other robots, the web, subtask labels and human verbal instructions into one model. And it makes that model "first say the subtask to do in text, then move according to what it said," so that it tidies kitchens and bedrooms in homes that were not in its training data.

---

## Introduction

The standard recipe for VLA (Vision-Language-Action) models is to take a VLM pretrained on the web and train it by imitation learning on robot demonstrations. RT-2, OpenVLA and π₀ all sit inside this frame, and have shown the ability to follow language instructions and to perform complex manipulation. But these models are mostly **evaluated in environments similar to their training data**. What happens when a robot leaves the lab and walks into a home it has never seen?

π₀.₅ is a paper that tries to answer this question head-on. To say it up front, the paper's contribution is not a new network module but **a training recipe**. The architecture is almost the same as π₀. What changed are three things: what the model is made to output, what it is trained on, and how actions are represented at each training stage.

This piece follows the paper's argument and builds up, with the concepts needed, **why** each design decision takes the shape it does.

---

## 1. The problem — why is an unseen home hard

### 1.1 Generalization has levels

Say a robot is told to tidy a kitchen it has never seen. The paper splits the generalization this requires into three levels.

| Level | Example | What it needs |
|---|---|---|
| **ⓐ Skill** | Picking up a knife or a plate | Data covering a wide enough range of scenes and objects |
| **ⓑ Composition** | Using learned skills in a new order or a new way | Understanding of task structure, recombination of skills |
| **ⓒ Semantics** | Judging which drawer to open, which thing on the counter is a drying rack | Scene understanding grounded in prior knowledge |

This split matters because the **source** of the knowledge needed differs by level. ⓐ can be obtained by collecting robot data in more diverse environments. But ⓑ demands task structure and ⓒ demands common sense about the world, and filling these two from robot demonstrations alone is hard.

### 1.2 Where existing approaches get stuck

The achievements and limits of prior work, as the paper's Related Work lays them out:

| Approach | Achievement | Limit |
|---|---|---|
| Wider environment diversity in robot data | Simple skills like grasping and opening drawers generalize to new environments | Long-horizon tasks like tidying a kitchen cannot have their possible scenarios covered by brute force |
| Task-specific assumptions (grasp prediction, model-based planning and control) | Generalize broadly, even to new homes | Confined to narrow primitives like picking up objects |
| Large multi-domain data + end-to-end learning | Simple tasks generalize to new environments | Usually simple tasks under a minute, with low success rates |
| VLA (built on web-pretrained VLMs) | Language following, complex manipulation | Evaluated only in environments close to the training distribution |

In short, simple skills generalize through data diversity alone, but there had been no case of accomplishing **long-horizon, multi-stage tasks in new environments**. Tasks like tidying a kitchen, where combinations of situations explode, cannot be covered by blindly growing robot data.

### 1.3 So the question the paper asks

> To obtain generalization at all three levels ⓐ, ⓑ and ⓒ at once with a single training recipe, what must be trained, and how?

The paper's hypothesis starts from human learning. People, too, do not learn everything by practicing it directly. They mix facts heard from others, knowledge read in books and experience from other contexts with direct experience in the target domain. Likewise, a robot **must transfer knowledge from multiple sources**.

Putting this hypothesis into practice means getting over three walls.

**⛔ Wall 1 — Coverage of target data.** Data collected directly on the target platform, a mobile manipulator, is about 400 hours across about 100 homes. Medium-sized, perhaps, but far too small to cover what can happen in a new home. And as Section 1.2 showed, blindly growing it is not the answer.

**⛔ Wall 2 — Heterogeneity of knowledge sources.** The knowledge sources that would help each come in their own format. There are actions from other embodiments, captions, VQA and bboxes from the web, subtask labels attached by people, and human verbal instructions. Their output spaces differ, the robots' degrees of freedom differ, and the levels of abstraction they deal with differ. The paper itself names this heterogeneity as the biggest obstacle.

**⛔ Wall 3 — The dilemma of action representation.** Write actions as discrete tokens and training is fast but inference is slow. Write them as continuous values and they suit real-time inference but training is slow. For a robot that must move at 50Hz, both are half an answer.

Walls 1 and 2 are intuitive. What Wall 3 really is only becomes visible once you know how a VLA handles actions, so Section 2 builds that background.

---

## 2. Background — how does a VLA "say" an action

### 2.1 VLA = maximizing the likelihood of action chunks conditioned on observation and instruction

VLAs are trained by imitation learning on diverse robot demonstration data. The objective is to maximize the log-likelihood of actions given observations and instructions.

$$\max_\theta\ \mathbb{E}_{(\mathbf a_{t:t+H},\ \mathbf o_t,\ \ell)\sim\mathcal D}\Big[\log\pi_\theta\big(\mathbf a_{t:t+H}\mid\mathbf o_t,\ \ell\big)\Big]$$

- $\mathcal D$ — the robot demonstration dataset
- $\mathbf a_{t:t+H}$ — an **action chunk**. Rather than one action at one time step, a bundle of actions over several time steps is predicted at once. π₀.₅'s chunks are 50 steps and control runs at 50Hz, so about 1 second's worth
- $\mathbf o_t=[\mathbf I^1_t,\dots,\mathbf I^n_t,\ \mathbf q_t]$ — $n$ camera images and the proprioceptive state $\mathbf q_t$ (joint angles, gripper pose, torso lift, base velocity)
- $\ell$ — the natural-language task instruction
- $\pi_\theta$ — the policy with parameters $\theta$

From an LLM perspective this has the same form as SFT. The prompt is "observation + instruction" and the response is "action." A VLA turns observations, instructions and actions all into tokens, recasts the problem as next-token prediction, and brings over modern LLM training tools as they are.

Images and text can simply use the VLM's tokenizer. The remaining question is **how to turn continuous-valued actions into tokens**, and the answer splits broadly into two branches.

### 2.2 Discrete representation — FAST

FAST compresses an action chunk in the frequency domain, like JPEG, and then tokenizes it with BPE.

```
 action chunk (50 x d)  ->  DCT per dimension  ->  quantize  ->  BPE  ->  discrete tokens
```

- Robot trajectories are smooth, so information concentrates in the low-frequency DCT coefficients. The result is a much shorter, less redundant token sequence than naive binning with "one token per step × dimension" (e.g., 50×18 = 900).
- Action tokens made this way are trained with cross-entropy exactly like text tokens.
- The detailed design is from the FAST paper (Pertsch et al., 2025); the π₀.₅ paper cites and uses it.

### 2.3 Continuous representation — flow matching and the action expert

π₀ generates actions with flow matching. It linearly interpolates between the ground-truth action chunk and Gaussian noise, and regresses the direction of that line (the velocity field).

$$\mathbf a^{\tau,\omega}_{t:t+H}=\tau\,\mathbf a_{t:t+H}+(1-\tau)\,\omega,\qquad\omega\sim\mathcal N(0,\mathbf I),\quad\tau\in[0,1]$$

$$\mathcal L_{\text{flow}}(\theta)=\mathbb E_{\tau,\omega}\Big[\big\|\,(\omega-\mathbf a_{t:t+H})-f^a_\theta(\mathbf a^{\tau,\omega}_{t:t+H},\ \mathbf o_t,\ \ell)\,\big\|^2\Big]$$

- $\tau$ — the flow time index. The interpolation ratio between noise and ground truth
- $\omega$ — Gaussian noise
- $\mathbf a^{\tau,\omega}_{t:t+H}$ — the intermediate, noise-mixed action chunk
- $f^a_\theta$ — the velocity field (vector field) the network predicts
- $\omega-\mathbf a_{t:t+H}$ — the regression target. The direction of the line joining noise and ground truth

At inference, starting from noise and integrating the learned velocity field 10 times yields an action chunk. From a diffusion background, it is the same **straight interpolation path + velocity regression** as Rectified Flow and SD3.

π₀ computes this velocity field with separate weights called the **action expert**. Action tokens are processed inside the same transformer as the VLM backbone but use a different set of weights, which the paper likens to a mixture-of-experts. The action expert specializes in flow-based action generation and can be made much smaller than the backbone.

Then why not just pick one of the two? In fact π₀ picked flow and π₀-FAST picked discrete tokens. The problem is that the strengths and weaknesses of the two representations **cross over exactly**.

> ### 💡 Discrete representations are good for learning; continuous ones are good for moving
>
> | | FAST (discrete) | Flow (continuous) |
> |---|---|---|
> | Training objective | cross-entropy — same form as VLM pretraining | velocity-field MSE — a new objective |
> | Training efficiency | Fast | Slow |
> | Inference | Autoregressive decoding — as many sequential forward passes as tokens | 10-step integration — the whole chunk generated in parallel, only the small expert repeated |
> | Output precision | Has quantization error | Continuous values |
> | Real-time control | Unsuitable | Suitable |
>
> The paper, citing the FAST paper, states two facts. Representing actions as discrete tokens makes VLA training much faster. But the cost of autoregressive decoding makes it unsuitable for real-time inference.
>
> The paper does not directly analyze why training is faster, but it can be read as follows. With discrete tokens, actions become "just another language" to the VLM, so the pretrained output layer and representations are reused as they are. With flow, by contrast, a randomly initialized module has to learn a regression target in a new space from scratch.
>
> On a mobile manipulator that must control 18–19 dimensions at 50Hz, this crossover becomes the unavoidable **Wall 3**. How π₀.₅ solves it is seen in Section 3.3.

---

## 3. Method

π₀.₅'s architecture is almost the same as π₀'s. What changed is **what it outputs** (subtask text), **what it trains on** (the data mixture), and **which action representation it uses at each training stage** (switching from discrete to continuous). The overall flow is as follows.

```
 init        : PaliGemma VLM (web-pretrained)
                 |
                 v
 PRE-TRAIN   : 280k steps, alpha = 0
               data = MM + ME + CE + HL + WD
               out  = text / bbox / subtask / FAST action tokens
                 |
                 v
 POST-TRAIN  : 80k steps, alpha = 10, + action expert (random init)
               data = MM + ME (filtered) + HL + WD + VI
               out  = text tokens + flow-matching action chunk
                 |
                 v
 INFERENCE   : subtask text (AR decode) -> 10 flow steps -> action chunk
```

- **pre-train** — every output is trained as discrete tokens. Actions are FAST tokens too.
- **post-train** — attaches the action expert and specializes in mobile manipulation.
- **inference** — first emits the subtask as text, then emits continuous actions conditioned on it.

The data abbreviations (MM, ME, CE, HL, WD, VI) and the meaning of $\alpha$ are defined in turn in Sections 3.3–3.5.

### 3.1 One model, two levels — hierarchical factorization

What π₀.₅ represents is the joint distribution of action chunks and text output. The paper factors it into a product of two conditional distributions.

$$\pi_\theta\big(\mathbf a_{t:t+H},\ \hat\ell\mid\mathbf o_t,\ \ell\big)=\underbrace{\pi_\theta\big(\mathbf a_{t:t+H}\mid\mathbf o_t,\ \hat\ell\big)}_{\text{low-level}}\ \underbrace{\pi_\theta\big(\hat\ell\mid\mathbf o_t,\ \ell\big)}_{\text{high-level}}$$

- $\ell$ — the overall task prompt (e.g., "put away the dishes")
- $\hat\ell$ — the text the model emits. It may be a subtask (e.g., "pick up the plate") or the answer to a web-data VQA question
- **High-level inference** $\pi_\theta(\hat\ell\mid\mathbf o_t,\ell)$ — generates the next subtask as text from the observation and the overall command
- **Low-level inference** $\pi_\theta(\mathbf a\mid\mathbf o_t,\hat\ell)$ — generates the action chunk conditioned on the subtask
- Both distributions are represented by **the same weights $\theta$**

It is the same structure as chain-of-thought in LLMs. First the "thought" (subtask) is emitted as text, then the "answer" (action) is emitted conditioned on that thought. But unlike the embodied-CoT line that generates a reasoning chain at every step, π₀.₅'s high-level inference runs at a lower frequency than low-level inference. Exactly how often it re-infers the subtask is not stated in the paper.

The high-level data is labeled not only with subtasks but also with **bboxes** of the relevant objects, and the model is trained to predict the bboxes **before** emitting the subtask. It is a short grounding step: first pin down what is where, then decide the next action.

There is one design worth noticing here. In the factorization, the low-level distribution $\pi_\theta(\mathbf a\mid\mathbf o_t,\hat\ell)$ **does not receive the original command $\ell$.** That means it moves seeing only "pick up the plate," without knowing the big picture "clean the kitchen." Why does this design, which deliberately discards information, pay off?

> ### 💡 The subtask string is the interface between the two levels
>
> Cut off $\ell$, and the two levels are connected only by **a single short natural-language command**. This narrow interface makes three things possible.
>
> **⓵ Low level — transfer at the unit of skills.** All the low level learns is a "short, concrete command → motion" mapping. The paper attached subtask labels to all of the robot data made up of multiple subtasks (MM, ME, CE). So dish-moving data collected in a lab and "pick up the plate" in a home kitchen meet within the same command vocabulary. Even when the overall tasks differ, the skills transfer.
>
> **⓶ High level — pure text generation.** High-level inference is a problem of producing text from images, so the web's VQA, caption and bbox knowledge is used as is.
>
> **⓷ A place for a person to step in.** A person can put commands into the same interface. The verbal-instruction data collection of Section 3.5 and the human-oracle comparison of Section 5.5 both use this slot.
>
> That is, the design that cuts off $\ell$ is **the structural solution to Wall 2 (heterogeneity)**. Every knowledge source need not teach everything; each only has to contribute at the level it fits. Seen from agentic AI, it is a structure that puts the planner and executor in one model and uses the subtask string like a tool call.

### 3.2 Architecture — per-token-type weights and the attention mask

The model is a transformer that takes $N$ multimodal tokens and emits $N$ multimodal outputs.

$$y_{1:N}=f\big(x_{1:N},\ A(x_{1:N}),\ \rho(x_{1:N})\big)$$

| Symbol | Meaning |
|---|---|
| $x_i$ | An input token. One of text $x^w_i\in\mathbb N$, image patch $x^I_i\in\mathbb R^{p\times p\times 3}$, or noisy action $x^a_i\in\mathbb R^d$ |
| $A(x_{1:N})\in[0,1]^{N\times N}$ | The attention mask. Decides which tokens can see which tokens |
| $\rho(x_i)$ | The token type. Decides which encoder and **which expert weights** process it |
| $y^\ell_{1:M}$ | $M$ text logits. Subtasks, VQA answers and FAST action tokens are sampled from here |
| $y^a_{1:H}$ | $H$ action-expert outputs. They become the velocity field after a linear projection |

$M+H\le N$. Some outputs carry no loss, such as the outputs at image and prompt positions. The components are as follows (Appendix E).

| Component | Spec | Notes |
|---|---|---|
| Vision encoder | SigLIP, 400M | Initialized from PaliGemma |
| VLM backbone | Gemma 2B — width 2048, depth 18, MLP 16,384 | Initialized from PaliGemma |
| Action expert | 300M — width 1024, MLP 4096, otherwise the same as the backbone | Randomly initialized at the start of post-training |
| Timestep injection | Sinusoidal encoding → swish MLP → injected into each layer via adaptive RMSNorm | π₀ combined $\tau$ with the noisy action as input |
| Proprioception | Discretized and put into the prefix as text tokens | |

From a diffusion background it is a familiar combination.

- **A structure resembling MM-DiT** — separate weights per token type, with attention alone computed jointly over one sequence. The same idea as SD3's MM-DiT (per-modality weights + joint attention). The difference is in the mask. MM-DiT's joint attention is fully bidirectional, but π₀.₅ applies a block mask like the one below.
- **Timestep conditioning like DiT** — adaptive RMSNorm injects the timestep into each layer the same way as DiT's adaLN.

| query ↓ · key → | prefix (images · prompt · state) | FAST tokens | expert tokens |
|---|---|---|---|
| **prefix** | Bidirectional | ✗ | ✗ |
| **FAST tokens** | ✓ | causal | ✗ |
| **expert tokens** | ✓ | ✗ | Bidirectional |

- The prefix (images, prompt, state) sees itself bidirectionally.
- FAST tokens see only the prefix and the FAST tokens before them. That is, autoregressive.
- Expert tokens see only the prefix and other expert tokens, not the FAST tokens.
- The prefix never sees the expert. Information flows **one way only, VLM → expert**.

The reason FAST and expert are blocked from each other is information leakage. During training, the FAST tokens are the **ground-truth actions** fed in by teacher forcing. If the expert could see them, it would learn to copy the answer instead of learning to generate actions from observations. Besides, FAST tokens are not even generated at inference.

Looking at this mask, it is easy to feel reassured. VLM-side tokens never see the action expert, so even when a randomly initialized expert is attached in post-training, isn't the VLM's pretrained knowledge safe?

> ### 💡 The forward pass is one-way, but the backward pass is not
>
> Writing out the attention of the expert tokens exposes the paths the gradient flows along.
>
> $$\mathrm{attn}_a=\operatorname{softmax}\!\Big(\tfrac{1}{\sqrt{d_h}}\,Q_a(X_a)\,\big[K_b(X_b);\ K_a(X_a)\big]^{\top}\Big)\,\big[V_b(X_b);\ V_a(X_a)\big]$$
>
> - $X_b$, $X_a$ — tokens processed by the backbone weights, tokens processed by the expert weights
> - $Q, K, V$ — the query, key and value projections of each weight set
> - $[\,\cdot\,;\,\cdot\,]$ — concatenation along the token axis
> - $d_h$ — the attention head dimension
>
> **Forward pass** — $K_a, V_a$ do not enter the attention of backbone tokens. So VLM-side activations are identical whether or not the expert exists.
>
> **Backward pass** — the expert's output depends on $K_b(X_b)$ and $V_b(X_b)$. So the gradient of the flow loss flows through these terms all the way to the backbone weights.
>
> $$\frac{\partial\mathcal L_{\text{flow}}}{\partial\theta_{\text{VLM}}}\ \neq\ 0$$
>
> The π₀.₅ paper does not mention stop-gradient. Instead it says only that it **preserves** the backbone's text abilities by co-training next-token prediction and web data in post-training. How much the random expert's gradient actually harms pretrained knowledge is tackled head-on by follow-up work, which we see again in Section 6.

### 3.3 Discrete and continuous in one loss — the device that breaks Wall 3

π₀.₅ puts the two action representations together into one loss.

$$\mathcal L(\theta)=\mathbb E_{\mathcal D,\tau,\omega}\Big[\ \mathrm{CE}\big(x_{1:M},\ f^\ell_\theta(\mathbf o_t,\ell)\big)\ +\ \alpha\,\big\|\,\omega-\mathbf a_{t:t+H}-f^a_\theta(\mathbf a^{\tau,\omega}_{t:t+H},\ \mathbf o_t,\ \ell)\big\|^2\ \Big]\qquad(1)$$

| Symbol | Meaning |
|---|---|
| $\mathrm{CE}(\cdot)$ | Cross-entropy between text tokens and predicted logits. **FAST action tokens are included here too** (the paper writes $H$) |
| $x_{1:M}$ | The ground-truth text token sequence |
| $f^\ell_\theta$ | The VLM's text logit output |
| $f^a_\theta$ | The action expert's velocity-field output |
| $\alpha$ | The trade-off weight between the two terms |

As $\alpha$ changes by training stage, the model's character changes.

| Stage | $\alpha$ | What is trained | Meaning |
|---|---|---|---|
| pre-train (280k steps) | 0 | CE only — text, bbox, subtasks, FAST actions | Ordinary VLM training that treats actions as text tokens |
| post-train (80k steps) | 10 | CE (including FAST) + flow | Attach the expert and train both action representations at once, blocked from each other by the mask |
| inference | — | — | Only the subtask is decoded autoregressively; actions come from the expert's 10-step integration. FAST tokens are not used |

This is the answer to Wall 3 previewed in Section 2. π₀.₅ does not pick one of the two. **Discrete when learning, continuous when moving.** The paper reports that this procedure leads to stable pretraining and excellent language following.

Following π₀, the flow timestep is sampled from a non-uniform distribution. Specifically, $p(\tau)=\mathrm{Beta}\big(\tfrac{s-\tau}{s};\,1.5,\,1\big)$, where $s=0.999$ is the upper bound on sampled $\tau$. This distribution emphasizes low $\tau$.

But if you try to implement Eq. (1) and the appendix's flow notation as written, there are a few places you will get caught.

> ### ⚠️ Fact check — notation inconsistencies you may hit when implementing
>
> | Where | Detail |
> |---|---|
> | Sign of the flow target | Differentiating the interpolation $\tau\mathbf a+(1-\tau)\omega$ with respect to $\tau$ gives $\mathbf a-\omega$, but the regression target is written $\omega-\mathbf a$. This sign is natural under a convention that puts $\tau=1$ on the noise side. The direction of integration is not in the paper, and whether "emphasizing low $\tau$" emphasizes the noise side or the data side also flips with this convention |
> | Overloaded symbols | $H$ is used both for cross-entropy and the horizon, and $\alpha$ both for the loss weight and a Beta parameter. The appendix writes the horizon of 50 as "$H=49$", because $t{:}t{+}H$ is an index that includes both ends |
> | Conditioning in Eq. (1) | In the factorization the low level is conditioned on $\hat\ell$, but the flow term of Eq. (1) is written conditioned on $\ell$. It should be read loosely as "the prompt contained in the prefix" |
> | Number of heads | The appendix's backbone setting num_heads=18 gives 18×256=4608, which does not match width 2048, and also differs from the original Gemma 2B setting (8 heads). It may be a typo |

### 3.4 Pretraining data — the material for getting over Walls 1 and 2

Pretraining runs for 280k steps on a mixture of the following data. The post-train column anticipates Section 3.5.

| Abbrev. | Content | What it supplies | post-train |
|---|---|---|---|
| **MM** (Mobile Manipulator) | Mobile manipulators, about 400 hours, about 100 homes | The target embodiment and target tasks | ✅ (success and length filter) |
| **ME** (Multi-Environment) | Static one- or two-arm robots, diverse homes | Environment diversity. Light and easy to move, so it can be collected in more homes | ✅ (filtered) |
| **CE** (Cross-Embodiment) | Lab tabletop settings, many robot types, including OXE | Task diversity. A mix of tasks related and unrelated to the evaluation | ❌ excluded |
| **HL** (High-Level) | Manual subtask labels on multi-stage robot data + bboxes of relevant objects | Subtask inference, grounding | ✅ (multi-environment data slice) |
| **WD** (Web Data) | Captions (CapsFusion, COCO), VQA (Cambrian-7M, PixMo, VQAv2), object localization (+ added indoor and household-object bboxes) | Semantic knowledge, object concepts | ✅ (to preserve abilities) |
| **VI** (Verbal Instruction) | A person directs the trained low-level policy step by step in language | Demonstrations of good high-level commands | New in post-train |

**97.6%** of the training examples used in the first training stage come not from mobile manipulators working in homes but from other sources such as other robots and the web (Introduction). It is a mixture in which target data is only 2.4%.

The devices for putting different robots and tasks into one token space are simple.

- **Normalization** — each dataset is fit to the $[-1,1]$ range using the 1% and 99% quantiles per action dimension.
- **Unified dimensionality** — dimensions are fixed to the largest action space, and lower-dimensional robots are zero-padded.
- **Control-mode tag** — whether the target is a joint pose or an end-effector pose is distinguished by a `<control_mode>` token in the prompt.
- **bboxes are text too** — locations are expressed as location tokens like `<loc0112>`.

In the end it is a strategy of making "everything a token." It is the same as how instruction-tuning mixtures for multimodal LLMs are put together. In short, Wall 1 is overcome by **detouring through other sources** instead of growing target data, and Wall 2 by putting every knowledge source into one sequence format.

### 3.5 Post-training — specialization, attaching the expert, and VI

After pretraining, it trains for another 80k steps with $\alpha=10$. This stage has two purposes. One is to specialize the model for mobile manipulation in homes; the other is to attach an action expert that produces continuous action chunks with flow matching.

- **Action data** — MM and ME are filtered down to episodes that succeeded and are below a certain length.
- **CE excluded** — lab data is dropped to focus on mobile manipulation and diverse environments.
- **WD kept** — kept in the mix to preserve semantic and visual abilities.
- **VI added** — an expert "teleoperates in language" the trained low-level policy in real time, choosing the appropriate subtask command step by step. What is collected this way becomes demonstrations of good high-level outputs for the trained policy.

VI is the "place for a person to step in" from Section 3.1, used for data collection. But MM and ME data already have subtask labels (HL) attached by people. Why collect VI separately?

> ### 💡 VI is on-policy data for the high-level policy
>
> The paper does not use the word "on-policy," but the difference between the two kinds of data reads this way. The key is **which states got labels**.
>
> | | HL labels | VI |
> |---|---|---|
> | How it is made | Post-hoc labeling of human teleop demonstrations | A person chooses commands in real time while the trained low-level policy runs |
> | States that get labels | States visited by an **expert operator** | States visited by **the system that will actually be deployed** |
> | What it teaches | What the expert did | What to tell this low-level policy right now for it to work |
>
> At deployment, what the high-level policy gives commands to is not an expert but **the trained low-level policy**. That policy slips differently and drifts differently from an expert. So the high-level policy has to choose, in the states that policy produces, commands that policy can actually carry out. This is a covariate-shift problem arising at the high level, and VI fills the gap by labeling the states the deployed system actually visits.
>
> The effect is large. VI is only about 11% of the high-level mobile-manipulation examples, but removing it drops performance significantly (Section 5.5, Fig. 13). Carried over to agentic AI, it is like training a planner not on idealized hypothetical traces but on **traces collected with the executor that will actually be deployed in the loop**.
>
> For reference, the examples in Fig. 4 show pairs like "Policy: put plate in sink / Relabeled: put plate on rack." It looks like cases where the command and the actual execution diverged were relabeled after the fact, but the paper does not describe this relabeling procedure.

### 3.6 Inference and control

```
 prompt l  ("clean the kitchen")
   |
   v
 4 cams + q_t  -->  VLM  --AR decode-->  l_hat  ("pick up the plate")
                     |
                     |  prefix = wrist + front cams, l_hat, q_t
                     v
 noise  -->  action expert (10 flow steps)  -->  action chunk (50 steps)
                                                    |
                                                    v
                                         PD controllers @ 50 Hz
```

- **Cameras** — high-level inference uses all 4 cameras (front, rear and both wrists); low-level inference uses only the 2 wrist cameras and the front camera.
- **Output** — target poses for the arms, grippers and torso lift and target velocities for the base are commanded directly at 50Hz (action chunking).
- **Controller** — the targets are tracked by simple PD controllers. There is no additional trajectory planning or collision checking. Both manipulation and navigation are end-to-end.

The dimensionality of the state and action space comes straight from the robot configuration.

$$\underbrace{2\times(6+1)}_{\text{two arms + grippers}}\ +\ \underbrace{3}_{\text{base } (v_x,\,v_y,\,\omega)}\ +\ \underbrace{1\sim 2}_{\text{lift}}\ =\ 18\sim 19$$

- Each arm is 6 DoF and each gripper 1 DoF
- The holonomic base is controlled with 2 axes of linear velocity and 1 axis of angular velocity
- The torso lift is 1 axis (up-down) or 2 axes (up-down and forward-back) depending on the platform

---

## 4. Why it works — taking stock of the three walls

First, where each of the three walls named in Section 1.3 was overcome.

| Wall | π₀.₅'s solution | Section |
|---|---|---|
| ⛔ Wall 1 — coverage | Rather than growing target data, cover more homes with light static arms (ME), more tasks with the lab (CE), and object concepts with the web (WD) | Section 3.4 |
| ⛔ Wall 2 — heterogeneity | Separate the levels with the subtask interface, and unify every knowledge source as a token sequence | Sections 3.1, 3.4 |
| ⛔ Wall 3 — representation | A two-stage switch: learn discrete, move continuous | Section 3.3 |

Matched against the three levels of Section 1.1, the paper's claim that **each knowledge source feeds a different level** corresponds to the experiments.

| Level of generalization | Main source | Supporting experiment |
|---|---|---|
| ⓐ Skill | MM, ME (environment diversity), CE (task diversity) | Scaling the number of environments (Fig. 8), no ME / no CE (Fig. 10) |
| ⓑ Composition | HL subtask prediction, VI | no HL / no VI / implicit HL (Fig. 13) |
| ⓒ Semantics | WD | no WD → drop in language following on OOD objects (Fig. 11), drop in high-level inference performance (Fig. 13) |

One question remains here. Does the gain from high-level inference come from actually saying the subtask at inference time, or from the process of learning to say it?

> ### 💡 Even without saying the subtask, having learned to say it already helps
>
> In the high-level inference comparison of Section 5.5, second place goes to **implicit HL**. It is a variant that generates no subtask at all at inference and feeds the overall command straight to the low level, but whose training mixture does include subtask-prediction data (HL). This variant is significantly better than no HL, which removes the HL data itself.
>
> The paper interprets this as: explicit subtask inference has a benefit too, but much of that benefit is already obtained just by putting subtask **prediction** data into the training mixture. That is, subtask prediction, as an auxiliary task, shapes the representation itself to fit the task structure.
>
> It resembles the phenomenon in LLMs where training on CoT data improves performance even when answering directly without CoT. But the paper has no evidence that it is the same mechanism, so it should be read only as an analogy.

Another axis is **language following**. In Appendix C's comparison (Fig. 15), π₀.₅ picks the instructed object at a rate slightly higher than π₀-FAST+Flow and much higher than π₀. The paper cites this as evidence that discrete-token training matters for language following. It is the property seen in the 💡 of Section 2, "discrete representations are good for learning," and the backpropagation interference issue seen in Section 3.2, viewed from another angle.

Seen this way, what makes π₀.₅ different from earlier models becomes clear.

> ### 📌 π₀.₅'s differentiator is not modules but placement
>
> π₀.₅ has almost no new network modules that π₀ lacked. The differentiator is **which knowledge is fed, to which level, at which training stage**.
>
> | | Low level (actions) | High level (subtasks) |
> |---|---|---|
> | pre-train | MM, ME, CE actions as FAST tokens | HL labels + WD |
> | post-train | MM, ME actions through the flow expert (FAST in parallel) | HL + WD + **VI** |
>
> Target data is only 2.4% of the pretraining examples. The remaining 97.6% each go into the slot that fits them and create generalization.

---

## 5. Experiments

### 5.0 Evaluation setup

All evaluation takes place in environments not in training. Controlled comparisons are run in 3 mock kitchens and 3 mock bedrooms, and the final evaluation in the kitchens and bedrooms of 3 real homes (12 locations in total). The standard evaluation is 4 tasks × 10 trials, or 40 episodes per policy, with significance judged by two-sided t-tests. Episodes canceled because of robot failures, time limits and the like are excluded.

The metric is rubric-based partial credit (progress %) (Appendix B).

| Task | Max score | Scoring |
|---|---|---|
| Dishes in Sink | 8 | +1 for picking up each of 4 dishes, +1 for putting each in the sink |
| Items in Drawer | 4 | +1 each for picking up, opening the drawer, putting in, closing |
| Laundry Basket | 3 | +1 each for moving to and picking up, placing on the basket, fully putting in |
| Make Bed | 5 | +1 for spreading the blanket, +1 for each of 2 pillows, +2 bonus for "very neat" |

Keep two things in mind. The metric is not binary success rate but **partial credit**. And the paper's results are all presented only as bar charts, so this piece carries over the direction and significance of the results rather than numbers.

### 5.1 Does it generalize in real homes (Fig. 7)

In 3 homes not in training, items in drawer, laundry basket and dishes in sink were each evaluated 10 times with two robot types. π₀.₅ consistently performed the various tasks in each home, and mock-environment performance represented real-home performance well. The model receives only simple commands like "place the dishes in the sink," and high-level inference decides steps like "pick up the cup" on its own.

But put this result side by side with the introduction's claim and a gap appears.

> ### ⚠️ Fact check — "10–15 minute long-horizon manipulation" is not what was quantitatively evaluated
>
> Fig. 1 and the introduction put forward 10–15 minute long-horizon manipulation that tidies an entire kitchen or bedroom. But the quantitatively evaluated tasks are mostly about 2–5 minutes long, and the metric is partial credit rather than binary success rate. Whole-room tidying at the 10–15 minute scale is presented only as qualitative demonstrations, not backed by numbers. The two should be kept apart when citing.

### 5.2 How does it scale with the number of training environments (Fig. 8, 9)

The number of locations where mobile-manipulation data was collected was increased 3 → 12 → 22 → 53 → 82 → 104 for comparison. All models were made to see the same number of unique samples over 40k steps, controlling for the effect of data quantity.

- **Task progress** — generally rises with the number of locations. The 104-location model reaches **a level similar to a control trained directly on the test homes' data**.
- **Baselines without co-training** — models trained only on the test homes' data, or only on the 104-location MM data, without other co-training tasks, fall far behind. The paper interprets this to mean that even having seen robot data from the test homes, other knowledge sources are essential for good generalization.
- **Language following** — a task of moving a specified one of 5 objects. The target object is placed farther away than the distractors, designed so that a policy that does not understand the instruction gets only about 20% right (Appendix C). As the number of locations grows both metrics rise, with new objects from training categories (ID) rising faster than objects from unseen categories (OOD: funnels, pill bottles, safety goggles, etc.).

There is a caution for reading this. Because of compute cost, this experiment is **a reduced recipe that pretrains without MM and then post-trains varying only the amount of MM**. It is not a scaling curve of the full π₀.₅ recipe, and the paper states this explicitly.

### 5.3 How much does each component of the data mixture matter (Fig. 10, 11, 16)

| Component removed | Mock task progress | Language following |
|---|---|---|
| no WD | **No significant difference** | **Significant drop on OOD objects** |
| no ME | Significant drop | Drop on both ID and OOD |
| no CE | Significant drop | Drop on both ID and OOD |
| no ME & CE | Largest drop | Drop |

Split by task (Fig. 16), the differences are sharper.

- **Items in Drawer** — a task that requires recognizing a wide range of everyday objects, so it drops when WD is removed, as well as ME and CE.
- **Dishes in Sink** — holds up without WD but drops without ME and CE. A task where general manipulation strategy is key.
- **Laundry Basket, Make Bed** — drop when cross-embodiment data is removed, and are less sensitive to other changes.

The correspondence in Section 4 — that the web data's benefit **concentrates at the semantic (ⓒ) level rather than the skill (ⓐ) level** — comes straight from this result.

### 5.4 How does it compare with π₀ (Fig. 12, 15)

| Model | How actions are trained | HL · WD data | High-level inference |
|---|---|---|---|
| π₀ | Flow expert from the start | ✗ | ✗ |
| π₀-FAST+Flow | The hybrid of Eq. (1) | ✗ (action data only) | ✗ |
| **π₀.₅** | FAST pretraining → attach expert | ✓ | ✓ |

The three models were trained on the same cross-embodiment robot data for a similar number of steps. π₀.₅ beats both significantly. The result holds even when π₀ is trained further to 300k steps, and the paper cites this as evidence that FAST token training is more compute-efficient than pure diffusion training.

The two comparisons isolate different effects.

- **π₀ vs π₀-FAST+Flow** — the effect of the training method (hybrid) on the same data
- **π₀-FAST+Flow vs π₀.₅** — the effect of HL and WD co-training and high-level inference (the two factors are mixed together)

### 5.5 How much does high-level inference matter (Fig. 13, 17)

The low level is fixed to π₀.₅ throughout, and variants that change only the high-level policy are compared. Of these, human HL is the variant in which a person directly puts commands into the interface of Section 3.1, and implicit HL is the variant seen in Section 4.

| Variant | Content | Result |
|---|---|---|
| **π₀.₅** | The same model does both high and low level | **1st** |
| implicit HL | No high-level inference, but trained with HL data | **2nd** |
| human HL | An "oracle" where an expert enters the high-level commands | Below π₀.₅ and implicit HL |
| no HL | Not even trained with HL data | Significant drop |
| no VI | VI data excluded | Significant drop |
| no WD | Web data excluded | Significant drop |
| GPT-4 | Zero-shot prompted with the task description and a list of frequently used labels | **Last** |

The drop for no VI supports the reading in Section 3.5 of VI as on-policy data. The drop for no WD shows that the web data's benefit leans heavily toward the high-level policy. But look at the ranking again and one thing is odd.

> ### ⚠️ Fact check — the human oracle that was supposed to be the upper bound is not the upper bound
>
> The paper defines human HL as an oracle that gives an upper bound on performance. But full π₀.₅ beats it, and implicit HL, which does no high-level inference at all, takes second place. In this experiment the oracle does not function as an upper bound, and the paper does not analyze why.
>
> The most plausible explanation is **interface alignment**. The low-level policy responds best to commands that match the vocabulary, granularity and switching timing of its training labels. A person's on-the-spot commands can stray from that distribution, and response latency gets in too. GPT-4 coming last even with the label list can be read in the same light. A good high-level policy is not a smart planner but **a planner aligned with this executor**.

---

## 6. Positioning — among neighboring work

The paper's Related Work sets up four lineages.

| Lineage | Representative | Limit π₀.₅ points out | π₀.₅'s position |
|---|---|---|---|
| Generalist policies · VLAs | RT-2, OpenVLA, π₀, etc. | Evaluated only in environments close to the training distribution | Evaluated in entirely new homes |
| Co-training on non-robot data | RT-2, PaLM-E, Magma | Limited to data for VLM training | Extended to other robots, subtasks, even verbal instructions |
| Language-based hierarchical reasoning | SayCan, YAY Robot, [Hi Robot](/notes/hirobot-hierarchical-vla-en/), HAMSTER / the ECoT line | Split into two models, or reasoning runs every step | One model + low-frequency high-level inference |
| Open-world robot systems | Roomba, AnyGrasp, RUM, "On Bringing Robots Home" | Narrow primitives, or simple tasks under a minute | Multi-stage long-horizon tasks |

The closest cousin is the hierarchical-reasoning lineage. It splits into two branches by how the model is composed, and π₀.₅ takes a combination between them.

| | Two-model hierarchy (SayCan, Hi Robot, etc.) | embodied CoT (ECoT, etc.) | **π₀.₅** |
|---|---|---|---|
| Number of models | 2 (VLM planner + low-level policy) | 1 | 1 |
| High-level inference frequency | Low | Same as the low level | Low |
| How the high-level policy is trained | Trained separately, or uses a pretrained LLM/VLM | Reasoning-chain labels | Co-trained with the low level (HL + WD + VI) |

The paper itself does not locate its contribution in "the idea of co-training." It admits that multitask learning and co-training themselves are not new, and locates its contribution in the fact that **a particular combination of data** made long-horizon behavior in new environments possible.

Section 3.2 left a question while pointing out the backpropagation path. If the random expert's gradient shakes the backbone, how much of a problem was that interference in practice?

> ### 🔗 Knowledge Insulation (Driess et al., 2025-05) — insulating the backbone from the flow gradient
>
> Follow-up work from the same team tackles this question head-on. Its analysis finds that naively attaching a new flow/diffusion action expert to a pretrained backbone **substantially harms both training speed and knowledge transfer**. The proposed fix is to put a stop-gradient on the path by which the expert references the backbone.
>
> $$\mathrm{attn}_a=\operatorname{softmax}\!\Big(\tfrac{1}{\sqrt{d_h}}\,Q_a\,\big[\operatorname{sg}(K_b);\ K_a\big]^{\top}\Big)\,\big[\operatorname{sg}(V_b);\ V_a\big]$$
>
> - $\operatorname{sg}(\cdot)$ — stop-gradient. Leaves forward values as they are and cuts only the backward pass
> - The remaining symbols are as in the equation of Section 3.2
>
> | | π₀.₅ paper | Knowledge Insulation |
> |---|---|---|
> | How the backbone is protected | Preserved by co-training next-token prediction and web data | Gradient from the expert to the backbone blocked with stop-gradient |
> | Training stages | 2 stages (FAST pretraining → attach expert) | FAST and expert trained together in a single stage |
> | Flow loss weight | $\alpha=10$ | $\alpha=1$ suffices (since the flow loss applies only to independent weights) |
>
> There is one important premise. Stop-gradient is valid **only when the backbone is also trained to predict discrete actions**. Only then does the backbone learn, on its own, the representations actions need even with the gradient cut. π₀.₅ keeping FAST token training through post-training in Eq. (1) is exactly this condition. PI describes this work as formalizing the method used in π₀.₅ and extending it to a single-stage recipe.
>
> Whether π₀.₅ v1 itself used stop-gradient, however, cannot be determined from the paper alone. In light of KI's explanation that $\alpha=1$ suffices with stop-gradient, π₀.₅'s $\alpha=10$ may be a trace of a setup in which the flow loss affected the shared weights (the author's inference).

---

## 7. Limitations

**Limitations the paper acknowledges**

- **Environment-specific difficulties** — unfamiliar drawer handles, cabinets that are physically hard to open
- **Partial observability** — cases where the robot arm hides the stain it needs to wipe
- **Distracted high-level inference** — cases like opening and closing a drawer several times while putting things away
- **Simple prompts** — the complexity of commands it can handle is set by the training labels
- **Short context, no memory** — weak on tasks that require moving between rooms or remembering where things are

**Further points to note**

- **The cost of human labeling has moved** — manual subtask labels on all multi-stage robot data, bbox labels and VI collection are required. Rather than eliminating cost, co-training partly moved teleop cost into annotation cost. The paper, too, names synthetic labeling as future work.
- **Pure imitation learning** — post-training data is filtered to successful episodes. There is no path for improving from its own failures.
- **Evaluation scale and reproducibility** — 3 real homes, 10 trials per task, a partial-credit metric, closed robots and data. The results are also presented only as charts, making numerical comparison hard.
- **A high-level policy without memory** — the drawer open-close loop looks like a typical failure of a high-level policy that picks the next subtask looking only at the current observation. It shares a root with the "short context" the paper acknowledges.

---

## 8. Closing — what this paper suggests

π₀.₅'s real contribution is not any particular module. It lies in showing, in real homes, that **a training recipe that places heterogeneous knowledge sources by level produces open-world generalization**. The result that a mixture with 2.4% target data accomplishes multi-stage tasks in homes it has never seen suggests that endlessly growing robot data collection is not the only path to scaling.

This paper reads especially well for someone with an LLM, diffusion and agentic AI background.

- **The data mixture is the contribution** — the same kind of research as LLM pretraining and mid-training mixture ablations.
- **Train with tokens, serve with a continuous decoder** — build representations first with an LLM-native objective (cross-entropy), then attach a continuous generator that fits the serving constraint (50Hz) later. The attached part is familiar to diffusion researchers: MM-DiT-style per-modality weights with shared attention, DiT-style adaptive-norm conditioning, and a rectified-flow objective with 10-step sampling.
- **Planner–executor inside one model** — the subtask string plays the role of a tool call. The result that a zero-shot GPT-4 planner comes last and even the human oracle loses is a lesson that carries straight over to agentic systems. **Alignment with the executor** comes before the planner's intelligence.
- **The training effect of reasoning data** — implicit HL shows that just putting reasoning data into training improves direct action. It is worth setting side by side with LLM research that uses CoT data as an auxiliary training signal.

---

## Appendix — Glossary

| Term | Definition |
|---|---|
| **co-training** | Training heterogeneous tasks, such as web VQA and subtask prediction, into one model alongside robot actions |
| **Action chunk** | The unit of predicting actions over several time steps at once. 50 steps for π₀.₅ (about 1 second at 50Hz) |
| **FAST** | A discrete action representation that compresses action chunks with the DCT and tokenizes them with BPE |
| **action expert** | A separate set of weights (300M) that processes only action tokens. Generates continuous actions with flow matching |
| **High-level inference (HL)** | Generating the next subtask as text from the observation and the overall command |
| **VI (verbal instruction)** | Demonstrations of high-level commands collected as a person directs the trained low-level policy step by step in language |
| **MM / ME / CE / WD** | Mobile manipulator / multi-environment static arms / lab cross-embodiment / web data |
| **implicit HL** | A variant that does not generate subtasks at inference but is trained with HL data |
| **knowledge insulation** | A follow-up technique that protects pretrained knowledge by cutting the gradient from the expert to the backbone with stop-gradient |

**Original** — [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) · **Project blog** — [pi.website/blog/pi05](https://pi.website/blog/pi05) · **Follow-up work** — [Knowledge Insulation, arXiv:2505.23705](https://arxiv.org/abs/2505.23705)
