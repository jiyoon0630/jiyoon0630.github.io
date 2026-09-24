---
layout: paper
lang: en
ref: cosmos-3-omnimodal-world-model
kind: paper-review
title: "Cosmos 3: Omnimodal World Models for Physical AI"
date: 2026-06-02 12:00:00 -0700
paper_date: 2026-06-01
venue: "NVIDIA Technical Report · arXiv:2606.02800"
tags: [World-Model, WAM, VLA, Omnimodal, Mixture-of-Transformers, Robot-Foundation-Model, Paper-Review]
authors: "NVIDIA (294 contributors, Appendix G of the paper)"
affiliations: "NVIDIA (Cosmos Lab)"
summary: "Split a VLM into an understanding tower and a generation tower, put video, audio and action on one physical time axis, and a single model serves as VLM, video generator, FD, ID and policy just by changing which tokens are given clean. Putting action into mid-training shows up as faster policy adaptation."
paper_url: "https://arxiv.org/abs/2606.02800"
code_url: "https://github.com/nvidia/cosmos"
---

> **Core claim** — The understanding (VLM), simulation (video generation, forward dynamics) and action (policy, inverse dynamics) that Physical AI needs do not have to be separate models. Split a VLM into two copies — one for understanding, one for generation — and bind video, audio and action into sequences on a single physical time axis, and one model plays every one of these roles just by changing **which tokens are given clean**. And putting action in as a first-class modality at the mid-training stage makes adaptation to a robot policy faster.

---

## Introduction

Cosmos 3 is NVIDIA's technical report on an "omnimodal world model". It claims to understand and generate language, images, video, audio and action inside one mixture-of-transformers, and to act as a VLM, a video generator, a world simulator and a world-action model just by changing its input/output configuration.

The report runs past 100 pages and also covers the data pipeline, training and serving infrastructure, and evaluations of image and audio generation. This piece focuses on the modeling argument — what goes wrong when you put heterogeneous modalities into one model, what mechanisms get past it, and whether that integration actually helps a robot policy. I build up the concepts where they are needed, so a thin robotics background is enough to follow along.

---

## 1. The problem — Physical AI built by chaining models

### 1.1 A fragmented pipeline

The paper holds that a Physical AI agent needs two capabilities together: **understanding**, which infers meaning and dynamics from partial observations, and **generation**, which pictures in advance how the world will unfold and how the agent should respond. Until now the two have developed separately. VLMs handle perception and reasoning; video generation models and forward dynamics models handle world simulation; VLAs and WAMs (World-Action Models) handle action prediction.

The paper's example is a household robot clearing the dining table after dinner. Done today's way, this robot has to chain three models together.

```
  task: "clean the dining table"

  [ VLM ]                  locate dishes, make a plan
     |
     v
  [ VLA / WAM ]            generate action sequence
     |
     v
  [ FDM / world model ]    simulate and evaluate future states
```

The authors argue that this separation is a fundamental limitation. Understanding requires reasoning about how the world will unfold and what actions will lead to; generation requires a compressed, structured representation of the world and of action. The two capabilities need each other, yet with the models split apart they cannot share representations, and computation is duplicated.

### 1.2 So the question the paper asks

> Instead of chaining models, can we design a **single model** that natively handles every capability a Physical AI agent needs?

### 1.3 Three walls

Naively pushing heterogeneous things into one transformer runs into three walls.

**⛔ Wall 1 — Interference between understanding and generation.** The understanding path is trained on clean tokens with causal attention and next-token cross-entropy. The generation path is trained on noisy tokens with bidirectional attention and velocity regression. Put both objectives on the same weights and the noise contaminates the understanding representations; the pretrained VLM's language and vision abilities can collapse.

**⛔ Wall 2 — Every modality runs on a different clock.** 24fps video becomes 6 tokens per second after the VAE's 4× temporal compression, audio is 25 tokens per second, and robot action follows the control rate (15Hz for DROID). With token index and physical time out of step, the model cannot learn the synchronization "this action produced this frame."

**⛔ Wall 3 — Disparate action spaces, and too little action data.** A car's steering, a robot arm's joints, a human's fingers and a camera's motion are all different control spaces. On top of that, action-labeled data is orders of magnitude scarcer than video.

Wall 1 is crossed in §3.2, Wall 2 in §3.3, Wall 3 in §3.4. Once past the walls, one question remains — does training in this unified way actually yield a reusable world-action prior? The answer comes in §4.

---

## 2. Background

Reading the walls and their fixes takes three concepts: the problems an action-conditioned world model solves (2.1), the generative objective (2.2), and position encoding (2.3). If they are familiar, skip ahead to §3.

### 2.1 Three questions for a world model with action

Let $v$ be a sequence of video latents, $l$ a language instruction, and $a$ the action. In this paper an action is defined as a causal variable that changes the world state; between consecutive video tokens, $a_t$ represents the transition from $v_{t-1}$ to $v_t$. A world model with action answers three questions.

$$\text{FD:}\quad p_\theta\big(v_{t:t+H}\mid v_{<t},\ a_{t:t+H},\ l\big)$$

$$\text{ID:}\quad p_\theta\big(a_{t:t+H}\mid v_{t-1:t+H},\ l\big)$$

$$\text{Policy:}\quad p_\theta\big(a_{t:t+H},\ v_{t:t+H}\mid v_{<t},\ l\big)$$

| Symbol | Meaning |
|---|---|
| $v_t$ | video latent token at time $t$ |
| $a_t$ | the action that caused the transition $v_{t-1}\to v_t$ |
| $l$ | language instruction |
| $H$ | length of the span (chunk) predicted at once |
| $v_{<t}$ | past video already observed |

| Name | Given | Asked | Role in a robot |
|---|---|---|---|
| FD (forward dynamics) | past video + action | future video | **Simulator** — what happens if I move like this |
| ID (inverse dynamics) | before-and-after video | action | **Labeler** — what action produced this change |
| Policy | past video + instruction | action + future video | **Actor** — imagines the action and its result together |

So do the simulator, the labeler and the policy each have to be trained separately?

> ### 💡 The three roles differ only in what you ask of one joint distribution
>
> All three distributions are different conditionals of a single joint distribution $p(v, a\mid l)$. Only what is given and what is asked changes.
>
> This setup is already familiar from language models. BERT, T5's span corruption and MaskGIT learn a joint distribution under many mask patterns so that one model can answer arbitrary conditionals. In diffusion, the mask becomes "the tokens given clean." Leave the given tokens clean, cover only the tokens being asked about with noise, and have the model denoise them.
>
> §3.1 implements exactly this idea.

### 2.2 The rectified flow objective

The generation side of Cosmos 3 trains every modality with a single rectified flow matching objective — the same family as SD3 and Wan.

$$x_\sigma=\sigma\,\epsilon+(1-\sigma)\,x_0,\qquad v^{*}=\epsilon-x_0$$

$$\mathcal{L}=\sum_{m}\lambda_m\,\mathbb{E}\Big[\big\|M_m\odot\big(v_\theta(x_{\sigma_m},\sigma_m,c)-v^{*}\big)\big\|_2^2\Big]$$

| Symbol | Meaning |
|---|---|
| $x_0$ | clean target latent — video, audio, **action vector** |
| $\epsilon\sim\mathcal{N}(0,I)$ | Gaussian noise |
| $\sigma_m\in[0,1]$ | noise level of modality $m$ |
| $v_\theta$ | denoiser that predicts the velocity |
| $c$ | conditioning (text, clean conditioning tokens) |
| $M_m$ | mask that removes clean conditioning tokens from the loss |
| $\lambda_m$ | per-modality loss weight |

Three points are worth noting.

- **Independent noise per modality** — $\sigma$ is sampled separately for each modality. Image, audio and action use logit-normal; video uses mode sampling.
- **Action is carved out like video** — the action vector is also denoised from noise. A normalized action vector has a small per-element MSE, so mid-training gives the action loss a 10× weight ($\lambda_{\text{act}}=10$).
- **Resolution-adaptive shift** — the sampled time variable is reparameterized to skew toward high noise.

$$\sigma=\frac{s\,\bar t}{1+(s-1)\,\bar t},\qquad \bar t=1-t$$

- $t$ — the time variable sampled from the distribution above
- $s\ge 1$ — the shift coefficient. 1/3/5 for 256p/480p/720p in pretraining, 3/5/10 in mid-training

### 2.3 From RoPE to MRoPE

RoPE encodes position as rotation, so that the attention score depends only on relative position.

$$\big\langle R(m)\,q,\ R(n)\,k\big\rangle=q^{\top}R(n-m)\,k$$

- $R(\cdot)$ — block-diagonal rotation matrix that rotates by an angle proportional to position
- $m, n$ — position indices of the query and the key

MRoPE (the Qwen family) splits the head dimensions into three groups $(t, h, w)$ and rotates each group by the index of its own axis. A video token then carries its frame number and its spatial position at the same time. But in the original MRoPE the time coordinate $t$ is the **frame index**. Why that leads to Wall 2 is covered in §3.3.

---

## 3. Method

The walls map to the mechanisms that cross them as follows.

| Wall | Mechanism | Section |
|---|---|---|
| ⛔ 1 Interference between understanding and generation | MoT dual tower + one-way joint attention + frozen reasoner | §3.2 |
| ⛔ 2 A different clock per modality | Absolute-time-axis MRoPE | §3.3 |
| ⛔ 3 Heterogeneous action spaces, scarce data | State-change pseudo-actions + per-domain projections + actions extracted from video | §3.4 |

Before that, the foundation every mechanism rides on: how the sequence is built.

### 3.1 One sequence; the mode is set by "what is clean"

The idea previewed in §2.1 is implemented here. The input sequence splits into two parts — an AR subsequence that handles understanding, followed by a diffusion (DM) subsequence. The same placement rules apply to every task.

- ⓵ AR tokens come before DM tokens
- ⓶ Within DM, for each modality, clean conditioning tokens come before noisy tokens
- ⓷ Both the conditioning part and the noisy part are ordered vision → audio → action

The AR prefix shared by every mode is:

$$S_{\text{AR}}\triangleq\big[\,l_1,\dots,l_n,\ \langle\text{EOS}\rangle,\ \langle\text{BOG}\rangle\,\big]$$

- $l_1,\dots,l_n$ — language tokens
- $\langle\text{EOS}\rangle$, $\langle\text{BOG}\rangle$ — end-of-sentence and begin-of-generation special tokens

Writing clean video, audio and action tokens as $v, s, a$ and noisy tokens as $\tilde v, \tilde s, \tilde a$, each mode is as follows (Eq. 3–6, Fig. 4).

| Mode | Sequence layout | What it does |
|---|---|---|
| Language | $S_{\text{AR}}$ only, generation parameters inactive | standard VLM |
| T2I | $[S_{\text{AR}},\ \tilde v_1]$ | image generation |
| T2V(+Audio) | $[S_{\text{AR}},\ \tilde v_{1:N},\ \tilde s]$ | video (+sound) generation |
| I2V / V2V | $[S_{\text{AR}},\ v_{1:P},\ \tilde v_{P+1:N}]$ | continue generating after the conditioning frames |
| Transfer | $[S_{\text{AR}},\ v^{\text{ctrl}}\_{1:N},\ \tilde v\_{1:N}]$ | generate RGB from a control video such as edge or depth |
| FD | clean $a$, noisy $\tilde v$ | action-conditioned future prediction |
| ID | clean $v$, noisy $\tilde a$ | action inference |
| Policy | noisy $\tilde v$, noisy $\tilde a$ | joint generation of action and outcome |

- $N$ — number of latent video frames, $P$ — number of conditioning frames ($P=1$ is I2V, $P>1$ is V2V)
- $v^{\text{ctrl}}$ — clean control-video tokens encoded by the VAE

Every mode shares the same architecture and the same objective. What separates the modes is only which tokens are given clean — the mask pattern. From the diffusion point of view, I2V is inpainting along time, and FD and ID are inpainting across modalities. In language mode the generation-side parameters are never switched on, so the model behaves exactly like a standard VLM.

### 3.2 Crossing Wall 1 — the MoT dual tower and one-way joint attention

Each decoder layer has two sets of parameters: a **reasoner** that processes the AR subsequence and a **generator** that processes the DM subsequence. LayerNorm, attention projections and FFN are all separate. Both paths start from the same pretrained VLM weights, and AR tokens are routed to the reasoner and DM tokens to the generator deterministically. The absence of a learned router is what distinguishes this from MoE.

The two towers meet in exactly one place, attention, and the direction is asymmetric (Eq. 7–8).

$$\mathbf{O}_{\text{AR}}=\text{Attn}_{\text{causal}}\big(\mathbf{Q}_{\text{AR}},\ \mathbf{K}_{\text{AR}},\ \mathbf{V}_{\text{AR}}\big)$$

$$\mathbf{O}_{\text{DM}}=\text{Attn}_{\text{full}}\big(\mathbf{Q}_{\text{DM}},\ [\mathbf{K}_{\text{AR}};\mathbf{K}_{\text{DM}}],\ [\mathbf{V}_{\text{AR}};\mathbf{V}_{\text{DM}}]\big)$$

- $\mathbf{Q}\_{\ast},\mathbf{K}\_{\ast},\mathbf{V}\_{\ast}$ — query, key and value that each tower produces with **its own projections**
- $[\cdot\,;\cdot]$ — concatenation along the sequence
- $\text{Attn}\_{\text{causal}}$ — attention that sees only preceding tokens, $\text{Attn}\_{\text{full}}$ — bidirectional attention

```
 token sequence
 [ l1 ... ln EOS BOG | v(1:P) | ~v(P+1:N) | ~s | ~a ]
 |<---- AR: reasoner ---->|<------- DM: generator ------->|

 attention mask (row = query, col = key)
              K_AR          K_DM
          +-------------+-------------+
   Q_AR   |   causal    |   blocked   |
          +-------------+-------------+
   Q_DM   |    full     |    full     |
          +-------------+-------------+
```

AR sees only AR, keeping the causal property inherited from the VLM. DM sees all of AR and DM bidirectionally, attending freely to the text condition and to every other conditioning and generated token. The paper states flatly that AR tokens are never updated by DM tokens. The point is to keep the conditioning path from being contaminated by the noisy generation process, and to preserve the pretrained VLM's text-generation ability.

There is one more mechanism on the training side. The reasoner is trained first (pretraining → Physical AI SFT), its weights are copied to initialize the generator tower, and **during generator pretraining only the generation-specific parameters are updated while the reasoner stays frozen.**

A question arises here. If the mask already shields AR from DM, why freeze the reasoner as well?

> ### 💡 The mask blocks the forward pass; freezing blocks the backward pass
>
> What the mask guarantees is only the forward direction: "AR's output does not depend on DM tokens." The other direction is open. DM's attention output takes AR's keys and values as input, so
>
> $$\frac{\partial\mathcal{L}_{\text{DM}}}{\partial W_K^{\text{AR}}}\neq 0,\qquad \mathbf{K}_{\text{AR}}=W_K^{\text{AR}}\,h_{\text{AR}}$$
>
> - $W_K^{\text{AR}}$ — the reasoner tower's key projection
> - $h_{\text{AR}}$ — the AR hidden state at that layer
>
> In other words, if the reasoner is not frozen, the gradient of the diffusion loss flows into the reasoner's K/V projections and every AR representation beneath them. Freezing closes this second channel. It solves the same problem as knowledge insulation on the VLA side (blocking the gradient from the action expert into the VLM), with a different tool.
>
> There is an inference benefit too. Because AR does not depend on noisy tokens, AR's KV can be computed once and reused across every denoising step — the same structure as prefix caching in LLM serving. The "Reasoner tower caching" item in the paper's serving section appears to correspond to this (an inference from the section title).

So when generating video or action, is the reasoner looking at the scene? The picture "the understanding tower interprets the scene and the generation tower draws from that interpretation" is natural, but following the equations gives something a little different.

> ### 💡 In generation mode the reasoner sees text — the split is by function, not by modality
>
> Video input splits into two encoders. The understanding ViT is trained together with the backbone; the generation VAE (Wan2.2-TI2V-5B) is used frozen. But in the generation-mode equations of §3.1 (Eq. 3–6), the AR prefix $S_{\text{AR}}$ holds only language tokens and special tokens. Visual conditions — input images, control videos, robot observations — all enter **on the DM side as clean VAE tokens**. Video goes through the ViT into AR only when the model is used as a VLM.
>
> Two things follow.
>
> - **The same vision goes to a different tower depending on its role.** For understanding it goes through the ViT to the reasoner; for generation it goes through the VAE to the generator. The split is not by modality, as in the original Mixture-of-Transformers, but by function: understanding versus generation.
> - **In generation and policy modes, pixel-level perception is the generator's job.** The reasoner is closer to a very strong instruction encoder that has been through Physical AI SFT.
>
> The paper does not give the AR layout for policy mode as an equation, so this reading is inferred from Eq. 3–6. It is also the decisive difference from π0-family VLAs (§6).

### 3.3 Crossing Wall 2 — absolute-time-axis MRoPE

Back to the problem previewed in §2.3. In the original MRoPE the time coordinate $t$ is the frame index. That is enough when handling video alone, but in Cosmos 3, which must generate video, audio and action simultaneously at different sampling rates, Wall 2 shows up directly: one step of index is a different amount of physical time for each modality.

Start with the coordinate assignment (Fig. 6).

| Token | $(t, h, w)$ assignment |
|---|---|
| Language (AR) | $t=h=w$, monotonically increasing — reduces to 1D RoPE |
| ViT video (AR) | same as Qwen3-VL's MRoPE |
| VAE video (DM) | all three axes vary, counting restarts from 0 for each segment |
| Audio, action (DM) | only $t$ increases, $h=w=0$ |

The key is the increment of $t$. The paper defines time steps per second (TPS). For video it is the frame rate divided by the VAE's temporal compression ratio of 4; for audio $48000/1920\approx25$; for action, the sampling frequency itself. The increment is then scaled to a base TPS (Eq. 9).

$$t_k=t_{\text{AR,end}}+g+k\cdot\delta t,\qquad \delta t=\frac{\text{TPS}_{\text{base}}}{\text{TPS}},\qquad \text{TPS}_{\text{base}}=\frac{24}{4}=6$$

- $k$ — index of the token within its modality segment
- $t_{\text{AR,end}}$ — time index of the last AR token
- $g$ — a fixed gap between AR and DM (explained below)
- $\text{TPS}_{\text{base}}$ — the TPS of 24fps video, the most common in the training data

| Modality | TPS | $\delta t$ | Increase in $t$ per second |
|---|---|---|---|
| Video 24fps | 6 | 1.0 | 6 |
| Video 16fps | 4 | 1.5 | 6 |
| Video 30fps | 7.5 | 0.8 | 6 |
| Audio 48kHz, hop 1920 | 25 | 0.24 | 6 |
| Action 15Hz (DROID) | 15 | 0.4 | 6 |

The last column is the point. Since $\text{TPS}\cdot\delta t=\text{TPS}_{\text{base}}=6$ always holds, **one physical second occupies 6 units of time index in every modality.** With the clocks unified, an action token and a video token from the same moment sit near the same phase inside attention. Video length and FPS are also written into the prompt, so the desired temporal characteristics can be given as a condition at inference time.

The gap $g$ is an empirical addition. When DM tokens were started at the position right after the last AR token, the first frame showed oversaturation and checkerboard artifacts, especially badly in Super. The paper's hypothesis is that the last language token and the first frame are at adjacent positions, so their temporal embeddings become nearly identical. That is plausible, since the low-frequency components of RoPE barely change phase over a one-step difference. The fix is to insert a fixed gap of $g=15000$ between the two subsequences, making the text-to-visual transition signal clear without any extra parameters.

### 3.4 Crossing Wall 3 — action redefined as "state change"

An action consists of up to three components (Fig. 3).

| Component | Meaning | Representation |
|---|---|---|
| ego pose | motion of the main observation frame (head camera, vehicle) | relative pose, 9D |
| effector pose | motion of the wrist / end effector | relative pose, 9D |
| grasp state | current manipulation state | fingertip coordinates 15D, or gripper open/close 1D |

To avoid embodiment-specific controller details (PID parameters, actuation interfaces), ego and effector are represented as **pseudo-actions** built from the difference between consecutive poses.

$$\Delta\mathbf{T}_t=\mathbf{T}_{t-1}^{-1}\,\mathbf{T}_t\ \ \longrightarrow\ \ \big[\,\Delta p\in\mathbb{R}^3,\ r\in\mathbb{R}^6\,\big]\in\mathbb{R}^9$$

- $\mathbf{T}_t\in SE(3)$ — pose at time $t$ (rotation + translation)
- $\Delta p$ — relative translation
- $r$ — 6D rotation representation (two columns of the rotation matrix). Unlike Euler angles or quaternions it is continuous, which helps regression. After output it is projected onto $SO(3)$ with an SVD

Grasp alone is not a difference; it holds the state at time $t$ as is. Combining these components gives an action vector whose dimension differs per embodiment.

| Embodiment | Composition | Dimension |
|---|---|---|
| Autonomous driving, camera motion | ego 9 | 9 |
| Single-arm robot | effector 9 + gripper 1 | 10 |
| Bimanual robot | (effector 9 + gripper 1) × 2 | 20 |
| Humanoid | ego 9 + (effector 9 + gripper 1) × 2 | 29 |
| Human egocentric | head 9 + (wrist 9 + fingertip 15) × 2 | 57 |

Vectors of different dimensions are lifted into and out of a shared latent space by a linear projection kept separately for each domain (Eq. 1–2). The backbone is shared.

$$\mathbf{z}=\mathbf{W}^{(k)}_{\text{in}}\,\mathbf{x}+\mathbf{b}^{(k)}_{\text{in}},\qquad \hat{\mathbf{x}}=\mathbf{W}^{(k)}_{\text{out}}\,\mathbf{z}+\mathbf{b}^{(k)}_{\text{out}}$$

- $k\in\{1,\dots,K\}$ — embodiment domain ID
- $\mathbf{x}\in\mathbb{R}^{d^{(k)}_{\text{in}}}$ — normalized action vector (scaled to roughly $[-1,1]$ per dimension)
- $\mathbf{z}\in\mathbb{R}^{d_{\text{model}}}$ — action token in the shared space
- $\mathbf{W}^{(k)},\mathbf{b}^{(k)}$ — domain-specific projection, trained from scratch

But what a robot policy ultimately has to output is joint commands. Why go to the trouble of defining action as "a difference of states"?

> ### 💡 Define action as state change, and a video corpus becomes an action corpus
>
> The bigger reason, more than controller independence, is **data**. Define action as joint commands and action data can only come from robot teleop logs. Define it as state change and both human-hand video (head and wrist poses plus 21-keypoint hand poses) and general video (estimated camera trajectories) become action data.
>
> The actual composition of the action mid-training data shows this (Fig. 9).
>
> | Domain | Hours | Share |
> |---|---|---|
> | Human egocentric (head camera + both hands) | 41.3K h | 67.4% |
> | Autonomous driving (in-house driving logs) | 10.0K h | 16.3% |
> | Robot (6 public datasets) | 5.4K h | 8.7% |
> | Camera motion (extracted from pretraining video) | 4.6K h | 7.5% |
>
> The total is 8.4M episodes and 61.3K hours. Camera motion was extracted by estimating camera poses on pretraining video with ViPE and DepthAnything3, and the robot data deliberately includes some failed episodes so the model also sees the consequences of actions that stray from the plan.
>
> Robot teleop is less than 10% of the whole. Half of Wall 3's "not enough data" was crossed like this — **by changing the definition of action**. In LLM terms, it is the same idea as turning unlabeled text into training data through pseudo-labeling.

The common action space is a lingua franca for mid-training. The actual deployment interface is attached fresh in post-training (§3.6).

### 3.5 Training curriculum

```
 Qwen3-VL-8B / 32B
      |
      v
 [Reasoner PT 22.0M] -> [Reasoner SFT 2.2M]
                              |
                              | init generator tower (copy)
                              v
 [Gen PT: img+video+audio] -> [Gen MT: +action +transfer] -> [Post-train]
  reasoner frozen              base: Nano / Super             T2I, I2V, Policy
```

- **Reasoner** — pretraining on 22.0M samples (OCR 42.9%, 2D grounding 16.5%, and so on). There is no separate alignment stage that fits only the projector first; everything is trained together from the start. SFT uses 2.2M samples, half of them video-text, concentrated on autonomous driving, robotics and smart infrastructure (Tab. 3).
- **Generator pretraining** — 767M images and 347.7M video clips, curated from 7.8B raw images and 3B videos. T2I/T2V/I2V/V2V are mixed at 20/56/16/8%, with audio generated alongside. Multi-resolution 256p/480p/720p is trained with 74K-token packing; Nano saw 31.05T tokens on 1024 GB200s and Super saw 17.86T tokens on 2048. The reasoner is frozen throughout.
- **Generator mid-training** — action and transfer enter for the first time. Nano 2.4T tokens, Super 1.9T. The output of this stage is the released base models, Cosmos3-Nano / Super.
- **Post-training** — T2I, I2V and Policy specialist models are each derived independently from the base. The architecture is identical to the base.

The mid-training data mix is as follows (Tab. 6).

| Stream | Modes | Share |
|---|---|---|
| Image | T2I | 10% |
| Video | T2V, I2V, V2V | 32% |
| Video + Audio | T2(V+A), I2(V+A), V2(V+A) | 8% |
| **Action** | FD, ID, Policy | **25%** |
| General Transfer | edge, blur, depth, segmentation control | 20% |
| Driving Transfer | world-scenario map control | 5% |

Bringing action in at mid-training rather than pretraining resembles mid-training in LLMs (the stage that raises the share of code and math). Solidify a broad visual prior first, then mix in a scarce but valuable modality later, with real weight.

There are three model sizes (Tab. 2).

| Variant | Total parameters | Base dense transformer | Initialization |
|---|---|---|---|
| Edge | 4B | 2B, 28 layers | trained from scratch in-house (architecture similar to Qwen3-1.7B) |
| Nano | 16B | 8B, 36 layers | Qwen3-VL-8B |
| Super | 64B | 32B, 64 layers | Qwen3-VL-32B |

"Total parameters" is the sum of the reasoner and generator towers. At the time of the report Nano and Super were released, and Edge was announced for a later release.

### 3.6 Specializing to a policy — Cosmos3-Nano-Policy-DROID

The paper frames this stage as a pilot study. The platform is DROID (a Franka Panda 7-DoF arm + a Robotiq 2F-85 gripper), and the data is 76K trajectories, 350 hours, 86 tasks and 564 scenes. It runs at 360×640 resolution, with idle-frame filtering and removal of failed demonstrations.

- **Initialization** — training continues from the mid-trained Nano, but the action encoder, the action-decoding MLP and the action embedding tokens are re-initialized. Here the action interface switches from the common pseudo-action to DROID's joint-position commands.
- **Input** — current proprioception and 3-view observations.
- **Output** — in policy mode, an action chunk and future video generated together.
- **Deployment** — after inference optimization, a policy server runs on two RTX Pro 6000s, and a Franky-based joint-position controller executes the 32 predicted actions at 15Hz.

```
  obs(t): 3 views + proprio
      |
      v
  [ Cosmos3-Nano-Policy-DROID ] --> 32 actions @ 15 Hz (~2.1 s) + predicted video
      |
      v
  execute chunk --> new obs --> replan
```

One chunk is about 2.1 seconds. Within a chunk it is open-loop; between chunks it is closed-loop, replanning from new observations.

---

## 4. Why it works

Back to the question left open in §1.3 — does unified training actually yield a reusable world-action prior?

The paper's key experimental design is a controlled comparison. It compares **PT-init**, which starts from a pretraining checkpoint that has never seen action data, with **MT-init**, which starts from a mid-trained checkpoint that has seen action across many domains and modes. Training recipe, model size, data and compute are all held fixed.

| Comparison | PT-init | MT-init | Source |
|---|---|---|---|
| Robot FD (Super, PSNR) | 22.69 dB | 26.04 dB | Tab. 18 |
| LIBERO-10 new embodiment, 500 iter | 0.0% | 24.6% | Tab. 20 |
| LIBERO-10 new embodiment, 2000 iter | 95.2% | 97.4% | Tab. 20 |
| RoboLab policy | behind | ahead | Tab. 19 |

The LIBERO-10 experiment uses a third-person camera and a wrist camera, and evaluates each checkpoint over 500 rollouts (50 per validation task).

What these results support more strongly is **"the starting point is different."** At 500 iter the gap is dramatic, but by 2000 iter it narrows to 2%p. The value of action mid-training shows up first in adaptation speed rather than in a higher ceiling.

Cross-domain transfer was measured separately. Modeled on cross-lingual transfer studies, the paper builds a transfer matrix among camera, autonomous driving, five robots and egocentric (Fig. 28). Two domains mixed 50/50 are trained for 4000 iter and a single domain for 2000 iter, matching the training exposure of the evaluation domain, and the change is read against the diagonal (single-domain) entry. For example, mixing in autonomous-driving data raised camera FD PSNR from 11.96 to 12.82. Whether a checkpoint warmed up on egocentric data can serve as the starting point for AgiBot robot adaptation is examined separately (Fig. 29). Positive transfer is uneven across domains, though, so one cannot say that cross-domain action data always helps.

The paper also checks how well the video the policy predicts matches reality. It executes a predicted action chunk in the RoboLab simulator from the same initial state and places that video side by side with the video the model imagined (Fig. 37).

> ### 📌 Action enters as a "mid-training modality," and its effect is isolated with a controlled comparison
>
> Existing VLAs attach action on top of a VLM after the fact, and WAMs fine-tune a video model into a policy. Cosmos 3 puts action into mid-training as training data on the same level as video and audio. And it isolates that effect with PT-init versus MT-init under the same recipe. This comparison is the core evidence behind the action part of the "omnimodal" claim.

---

## 5. Experiments

### 5.1 Summary across all capabilities

Tab. 1 gathers every capability in one table (* post-trained, † closed model).

| Capability | Super | Nano | Compared against |
|---|---|---|---|
| Reasoning · General | 73.7 | 69.6 | Qwen3-VL-32B 72.8 / Gemini 3.1 Pro† 77.5 |
| Reasoning · Robotics | 57.8 | 55.1 | Qwen3-VL-32B 52.6 / Gemini 3.1 Pro† 58.2 |
| Reasoning · Driving | 79.3 | 76.0 | Qwen3-VL-32B 40.7 / Gemini 3.1 Pro† 47.2 |
| Text2Image | 91.36* | 84.61 | Gemini 3 Pro Image† 90.85 / Qwen-Image-2512 84.25 |
| Text2Video | 80.0 | 79.4 | Veo-3.1† 79.1 / Wan2.2-A14B 78.0 |
| Image2Video | 82.8 | 82.7 | Veo-3.1† 82.6 / Wan2.2-A14B 81.3 |
| Audio | 7.31 | 7.34 | Veo-3.1† 7.45 |
| FD · Robot | 26.0* | 25.5* | Ctrl-World 23.0 |
| Policy · Robot | – | 39.7* | π0.5 28.1 |

Among the individual action results, on camera FD MT-init reached RRE 0.142°, RTE 0.026m and ATE 0.99m, beating Lingbot-World and HY-World 1.5 on all three metrics (Tab. 18).

### 5.2 Policy

- **RoboLab (simulation, Tab. 19)** — 120 tasks × 10 rollouts. Under the most specific instruction condition it averages 39.7%, ahead of π0.5's 28.1% and DreamZero's 25.2%. The paper states that it leads at every instruction granularity and difficulty.
- **RoboArena (real world, Fig. 26)** — anyone with a DROID compares two policies in a double-blind A/B, and pairwise preferences are aggregated into a score. First place as of 2026-05-30.

| Policy | RoboArena score |
|---|---|
| Cosmos3-Nano-Policy | 1870 |
| Spirit v1.6 | 1785 |
| DreamZero X | 1732 |
| WALL-OSS | 1657 |

- **MolmoSpaces (simulation, Fig. 27)** — first place as of 2026-06-20. Submitted with the same model and hyperparameters as RoboLab and RoboArena, with no per-benchmark tuning.

> ### ⚠️ Fact check — the qualifiers on the headline numbers
>
> ⓵ **It holds only "among open-source models."** The Tab. 1 caption says it beats specialized open-source baselines on every capability. Include closed models and it trails on general reasoning (Gemini 3.1 Pro 77.5 vs 73.7), robotics reasoning (58.2 vs 57.8) and audio (Veo-3.1 7.45 vs 7.34). The Artificial Analysis rankings are likewise first among open-weight models, but with proprietary models included it is 4th on T2I and 22nd on I2V (Fig. 18–19, as of 2026-05-28).
>
> ⓶ **The 39.7% policy figure is the value under the "specific" instruction condition.** Tab. 1 lists it without this qualifier. A third-party comparison on the full RoboLab-120 (BFL, 2026-09) puts Cosmos 3 Nano at 36.8%.
>
> ⓷ **Driving 79.3 vs 40.7 has to account for in-domain SFT.** The reasoner SFT included about 1.1 million decision-making videos auto-labeled from in-house driving logs. Without looking at the distance between the benchmark and the SFT distribution, it is hard to read this as generalization ability.
>
> ⓸ **"First place" is a dated snapshot.** The paper itself notes that more community evaluations will accumulate on RoboArena. On RoboLab, BFL later self-reported 42.92% with the 7B-scale FLUX 3 Action (2026-09).

---

## 6. Positioning — among neighboring work

The paper's own coordinate system is the three model classes of §1 — VLMs for perception and reasoning, video generation and FD models for simulation, VLAs and WAMs for action. Its experimental baselines also field a representative from each class: π0.5 as the VLA, DreamZero as the WAM, Ctrl-World as the FD model.

Architecturally, the work the paper acknowledges as closest is BAGEL (Deng et al., 2025). The decoder-layer structure is similar, but the paper distinguishes itself on training strategy, position embedding and overall range of capabilities. The dual tower of §3.2 and the absolute-time MRoPE of §3.3 are the substance of that difference.

The difference from the π0 family previewed in §3.2 becomes sharp when three designs on the same DROID are placed side by side.

> ### 🔗 Three designs on the same DROID — VLA, WAM, omnimodal world model
>
> | Axis | π0.5 (VLA) | DreamZero (WAM) | Cosmos 3 |
> |---|---|---|---|
> | Backbone origin | VLM | video diffusion model | two copies of a VLM (Qwen3-VL) |
> | Where observation images are seen | VLM | video backbone | generator (VAE path, inferred in §3.2) |
> | Future video prediction | none | yes | yes |
> | Text understanding / output | VLM | none | reasoner |
> | FD / ID modes | none | n/a | possible with the same weights |
> | RoboLab (specific) | 28.1 | 25.2 | 39.7 |
>
> The π0 family and Cosmos 3 share the same skeleton: "separate weights + asymmetric attention." The difference is who looks at the observation. In π0 the VLM sees the observation images and the action expert reads its KV. In Cosmos 3's generation and policy modes, the generator sees the observation directly as VAE latents, and the reasoner encodes the instruction.

From the diffusion side, the closest ancestor is SD3's MM-DiT. The skeleton of per-modality weights plus joint attention is the same. Two things differ. The text stream is not a T5-level encoder but **an entire VLM**, and attention is not bidirectional but **one-way, with AR not seeing DM**. The idea of "setting the mode by the mask" also connects to the UWM line of work, which decouples the diffusion timesteps of video and action to get policy, FD and ID from one model.

---

## 7. Limitations

**Acknowledged by the paper and official materials**

- **Generation quality** — the model card states temporal inconsistency, inaccurate physical interactions, and action-state drift, especially in long and high-resolution outputs. With no explicit physics simulator, contact dynamics and physical laws are only approximated.
- **Scope of policy validation** — the policy is a pilot on a single embodiment, DROID. No policy results are reported for Super ("–" in Tab. 1).
- **Cross-domain transfer** — positive transfer is uneven across domains (§4).

**Further points to raise**

- **Direct evidence that "omnimodal integration helps the policy" is narrow.** The controlled comparison is in effect the presence or absence of action mid-training (PT-init vs MT-init). Judging by their titles, the appendix ablations (E.1–E.5) do not include an item that directly isolates the effect of audio or transfer data, or of having a reasoner, on policy success rate.
- **Deployment cost** — the full 16B generator runs for every chunk. By a third-party measurement (BFL), the real-time factor is 0.150 for Cosmos 3 Nano and 0.032 for π0.5 — about 5× the compute per second of motion. The measurement conditions differ, so treat it as a reference figure only.
- **Asymmetry in failure data** — failed episodes were put into mid-training, but DROID post-training is pure BC with failed demonstrations removed. The policy stage has no mechanism for producing recovery data.

---

## 8. Closing — what this paper suggests

Cosmos 3's contribution lies less in first place on any particular benchmark than in **assembling a set of mechanisms for putting heterogeneous modalities into one model, and releasing it with open weights**. A function-based dual tower, one-way attention, absolute-time MRoPE, state-change action representation — each may not be new on its own, but this combination runs VLM, video generation, FD, ID and policy from one checkpoint.

For a reader with an LLM and diffusion background, several familiar bridges are visible.

- **"Two copies of a VLM, with generation trained on only one"** is in the lineage of BAGEL and MM-DiT. Because AR does not see DM, prefix caching from LLM serving carries over directly as an optimization of the denoising loop.
- **"Mode = mask"** extends T5 span corruption and MaskGIT. That FD (simulator) and ID (labeler) come out of the same weights as the policy with no extra heads has a practical meaning: policy evaluation and data labeling can be handled by one model.
- **"The definition of data is the scale of data"** — the moment action was redefined as geometric state change, human-hand video and general video became action data. An action-data mix in which robot teleop is under 10% is a strong signal for data strategy in physical AI.

Above all, what this paper shows most convincingly is not the effect of "integration" but the effect of "bringing action up into the pretraining stages." What omnimodal integration, audio and transfer included, adds to the policy remains an open question.

---

## Appendix — Glossary

| Term | Definition |
|---|---|
| **AR / DM subsequence** | the autoregressive token span that handles understanding and the diffusion token span that handles generation. AR always comes first |
| **MoT dual tower** | a structure with two parameter sets per layer, reasoner and generator, routed deterministically by token type |
| **Dual-stream joint attention** | asymmetric attention in which AR sees only AR, causally, and DM sees all of AR and DM bidirectionally |
| **Absolute-time MRoPE** | a position encoding that scales the time-coordinate increment to $\text{TPS}_{\text{base}}/\text{TPS}$, so that one physical second has the same position width in every modality |
| **Pseudo-action** | a controller-independent action defined as the relative transform of consecutive poses ($\mathbf{T}_{t-1}^{-1}\mathbf{T}_t$) |
| **FD / ID / Policy modes** | three action-related generation modes determined by which of video and action is given clean |
| **PT-init / MT-init** | initialization from a pretraining checkpoint that has not seen action / from a mid-trained checkpoint that has |

**Original** — [arXiv:2606.02800](https://arxiv.org/abs/2606.02800) · **Project page** — research.nvidia.com/labs/cosmos-lab/cosmos3 · **Code** — github.com/nvidia/cosmos
