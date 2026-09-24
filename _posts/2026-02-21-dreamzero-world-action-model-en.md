---
layout: paper
lang: en
ref: dreamzero-world-action-model
kind: paper-review
title: "World Action Models are Zero-shot Policies (DreamZero)"
date: 2026-02-21 12:00:00 -0800
paper_date: 2026-02-17
venue: "Preprint · arXiv:2602.15922"
tags: [VLA, World-Model, World-Action-Model, Video-Diffusion, Robot-Foundation-Model, Paper-Review]
authors: "Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi \"Jim\" Fan, Joel Jang"
affiliations: "NVIDIA"
summary: "Let a video diffusion model pretrained on web video generate future frames and actions together, and action learning turns from imitation into inverse dynamics, generalizing to new motions and new environments from heterogeneous, non-repetitive data alone."
paper_url: "https://arxiv.org/abs/2602.15922"
code_url: "https://github.com/dreamzero0/dreamzero"
---

> **Core claim** — What a VLA lacks is not the volume of repeated demonstrations but **a prior on "how the world moves."** Have a video diffusion model pretrained on web video generate future video and actions together, and action learning changes from "imitating actions from states" into "reading actions off an imagined future (inverse dynamics)." As a result, it generalizes to new motions and new environments from heterogeneous, non-repetitive data alone.

---

## Introduction

The standard VLA (Vision-Language-Action) recipe is to attach an action head to a pretrained VLM and train it on robot demonstrations. RT-2, OpenVLA, π0.5 and GR00T all sit inside this frame. The strategy is to carry the semantic knowledge a VLM learned from the web over to the robot.

DreamZero changes the starting point. Instead of a VLM it takes an **image-to-video diffusion model** (Wan2.1-I2V-14B) as the backbone, and generates future video and actions together in one model. The paper calls this class a **World Action Model (WAM)**. This piece follows the paper's argument, but builds up the needed concepts as it goes, so that a reader with a thin robotics background can still be convinced of the **reason** behind each design decision.

---

## 1. The problem — VLAs know "what" but not "how"

### 1.1 The gap in the VLM prior

VLMs are trained on static image-text pairs. So they are rich in semantic knowledge, but the **spatiotemporal prior** — how objects move, what changes on contact — is empty. The paper's example makes the difference vivid.

| Instruction | VLA result | Why |
|---|---|---|
| "move coke can to Taylor Swift" | Success | Identifying the target is web knowledge; "move" is a skill present in the robot data |
| "untie the shoelace" | Failure | If the **motion** of "untying" is absent from robot data, the VLM prior cannot fill it in |

The VLM prior encodes *what*, but not a *how* aligned with geometry, dynamics and motor control.

### 1.2 The price of filling that gap with data

According to the paper's related work, existing VLAs fill this gap with data collection.

- **⓵ Environment generalization** — collect a given task by teleoperation across hundreds of environments (π0.5).
- **⓶ Task generalization** — grow a library of language-conditioned motion primitives (Gemini Robotics). But covering the whole space of possible physical interactions with an episode-level task list is inherently impractical.

This strategy even dictates how data is collected. VLAs depend on **structured demonstrations that repeat the same task under similar conditions**. The paper's hypothesis is this: a model that predicts only actions must implicitly infer dynamics from noisy state–action pairs, so it learns poorly from heterogeneous, non-repetitive data. A video world model, by contrast, draws a learning signal from **every consecutive frame pair** in the data, and has already learned physical dynamics from web-scale video.

Side by side, the information flow of the two approaches looks like this.

```
  VLA :  (o, c) --> [ VLM backbone ] --> [ action head ] --> a
                          ^
                          prior = static image-text

  WAM :  (o, c) --> [ video diffusion backbone ] --+--> future video o'
                          ^                        |
                          prior = web video        +--> a
```

- A VLA goes straight from observation to action
- A WAM also draws the future video, and the action is generated to be aligned with that future

### 1.3 So the question the paper asks

> If a policy directly inherits the physical-dynamics prior of web video, can it generalize to new motions and new environments **from heterogeneous data alone, without repeated demonstrations**?

The idea of using a video model as a policy is not itself new (Section 6). The paper says that making it an actually working WAM means getting over three walls.

**⛔ Wall 1 — Video–action alignment.** Future video and motor commands must mesh tightly. Naively attaching a separate video head and action head lets the two drift apart.

**⛔ Wall 2 — Architecture choice.** It is unclear whether bidirectional (BD) or autoregressive (AR) is right. Modality alignment, error accumulation and inference efficiency all hang on this choice.

**⛔ Wall 3 — Real-time inference.** A 14B model doing iterative denoising in a high-dimensional latent is too slow for closed-loop control. Implemented naively, a single action chunk takes about 5.7 seconds.

Section 2 builds the concepts needed to understand the three walls, and Section 3 is the process of getting over them one by one.

---

## 2. Background — the minimum concepts for reading the three walls

### 2.1 Forward dynamics, inverse dynamics, policy

In robot learning, a "model" falls into three kinds depending on what it conditions on and what it predicts.

| Model | Distribution | Question it asks |
|---|---|---|
| Forward dynamics (classic world model) | $p(s_{t+1}\mid s_t,a_t)$ | If I move like this, what happens to the world |
| Inverse dynamics (IDM) | $p(a_t\mid s_t,s_{t+1})$ | If the world changed like this, what action was taken |
| Policy | $p(a_t\mid s_t,g)$ | If this is the goal, what should I do |

- $s_t$ — state at time $t$, $a_t$ — action, $g$ — goal

DreamZero's key sentence is "shift action learning from dense state–action imitation to inverse dynamics." But policy or IDM, both end up outputting actions. **Why does switching to inverse dynamics make the problem easier?**

> ### 💡 A VLA learns "the average integrated over futures"; a WAM learns with that integral unpacked
>
> A policy can be factored over possible future observations $o'$ as follows.
>
> $$\pi(a\mid o,c)\;=\;\int \underbrace{p(a\mid o,o')}_{\text{IDM}}\;\underbrace{p(o'\mid o,c)}_{\text{future prediction}}\;do'$$
>
> - $o$ — observations so far, $o'$ — future observation, $c$ — language instruction, $a$ — action chunk
>
> **A VLA learns the left-hand side directly.** When several strategies are valid for the same $(o,c)$ (left hand/right hand, grasp from above/from the side), all of that multimodality folds into the action distribution. That is why each mode needs repeated demonstrations.
>
> **A WAM learns the two factors separately.** The multimodality is carried by $p(o'\mid o,c)$, and web video pretraining already does this part well. Once the future is given, $p(a\mid o,o')$ is nearly deterministic, with low entropy.
>
> By analogy to diffusion, it is the difference between text-to-image (weak condition, broad distribution) and sketch-conditioned generation (strong condition, narrow distribution). In a WAM, the predicted future video plays the role of the sketch.
>
> This integral does not appear in the paper directly. It spells out the grounds for the claim "shift to inverse dynamics." And for this factorization to hold, the IDM must not need the language $c$. We meet this assumption again in Eq. (1) in Section 3.1.

### 2.2 AR video diffusion and teacher forcing — background for Wall 2

A video diffusion model makes video in one of two ways.

| | Bidirectional (BD) | Autoregressive (AR, chunk-wise) |
|---|---|---|
| Unit of generation | A whole fixed-length clip at once | Chunk by chunk, sequentially |
| Past context | Only within the same clip | Arbitrary-length context via KV cache |
| Training | Whole clip at the same noise level | Independent timestep per chunk, previous chunks clean (teacher forcing) |
| Weakness | Fixed length, so a subsampling problem | Errors accumulate at inference as it re-consumes its own output |

AR has the same structure as LLM training. At training time it conditions on the ground-truth prefix (clean previous chunks); at inference it conditions on what it generated itself. So it has the same weakness too: **exposure bias**, where training and inference conditions differ. This is why work the paper cites, like Self Forcing, deals separately with this train-test gap in AR video diffusion. How DreamZero avoids this weakness comes in Section 3.3.

### 2.3 Flow matching notation, and step count is latency — background for Wall 3

DreamZero trains with flow matching. The paper's notation is as follows.

$$z_t = t\,z_1 + (1-t)\,z_0,\qquad z_0\sim\mathcal{N}(0,I),\qquad v = z_1 - z_0$$

- $z_1$ — clean latent, $z_0$ — gaussian noise
- $t\in[0,1]$ — timestep. **$t=1$ is clean, $t=0$ is pure noise** (some literature uses the opposite convention, so watch out)
- $v$ — the target velocity the network regresses

Inference starts from noise at $t=0$ and integrates the velocity toward $t=1$. Each step runs one forward pass of the whole DiT, so latency is roughly **(number of denoising steps) × (DiT forward cost)**. With 16 steps on a 14B DiT, it is immediately clear why Wall 3 arises. What breaks when you try to cut the step count is covered in Section 3.4.

---

## 3. Method — DreamZero

The overall structure is as follows.

```
  o_0:l    -- VAE (frozen) ---------+
  c        -- text enc (frozen) ----+
  q_l      -- state enc (new) ------+----> AR DiT : Wan2.1-I2V-14B (trainable)
  noisy a  -- action enc (new) -----+            |
                                                 +--> video latent o_l:l+H
                                                 +--> action dec (new) --> a_l:l+H
```

- The inputs are the observed video, language, proprioception, and the noisy action to be denoised
- A single AR DiT jointly denoises the future video latent and the action, chunk by chunk
- After execution, the real observation goes into the KV cache and it moves on to the next chunk

### 3.1 Eq. (1): the factorization and a single model — getting over Wall 1

The paper formalizes the problem as follows.

$$\pi_\theta(o_{l:l+H},a_{l:l+H}\mid o_{0:l},c,q_l)=\underbrace{\pi_\theta(o_{l:l+H}\mid o_{0:l},c,q_l)}_{\text{video prediction}}\;\underbrace{\pi_\theta(a_{l:l+H}\mid o_{0:l+H},q_l)}_{\text{IDM}}$$

- $o_{0:l}$ — observed video so far, $o_{l:l+H}$ — future video to predict
- $a_{l:l+H}$ — action chunk, $H$ — fixed horizon
- $c$ — language instruction, $q_l$ — proprioception
- $l$ — a time step sampled at random from the trajectory
- The original writes this distribution as $\pi_0$; here it is written $\pi_\theta$ to avoid confusion with the π0 model

**The answer to Wall 1 is not to implement this factorization as two models.** Instead of a separate video model and IDM (the approach of Li et al., 2026; Pai et al., 2025), a single model jointly denoises [video, action] under a shared objective. The idea is to secure alignment not by combining two heads after the fact but by generating them together inside the same attention. The paper's grounds, though, are "We believe." There is no ablation against the separated variant.

The concrete design is as follows.

| Item | Choice | Reason |
|---|---|---|
| Added parameters | Only a state encoder and an action encoder/decoder | Preserve the video model's generalization |
| Multi-view | Concatenate the views into one frame | Leaves the backbone architecture untouched |
| What trains | Whole DiT trained; text/image encoders and VAE frozen | LoRA did worse (footnote 7) |
| Action representation | Relative joint position, idle actions removed | — |
| Timestep | Video and action share the same $t$ within a chunk | Faster convergence early in training |

The last row deserves attention. Recent WAMs (Kim et al., 2026; Li et al., 2025a, etc.) separate the timestep per modality, but DreamZero deliberately shares it. This choice is reversed again in Section 3.4.

But look at the right-hand side of Eq. (1) again and something is odd. **The language $c$ has vanished from the IDM term.** Applying the chain rule as is, it should remain.

> ### 💡 Eq. (1) is not an identity but a factorization that carries an assumption
>
> Applying only the chain rule, the second term becomes:
>
> $$\pi_\theta(a_{l:l+H}\mid o_{0:l+H},\,c,\,q_l)$$
>
> Eq. (1) dropped $c$ from it. That is, it contains the conditional independence assumption $a\perp c\mid(o_{0:l+H},q_l)$: "given the future video and the pose, language adds no information about the action." The integral in Section 2.1 stands on the same assumption.
>
> | Case | Does the assumption hold |
> |---|---|
> | The video specifies the motion visually (most pick-and-place) | Holds |
> | Same visual trajectory, different force control ("press gently") | Can break |
> | The contact point is occluded, so the video underdetermines the motion | Can break |
>
> Note, however, that the actual model does not train the two terms separately, so the action tokens also see $c$ through attention. Eq. (1) is closer to **a justification of the design** than a structural constraint.

### 3.2 AR chunk-wise training — getting over Wall 2

The paper names three advantages of AR. Inference is fast thanks to the KV cache, the observation history can serve as guidance for the next generation, and it avoids BD's modality alignment problem. The third is decisive; with the example from Appendix B, it goes like this.

Say a particular segment of a long demo carries the caption "put the black objects into the drawer."

| Approach | What happens | Result |
|---|---|---|
| BD, no subsampling | The fixed-length clip covers only part of the task segment, so the language describes motion not yet in the video | Worse language following |
| BD, subsampled to fit the captioned segment | Closed-loop training must start from arbitrary points mid-task, and fitting the segment from the middle distorts the native FPS | Worse video–action alignment |
| AR | The earlier part goes in as context and only the later part is generated | Keeps both caption correspondence and native FPS |

BD falls into a dilemma where it must give up either "language alignment" or "temporal resolution." AR, by virtue of a structure that takes context, sidesteps the dilemma altogether.

The chunk spec (Appendix C, for AgiBot) is as follows.

```
  |<-- chunk 1 -->|<-- chunk 2 -->|<-- chunk 3 -->|<-- chunk 4 -->|
  | 2 latent fr.  | 2 latent fr.  | 2 latent fr.  | 2 latent fr.  |
  | 48 actions    | 48 actions    | 48 actions    | 48 actions    |
  | 1.6 s         | 1.6 s         | 1.6 s         | 1.6 s         |
  |<------------ max context: 8 latent = 33 raw frames = 6.6 s -->|
```

- Video at 5 FPS, actions at 30Hz. With $K=2$ latent frames per chunk and action horizon $H=48$, each chunk is 1.6 seconds
- They report $K=2$ was better than $K=1$
- DROID uses 15Hz actions with $H=24$, also 1.6 seconds per chunk
- The default is 4 chunks, and the maximum visual context is 6.6 seconds

There is one more design choice. **AR is applied only to the video modality.** The reason the paper gives is to avoid error propagation from closed-loop action prediction.

The training objective is per-chunk flow matching. First, build the noisy input of chunk $k$.

$$z^k_{t_k}=t_k\,z^k_1+(1-t_k)\,z^k_0,\qquad a^k_{t_k}=t_k\,a^k_1+(1-t_k)\,a^k_0$$

- $z^k_1$ — clean video latent of chunk $k$, $a^k_1$ — normalized clean action
- $z^k_0,\ a^k_0\sim\mathcal{N}(0,I)$ — noise
- $t_k$ — timestep of chunk $k$. **Shared by video and action within a chunk, independent across chunks**

Then regress the velocities of both modalities together.

$$\mathcal{L}(\theta)=\mathbb{E}\Big[\frac{1}{K}\sum_{k=1}^{K}w(t_k)\,\big\lVert u_\theta([z^k_{t_k},a^k_{t_k}];\,\mathcal{C}_k,c,q_k,t_k)-v^k\big\rVert^2\Big],\qquad v^k=[z^k_1,a^k_1]-[z^k_0,a^k_0]$$

- $u_\theta$ — the DiT that outputs the video and action velocities together
- $\mathcal{C}_k$ — clean context from previous chunks (teacher forcing)
- $q_k$ — proprioception of chunk $k$, $c$ — language
- $w(t_k)$ — timestep weight, $v^k$ — target velocity

Training is done over a whole trajectory at once, with an attention mask so that the current noisy chunk sees only earlier clean chunks. It is the same way an LLM trains variable-length sequences in one pass with a causal mask.

> ### ⚠️ Fact check — inconsistent notation around the training objective
>
> Three places disagree.
>
> - **Does the context include actions?** Eq. (3) and Algorithm 1 define $\mathcal{C}_k$ as pairs of clean video latents and clean actions $a^j_1$ from past chunks. But the main text says "AR only for video," and the inference context in Fig. 14 and Algorithm 2 is video only. The main text and Fig. 14 look like the intent.
> - **$K$ used twice.** In Appendix C, $K$ is the number of latent frames per chunk (=2), while in Eq. (3) it is the range of the sum over chunks. Algorithm 1 uses $M$ in the same place.
> - **Timestep convention.** Algorithm 2 attaches $t=0$ to the call that injects clean GT, but under the convention of Eq. (2), $t=0$ is pure noise.

### 3.3 Inference — feeding real observations back into the KV cache

```
  [chunk k]      KV cache = {GT_0, ..., GT_k}
                    |
                    v
               joint denoise  --+--> video latent Z_k+1  --> discarded
                                +--> action Y_k+1        --> filter --> robot (async)
  [chunk k+1]  real obs --VAE--> GT_k+1 --forward(update KV)--> KV cache = {GT_0, ..., GT_k+1}
```

- **⓵ Generate** — starting from noise, jointly denoise [video, action]
- **⓶ Execute** — take out only the clean actions, smooth them, and execute them asynchronously on the robot. **The predicted video latent is discarded**
- **⓷ Feed back** — encode the real camera observation with the VAE, run it forward, and write its KV into the cache

But recall AR's weakness from Section 2.2 and a question arises. In AR video generation the model re-consumes frames it drew itself as conditions, so small errors snowball. **DreamZero is AR too, so why does it say it doesn't suffer from this?**

> ### 💡 In closed loop, the environment is the teacher even at inference
>
> | Stage | What goes into the context | Result |
> |---|---|---|
> | Training (teacher forcing) | GT frames | — |
> | Inference for pure video generation | Frames it generated itself | Train-test mismatch → error accumulation |
> | DreamZero inference | **Real camera observations** | Same as the training condition |
>
> In LLM terms, it is as if every time a chunk of text is generated, it gets swapped for the "ground-truth continuation" before the next step. Impossible in text, but natural for a robot, because the world renders the real next frame for free. So the predicted video becomes **a use-and-discard plan, not state**.
>
> Strictly speaking, rather than "replacing predicted frames with GT," the predicted frames **never enter the cache in the first place** and only GT is used. The paper calls this "an advantage unique to WAMs" and puts it forward as one of the three key designs in Section 3. There is, however, no ablation isolating this mechanism.

Thanks to this structure, DreamZero becomes a **stateful policy** that uses visual history as memory. But tasks that can only be solved with memory were not evaluated (footnote 2).

### 3.4 Making it real-time — getting over Wall 3

The naive implementation takes 5.7 seconds per chunk for three reasons.

| Bottleneck | Detail |
|---|---|
| Iterative denoising | 16 steps for smooth actions |
| Backbone size | 14B DiT |
| Sequential execution | The robot stops while inference runs |

A natural question arises here. Since the predicted video is discarded anyway, **wouldn't generating only actions be faster?** The paper denies this in footnote 3. At 14B scale the gain is negligible, because, as seen in Section 2.3, latency is dominated by the step count and the number of DiT blocks. Moreover, since the two modalities were trained together, naively cutting only the action steps degrades quality. In the end what has to shrink is the step count itself, and that is the motivation for DreamZero-Flash, which we will see below.

**Asynchronous execution.** The first step is to decouple inference from execution. The motion controller keeps executing the most recent action chunk while inference runs concurrently on the latest observation. The constraint then relaxes from "inference must finish before the robot moves" to "inference must finish before the current chunk (1.6 s) runs out." The paper targets roughly 200ms or less for enough overlap.

**Optimizing the whole stack.** On top of that, optimizations stack up (Table 1, H100 baseline = 1×).

| Layer | Technique | Cumulative (GB200) |
|---|---|---|
| — | baseline | 1.1× |
| System | CFG parallelism — split conditional/unconditional forward across 2 GPUs (−47% latency per step) | 1.8× |
| System | DiT caching — reuse when the cosine similarity of consecutive velocities exceeds a threshold (effective 16→4 steps) | 5.4× |
| Implementation | torch.compile + CUDA Graphs | 10.9× |
| Implementation | cuDNN attention, moving scheduler ops to GPU | 14.8× |
| Implementation | NVFP4 quantization (FP8 for QKV·Softmax, FP16 for nonlinear ops) | 16.6× |
| Model | **DreamZero-Flash** (1-step) | **38×** |

The upshot is that 5.7 seconds becomes 150ms, enabling roughly 7Hz control. Apart from DiT caching and quantization, everything is mathematically equivalent to the baseline.

**DreamZero-Flash — separate noise schedules per modality.** What breaks when you cut the step count? In few-step inference you get a situation where "the video is still noisy but the action must be clean." Yet during training the two modalities were always at the same noise level (the shared timestep of Section 3.1). Flash closes this train-test mismatch through the training distribution.

$$t^{\text{video}}_k = 1-\eta,\quad \eta\sim\text{Beta}(\alpha,\beta),\ \alpha>\beta,\qquad t^{\text{action}}_k\sim\mathcal{U}(0,1)$$

- $\eta$ — a sample from a Beta distribution. With $\alpha>\beta$ it concentrates near 1
- The actual setting is $\text{Beta}(7,1)$. $\mathbb{E}[\eta]=0.875$, so $\mathbb{E}[t^{\text{video}}_k]=0.125$, which under the convention of Section 2.3 is **mostly high noise** (the mean is 0.5 in the shared setting)
- The action timestep stays uniform

This way training frequently encounters "predicting clean actions from a noisy visual context," matching the 1-step inference regime. Flash is applied as a final stage after main training is done.

| Table 3 (table bussing) | steps | task progress | latency |
|---|---|---|---|
| DreamZero | 4 | 83% ± 6.1 | 350ms |
| DreamZero | 1 | 52% ± 10.2 | 150ms |
| DreamZero-Flash | 1 | **74%** ± 10.1 | 150ms |

The scope of what is new needs to be stated precisely. As seen in Section 3.1, separating timesteps per modality is what earlier WAMs already did, and the DreamZero main model actually reverted to sharing for convergence speed. So Flash's contribution is not the separation itself but **the distribution design that biases the video noise toward high noise to fit the 1-step inference regime**.

But with 1 step, the current chunk's video tokens start from pure noise ($t=0$). **Then what "future" does the action look at to do IDM?**

> ### 💡 In Flash, the visual plan moves from "a drawn video" to "an internal network representation"
>
> In a 1-step forward pass, the input video tokens themselves carry no pixel information. But inside the DiT, the hidden states of those tokens are computing "which future to send this noise to" — that is, the velocity toward the clean latent. The action tokens see these hidden states through attention. **The plan exists not as a denoised video but as features.**
>
> Seen this way, Flash is a step away from explicit visual planning toward extracting actions from "clean past context + an implicit future." The drop from 83% at 4 steps to 74% for 1-step Flash is consistent with this reading. The IDM reading of Eq. (1) holds most literally with multi-step inference.
>
> This interpretation is not in the paper, and it is not backed by any experiment analyzing internal representations.

Finally, **action chunk smoothing** is applied. The action chunk is upsampled 2× with cubic interpolation, high-frequency noise is suppressed with a Savitzky-Golay filter (window 21, order 3), and it is downsampled back to the original resolution.

---

## 4. Why it works

The paper's own explanation comes down to four points.

| Explanation | Content | Evidence |
|---|---|---|
| ⓵ Inheriting the prior | Video prediction is already optimized on web data, so only video prediction for the robot embodiment and action extraction need to be learned on top | Section 3.1 |
| ⓶ Failures come from the video | Most failures stem not from action extraction but from video generation errors. The policy faithfully executes even a wrong video plan | Fig. 16 |
| ⓷ Diversity makes the IDM | The video side is mostly inherited, so the bottleneck is the IDM. A robust IDM needs state–action correspondences from diverse contexts, which repetitive data lacks | Table 4 |
| ⓸ Backbone size | Smaller models hallucinate visually more often, and that propagates into wrong actions | Table 4 |

The cases for ⓶ are concrete. For "pick up the marker and draw a line on the whiteboard," the model drew a video of handing the marker to the other hand, and the robot handed it over just so. For "bake the croissant in the oven," it drew a video of picking up the bread before opening the oven, and the robot stopped in front of the oven holding the bread. From this the paper concludes that "improving robot capability reduces to improving video generation capability."

Two things can be added.

- **The integral in Section 2.1 explains "why non-repetitive data works only for WAMs."** A VLA needs samples covering the modes for each $(o,c)$. A WAM's IDM, by contrast, gets **every consecutive frame pair as supervision** regardless of task labels. That is why the repetition of task labels matters less.
- **⓶ looks like a weakness report, but it is actually the strongest evidence for the Eq. (1) factorization.** If the action were not dependent on the video, video errors would not carry over to actions this faithfully. At the same time, it means the policy's ceiling is tied to the physical plausibility of the video model.

> ### 📌 Robot capability = video generation capability + a thin action readout
>
> DreamZero's biggest differentiator is that it makes the robot-specific learned part thin: "an action readout synchronized to video." In this design, **scaling the video backbone is scaling the policy.** On the same data, DreamZero went from 21% → 50% at 5B → 14B, while VLAs scaled to the same sizes stayed at 0% at both 5B and 14B (Section 5.5). It is a signal that the scaling axis of robot foundation models may partly shift from "the amount of robot data" to "the quality of the video model."

---

## 5. Experiments

### 5.1 Setup

| Item | Detail |
|---|---|
| Pretraining data | About 500 hours of AgiBot G1 (mobile bimanual). 22 environments, 7,193 episodes, 4.4 minutes and about 42 subtasks per episode on average. DROID for Franka |
| Collection | Once a task reaches 50 episodes it is removed from the list and operators propose new tasks, forcing a long tail |
| Pretraining | Separate training per embodiment, 100K steps, batch 128 |
| Post-training | Shirt folding 33 hours, fruit packing 12 hours, table bussing 40 hours. 50K steps per task |
| Baselines | GR00T N1.6 and π0.5, each prepared as ⓐ scratch (VLM weights only) and ⓑ pretrained (official checkpoint continually trained on the same data). Batch and step count matched |
| Evaluation conditions | Default is **unseen environment + unseen object**. Evaluation sites differ from the data-collection sites |
| Seen / unseen tasks | Defined by the "motion + object type" combination. A shirt in a different color is seen; folding socks is unseen |
| Metrics | 8 rollouts per task (4 robots), 80 rollouts per checkpoint. The main metric is **task progress** (partial credit) |

### 5.2 Generalization — seen, unseen, post-training

AgiBot G1 results (average task progress, %).

| | GR00T N1.6 scratch | GR00T N1.6 pretrained | π0.5 scratch | π0.5 pretrained | DreamZero |
|---|---|---|---|---|---|
| Seen tasks (Fig. 8) | 0.6 | 8.4 | 0 | 27.4 | **62.2** |
| Unseen tasks (Fig. 9) | 0.7 | 5.0 | 0 | 16.3 | **39.5** |
| Post-training, mean of 3 (Fig. 10) | 9.8 | 53.3 | 0.5 | 79.8 | **90.5** |

- Scratch VLAs are near 0% on the same heterogeneous data. Even on easy pick-and-place they only reach toward the right object without properly interacting with it
- Among the 10 unseen tasks, the strong ones are "Remove Hat from Mannequin" at 85.7 and "Shake Hands" at 59.2
- Post-training: shirt folding tied (92.5 vs 92.5), fruit packing a large lead (96 vs 71), table bussing a small lead (83 vs 76). Since evaluation is in unseen environments, this is evidence that environment generalization survives post-training
- Qualitatively, pretrained VLAs tended to reach for and try to grasp objects regardless of the instruction. The reading is that they overfit to the dominant training behavior (pick-and-place)

On DROID-Franka unseen tasks (20 verbs absent from DROID), task progress is 49 / 31 / 33 and success rate 22.5 / 12.5 / 7.5 (DreamZero / GR00T N1.6 / π0.5).

> ### ⚠️ Fact check — "over 2×" is on AgiBot, and the metric is partial credit
>
> The abstract claims a more than 2× generalization gain over VLAs in real-robot experiments. On AgiBot, seen is 2.27× (62.2/27.4) and unseen 2.42× (39.5/16.3), consistent with the claim. But DROID unseen task progress is 49/33 ≈ 1.5×. "Over 2×" should be read as a statement about the AgiBot results.
>
> Also, the headline numbers are all task progress, which is not completion rate. As DreamZero's task progress of 49 against a success rate of 22.5 on DROID shows, the two can diverge widely.

### 5.3 Cross-embodiment — learning from video without actions

Video from other embodiments gets **only the video prediction objective**, with no action labels, mixed 1:1 with the pretraining data and co-trained for 10K steps. In terms of the Section 3.1 factorization, this data strengthens only the first term of Eq. (1) (video prediction).

| Table 2 (9 unseen tasks) | task progress |
|---|---|
| DreamZero | 38.3% ± 7.6 |
| + Human2Robot (12 minutes of egocentric video) | 54.3% ± 10.4 |
| + Robot2Robot (20 minutes of YAM video) | 55.4% ± 9.5 |

- The largest gain is Robot2Robot. The paper attributes it to YAM and AgiBot both being bimanual parallel grippers, so the embodiment gap is narrow
- Human video, with a large difference in form and a shaky viewpoint, improved things by nearly the same margin

> ### ⚠️ Fact check — the scope of what this experiment shows
>
> - **The transfer video is the evaluation tasks themselves.** It is 72 trajectories demonstrating the 9 unseen tasks, 8 per task. It is the effect of "showing, as video, tasks unseen in terms of action data," not zero-shot transfer.
> - **No control for extra training.** The baseline is a checkpoint that did not go through the extra 10K steps. There is no control that ran another 10K steps on the pretraining data alone.
> - **The exact value of "over 42%."** The relative gains are +44.6% for Robot2Robot and +41.8% for Human2Robot.

### 5.4 Few-shot embodiment adaptation

The checkpoint pretrained on AgiBot was post-trained on a new robot (YAM) with 55 trajectories, 11 tasks, about 30 minutes of play data. They report that language following held on pick-and-place variants featuring new objects (a pumpkin, a teddy bear, cup noodles, a paper bag, and so on).

The paper gives two reasons for this efficiency: the visual similarity of the two embodiments, and more fundamentally, that **IDM learning may be inherently more sample-efficient than direct policy learning** (Section 2.1). Failures, again, came mainly from video prediction errors rather than action extraction.

This result, however, is presented **only as qualitative results in Fig. 12, with no quantitative table**. "Maintained zero-shot generalization" is also an observation about object-level generalization, not new motions.

### 5.5 Ablation

Owing to compute constraints, all ablations were trained for 50K steps at batch 32 and evaluated only on the PnP Easy task (Table 4).

| Question | Comparison | task progress |
|---|---|---|
| Data diversity | Repetitive data (70 tasks) vs diverse data, 500 hours each | 33% vs **50%** |
| Model size | DreamZero 5B vs 14B | 21% vs **50%** |
| Same-size VLA | 5B / 14B (first half of the blocks of an 8B/32B VLM + DiT action module) | 0% / 0% |
| BD vs AR | 14B | 50% vs 50% |

- **Diversity** — for the same hours, diverse data is better. Even on easy pick-and-place
- **Size** — VLAs cannot digest heterogeneous data even when scaled up, and just hover near the object. Model capacity alone does not solve it
- **BD vs AR** — task progress is the same, but AR's motion is noticeably smoother, and inference is 3–4× faster thanks to the KV cache

> ### ⚠️ Fact check — cautions when reading the ablation
>
> - **HTML and PDF disagree.** Table 4 in the arXiv HTML version shows the VLA row as 50%, but the main text and PDF say 0%. It looks like an HTML conversion error.
> - **The "scaling" evidence is two points.** 5B and 14B, and in the single PnP Easy category at that. The paper itself admits the absence of a scaling law in Section 6.
> - **The identity of the 5B backbone is unclear.** The paper writes "Wan2.1-I2V-5B-480P," but Wan2.1's official checkpoints are 1.3B and 14B; 5B exists as Wan2.2-TI2V-5B (which uses a VAE with a different compression rate). If it is the latter, a difference in model family is mixed into the size comparison.

---

## 6. Positioning — among neighboring work

DreamZero sits at the intersection of several lineages.

| Lineage | Representative work | Relation to DreamZero |
|---|---|---|
| VLA | RT-2, OpenVLA, π0/π0.5, GR00T N1, Gemini Robotics | Baselines. The gap in the VLM prior is the starting point |
| Generate video, then extract actions at test time | UniPi (Du et al., 2023), AVDC (Ko et al., 2024), VLP (Du et al., 2024) | Pipelines where video generation and action extraction are separate |
| Video models as synthetic-data generators | DreamGen (Jang et al., 2025) | Uses video as a data generator. DreamZero uses it as the policy itself |
| Joint video-action, from scratch or VLA-based | GR-1, GR-2, UVA, etc. | Showed the benefit of joint prediction but do not use a web video prior |
| **Joint video-action, based on pretrained video diffusion** | Cosmos Policy, Video Generators are Robot Policies, mimic-video, Genie Envisioner, VPP | **The closest cousins.** The same WAM family |

Within the same WAM family, the axes on which the paper distinguishes itself are these.

| Design axis | DreamZero | Alternatives the paper names |
|---|---|---|
| Model composition | Single model, joint denoising | Video model + separate IDM (Li et al., 2026; Pai et al., 2025) |
| Timestep | Shared by video and action (separated only in Flash) | Separated per modality (Kim et al., 2026; Li et al., 2025a; Liao et al., 2025; Zhu et al., 2025) |
| Data | 500 hours, non-repetitive and heterogeneous | Mostly centered on repeated demonstrations |
| Generation | AR chunks + GT observation KV injection | BD (the ablation control) |

The name "world model" is used for several branches besides WAMs. So how do WAMs differ from world models like V-JEPA 2 or Dreamer?

> ### 🔗 A WAM is not a forward model but a joint distribution of "the future and the actions that produce it"
>
> Appendix A lays out the difference.
>
> | | What is modeled | Action output at inference |
> |---|---|---|
> | Latent world model (JEPA / V-JEPA 2, Dreamer) | $p(s_{t+1}\mid s_t,a_t)$ — forward dynamics | Requires goal-conditioned planning or search |
> | 3D point world model (PointWorld) | Action-conditioned 3D point flow | Requires explicit optimization such as MPPI |
> | **WAM** | $p(o_{t:t+H},a_{t:t+H}\mid o_{0:t},c)$ — joint | **Output directly**, no test-time optimization |
>
> A forward model answers "what happens if I take this action," so finding a good action requires search at inference. A WAM generates actions together with the future, so there is no search, and that is also what makes 7Hz control possible. In exchange, a WAM does not do the kind of planning that hypothesizes several candidate actions and compares their outcomes.

---

## 7. Limitations

**Limitations the paper acknowledges**

| Item | Detail |
|---|---|
| Scaling law | No systematic scaling curve over model size, data and compute |
| In-the-wild human data | Confirmed only with 12 minutes of in-lab data |
| Inference cost | 7Hz needs 2 GB200s. VLAs run at 20Hz+ on consumer GPUs |
| Long-horizon | A System 1 model with about 6 seconds of visual memory (at most 6.6 seconds per Appendix C). Needs a System 2 planner or a much longer context |
| High-precision work | Sub-cm tasks such as key insertion and precise assembly. Diversity-first data underrepresents dense demos |
| Embodiment | Hypothesis that higher-DOF robots will need more play data for IDM learning. No multi-embodiment pretraining |
| Memory | Stateful, but tasks requiring memory were not evaluated |

**Further points to note**

- **Evaluation design** — in-house real-robot evaluation, a partial-credit metric, 8 rollouts per task. The AgiBot data is not yet public, so reproduction is possible only via the DROID path.
- **Key designs unverified** — neither single vs separated model (Section 3.1) nor the contribution of GT observation injection (Section 3.3) has an isolating ablation.
- **Width of the embodiment gap** — both robots in the few-shot adaptation are bimanual parallel grippers. Adaptation to robots of very different form is unverified.
- **Reading the numbers** — see the fact check in Section 5.2 for the abstract's "2×," and Section 5.3 for the scope of the transfer experiment.

---

## 8. Closing — what this paper suggests

DreamZero's real contribution is not any particular trick. It is showing on real robots that **you can hand most of a robot policy to a video generation model and shrink the robot-specific learned part to a thin action readout**. The role of robot data then changes too: not filling modes by repeating the same task, but widening the IDM with frame pairs from diverse situations.

And this paper reads unusually well for someone with an LLM and diffusion background.

| Background | Counterpart seen in DreamZero |
|---|---|
| Diffusion | The action is **another track** synchronized to the video — the structure of joint video-audio generation with actions in place of audio. Flash extends the familiar technique of designing the training timestep distribution for few-step inference to a mismatch between modalities |
| LLM | Teacher forcing, causal masks and KV cache carried over as is. The difference is that the environment supplies GT even at inference, structurally eliminating exposure bias |
| Agentic AI | DreamZero is pure System 1. Tasks needing multi-step procedures still need a planner–executor split, and the paper itself names dual-system structures like Hi Robot as a complement |

Finally, there is an implication for data strategy. The proposition "task diversity over demonstrations per task" held in this paper **only when combined** with an architecture prior. On the same data, scratch VLAs learned almost nothing. How data is collected and how the model is built are a pair that cannot be evaluated separately.

---

## Appendix — Glossary

| Term | Definition |
|---|---|
| **WAM (World Action Model)** | A foundation model that jointly predicts future world states and actions in an aligned way. This name is used instead of "Video Action Model" in the sense that it can extend beyond video to touch, force and so on |
| **IDM (inverse dynamics model)** | A model that estimates, from a change of state, the action that caused it. $p(a\mid o,o')$ |
| **Teacher forcing** | In AR training, conditioning on the ground truth of the previous step (clean chunk) |
| **GT observation injection** | Writing real camera observations, instead of predicted video, into the KV cache at inference. Aligns inference conditions with training conditions |
| **DreamZero-Flash** | A variant trained further with a schedule that biases the video timestep toward high noise, enabling 1-step inference |
| **Task progress** | An evaluation metric that gives partial credit for stages of progress rather than completion |
| **Action chunk** | A run of consecutive actions generated in one inference pass. 48 steps (1.6 s) for AgiBot |

**Original** — [arXiv:2602.15922](https://arxiv.org/abs/2602.15922) · **Project page** — dreamzero0.github.io · **Code** — github.com/dreamzero0/dreamzero
