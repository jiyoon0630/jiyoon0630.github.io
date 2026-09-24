---
layout: paper
lang: en
ref: lapa-latent-action-pretraining
kind: paper-review
title: "Latent Action Pretraining from Videos (LAPA)"
date: 2026-02-03 12:00:00 -0800
paper_date: 2024-10-15
venue: "ICLR 2025 · arXiv:2410.11758"
tags: [VLA, Latent-Action, Video-Pretraining, VQ-VAE, Robot-Foundation-Model, Paper-Review]
authors: "Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, Minjoon Seo"
affiliations: "KAIST · University of Washington · Microsoft Research · NVIDIA · Allen Institute for AI"
summary: "Tokenize the change between two frames with a VQ-VAE to give action-free video pseudo-actions, pretrain a VLM on those tokens, then connect it to real actions with a small amount of robot labels. When the embodiment changes, this can beat pretraining on ground-truth actions."
paper_url: "https://arxiv.org/abs/2410.11758"
---

> **Core claim** — A VLA can be pretrained even on video with no robot action labels. Discretize the change between two frames into tokens with a VQ-VAE to attach pseudo-actions to the video, train a VLM to predict those tokens, and then connect it to real actions with a small amount of robot labels. And when the embodiment changes, this approach can beat pretraining on GT actions.

---

## Introduction

The standard recipe for a VLA is to fine-tune a large VLM on robot action data. RT-2 and OpenVLA sit within this frame, and OpenVLA was pretrained on about 970K trajectories from Open X-Embodiment. The problem is that nearly all of this data is collected by human teleoperation. Growing it takes robots, people and time, all at once.

The internet, on the other hand, holds a practically endless supply of video of people picking things up, covering them and knocking them over. LAPA proposes a way to use this video for VLA pretraining. This piece follows the paper's argument, building up concepts such as IDM and VQ-VAE where they are needed, so that a reader with a thin robotics background can see the reason behind each design.

---

## 1. The problem — video is abundant, so why can't VLAs use it?

### 1.1 Two walls

The paper states that there are two obstacles to using internet video for robot learning.

**⛔ Wall 1 — There are no action labels.** Video has only pixels. BC (behavior cloning) requires (observation, action) pairs, and in video the "action" slot is empty.

**⛔ Wall 2 — The bodies and environments differ.** A human hand and a 7-DoF gripper move differently, and scenes in web video differ from a robot workbench. Among robots the situation is similar. Each dataset has its own action representation (EE delta or joint, coordinate frame, control rate), so the paper points out that pretraining on GT actions can reduce positive transfer across datasets.

### 1.2 How existing approaches dealt with the walls

Organizing the paper's related work by Wall 1 gives the following.

| Line | Representative | What it gets from video | Limitation |
|---|---|---|---|
| Visual representation pretraining | R3M (Ego4D) | a good visual encoder | does not learn actions themselves |
| Video generation + IDM | UniPi | generates future frames (a video plan) and extracts actions with an IDM | plans go wrong over long horizons; the IDM needs labels |
| Human motion retargeting | based on hand pose estimation | hand trajectory → robot action | task-specific, or needs aligned human-robot data from the same environment |
| IDM pseudo-labels | VPT | an IDM trained on labels attaches actions to video | training the IDM itself needs labels, and the IDM collapses when the environment changes |
| Latent action | Genie, LAPO | latent actions extracted from observations | game domains, and not VLAs |

### 1.3 So the question the paper asks

> Can a VLA be pretrained on video alone, without robot action labels? Can that pretraining catch up with GT-action pretraining, or surpass it when the embodiment changes? Does it work with human video alone?

Wall 1 is broken through in §2–§3, and Wall 2 is gathered back in §4. That process is this paper's narrative.

---

## 2. Background — IDM, FDM, and treating action as a latent variable

### 2.1 The three functions of robot learning

| Function | Input → output | Question | Generative-model / LLM counterpart |
|---|---|---|---|
| **IDM** (inverse dynamics model) | $(x_t,\ x_{t+1}) \to a_t$ | what action caused this change? | labeler, tokenizer encoder |
| **FDM** (forward dynamics model) | $(x_t,\ a_t) \to x_{t+1}$ | what happens if I take this action? (= world model) | conditional generator |
| **Policy** (BC) | $(x_t,\ \ell) \to a_t$ | what should I do now to carry out the instruction? | language model |

- $x_t$ — image observation at time $t$
- $a_t$ — action (e.g., a 7-DoF end-effector delta)
- $\ell$ — language instruction

### 2.2 VPT's chicken and egg

The most direct way to use video is the VPT way. Use an IDM to attach action labels to every frame of the video, then train a policy with BC on those labels. But training the IDM ultimately requires $(x_t,\ x_{t+1},\ a_t)$ labels. The tool built to get rid of labels demands labels. On top of that, the IDM is trained on the few domains that have labels, so applied to differently shaped video it attaches wrong labels. This problem is actually confirmed in the experiments of §5.2.

### 2.3 LAPA's way out — action as a latent variable

LAPA's idea is simple. If $a$ is unknown, treat $a$ as a **latent variable $z$**, and bundle the IDM and FDM into one autoencoder trained jointly.

```
  (x_t, x_{t+H}) --[ IDM = encoder ]--> z_t --[ FDM = decoder, given x_t ]--> x_hat_{t+H}
                                         ^
                                         discrete bottleneck: s tokens, |C| codes each
```

- The encoder compresses what happened between the two frames into $z_t$ (the IDM role)
- The decoder reconstructs $x_{t+H}$ from $x_t$ and $z_t$ alone (the FDM role)
- No labels are used. The only loss is reconstruction error

But there is something odd here. The decoder already receives $x_t$ as input. Then couldn't the encoder stuff all of $x_{t+H}$'s information into $z_t$ and pass it along, driving the reconstruction loss to zero? In that case there would be no reason for $z_t$ to become an "action."

> ### 💡 What makes $z$ an "action" is the information bottleneck
>
> The key is the capacity of $z_t$. In the default setting, $z_t$ is a sequence of $s=4$ tokens, each chosen from a codebook of size $\lvert C\rvert=8$.
>
> $$\lvert\mathcal{Z}\rvert\ =\ \lvert C\rvert^{s}\ =\ 8^4\ =\ 4096\qquad\Rightarrow\qquad \log_2\lvert\mathcal{Z}\rvert\ =\ s\log_2\lvert C\rvert\ =\ 12\ \text{bits}$$
>
> - $\mathcal{Z}$ — the set of all possible latent actions
> - $\lvert C\rvert$ — codebook size (vocab), $s$ — number of tokens
>
> The information needed to reconstruct one image is incomparably larger than 12 bits. So the encoder cannot pass along all of $x_{t+H}$, and instead picks **only the 12 bits that reduce reconstruction loss the most**. It need not carry the static scene, which the decoder can already know from $x_t$. What remains is the largest change between the two frames, and in robot manipulation video that change is mostly the motion of the arm and hand.
>
> The structure points the same way. Since the vector being quantized is the **difference** of the two frame embeddings, $d_t = e_{t+H} - e_t$ (§3.1), the codes carry an inductive bias toward looking at change from the start.
>
> To be precise, though, what $z$ holds is **not "action" but "the most salient visual change."** In video where the camera moves, camera motion takes up the codes. This difference becomes a problem again in §5.6 and §7.

In a word, this model is **a tokenizer for actions**. The paper, too, likens it to BPE: it learns a vocabulary of "atomic motions" from data, without predefined action units such as EE position or joint angles. From the image-generation side, the difference is that a tokenizer like VQGAN compresses **a single image**, whereas this model compresses **the conditional change given the first frame**.

Turning the IDM from a label-regression problem into a conditional-compression problem — that is LAPA's key trick, and Wall 1 is broken through here.

---

## 3. Method — three stages

```
  STAGE 1: LAQ                STAGE 2: LATENT PRETRAIN      STAGE 3: ACTION FINETUNE
  ----------------------      ------------------------      ---------------------------
  video only, no labels       video + language              small robot set w/ actions
  (x_t, x_{t+H}) -> z_t       VLM(x_t, l) -> z_t            VLM(x_t, l) -> a_t
  VQ-VAE (~300M)              7B LWM + latent head          latent head dropped
  enc = IDM, dec = FDM        BC on pseudo labels           new head: 7 dims x 256 bins
```

- **Stage 1 (LAQ)** — train a latent action tokenizer on unlabeled video
- **Stage 2 (Latent Pretraining)** — use that tokenizer to attach pseudo-actions to the video, and train a VLM with BC to predict them
- **Stage 3 (Action Finetuning)** — connect the latents to real actions with a small amount of robot labels

Stages 1 and 2 use the same pretraining dataset. The paper calls any model that has gone through latent pretraining LAPA.

### 3.1 Latent Action Quantization (LAQ)

A model of about 300M parameters that reproduces Genie's latent action model with a C-ViViT architecture and changes a few things.

```
  x_t, x_{t+H} --> [patch emb -> spatial tf -> causal tf] --> e_t, e_{t+H}
                                                                   |
                                                      d_t = e_{t+H} - e_t
                                                                   |
                                                     [VQ: nearest code] --> z_t
                                                                             |
  x_t --> [patch emb, stop-grad] --> [DECODER: cross-attn + spatial tf] <----+
                                                     |
                                                     v
                                               x_hat_{t+H}    (L2 loss vs x_{t+H})
```

**⓵ Difference vector and quantization**

$$d_t\ =\ e_{t+H}\ -\ e_t,\qquad z_t\ =\ \arg\min_{z_k\in C}\ \lVert d_t - z_k\rVert^2$$

- $x_t,\ x_{t+H}$ — the current frame and the frame a window $H$ later
- $e_t,\ e_{t+H}$ — continuous embeddings of the two frames after the patch embedding, spatial transformer and causal transformer
- $d_t$ — the difference of the two embeddings; the target of quantization
- $C=\{z_k\}$ — the codebook, $z_t$ — the code closest to $d_t$
- The number of tokens $s$ is set by the kernel, stride and padding of the CNN just before quantization

**⓶ NSVQ — replace quantization error with noise**

argmin is not differentiable. Standard VQ-VAE works around this with the straight-through estimator (STE), but LAPA uses NSVQ to avoid the gradient collapse common in VQ-VAEs.

$$\hat d_t\ =\ d_t\ +\ \frac{\lVert d_t - z_t\rVert}{\lVert v\rVert}\,v,\qquad v\sim\mathcal N(0,\ I)$$

- $\hat d_t$ — the vector actually fed to the decoder during training
- $v$ — a random direction vector, normalized and then scaled to the size of the quantization error

| | Standard VQ-VAE (STE) | NSVQ |
|---|---|---|
| Forward pass | quantized $z_t$ | $d_t$ + random noise of **the same size** as the quantization error |
| Backward pass | an approximation that copies $z_t$'s gradient to $d_t$ | flows directly to $d_t$; the codebook is updated through the $\lVert d_t - z_t\rVert$ term |
| Codebook utilization | commitment loss | **codebook replacement** early in training relocates unused codes |

The design mimics the size of the error that quantization creates without cutting the gradient path.

**⓷ Decoder and loss**

$$\hat x_{t+H}\ =\ D\Big(\mathrm{CrossAttn}\big(\mathrm{sg}[p_t];\ \hat d_t\big)\Big),\qquad \mathcal L_{\text{LAQ}}\ =\ \lVert x_{t+H} - \hat x_{t+H}\rVert_2^2$$

- $p_t$ — patch embedding of $x_t$, $\mathrm{sg}[\cdot]$ — stop-gradient
- $\mathrm{CrossAttn}(\cdot\,;\,\cdot)$ — query is $\mathrm{sg}[p_t]$; key and value are $\hat d_t$
- $D$ — a decoder made of spatial transformers only. With only two frames as input, there is no temporal part

**⓸ Reasons for the design choices**

| Choice | Reason |
|---|---|
| Cross-attention (Genie uses additive embedding) | empirically captures more meaningful latent actions (as the paper reports) |
| $\mathrm{sg}[p_t]$ | prevents representation collapse. The specific mechanism is not explained in the paper |
| Only two frames as input (Genie uses several past frames) | computational cost. Adding past observations is left for future work |
| Window $H$ — 0.6 s for robots, 2.4 s for human video | human video has many static frames. $H=3$ on Bridge (5Hz) |

Once training is done, **the encoder becomes a latent IDM and the decoder a latent world model**. The encoder is used as the labeler in Stage 2, and the decoder for the neural rollouts of §5.6.

### 3.2 Latent Pretraining

The LAQ encoder attaches a $z_t$ label to every frame of the pretraining video, and a VLM is trained to predict it.

$$\mathcal L_{\text{latent}}(\theta)\ =\ -\,\mathbb E_{(x_t,\ \ell,\ z_t)}\Big[\sum_{k=1}^{s}\log p_\theta\big(z_t^{(k)}\mid x_t,\ \ell,\ z_t^{(<k)}\big)\Big]$$

- $z_t^{(k)}$ — the $k$-th token of the latent action
- $\ell$ — the video clip's language instruction
- $\theta$ — VLM parameters. The backbone is LWM-Chat-1M (7B)

The paper does not state the loss explicitly. The equation above translates its description, "BC in which the VLM predicts latent actions," into causal-LM token prediction. In the implementation, instead of the LM head, a **separate latent action head** (a single MLP layer, output size $\lvert C\rvert$) is attached; the vision encoder is frozen and the language model is unfrozen. The input is one current image and the instruction.

It is the same structure as LLM pretraining. The only difference is that the next "word" is not text but a token representing **the change that will happen next**. Since no action labels are needed, any video with a language instruction attached can become pretraining data.

> ### ⚠️ Fact check — inconsistent notation between $x_{t+1}$ and $x_{t+H}$
>
> §3.1 defines the encoder input as $(x_t,\ x_{t+H})$, but §3.2 writes that "$x_t$ is labeled given $x_{t+1}$." Since Appendix F states $H=3$ as the default on Bridge, the $x_{t+1}$ of §3.2 should be read as loose notation for "the next window frame."

### 3.3 Action Finetuning

A real robot cannot execute latent actions. So the model is fine-tuned on a small number of labeled trajectories (delta EE).

$$\mathcal L_{\text{FT}}(\theta)\ =\ -\,\mathbb E_{(x_t,\ \ell,\ a_t)}\Big[\sum_{j=1}^{7}\log p_\theta\big(b_t^{(j)}\mid x_t,\ \ell,\ b_t^{(<j)}\big)\Big],\qquad b_t^{(j)}\ =\ \mathrm{bin}_j\big(a_t^{(j)}\big)$$

- $a_t^{(j)}$ — the $j$-th dimension of the 7-DoF action
- $\mathrm{bin}_j$ — per-dimension discretization that puts an equal amount of data into each bin (the OpenVLA / RT-2 way, 256 bins)
- $b_t^{(j)}$ — the discretized action token

This equation is not stated in the paper either; it translates the description that the paper follows the OpenVLA approach.

There is a choice worth noting here. **The latent action head is discarded and a new action head is initialized.** The paper also tried the LAPO-style approach of keeping the latent head and adding a "latent → real action" decoding head on top, but reports that re-initialization worked better (speculating that this is thanks to the 7B model size). Freezing the vision encoder and unfreezing the whole language model is the same as in pretraining.

A question follows. If the latent head learned so carefully in pretraining is thrown away, what did pretraining leave behind?

> ### 💡 What transfers is not $z$ but the backbone's representation of "what should happen next"
>
> The output of latent pretraining can be split into two parts.
>
> $$p_\theta\big(z_t^{(k)}\mid x_t,\ \ell,\ z_t^{(<k)}\big)\ =\ \mathrm{softmax}\Big(\underbrace{W}_{\text{latent head}}\ \underbrace{h_\theta\big(x_t,\ \ell,\ z_t^{(<k)}\big)}_{\text{backbone}}\Big)$$
>
> - $h_\theta$ — hidden state of the 7B backbone. **Kept in finetuning**
> - $W$ — the latent action head. **Discarded in finetuning**
>
> The head is only a thin reader; the knowledge lives in $h_\theta$. It is a representation from which "what change should happen next to carry out this instruction in this scene" can be read out easily. The new head only has to learn **how to read the same intent in this robot's 7-DoF coordinates** on top of that representation. The backbone is unfrozen too, so the representation itself also adjusts a little.
>
> In LLM terms, it is the same structure as removing the LM head after next-token pretraining and fine-tuning with a new head attached. The output space of the pretraining objective is discarded, and only the representation remains.
>
> This interpretation makes one testable prediction. Abilities that come from pretraining should be strong in coarse intent such as **which object, in which direction**, while fine motor skill such as **exactly when to grasp** should depend on the amount of finetuning labels. The real-world results of §5.4 show exactly this pattern.

---

## 4. Why it works

The paper offers three explanations.

**⓵ A shared action space independent of embodiment** — Latent actions are defined by pixel change, so WidowX, Franka or a human hand are all expressed in the same vocabulary. In fact, feeding the same code to different embodiments in Open-X reconstructs similar motions (Fig. 6). GT-action pretraining, by contrast, must mix a different action space for each dataset, which reduces positive transfer.

**⓶ No prior assumptions about action units are needed** — It does not decide whether to use EE or joints, or at what resolution. It is simply trained end-to-end to best capture the delta between consecutive observations.

**⓷ Small output space → fast learning** — The pretraining output space is $8^4$, while OpenVLA's action space is $256^7$. The paper reports that every LAPA model reached its best performance within one epoch.

The three explanations are not separate. Tie them together with the division of labor seen in §3.3 — "coarse intent from pretraining, fine motor skill from finetuning" — and what LAPA is can be summed up in one sentence.

> ### 📌 The core of LAPA — learn "what to do" and "how to do it with this body" from different data
>
> | Knowledge | Where it is learned | Labels | Experimental evidence (§5.4) |
> |---|---|---|---|
> | What to do — target selection, language conditioning, rough trajectory | pretraining (any video) | not needed | unseen instruction 48.5 vs OpenVLA 43.4, reaching 83.3% vs 66.7% |
> | How to do it with this body — grasp timing, contact | finetuning (robot labels) | needed | early-grasp failures, pick&place 45.8 vs 54.2 |
>
> The two walls from the introduction are gathered back here.
>
> - **Wall 1 (no labels)** — LAQ builds an IDM without labels and gets around it. VPT's chicken-and-egg problem disappears.
> - **Wall 2 (embodiment gap)** — Pretraining happens in an embodiment-independent latent space, and the embodiment-specific part is **deferred** to the new head in finetuning. It does not remove the gap; it moves only the part where the gap matters to where the labels are.

---

## 5. Experiments

### 5.1 Setup and baselines

| Environment | Action | Pretraining data | Finetune | What it measures |
|---|---|---|---|---|
| Language Table (sim) | 2-DoF pushing | sim 181k / real 442k | 1k or 7k | in-domain, cross-task, cross-env |
| SIMPLER (sim) | 7-DoF WidowX | Bridgev2 60k / Sthv2 | 100 | in-domain, human → robot |
| Real world | 7-DoF Franka | Bridgev2 / Open-X 970k / Sthv2 | 450 (3 tasks × 150) | cross-embodiment, multi-embodiment |

| Baseline | Pretraining | What it controls for |
|---|---|---|
| Scratch | none (same LWM backbone) | the net effect of pretraining |
| UniPi | video diffusion + IDM | the "predict pixels" alternative |
| VPT | pretrains **the same VLM** on pseudo-labels from an IDM trained on labels | latent action vs pseudo-GT action |
| ActionVLA | pretrains **the same LWM backbone** on GT actions | the de facto upper bound |
| OpenVLA | Open-X GT actions, Prismatic backbone | external SOTA |

### 5.2 Language Table — how far 0.5% of the labels can go

Table 1, success rate (%).

| | In-domain Seen | In-domain Unseen | Cross-task Seen | Cross-task Unseen | Cross-env Seen | Cross-env Unseen |
|---|---|---|---|---|---|---|
| Scratch | 15.6 | 15.2 | 27.2 | 22.4 | 15.6 | 15.2 |
| UniPi | 22.0 | 13.2 | 20.8 | 16.0 | 13.6 | 12.0 |
| VPT | 44.0 | 32.8 | 72.0 | **60.8** | 18.0 | 18.4 |
| **LAPA** | 62.0 | 49.6 | 73.2 | 54.8 | 33.6 | 29.6 |
| ActionVLA | 77.0 | 58.8 | 77.0 | 58.8 | 64.8 | 54.0 |

- **In-domain** — pretrained on sim 181k, fine-tuned on 1k across 5 tasks
- **Cross-task** — same pretraining, fine-tuned only on 7k of a separate task, then evaluated on all 5 tasks
- **Cross-env** — pretrained on real 442k, fine-tuned on sim 1k (the real-to-sim gap)

In-domain, LAPA uses only 0.5% (1k/181k) of the labels yet far exceeds Scratch and narrows the gap to ActionVLA. What is interesting is the relationship with VPT. In cross-task unseen, VPT beats LAPA (60.8 vs 54.8), but in cross-env VPT collapses to Scratch level (18.0). It is the same "pseudo-label → BC" structure, so why do they split so sharply?

> ### 💡 What separates VPT from LAPA is "what data the IDM was trained on"
>
> VPT's IDM is trained on **the labeled finetuning data** (1k or 7k trajectories) and then applied to **the pretraining video**. LAPA's LAQ is trained on **the pretraining video itself**.
>
> | | VPT | LAPA |
> |---|---|---|
> | IDM training data | labeled trajectories from the target domain | pretraining video (no labels) |
> | Label space | the target robot's GT actions | a latent learned from data |
> | What gets labeled | may be OOD for the IDM | in-distribution by definition |
> | When domains are aligned | at labeling time | at finetuning time (new head) |
>
> - **cross-task** — pretraining and finetuning are in the same sim environment, and there are plenty of labels (7k). VPT's IDM can produce accurate labels in the GT space, so VPT has the advantage. The paper, too, explains it as the IDM having become accurate thanks to many labels.
> - **cross-env** — an IDM trained on sim 1k is applied to 440K real trajectories, and the labels break. The paper considers the IDM not robust to environment shift.
>
> VPT aligns domains at labeling time, so when the domains are misaligned, the entire pretraining dataset is contaminated. LAPA defers the alignment to finetuning, so it carries no such risk. This is the VPT weakness previewed in §2.2, and also the cleanest evidence that the design of §4, "defer Wall 2 to finetuning," works.

Still, ActionVLA on the same backbone is higher than LAPA in every setting, and the gap is large in cross-env (64.8 vs 33.6). If the embodiment is the same, GT labels are still better.

> ### ⚠️ Fact check — cross-task ActionVLA is not a separate run
>
> The ActionVLA columns of Appendix Tables 7 and 8 are identical, down to the per-task numbers, to in-domain Tables 5 and 6. The cross-task "upper bound" appears to reuse the in-domain result rather than being rerun under the separate-7k finetuning condition. Table 1 and the appendix averages also differ slightly (ActionVLA 77.0 vs 76.8, VPT 44.0 vs 43.6).

### 5.3 SIMPLER — predict pixels, and the translator becomes the bottleneck

Table 11. Pretrained on Bridgev2, then fine-tuned on 100 trajectories, success rate (%).

| Scratch | UniPi | VPT | **LAPA** | ActionVLA | OpenVLA |
|---|---|---|---|---|---|
| 34.4 | 1.3 | 51.0 | **57.3** | 63.5 | 36.4 |

UniPi is near zero. According to the paper, the video plans themselves are fairly accurate, but the IDM trained on 100 trajectories cannot properly predict continuous 7-DoF actions, so it fails to grasp the object within the step limit. **The strategy of predicting pixels and translating them into actions later collapses when the translator (IDM) lacks labels.** The paper explains OpenVLA's low score as an already known phenomenon: SIMPLER's real-to-sim transfer problem.

One point is worth noting. SIMPLER's 100 finetuning trajectories were filtered from the successful rollouts of an LWM-based VLA trained on Bridgev2. They look like, in effect, the action distribution of the ActionVLA family, so this bias works against LAPA.

### 5.4 Real-world Franka — the headline result

3 tasks (knock, cover, pick&place) × 3 generalization types (unseen combinations, unseen objects, unseen instructions) × 6 rollouts, for 54 rollouts per model. Scores use partial credit (e.g., for pick&place: reach 0.25, grasp 0.5, move 0.75, success 1).

Fig. 3 · Tables 13–16, success rate (%).

| Model | Pretraining labels | Knock | Cover | Pick&Place | **Avg** |
|---|---|---|---|---|---|
| Scratch | — | 13.9 | 38.7 | 11.1 | 21.2 |
| ActionVLA (Bridge) | GT | 33.3 | 42.3 | 22.2 | 32.6 |
| OpenVLA (Bridge) | GT | 25.0 | 47.8 | 19.4 | 30.8 |
| **LAPA (Bridge)** | none | 25.0 | 42.4 | **43.1** | **36.8** |
| OpenVLA (Open-X) | GT | 38.9 | 38.6 | **54.2** | 43.9 |
| **LAPA (Open-X)** | none | **52.8** | **51.7** | 45.8 | **50.1** |

**⓵ Bridge pretraining (WidowX → Franka)** — On the same LWM backbone, LAPA beats ActionVLA (36.8 vs 32.6). The paper's hypothesis is that since most of Bridge is pick-and-place, GT-action pretraining overfit to the WidowX action space, which got in the way when moving to Franka. Indeed, the gap comes almost entirely from the single pick&place task (43.1 vs 22.2). Since the backbone and data are the same and only the embodiment changes, this is **the cleanest evidence for the claim that "the side that dropped the labels wins."**

**⓶ Open-X pretraining** — Growing the data from Bridge to Open-X improves both models, and LAPA beats OpenVLA on average (50.1 vs 43.9). By generalization type it leads in all three (Table 2: unseen combinations 57.8 vs 46.2, unseen objects 43.9 vs 42.1, unseen instructions 48.5 vs 43.4). The biggest difference comes on the language-conditioning side, while the difference on unseen objects is only 1.8%p.

But it loses on pick&place (45.8 vs 54.2). The paper reports that most failures are grasps that come too early, and that on reaching LAPA is actually higher (83.33% vs 66.67%). The prediction made in §3.3 holds exactly. Pretraining tells the model which object to go to, but a grasp, which happens only once or twice per trajectory, is too little to learn from 150 labels.

> ### ⚠️ Fact check — how to read "+6.22% over the SOTA VLA"
>
> The +6.22 in the contributions list is 50.09 − 43.87 from Table 16. The number is correct, but four things must be considered with it.
>
> - **⓵ Units** — it is not a relative % but **%p** on partial-credit scores. By strict success rate it is 35.19 vs 27.78.
> - **⓶ Backbone confound** — LAPA uses LWM-Chat-1M, OpenVLA the Prismatic backbone. There is no ActionVLA GT-pretrained on Open-X with the same backbone. The paper, too, attributes part of the efficiency to the LWM backbone (§5.6).
> - **⓷ Differences in finetuning recipe** — OpenVLA was trained with LoRA, batch 32, up to 95% train action accuracy; LAPA unfroze the entire LM and used batch 128 with image augmentation (Appendix C). The paper states that LoRA and full FT were similar for OpenVLA, but the augmentation difference remains.
> - **⓸ Sample size** — 6 rollouts per cell. In the paired comparison (Appendix D), LAPA wins 31.5%, OpenVLA wins 16.7%, and 51.9% are ties.
>
> So rather than "it beat OpenVLA," the more robust claim in this paper is **"on the same backbone, it beats GT pretraining when the embodiment changes"** (the Bridge comparison above).

### 5.5 Human video alone — Something-Something V2

This section is the paper's original motivation. Pretraining uses only Something-Something V2 (Sthv2, about 220K clips of people handling everyday objects), with the window set to 2.4 seconds.

| Evaluation | Scratch | UniPi | VPT | **LAPA (Human)** | Reference |
|---|---|---|---|---|---|
| SIMPLER (Table 12) | 34.4 | 0.7 | 45.8 | **52.1** | LAPA (Bridge) 57.3, 10% data 50.0 |
| Real-world average (Table 16) | 21.2 | — | — | **34.0** | OpenVLA (Bridge) 30.8 |

ActionVLA cannot be trained at all, because human video has no robot action labels.

LAPA pretrained on human video alone beats OpenVLA pretrained on robot data (Bridge) in the real-world average. By task, though, the spread is large: knock 30.6, cover 47.9, pick&place 23.6. The paper explains this by the task distribution of the pretraining data.

Table 4, number of pretraining trajectories for the same tasks as the evaluation tasks (by lexical matching).

| Task | Bridgev2 | Open-X | Sthv2 |
|---|---|---|---|
| Knocking | 2 | 7,969 | 6,655 |
| Covering | 898 | 5,026 | 6,824 |
| Pick & Place | 10,892 | 911,166 | 3,272 |

Bridge has practically no knocking, and Sthv2 has a lot. So even though the embodiment gap is larger, Sthv2 does better on knocking (30.6 vs LAPA (Bridge) 25.0). Conversely, pick&place is far more common in Bridge, and the result flips (23.6 vs 43.1). The observation is that **how much of a skill the pretraining video contained may matter more than matching embodiment**. The paper acknowledges that lexical matching is a rough estimate.

> ### ⚠️ Fact check — the distance between "internet-scale video" and the actual experiments
>
> The abstract and intro put forward internet-scale video as the motivation, but the human video actually used is Sthv2: short, curated, crowdsourced clips with templated captions, some distance from unedited web video. Scaling on human video, too, has only two points, 10% and 100% (which the paper also acknowledges as future work). Expansion to web scale is not something this paper showed; it is an extrapolation.

### 5.6 Efficiency, scaling, and what the latent actually holds

**Pretraining efficiency** — LAPA (Open-X) took 34 hours on 8 H100s (272 H100-hours); OpenVLA took 21,500 A100-hours. The paper sees this efficiency as coming from two places. One is that the LWM backbone was pretrained on video next-frame generation: with the same Bridge data and the same objective, the LWM-based ActionVLA reached its optimum in 3 epochs, while the Prismatic-based OpenVLA took 30. The other is the small latent output space.

> ### ⚠️ Fact check — the basis for "more than 30× efficiency"
>
> The paper assumes an H100 is 2–3× faster than an A100. Calculating as is gives between 21,500 / (272 × 3) ≈ 26× and 21,500 / (272 × 2) ≈ 40×, and the "over 30x" in the contributions list takes the middle of this range or above. Whether the 272 hours include the cost of training LAQ is not stated. Also, the $8^4$ vs $256^7$ comparison applies only to the pretraining output space; after finetuning, LAPA outputs the same $256^7$ space as OpenVLA.

**Scaling (Fig. 5)** — Four axes are scaled up: LAQ model size, data fraction, latent token length and vocab. Note that "model scaling" here refers not to the 7B VLA but to **the size of LAQ** (30M → 300M). Every axis improves as it grows, but the optimal latent space depends on the action complexity of the data. On the visually simple Language Table (2-DoF), increasing the vocab was far more effective than increasing the token length (Fig. 16). The window $H$ gives stable results unless it is extremely large, which the paper attributes to the 300M-scale LAQ being unable to model very large visual changes. Even with less finetuning data, LAPA consistently leads Scratch (Fig. 15b).

**What the latent actually holds (paper §5.2, Appendix E)** — The difference between "action" and "visual change" previewed in the callout of §2.3 shows up here.

| Data | Observation |
|---|---|
| Language Table (vocab 8, length 1) | the 8 codes correspond to 8 directions (left-forward, left-back, right-back, slightly right, right, back, stop, forward) and cluster cleanly in the GT 2D action space (Figs. 12–13) |
| Open-X | feeding the same code reconstructs similar motions even across different embodiments (Fig. 6) |
| Sthv2 (egocentric) | the codes capture not only hand motion but also **camera motion** (Fig. 14) |

The third row is direct evidence for the information-bottleneck interpretation. $z$ holds not action but the most salient visual change, and in egocentric video the camera movement that comes from the head moving is part of that change. The paper presents this as an advantage that could extend to navigation and the like, but from a manipulation policy's standpoint it also means part of the code capacity goes to signals unrelated to action.

**Neural rollouts (Fig. 7)** — LAPA with pretraining only emits latent actions, and the LAQ decoder generates the next frame, closing the loop. Given the instruction "take the broccoli out of the pot," it generates a video of the arm approaching the broccoli and lifting it. The dual role of §3.1, "encoder = IDM, decoder = world model," combines with the policy to become **a purely neural simulator**. The paper even looks ahead to scaling test-time compute by generating several plans and picking the best one, but the evidence presented is a single qualitative example.

---

## 6. Positioning — among neighboring work

LAPA sits at the intersection of three lineages.

**⓵ VLA** — RT-2, OpenVLA, Octo. Their dependence on action labels is LAPA's starting point.

**⓶ Robot learning from video** — representation pretraining (R3M), video generation (UniPi), human motion retargeting, IDM pseudo-labels (VPT).

**⓷ Latent action** — Genie (an interactive world model), ILPO and LAPO (game policies).

The closest cousins are VPT and Genie / LAPO.

| | UniPi | VPT | Genie / LAPO | **LAPA** |
|---|---|---|---|---|
| What it extracts or predicts from video | future **pixels** | **pseudo-GT actions** | **latent actions** | **latent action tokens** |
| Labels for IDM training | needed | needed | not needed | not needed |
| Final model | video diffusion + IDM | BC policy | Genie a world model, LAPO a game policy | 7B monolithic VLA |
| Domain | robots | Minecraft in the original paper (reimplemented with the same VLM in this paper) | 2D games, Procgen | real-robot manipulation |

To place it in one sentence: **LAPA takes VPT's pipeline (label the video → BC pretraining), replaces the label-hungry IDM with Genie's unsupervised latent action model, and scales the policy up to a VLM.**

It should also be distinguished from another line that compresses actions into latents (Play-LMP, VQ-BeT, QueST). Those tokenize **GT actions** with the aim of handling multimodality, while LAPA derives actions from **observations**. Both can be called "action tokenizers," but their inputs differ.

Viewed along the axis "what to predict from video," the tradeoff becomes sharp.

| Prediction target | Information | When labels are needed | Weakness (by this paper's experiments) |
|---|---|---|---|
| Pixels (UniPi) | richest | at the translation (IDM) stage | collapses when the translator lacks labels (SIMPLER 1.3) |
| Pseudo-GT actions (VPT) | the target robot's space | at the labeling stage | labels are contaminated when domains misalign (cross-env 18.0) |
| Latent action (LAPA) | 12 bits per step | at the finetuning stage | weak on fine motor skill (early grasp) |

So how has the ingredient called latent action been used since? LAPA kept the latent as discrete tokens and threw it away, head and all, at finetuning. Discretization is necessary as the bottleneck for training LAQ, but there is no separate reason for the policy to predict those discrete tokens as is.

> ### 🔗 GR00T N1 (NVIDIA, arXiv:2503.14734) — a later adoption that carried latent actions as continuous embeddings
>
> GR00T N1 applies LAPA's VQ-VAE latent action model as is to human egocentric video without action labels and to generated neural trajectories. What it hands to the policy, though, is different.
>
> | | LAPA | GR00T N1 |
> |---|---|---|
> | Latent the policy receives | discrete codes ($8^4$) | the continuous embedding **before** quantization |
> | Policy loss | token cross-entropy (autoregressive) | flow matching (the same loss as for real actions) |
> | Relation to real action data | pretraining on latents only, head replaced in finetuning | mixed into the same pretraining, but treated as a separate "LAPA" embodiment |
>
> The inductive bias of the discrete bottleneck was already obtained at the LAQ training stage, so it can be read as an evolution toward handing the policy a continuous representation with less information loss.

---

## 7. Limitations

Here are both what the paper states itself and what is worth adding while reading.

**Limitations the paper acknowledges**

- **Fine motor skill** — it is weaker than GT-action pretraining on fine-grained motions such as grasping. The paper expects a larger latent space to help.
- **Inference latency** — as a 7B autoregressive VLA, it has real-time inference latency. It mentions as an alternative a hierarchical structure in which a small head emits actions at a higher frequency.
- **Video beyond manipulation** — application to autonomous driving, navigation, landscape video and the like was not explored.

**Further points to raise**

- **"Action" ≠ "visual change"** — as seen in §2.3 and §5.6, $z$ holds the most salient visual change. Camera ego-motion and the movement of other people and objects also take up codes, and this contamination will grow worse the more unedited the web video. It is a real obstacle to "web-scale expansion."
- **The bottleneck was moved, not removed** — latent pretraining needs "video + language instruction" pairs. Sthv2 has templated captions, but web video lacks not only action labels but also task-level instructions. The label bottleneck moved from action to language.
- **Hand tuning per data source** — the window $H$ was set separately: 0.6 s for robots, 2.4 s for human video. With heterogeneous web video, this knob would have to be set per source.
- **Confounds in the headline comparison** — as seen in §5.4, backbone and finetuning recipe are mixed in. The only case where latent pretraining beat GT pretraining on the same backbone is Bridge → Franka, and even that gap comes essentially from one task.
- **Little input information** — LAQ sees only two frames, and the policy sees only one image and the instruction. There is no validation on situations that need temporal context, such as occlusion or velocity.

---

## 8. Closing — what this paper suggests

LAPA's contribution is neither the VQ-VAE nor the VLA; both were existing ingredients. The real contribution is **turning the IDM into a label-free conditional-compression problem, and thereby turning unlabeled video into a VLA pretraining corpus**.

And this paper reads unusually well for someone with an LLM and generative-model background.

- **LAQ = an action tokenizer** — where VQGAN compresses a single image, LAQ compresses the change given the first frame.
- **Latent pretraining = next-token pretraining on an unlabeled corpus** — except that instead of text it predicts "next change" tokens.
- **Action finetuning = SFT with the head swapped** — the same even down to discarding the pretraining output space and keeping only the representation.
- **Encoder = IDM, decoder = world model** — the policy and the world model come out of one round of unsupervised learning together. From the world-model side, it is also a way to obtain an action-conditioned video generator without action labels.

Above all, the question Table 4 of §5.5 raises is a big one. If downstream performance follows **the skill distribution of the pretraining video** more than embodiment match, the data strategy for robot foundation models shifts from "how much robot data to collect" to **"how much video containing which skills to collect."** This is especially so in domains where unlabeled but skill-dense video has piled up, such as footage of on-site work.

---

## Appendix — Glossary

| Term | Definition |
|---|---|
| **latent action** | a token sequence that discretizes the change between two frames with a VQ-VAE. Default setting: length 4, vocab 8 ($8^4$) |
| **LAQ** (Latent Action Quantization) | a VQ-VAE that learns latent actions without supervision. The encoder is a latent IDM, the decoder a latent world model |
| **IDM** (inverse dynamics model) | $(x_t,\ x_{t+1}) \to a_t$. Estimates the action from the change in observation |
| **FDM** (forward dynamics model) | $(x_t,\ a_t) \to x_{t+1}$. A world model that predicts the result of an action |
| **NSVQ** | a VQ training technique that replaces quantization error with random noise of the same size to keep the gradient path |
| **latent pretraining** | BC pretraining in which the VLM predicts latent actions from $(x_t,\ \ell)$ |
| **action finetuning** | discarding the latent head and learning real robot actions (256 bins × 7 dimensions) with a new head |
| **ActionVLA** | a baseline that pretrains the same LWM backbone on GT actions; the de facto upper bound |
| **partial success** | a real-world evaluation scheme that gives partial credit for sub-stages (reach, grasp, move, success) |

**Original** — [arXiv:2410.11758](https://arxiv.org/abs/2410.11758) · **Project page** — latentactionpretraining.github.io
