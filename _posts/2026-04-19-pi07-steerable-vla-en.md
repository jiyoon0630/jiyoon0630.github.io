---
layout: paper
lang: en
ref: pi07-steerable-vla
kind: paper-review
title: "π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities"
date: 2026-04-19 17:00:00 -0700
paper_date: 2026-04-16
venue: "arXiv preprint · arXiv:2604.15483"
tags: [VLA, Robot-Foundation-Model, Context-Conditioning, World-Model, Cross-Embodiment, Paper-Review]
authors: "Physical Intelligence — Bo Ai, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Greg Balke, Kevin Black, George Bokinsky, Shihao Cao, Thomas Charbonnier, Vedant Choudhary, Foster Collins, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Maitrayee Dhaka, Jared DiCarlo, Danny Driess, Michael Equi, Adnan Esmail, Yunhao Fang, Chelsea Finn, Catherine Glossop, Thomas Godden, Ivan Goryachev, Lachlan Groom, Haroun Habeeb, Hunter Hancock, Karol Hausman, Gashon Hussein, Victor Hwang, Brian Ichter, Connor Jacobsen, Szymon Jakubczak, Rowan Jen, Tim Jones, Gregg Kammerer, Ben Katz, Liyiming Ke, Mairbek Khadikov, Chandra Kuchi, Marinda Lamb, Devin LeBlanc, Brendon LeCount, Sergey Levine, Xinyu Li, Adrian Li-Bell, Vladislav Lialin, Zhonglin Liang, Wallace Lim, Yao Lu, Enyu Luo, Vishnu Mano, Nandan Marwaha, Aikys Mongush, Liam Murphy, Suraj Nair, Tyler Patterson, Karl Pertsch, Allen Z. Ren, Gavin Schelske, Charvi Sharma, Baifeng Shi, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, Will Stoeckle, Jiaming Tang, Jimmy Tanner, Shalom Tekeste, Marcel Torne, Kyle Vedder, Quan Vuong, Anna Walling, Haohuan Wang, Jason Wang, XuDong Wang, Chris Whalen, Samuel Whitmore, Blake Williams, Charles Xu, Sukwon Yoo, Lili Yu, Wuming Zhang, Zhuoyang Zhang, Ury Zhilinsky"
affiliations: "Physical Intelligence"
summary: "Describe, don't filter — label each episode with how it was done (quality, speed, mistakes, control mode, subgoal images), and one generalist VLA absorbs even failures and RL rollouts, reaching specialist-level skill and compositional generalization together."
paper_url: "https://arxiv.org/abs/2604.15483"
---

> **Core claim** — What it takes to use heterogeneous robot data is not stricter filtering but **richer context**. Condition each episode not only on "what" but on "how" (quality, speed, mistakes, control mode, goal images), and even failed episodes and RL rollouts can be absorbed into one model instead of thrown away. Specify that context at inference time, and a single generalist shows specialist-level dexterity and compositional generalization together.

---

## Introduction

An LLM that knows how to translate and knows how to output JSON will return a translation as JSON, even if that combination never appeared in its training data. This kind of **compositional generalization** is the reason we call foundation models generalists. Robot foundation models, that is, VLAs, have not yet shown this ability.

π0.7 is a 5B-scale VLA from Physical Intelligence, released with the claim of the "first signs" of it. It is not a paper about a new architecture. The paper defines its own contribution as "a methodology that lets VLAs use more diverse data, and an empirical analysis of it." The core idea is one thing: put "how" into the prompt, not just "what."

This post follows the paper's argument, assumes a reader with an LLM and diffusion background, and builds up the robot-learning concepts as they are needed. Readers who know diffusion will notice that many of this paper's design choices are devices already seen in text-to-image. I lay down those bridges along the way.

---

## 1. The problem — why generalist VLAs are not yet generalists

### 1.1 Two symptoms

The paper diagnoses the limits of existing VLAs as two symptoms.

**⓵ No composition** — they cannot recombine learned skills to solve a new task. Operating an unfamiliar kitchen appliance or using a new tool falls here.

**⓶ Dependence on specialists** — even tasks they were trained on are not performed fluently without task-specific fine-tuning. In practice, the best-performing policies are specialists refined per task on top of generalist pretraining (π\*0.6 refined with RL, π0.6 with per-task SFT). The PI blog compares this to the era when early language models (BERT) were fine-tuned for every domain.

### 1.2 The prescription is broader data — and two walls

By the logic of foundation models the prescription is clear: broader, more diverse data. In robotics, "diverse data" means the following.

| Data | Source of diversity |
|---|---|
| Demonstrations from many robots | Different morphologies, degrees of freedom, and control schemes |
| Demonstrations from many operators | Different strategies even for the same task |
| Autonomous execution logs | Evaluation rollouts of earlier policies, failures included |
| RL agent rollouts | Good and bad attempts from the learning process, mixed together |
| Human first-person video, web data | No robot action labels |

Mixing these naively, however, runs into two walls.

**⛔ Wall 1 — Diversity is ambiguity.** The same observation and the same instruction ("fold the shirt") correspond to a fast, clean episode, a slow episode with many regrasps, and a failed episode. In the paper's words, a naively trained model "averages" these modes, and the result is suboptimal. So the common practice is filtering: use only high-quality demonstrations with a consistent strategy. The paper points out that this filtering is labor-intensive, uses different criteria for every task, and throws away valuable information. It amounts to collecting data for its diversity and then using it by cutting that diversity away.

**⛔ Wall 2 — Language cannot write down all of "how."** To resolve the ambiguity, you can attach descriptions to the data. It is the same idea as image and video generation models enriching captions through prompt expansion to raise quality. But the paper says that in robotics, text captions alone are not enough, because the information that decides success is ⓐ subtle, like the overall quality of an episode, or ⓑ hard to put into words, like what a neatly folded T-shirt looks like or how to grasp a fridge handle.

Wall 1 is dissected in §2.3 and broken through in §3.3. The device that breaks through Wall 2 is in §3.4.

### 1.3 So the question the paper asks

> If you add "how" it was done to "what" is done as context, can you train on heterogeneous, low-quality data without throwing it away? And does data broadened that way lead to specialist-level performance without fine-tuning, and to compositional generalization?

---

## 2. Background — the minimum needed to read the two walls

### 2.1 The training objective of a flow-based VLA

A VLA starts from a pretrained VLM backbone and is adapted to robot control. Its training objective is:

$$\max_\theta\ \mathbb{E}_{\mathcal{D}}\big[\log \pi_\theta(\mathbf{a}_{t:t+H}\mid \mathbf{o}_{t-T:t},\ \mathcal{C}_t)\big]$$

| Symbol | Meaning |
|---|---|
| $\theta$ | VLA parameters |
| $\mathcal{D}$ | Dataset of training trajectories |
| $\mathbf{a}_{t:t+H}$ | Action chunk: the actions for the next $H$ steps, with $H=50$ in π0.7. Only the first $\hat H\in\lbrace 15,25\rbrace$ steps are actually executed |
| $\mathbf{o}_{t-T:t}$ | Recent observation history. $\mathbf{o}_t=[\mathbf{I}^1_t,\dots,\mathbf{I}^n_t,\mathbf{q}_t]$ is $n$ camera images plus the joint state $\mathbf{q}_t$ |
| $\mathcal{C}_t$ | Prompt (context): all the side information that conditions the actions |

- The action expert is a small transformer that attends to the VLM backbone's activations and generates the action chunk with flow matching. Flow matching optimizes an approximate lower bound instead of a closed-form log-likelihood.
- In earlier models (π0, π0.5, π0.6) the context was a single short task description, i.e. $\mathcal{C}_t=(\ell_t)$.

The only thing π0.7 changes in this objective is **$\mathcal{C}_t$**. What goes into it is all of §3.

### 2.2 Knowledge Insulation — division of labor between backbone and action expert

π0.7 follows the Knowledge Insulation (KI) recipe. Putting the paper's description into an equation gives the following (notation mine).

$$\mathcal{L}\ =\ \underbrace{\mathcal{L}_{\text{CE}}\big(\text{FAST tokens}\mid h_{\text{VLM}}\big)}_{\text{backbone}}\ +\ \underbrace{\mathcal{L}_{\text{FM}}\big(\mathbf{a}_{t:t+H}\mid \mathrm{sg}[h_{\text{VLM}}]\big)}_{\text{action expert}}$$

- $h_{\text{VLM}}$ — activations of the VLM backbone
- FAST tokens — the action chunk compressed into discrete tokens
- $\mathcal{L}_{\text{CE}}$ — cross-entropy over the discrete tokens
- $\mathcal{L}_{\text{FM}}$ — flow matching loss over continuous actions
- $\mathrm{sg}[\cdot]$ — stop-gradient. The action expert attends to backbone activations but sends no gradient back

There is an easy point to misread here. Under KI the backbone is not frozen. It keeps learning actions through the FAST-token cross-entropy. What is blocked is only the gradient path from the action expert's continuous flow loss into the backbone. The paper's reason is to train the backbone with the comparatively stable discrete cross-entropy. From an LLM perspective, the backbone is an LLM doing next-token prediction, and the action expert is a diffusion decoder that reads its representations and produces continuous actions.

### 2.3 What Wall 1 really is — mode averaging

Drawn as a picture, Wall 1 looks like this.

```
 same (o, l = "fold the shirt")
   episode A : fast, clean         -> a^A
   episode B : slow, 2 regrasps    -> a^B
   episode C : failed grasp        -> a^C

 no m   : learn mixture of A/B/C, sample in data proportion
 with m : a ~ p(a | o, l, m* = {Quality 5, Mistake false, Speed 2000})
```

- Three ways of performing the task are mixed under the same observation and instruction
- `m` is the variable that names that way of performing (the episode metadata of §3.3)

A reader who knows diffusion will have a question here. Flow matching is a generative model used precisely to represent multimodal distributions. It should not average modes the way MSE regression does, so why is "averaging" a problem?

> ### 💡 The problem is not the 'average' but 'sampling from the mixture'
>
> Decompose the conditional distribution the data creates using a latent variable $m$ for the way of performing.
>
> $$p_{\mathcal{D}}(\mathbf{a}\mid\mathbf{o},\ell)\ =\ \sum_{m}\ p_{\mathcal{D}}(m\mid\mathbf{o},\ell)\ p_{\mathcal{D}}(\mathbf{a}\mid\mathbf{o},\ell,m)$$
>
> - $m$ — the way of performing: strategy, quality, speed. A latent variable not recorded in the data
> - $p_{\mathcal{D}}(m\mid\mathbf{o},\ell)$ — the share each way takes up in the data
> - $p_{\mathcal{D}}(\mathbf{a}\mid\mathbf{o},\ell,m)$ — the action distribution under way $m$
>
> Even if the flow model is perfect, what it learns is the left-hand side, the mixture. Three problems follow.
>
> - **⓵ It samples by frequency** — if 30% of the data is sluggish demonstrations, 30% of chunks come from that mode. Put failure data in and failures are reproduced at that rate too.
> - **⓶ The mode switches from chunk to chunk** — since it resamples every $\hat H$ steps, it can produce an inconsistent trajectory that starts with strategy A and switches to strategy B.
> - **⓷ In practice, interpolation happens too** — with finite capacity and few denoising steps (5 in π0.7), awkward in-between actions do come out.
>
> The root cause is that $m$ is a **confounder** that cannot be inferred from the observation $\mathbf{o}$. Who operated the robot, or which evaluation run it was, is not visible in the camera images. The fix, then, is to turn $m$ into an observed variable, learn the right-hand side directly, and pick the desired $m$ at inference time. That is the episode metadata of §3.3.
>
> The paper only writes that the model "averages the modes." The decomposition above and the three paths are my interpretation.

---

## 3. Method — context as "what + how"

### 3.1 At a glance

π0.7 widens the context to five components.

$$\mathcal{C}_t=(\ell_t)\quad\longrightarrow\quad \mathcal{C}_t=\{\ell,\ \hat\ell_t,\ \mathbf{g}_t,\ m,\ c\}$$

| Component | Content | Source at training | Source at inference | Wall targeted |
|---|---|---|---|---|
| $\ell$ | Overall task ("clean the kitchen") | Human annotation | User | — |
| $\hat\ell_t$ | Next subtask ("open the fridge door") | Segment-level annotation | High-level policy or human coaching | Wall 2 (partly) |
| $\mathbf{g}_t$ | Multi-view subgoal images | Real future frames + world-model-generated images | World model | Wall 2 |
| $m$ | Episode metadata (speed, quality, mistake) | Measurements + human annotation | Fixed at the "best" values + CFG | Wall 1 |
| $c$ | Control mode (`joint` / `ee`) | The data's control scheme | Chosen per task | Wall 1 (heterogeneity of control schemes) |

The prompt is concatenated roughly in this order (the structure of the example in the paper's §V-E).

```
<multi-view obs><multi-view subgoals> Task: ... Subtask: ... Speed: ... Quality: ... Mistake: ... Control Mode: ... <proprio>
```

The training and inference flow on one page:

```
  robot demos | autonomous evals (incl. failures) | RL rollouts | human video | web
                                   |
                                   |  annotate each sample:  l, l^, g, m, c
                                   v
                 TRAIN   pi_0.7 learns  p(a | o, C),  random dropout on each part of C
                                   |
                                   v
                 INFER   C* = { l, l^ from high-level policy,
                                g* from world model,
                                m* = best metadata, c }
                         a ~ p(a | o, C*)   (+ CFG on m)
```

- During training, each component of the context is randomly dropped out (§3.5)
- At inference, `l^` comes from the high-level policy, `g*` from the world model, and `m*` is fixed at the "best" values (§3.7)

**Training data mix** (paper §VI-A)

| Data | Notes |
|---|---|
| Robot demonstrations | Static and mobile, single-arm and bimanual robots / lab-style and home-style environments, and real homes |
| Autonomous evaluation data | Policy evaluation rollouts of earlier models. Includes data collected during π\*0.6's RL training, and failed episodes |
| Human interventions | Human interventions during policy rollouts |
| Open-source robot data, human first-person video | — |
| Web multimodal data | Object location and attribute prediction, VQA, text-only prediction, captioning of robot and web video |

Autonomous data from the tasks used for generalization evaluation was excluded from training (paper footnote). It is a guard against evaluation leakage.

### 3.2 Subtask instructions $\hat\ell$ — a channel for teaching in words

Next to the overall task description $\ell$ ("clean the kitchen"), the next-step subtask $\hat\ell_t$ ("open the fridge door") goes in too. The structure itself comes from π0.5. What changed is its role. Once the low-level policy follows fine-grained language well, two uses emerge depending on who fills in $\hat\ell$.

| Who fills $\hat\ell$ | Use |
|---|---|
| A learned high-level policy | Performs long-horizon tasks autonomously |
| A human | **Coaching**: calling out an unseen task step by step |

Coaching logs become training data in their own right. The logs are collected and π0.7 is fine-tuned into a high-level policy (notation mine).

$$\hat\ell_{k+1}\ \sim\ \pi^{\text{HL}}_\phi\big(\cdot\mid \mathbf{o}_t,\ \ell,\ \hat\ell_{1:k}\big)$$

- $\pi^{\text{HL}}_\phi$ — the high-level policy. Fine-tuned from the same architecture as π0.7 (Gemma3 4B based)
- $\mathbf{o}_t$ — the current robot observation
- $\hat\ell_{1:k}$ — the history of subtask instructions given so far
- $\hat\ell_{k+1}$ — the next subtask instruction

The upshot is that a new task can be automated without action-level data such as teleop. The results are in §5.4.

### 3.3 Episode metadata $m$ — the device that breaks through Wall 1

As §2.3 showed, Wall 1 comes from the way of performing, $m$, being unobserved. π0.7 attaches it directly as a label.

| Field | Definition | Label source |
|---|---|---|
| Speed | Episode length (timesteps), discretized in units of 500 steps (1750–2250 → "2000") | Measurement |
| Quality | Execution quality, 1–5 (5 is best) | Human annotation |
| Mistake | Whether there was a mistake in that **action segment** (failed grasp, wrong subtask, etc.) | Coarse human annotation |

Two details matter.

- **Mistake is per segment.** For an episode that succeeded overall but failed one grasp midway, only the failed stretch can be marked.
- **Speed correlates with quality.** The paper notes that fast episodes usually have fewer mistakes too.

At inference it always asks for "the best."

| Field | Value at inference |
|---|---|
| Speed | 15th percentile of episode length for the task |
| Quality | 5 |
| Mistake | false |

But episode length is a value you only know once the episode is over. How can a future statistic that does not exist at the moment of choosing an action be used as a condition?

> ### 💡 A value known only in hindsight can still be a condition — the robot version of return-conditioning and micro-conditioning
>
> During training, hindsight labels are attached to finished episodes, and the model learns "this is what the actions of an episode with this outcome look like." At inference, you declare the desired outcome first.
>
> $$\text{train: }\ p_\theta(\mathbf{a}\mid\mathbf{o},\ell,m),\ \ m=f(\tau)\qquad\quad \text{test: }\ \mathbf{a}\sim p_\theta(\mathbf{a}\mid\mathbf{o},\ell,m^\star)$$
>
> - $\tau$ — the whole episode trajectory
> - $f$ — hindsight statistics of the trajectory (length, quality score, whether there were mistakes)
> - $m^\star$ — the value requested at inference
>
> The same idea already exists in other fields.
>
> | Field | Hindsight condition | Requested at inference |
> |---|---|---|
> | RL (return-conditioning, Decision Transformer family) | Trajectory return | High return |
> | Image generation (SDXL micro-conditioning) | Original resolution, crop coordinates | High resolution, no crop |
> | π0.7 | Length, quality, mistakes | Fast, 5 points, no mistakes |
>
> It is the same attitude as SDXL feeding in "original size" as a condition to make use of low-resolution training images instead of discarding them. **Don't throw it away; describe it and put it in.**
>
> The paper does not say why it asks for the 15th percentile rather than the minimum length. Condition values in the tail of the distribution have thin training support, so the condition itself becomes OOD; I read it as asking for a range that is fast yet well observed (my interpretation).

### 3.4 Subgoal images $\mathbf{g}$ — the device that breaks through Wall 2

The instruction "open the fridge door" does not specify how to grasp the handle. A subgoal image spells out, as a picture, how the scene should look in the near future.

$$\mathbf{g}_t=[G^1_t,\dots,G^n_t]$$

- $G^i_t$ — the near-future goal image as seen from camera $i$
- Why multi-view — the base view makes it easy to specify the outcome for the environment and objects, the wrist views the outcome for the arm and gripper

**World model — the model that draws the subgoals**

At inference, subgoals are generated by a separate world model $g_\psi$. Its input is the same subtask instruction π0.7 receives.

$$\max_\psi\ \mathbb{E}_{\mathcal{D}_g}\Big[\mathcal{L}_{\text{CFM}}\big(\mathbf{g}^\star_t,\ g_\psi(\mathbf{o}_t,\hat\ell_t,m)\big)\Big],\qquad \mathbf{g}^\star_t=\mathbf{o}_{t_{\text{end}}}$$

| Symbol | Meaning |
|---|---|
| $\psi$ | World model parameters. Initialized from BAGEL (a 14B mixture-of-transformers capable of image understanding, editing, and generation) |
| $\mathcal{D}_g$ | The subset of segments whose subtask labels are especially good |
| $\mathbf{g}^\star_t$ | Ground-truth subgoal: the last frame of the current segment, $\mathbf{o}_{t_{\text{end}}}$ (3 views) |
| $m$ | Episode metadata. The world model also takes quality and speed conditions |
| $\mathcal{L}_{\text{CFM}}$ | Standard conditional flow matching loss |

> ⚠️ The original writes the loss with max; it looks like a typo for min.

The key point is that, following the SuSIE lineage, it starts from an image-editing model pretrained on the web. Trained on robot data mixed with web data, human first-person video, and other video, it passes the semantic and physical concepts it gained there to π0.7 in the form of subgoal images. According to Appendix C, following the BAGEL architecture, inputs are encoded along two paths, a ViT (semantic understanding) and a VAE (fine detail), and the quality of the labels' temporal segmentation strongly determines subgoal quality.

The paper consistently calls this model a "lightweight world model." How light is it actually?

> ### ⚠️ The 'lightweight' world model is 14B
>
> The BAGEL-based world model is a 14B model made of a 7B backbone for understanding and a 7B backbone for generation. Producing one set of subgoals with 25 denoising steps takes 1.25 seconds even with 4×H100 tensor parallelism, 8-bit quantization, and a modified SageAttention all brought to bear (Appendix D). That is far heavier than the π0.7 model itself, which runs in 38–127ms on a single H100. "Lightweight" seems better read as relative to video generation models (my interpretation), and real-time operation is secured by asynchronous execution (§3.7).

**How subgoals are fed to π0.7**

| Setting | Value |
|---|---|
| Share of samples with a subgoal | 25% of the batch |
| Choice of real future frame | With p=0.25 the segment's end frame (matching the world model's prediction target); with p=0.75 sampled uniformly between 0 and 4 seconds ahead |
| Generated images | Samples are added in which subgoals mass-generated by the world model replace the real frames. This is to reduce the train–test mismatch between real and generated images |
| Subtask removal | $\hat\ell$ is erased in 30% of the samples that have a subgoal |

How much the subgoals contribute is seen in §5.2 and §5.3 by comparing π0.7 with π0.7 (GC) (the setting prompted with world-model subgoals).

But according to the paper, the model learns noticeably faster when given subgoals. Then why limit them to 25% instead of putting them in every sample?

> ### 💡 Subgoals are a shortcut, so they are used sparingly
>
> As the paper explains, once a subgoal is given, action prediction effectively becomes an inverse dynamics problem.
>
> $$\mathbf{a}_{t:t+H}\ \approx\ f_{\text{ID}}(\mathbf{o}_t,\ \mathbf{g}_t)$$
>
> - $f_{\text{ID}}$ — an inverse dynamics function that infers the actions between the current frame and the goal frame
>
> Easy problems get solved quickly, and that is exactly why it is risky. If subgoals are always there, the path that reads the language and metadata and infers on its own "how the world should look next" receives less learning signal. It is like a student who always gets the answer hint and stops reading the problem. 25% is a mix that keeps both paths alive. The paper only goes as far as "it learns quickly, so we include it in only 25%"; the shortcut interpretation is my inference.
>
> Erasing $\hat\ell$ in 30% of the samples with a subgoal is a design in the opposite direction. This time the model is forced to read the intent from the image alone. The use in §5.3's UR5e shirt folding, prompting with only subgoals and metadata and no language, relies on this design.

### 3.5 Control mode and the dropout design

The coordinate frame of the actions also goes into the context. Joint-control and end-effector-control data are trained together and distinguished by a text identifier $c\in\lbrace \texttt{joint},\texttt{ee}\rbrace$. At inference, it is chosen to fit the task. End-effector commands are converted into joint targets by numerical inverse kinematics (IK) and passed to a PD controller.

Every component is randomly dropped during training. Control mode is the one exception.

| Component | Treatment during training |
|---|---|
| Subgoal images | Included in only 25% of the batch |
| Subtask $\hat\ell$ | Removed in 30% of the samples with a subgoal |
| Metadata as a whole | Removed entirely with 15% probability |
| Individual metadata fields | Speed, quality, and mistake each removed with 5% probability |
| Control mode | Never removed |
| History frames | Removed entirely with 30% probability |
| Rear camera image | Removed with 30% probability |

The paper does not say why control mode is never dropped. Since it is information that changes the very coordinate frame of the actions, sampling without knowing it would produce meaningless actions (my interpretation).

Dropout has two effects.

- **⓵ Flexibility** — at inference you can prompt with any subset. It works with language alone and no subgoal, or with subgoals alone and no language.
- **⓶ Learning conditional and unconditional together** — one model learns both $p(\mathbf{a}\mid\mathbf{o},\mathcal{C})$ and $p(\mathbf{a}\mid\mathbf{o},\mathcal{C}^{\text{uncond}})$ with some components removed. It is the same device as null-prompt dropout in text-to-image, and it is what makes §3.7's CFG possible.

### 3.6 Architecture — context added on top of π0.6-MEM

π0.7 adds multimodal context conditioning on top of π0.6's VLA structure and the MEM memory system. It has about 5B parameters in total.

```
[obs: <=4 cams x <=6 hist frames -> 1-frame tokens]  bidirectional
[subgoals: <=3 views, same encoder]                  bidirectional, attends to obs
[text: task | subtask | metadata | control mode]     causal
[proprio + history states, linear projection]
            |
            |  attend only (no gradient into backbone: KI)
            v
[action expert 860M: 50 action tokens, flow matching, adaRMSNorm]
```

- **Backbone** — initialized from the Gemma3 4B VLM (including a 400M vision encoder)
- **Observations** — up to 4 cameras (front, both wrists, optionally rear), up to 6 history frames per camera (1 second apart). A MEM-style encoder compresses the history spatiotemporally into one frame's worth of tokens. Input resolution is 448×448
- **Subgoals** — up to 3 images (no rear), processed by the same encoder as the observations
- **Attention** — block-causal. The observation block and the subgoal block are each bidirectional internally, subgoals attend to observations, and text is causal
- **Proprio** — instead of discrete text tokens as in π0.6, it is embedded by a linear projection. One token per history state
- **Action expert** — an 860M transformer. The 50 action tokens see each other bidirectionally and attend to backbone activations; flow-time information is injected with adaptive RMSNorm
- **Training-time RTC** — simulates an inference delay of 0–12 steps (up to 240ms at 50Hz) during training. The model learns to produce a chunk that continues the action prefix already committed for execution while inference runs. Unlike test-time RTC, it has no inference overhead (Appendix D)

### 3.7 Runtime — three models combined asynchronously

At inference, three trained models mesh through loops running at different speeds.

```
 high-level policy (4B, same arch)  or  human coaching       [slow, async]
        |  l^  (next subtask)
        v
 world model g_psi (BAGEL 14B)                               [slow, async]
        |  g*  (3-view subgoal; refresh on new l^ or every 4 s)
        v
 pi_0.7 VLA (5B)  <-- also: l, l^, m*, c, obs history        [fast]
        |  a_{t:t+50}  (5 denoise steps, CFG on m, exec 15 or 25 steps)
        v
 robot (PD controller; numerical IK if ee)
```

- **High-level policy** — produces the next subtask $\hat\ell$. During coaching, a human takes over this role
- **World model** — redraws the subgoal when $\hat\ell$ changes or when $\Delta=4$ seconds have passed since the last generation
- **π0.7** — produces a 50-step chunk in 5 denoising steps and executes only 15 or 25 steps. 38ms in the minimal configuration, 127ms in the worst case with both the MEM encoder and subgoals on (single H100)
- **Asynchronous** — subtask and subgoal generation run on separate threads, and the VLA uses the latest values ready at that moment
- **Always-present components** — control mode and metadata go into the prompt for every task

Metadata has one more step. CFG is applied at every step of action denoising.

$$\tilde\nabla_{\mathbf{a}}\ =\ \nabla_{\mathbf{a}}\log\pi_\theta(\mathbf{a}\mid\mathbf{o}_t,\mathcal{C}_t)\ +\ \beta\Big(\nabla_{\mathbf{a}}\log\pi_\theta(\mathbf{a}\mid\mathbf{o}_t,\mathcal{C}_t)\ -\ \nabla_{\mathbf{a}}\log\pi_\theta(\mathbf{a}\mid\mathbf{o}_t,\mathcal{C}^{\text{uncond}}_t)\Big)$$

| Symbol | Meaning |
|---|---|
| $\mathbf{a}$ | Shorthand for the action chunk $\mathbf{a}_{t:t+H}$ |
| $\mathcal{C}_t$ | The full context, including metadata |
| $\mathcal{C}^{\text{uncond}}_t$ | The "unconditional" context. In π0.7, the context without metadata |
| $\beta$ | CFG weight. The paper uses $\beta\in\lbrace 1.3,\ 1.7,\ 2.2\rbrace$ |
| $\tilde\nabla_{\mathbf{a}}$ | The guided score actually used for denoising |

But quality 5 and mistake false are already in as conditions. Why push once more with CFG?

> ### 💡 Metadata CFG sharpens the distribution toward an implicit 'quality discriminator'
>
> Setting $\mathcal{C}_t=\mathcal{C}^{\text{uncond}}_t\cup\lbrace m^\star\rbrace$ and applying Bayes' rule, the difference in parentheses becomes $\nabla_{\mathbf{a}}\log p_\theta(m^\star\mid\mathbf{a},\mathbf{o},\mathcal{C}^{\text{uncond}})$. So the guidance above is equivalent to sampling from the following distribution.
>
> $$\tilde\pi(\mathbf{a}\mid\mathbf{o},\mathcal{C})\ \propto\ \pi_\theta(\mathbf{a}\mid\mathbf{o},\mathcal{C}^{\text{uncond}})\cdot p_\theta(m^\star\mid\mathbf{a},\mathbf{o},\mathcal{C}^{\text{uncond}})^{\,1+\beta}$$
>
> - $m^\star$ — the requested metadata (quality 5, mistake false, fast speed)
> - $p_\theta(m^\star\mid\mathbf{a},\mathbf{o},\cdot)$ — "the probability that this action chunk came from such an episode." A classifier never trained separately, but defined implicitly by the ratio of the conditional and unconditional models
> - $1+\beta$ — the exponent that sets how sharply the distribution is pushed toward the classifier
>
> With the conditional model alone, the condition is reflected only weakly. It is the same phenomenon as the drop in prompt fidelity when sampling text-to-image without CFG. The exponent $1+\beta$ drives the distribution toward the discriminator. And what the discriminator picks out here is **quality**. As a result, metadata CFG works like policy improvement carried out without a critic. It has the same shape as policy improvement that weights high-advantage actions (my interpretation).
>
> In the implementation, the conditional and unconditional branches are packed into one sequence with an attention tree and handled in a single forward pass (Appendix B).

---

## 4. Why it works

The paper's reasons come down to four.

**⓵ Resolving ambiguity** — because the context distinguishes quality and strategy, low-quality data is absorbed without hurting performance.

**⓶ State coverage** — suboptimal data widens the states and scenarios visited within a task. The paper explains that this raises robustness and is why it sometimes surpasses the RL specialist.

**⓷ Specialist distillation** — data π\*0.6 collected during RL training goes in with good and bad attempts mixed together. Metadata becomes the handle that calls up the good behavior among them, so the generalist inherits the specialist's abilities.

**⓸ An interface to web knowledge** — subgoal images are the channel that carries the semantic and physical knowledge of the web-pretrained world model into the policy.

Compressed into one sentence, π0.7 **turns "how," which was a confounder, into an observed variable.** In the decomposition of §2.3, it now learns the right-hand side instead of the left. There is one side effect. Metadata is a task-agnostic vocabulary: "quality 5" points the same way for espresso as for shirt folding. So the "how" axis is shared across tasks, and the "what" axis is learned with the noise of performing style cleared away. It is a structure that favors composition (my interpretation).

Then is it the context or the data diversity that produced compositional generalization? Putting side by side the two experiments examined in detail in §5.5 separates the roles.

> ### 📌 Context is the necessary condition; the fuel for composition is task diversity
>
> | Experiment (Fig. 18) | Manipulation | Result | Role revealed |
> |---|---|---|---|
> | Left | Add more and more low-quality data | With metadata it keeps improving; without, it can degrade | Context makes diverse data **usable** |
> | Right | Remove the same amount: the most diverse 20% vs a random 20% | Unseen-task performance drops sharply only when the most diverse 20% is removed | The **fuel** for compositional ability is task diversity |
>
> π0.7's contribution is not a device that directly creates new abilities, but one that makes learnable the diversity that filtering used to throw away. That is why the paper calls itself "a methodology, not an architecture."

The paper's Discussion is on the same line. Its position is that generalization is ultimately a remix of behaviors already seen, and that this is the very essence of compositional generalization.

Here is how the two walls named in §1.2 were resolved.

| Wall | Prescription | Evidence |
|---|---|---|
| ⛔ Wall 1 — Diversity is ambiguity | Episode metadata $m$ (+ control mode $c$, subtask $\hat\ell$), dropout, metadata CFG | Fig. 7 metadata-removal ablation, Fig. 18 left |
| ⛔ Wall 2 — The limits of language | Subgoal images $\mathbf{g}$ generated by the world model | Gains of π0.7 (GC) in Figs. 10, 11, 12 (right), 15 |

---

## 5. Experiments

Evaluation took place on several robots: a mobile bimanual robot (two 6-DoF arms), a static bimanual robot with lightweight 6-DoF arms, a bimanual UR5e system with Robotiq grippers, and a single-arm robot. The UR5e runs at 20Hz, the others at 50Hz. Results are mostly presented as bar charts; the numbers stated explicitly in the text are roughly the human comparison experiment and the success-rate ranges in the Discussion. Below they are summarized with figure numbers.

| Axis | Tasks | Compared against | Gist of the result | Figure |
|---|---|---|---|---|
| Dexterity | Laundry (T-shirts & shorts / button-up shirts), espresso, box assembly | π\*0.6 RL specialist | Comparable success rate; higher throughput on button-up laundry and box assembly | Fig. 6 top |
| Dexterity | Peanut butter sandwich, turning a shirt inside out, driving through a door, slicing zucchini, peeling vegetables, replacing a trash bag | π0.6 SFT specialist | Close | Fig. 6 bottom |
| Memory | Swapping 3 mugs, finding an object, scooping coffee, wiping a window | π0.6-MEM specialist | On par or better | Fig. 8 |
| Instruction following | 4 unseen kitchens + 2 bedrooms, 14 scenarios | π0.5, π0.6 | Large margin overall | Fig. 9 |
| Referential instructions | "The thing you use when eating soup," "the fruit on the largest plate" | π0.5, π0.6 | Ahead on complex instructions, further gains with GC | Fig. 10 |
| Going against data bias | Reverse Bussing, Reverse Fridge to Microwave | π0.5, π0.6 | Ahead; GC decisive for the latter | Fig. 11 |
| Cross-embodiment | 6 tasks (§5.3) | π0.5, π0.6 | The larger the morphology gap, the larger π0.7's lead | Fig. 12 |
| Composition (short-horizon) | Pressing a French press, scooping rice into a rice cooker, wiping office supplies, turning articulated objects | π0.5, π0.6 | Done by prompting alone; similar to GC | Fig. 17 |
| Composition (long-horizon) | Loading and emptying an air fryer, toasting a bagel | π0.5, π0.6 (coaching) | Earlier models cannot even follow the coaching instructions | Fig. 15 |
| Coaching → autonomy | 5 tasks | Coaching episodes | Autonomous performance ≈ coached performance | Fig. 16 |

### 5.1 Specialist-level dexterity — how well does it do the tasks it trained on

The first question is whether it performs dexterous tasks already in the training data as fast and as robustly as a specialist, without fine-tuning. It is the direct answer to symptom ⓶ of §1.1.

- **Against the RL specialist (Fig. 6 top)** — on laundry (T-shirts & shorts / button-up shirts, the hardest item), espresso, and box assembly, which were used to evaluate π\*0.6, the success rate is comparable. On button-up laundry and box assembly, successes per hour (throughput) are higher than the specialist's.
- **Against SFT specialists (Fig. 6 bottom)** — on various dexterous tasks including the "Robot Olympics" tasks, it comes close to π0.6-based SFT specialists.
- **Memory tasks (Fig. 8)** — on tasks that require remembering past observations too, it matches or beats the memory-fine-tuned π0.6-MEM specialist.
- **Ablation (Fig. 7)** — removing metadata or removing autonomous evaluation data lowers performance on every task, and the gap is largest in throughput. Speed is information carried by the speed metadata and the RL rollouts, so the result fits the design intent.

But can this be read as "the generalist beat the specialist"?

> ### ⚠️ "Out-of-the-box" does not mean "no task data"
>
> The phrase means there is no per-task post-training stage. For the four tasks in Fig. 6 top, the rollouts π\*0.6 accumulated while learning with RL are in π0.7's training data, and the paper explicitly calls this "distillation." So this result is more accurately read not as "it beat the specialists" but as "it folded several specialists into one generalist without loss." By contrast, autonomous data from the generalization-evaluation tasks was excluded from training, so the results of §5.2–§5.4 do not have this issue.

### 5.2 Instruction following — does it actually read the language

- **Open-ended instructions (Fig. 9)** — in 4 unseen kitchens and 2 bedrooms, 14 scenarios were evaluated in which it follows 3–6 open-ended instructions in order: tidying, manipulating furniture, cleaning up spills, and so on. In instruction completion it leads π0.5 and π0.6 by a large margin overall.
- **Referential instructions (Fig. 10)** — in an office-desk tidying task, every model succeeds at standard instructions ("pick up the spoon"). On complex referential instructions ("pick up the thing you use when eating soup," "pick up the fruit on the largest plate"), π0.7 leads, and it goes higher when given subgoals (GC).
- **Going against data bias (Fig. 11)** — a model trained on data where a scene always came with the same behavior ignores language in that scene and follows habit. In "Reverse Bussing" (trash into the bus tub, dishes into the trash can) and "Reverse Fridge to Microwave" (the data only has the fridge-to-microwave direction), π0.7 goes against the bias and follows the instruction.

In "Reverse Fridge to Microwave" in particular, GC was decisive. The paper explains that thanks to web-scale image-generation pretraining, the world model produces good subgoals from text instructions. When the world model draws a scene that runs against the data's habit, the policy only has to follow that picture. It is the scene where the subgoal contribution foreshadowed in §3.4 shows most clearly.

### 5.3 Cross-embodiment — moving a skill to a robot that has never done it

Can a robot with no data at all for a task do that task? The paper experiments while progressively widening the morphology gap between the source and target robots (Fig. 12).

| Task | Robot the data came from | Robot evaluated on | Result |
|---|---|---|---|
| Table Setting | Several robots (mobile, static, single-arm) | Static bimanual | All models do well |
| Bag In Backpack, Organize Tupperware | Bimanual UR5e (large and heavy) | Small static bimanual | π0.5 drops sharply; π0.6 and π0.7 do well |
| Shirt Bagging | Small static bimanual | Single-arm UR5e | Clear π0.7 lead |
| Towel / Shirt Folding | Small static bimanual | Bimanual UR5e | π0.7 succeeds, further gains with GC |

The UR5e is much longer and heavier than the source robot, differently shaped, and mounted on both sides of the table rather than at one end. So the manipulation strategy itself has to change. What is interesting is that π0.7 does not imitate the source robot's behavior but writes a new strategy suited to the target robot (Fig. 13).

| Situation | Human operator's strategy on the source robot | π0.7's strategy on the UR5e |
|---|---|---|
| Putting a shirt in a bag | Holds the bag open with one arm and puts it in with the other | Picks it up and puts it in in one go with the long arm |
| Grasping a shirt | Tilts the end-effector to press the cloth against the table and grab it | Uses a vertical grasp suited to the arm placement |

As foreshadowed in §3.4, UR5e shirt folding is prompted with only subgoals and metadata, no language. The paper explains that the world model builds a visual analogy between the source and target robots, presenting a grasp and cloth placement suited to the target robot as the subgoal.

To gauge the result, it was also compared with humans. Ten skilled operators in the top 2% by manipulation experience (about 375 hours on average across all robots) each attempted UR5e shirt folding three times without practice. The humans recorded 90.9% task progress and an 80.6% success rate; π0.7 recorded 85.6% and 80%. Two conditions have to be stated whenever this result is cited.

> ### ⚠️ What faced the humans was π0.7 (GC), and the UR5e is not an unseen robot
>
> The opponent in the human comparison (Fig. 22, Appendix F) is π0.7 (GC), using world-model subgoals. The main text just says "π0.7." Also, bimanual UR5e data does exist in training (the source robot for Bag In Backpack and Organize Tupperware). What is new is the combination "UR5e × laundry folding." The basis for choosing joint-space control is also a joint vs end-effector comparison of π0.5 and π0.6 (Fig. 20); no comparison for π0.7 itself is reported.

The practical implication the paper draws is clear: skills collected on cheap, lightweight arms that are easy to teleop may be transferable to heavy-payload industrial arms that are hard to teleop and expensive to collect demonstrations on.

### 5.4 Compositional task generalization — tasks never seen before

This is the axis the paper calls the "grand challenge" of robot foundation models. Earlier models showed semantic-level generalization, such as picking up an object with an unseen name, but performing a new task was hard.

**Short-horizon tasks by prompting alone (Fig. 17)** — pressing a French press, scooping rice into a rice cooker, wiping office supplies such as headphones and rulers, and turning articulated objects such as a gear set or a desk fan, all without robot data for those tasks. π0.7 prompted with language only and π0.7 (GC) with subgoals are at a similar level.

**Long-horizon tasks by coaching (Figs. 14, 15)** — roasting sweet potatoes in an air fryer, emptying an air fryer, and toasting a bagel in a toaster are tasks where several steps run for up to 5 minutes, so a one-line prompt is not enough. According to the PI blog, given only the zero-shot prompt "put the sweet potatoes in the air fryer," it fumbles a few times and ends up doing only part of it. This is where the coaching foreshadowed in §3.2 comes in. When a human calls out steps such as "pick up the sweet potato" and "open the air fryer," π0.7 follows them and completes the task. Earlier models cannot follow these instructions at all and stay at very low performance.

**From coaching to autonomy (Fig. 16)** — train a high-level policy on the coaching logs, and on five tasks it performs autonomously, with no human, close to the coached level. Nowhere in this process is there action-level data such as teleop.

Where did the knowledge of air fryers come from? The paper's main text only says that similar appliances appeared in other contexts in human data and external datasets. The PI blog is more specific: the closest data were two episodes of a robot *closing* an air fryer in a home and Franka data from the open-source DROID dataset, and they looked quite different from the motions in the experiment. It can be read as a case of recombining scattered pieces.

### 5.5 Does it actually learn from heterogeneous data

The last is a controlled ablation (Fig. 18). The two experiments whose roles were separated in §4 come from here.

**Left — what if you add more mixed-quality data?** Laundry (T-shirts & shorts) data was split by quality and speed into four buckets, top 30%, 50%, 80%, and all, and 8 models were trained from scratch combining these with and without metadata. The bigger the bucket, the more data but the lower the average quality. Models without metadata could actually get worse, but models with metadata kept improving as the data grew.

**Right — what if you remove diversity?** A model with the 20% highest in task diversity removed and a model with the same amount removed at random were compared on the short-horizon unseen tasks of §5.4. Only the former dropped sharply.

But how far can the conclusion of the left experiment be generalized?

> ### ⚠️ Fig. 18 left is a single seen-task experiment
>
> The caption groups the two panels as "scaling of generalization performance," but the left panel is the performance on one task already trained on. The conclusion "metadata enables data scaling" is confirmed on this one task, and reading it as a general scaling law is over-interpretation. The paper itself admits that with huge datasets it is hard to cleanly isolate "diversity," so such questions are hard to answer definitively.

---

## 6. Positioning — among neighboring work

Organizing the paper's related work around the place π0.7 occupies:

| Lineage | Representative work (cited in the paper) | π0.7's place |
|---|---|---|
| Generalist policies initialized from VLMs | RT-2, OpenVLA, π0, π0.5, Gemini Robotics | Same skeleton. Built on π0.6-MEM |
| VLA components | Memory (MEM), hierarchy for long-horizon planning (Hi Robot, π0.5), goal-image conditioning (CoT-VLA) | Integrates all three in one model |
| Pretraining on non-standard data | Web data (RT-2), human first-person video, autonomous experience (π\*0.6, RLDG, PLD) | Puts all of it in, distinguished by context |
| Task and embodiment generalization | Representation learning from human video, 2D point tracks, Open X-Embodiment, handheld collection devices (UMI) | Uses robot, human, and internet data together through the right prompts |
| Goal-image prompts | User-provided goals, goals generated by a separate model (SuSIE, VLP), chain-of-thought-style generation (CoT-VLA) | Subgoals generated by an external world model, as an optional condition |

The closest cousins are the generated-subgoal line.

| | Goals generated by a separate model (SuSIE, etc.) | Chain-of-thought-style generation (CoT-VLA) | π0.7 |
|---|---|---|---|
| Where the subgoal is made | A generative model outside the policy | Inside the policy model | A 14B world model outside the policy, asynchronous |
| Status of the subgoal in the policy | A goal that conditions the policy | An intermediate product made before generating actions | An optional condition alongside language, metadata, and control mode |

Following SuSIE, π0.7 initializes the world model from a web-pretrained image-editing model, and matched the 4-second subgoal regeneration interval to SuSIE too (Appendix C). The paper defines its contribution as complementary to this line: not a new architecture, but a methodology that lets VLAs use more diverse data. It claims that zero-shot robot-to-robot transfer of laundry folding and interaction with new objects like the air fryer go well beyond the quantitative improvements of prior work.

But is π0.7 the first to use mixed-quality data by putting quality in as a condition? π\*0.6, which produced the data π0.7 distilled, already did something similar.

> ### 🔗 π\*0.6 (RECAP) — the same organization's immediately preceding work on conditioning on "how"
>
> RECAP, π\*0.6's training method, also conditions the policy on "how good was this action." But the source and purpose of the signal differ.
>
> | | π\*0.6 (RECAP) | π0.7 |
> |---|---|---|
> | Purpose | Improve per-task RL performance | Let a generalist absorb heterogeneous data |
> | "How" signal | Advantage estimated by a learned value function, as a binary indicator | Measured episode length + human-annotated quality and mistakes (3 attributes) |
> | Requested at inference | High advantage | Quality 5, mistake false, fast speed + CFG |
> | Relationship | Rollouts generated during RECAP training → | Distilled as π0.7 training data |
>
> π0.7's metadata can be seen as generalizing RECAP's "learned 1-bit condition" into human-readable multi-dimensional labels (my interpretation). And the two methods connect in series within PI's pipeline: build specialists with RL, then fold all that experience, with labels, into the generalist. The PI blog also explains that the experience generated during RECAP training was distilled into π0.7 together with strategy metadata.

---

## 7. Limitations

Here are the limitations the paper states itself, together with further points worth noting while reading.

**Limitations the paper acknowledges**

- **Gap in generalization success rates** — trained tasks often exceed 90%, but unseen tasks and unseen task–robot combinations are in the 60–80% range.
- **Blurry seen/unseen boundary** — the data is so vast that related skills may already have been in it under a different label, or as part of another task. It is the same setup as the LLM generalization debate. The paper responds that recombining existing parts is the very essence of compositional generalization.
- **Long-horizon new tasks** — a one-line prompt does not work; they have to go through coaching.

**Further points to note**

- **Empirical scope of "Steerable"** — in the experiments, metadata was only ever used with fixed "best" values and CFG. No steering-accuracy (controllability) evaluation was reported, such as whether changing the speed value actually changes episode length accordingly. The axes on which real-time steering was demonstrated are language (coaching) and subgoal images.
- **The bottleneck moves** — the cost of filtering shifts to the cost of annotation. Quality and mistake are human annotations, and the world model is sensitive to labels with accurate temporal segmentation (Appendix C).
- **Incomplete isolation of GC's contribution** — π0.7 and π0.7 (GC) are the same model and differ only in whether subgoals are given at inference. But this model saw world-model-generated images as context during training. The world model's knowledge may be indirectly present in the performance of π0.7 run without subgoals as well.
- **Baselines** — all are PI's own models (π0.5, π0.6, specialists). There is no external VLA baseline, and the number of trials and confidence intervals are not given in the text or figure captions.
- **Runtime cost** — it consists of a 5B VLA, a 4B high-level policy, and a 14B world model (4×H100). The world model is optional, but the hardest generalization (going against bias, UR5e shirt folding) relies on GC.

---

## 8. Closing — what this paper suggests

π0.7's real contribution is not a particular module. **It is making data explained instead of filtered.** Failed episodes, slow demonstrations, RL rollouts, human video: data that until now was thrown away or used separately enters a single context grammar.

And this paper reads unusually familiarly to a reader with a generative-model background.

- **It is a road text-to-image already walked.** ⓐ Condition on attributes instead of filtering (SDXL micro-conditioning). ⓑ Enrich the captions (prompt expansion, DALL·E 3's recaptioning). ⓒ Push toward quality at inference with condition dropout and CFG. What is new in robotics is that text alone was not enough for the "caption," so it expanded into images (subgoals) and numbers (metadata).
- **It overlaps the trajectory of LLMs.** The shift from per-task fine-tuning to prompting, and the flow of feeding the experience of RL-built expert models back as training data to fold into a general model, both show up as they are.
- **It is the structure of an agentic system.** The runtime splits into a planner (high-level policy), a tool (the world model that draws visual specs), and an executor (the VLA). Fine-tuning the high-level policy on coaching logs is the same as SFT-ing a planner on human-written plan traces.

The biggest implication is in the last item. The unit of teaching a new task has moved up one layer, **from action demonstrations to instructions**. The PI blog goes a step further and envisions that a model that follows prompts precisely could become a channel for grounding a foundation model's semantic reasoning in physical action. It means the center of gravity of learning may shift from collecting more action data for robots to explaining, instructing, and teaching them.

---

## Appendix — glossary

| Term | Definition |
|---|---|
| **context $\mathcal{C}_t$** | All the side information that conditions the actions. In π0.7, $\lbrace \ell,\hat\ell,\mathbf{g},m,c\rbrace$ |
| **episode metadata** | Labels for episode length (speed), quality score (quality), and per-segment mistakes (mistake) |
| **subgoal image** | A multi-view goal image showing how the scene should look in the near future |
| **world model** | The BAGEL-based 14B model that generates subgoal images from subtask instructions |
| **π0.7 (GC)** | The setting prompted at inference with subgoals generated by the world model |
| **high-level policy** | A model that generates the next subtask instruction from the observation, the task, and the subtask history |
| **language coaching** | A human guiding a new task through step-by-step language instructions. The logs become training data for the high-level policy |
| **Knowledge Insulation** | A training recipe in which the backbone learns from discrete FAST tokens and the action expert's gradient does not flow into the backbone |
| **metadata CFG** | An inference technique that sharpens the action distribution toward quality using the difference from the unconditional prediction without metadata |
| **training-time RTC** | A technique that simulates inference delay during training so the model produces a chunk continuing an already-committed action prefix |

**Paper** — [arXiv:2604.15483](https://arxiv.org/abs/2604.15483) · **Project page** — [pi.website/pi07](https://pi.website/pi07) · **PI blog** — [A Steerable Model with Emergent Capabilities](https://www.pi.website/blog/pi07)
