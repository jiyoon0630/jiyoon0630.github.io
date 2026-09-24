---
layout: paper
lang: en
ref: rfm-vla-development
kind: tech-review
title: "Robot Foundation Model Technology: How VLAs Developed and How They Are Trained"
date: 2026-04-10 12:00:00 -0700
tags: [Robot-Foundation-Model, VLA, Diffusion-Policy, Imitation-Learning, Reinforcement-Learning, Tech-Review]
summary: "How VLAs started from regression, split into two branches — diffusion and web-scale VLMs — and merged back into one; and the data and the pre-training and post-training methods used to train them."
---

> **In short** — **The frontier of VLA today is an architecture that keeps a VLM's visual and language knowledge while generating continuous actions.** From the two limits of the starting point, predicting actions directly by regression (Era 1), two branches split off: one that captures multimodality with diffusion (Era 2A) and one that draws on the knowledge of web-scale VLMs (Era 2B). The two merged in Era 3. This piece traces that development through representative research, along with the data used to train VLAs and the pre-training and post-training methods.

---

## Section 1. How VLA (Vision Language Action) models developed

### 1-1. [Era 1] Generating actions with naive regression

> ### 💡 In one line
>
> **The simplest possible approach: "look at the image and predict directly how far to move each joint." It was the first starting point, but it hit limits on precise motion and on large-scale training.**

Robot manipulation was traditionally built as a "perception → state estimation → planning → control" pipeline. Each module had to be designed and tuned by hand, so every new task carried a large engineering cost, and errors accumulated across modules. As deep learning took off, some computer scientists and engineers began asking a fundamental question at exactly this point.

> *What if we trained a single neural network end to end that takes the camera image and the robot's sensor values and directly outputs the robot's joint actions?*

[![Era 1 — Generating actions with naive regression](/assets/img/notes/rfm/en/era1.svg)](/assets/img/notes/rfm/en/era1.svg)
*[Era 1] Generating actions with naive regression (click to open full size)*

#### 1-1-1. End-to-End Visuomotor (JMLR 2016)

In 2016, Sergey Levine, then a postdoc in the UC Berkeley lab (BAIR) of Professors Pieter Abbeel and Trevor Darrell, proposed with PhD student Chelsea Finn an algorithm that learns everything from perception to control end to end.

They built **the first image-based policy model (CNN-based), which combines the scene seen by the camera with the robot arm's current pose to predict how far each joint should move next.**

> ### 💡 Policy model
>
> A decision-making model that takes what the robot currently sees and senses (camera images, joint sensors) as input and outputs the next action to take (joint movements).

A robot's actions are continuous real values rather than discrete class labels, so the natural way to train is regression.

That is, it is supervised learning that reduces, via an MSE (mean squared error) loss, the gap between the correct action a human demonstrated (the ground truth) and the action the model predicted. Put simply: "if the correct joint torque is 5 and the model predicted 3, train repeatedly to shrink that difference of 2."

It was limited to a single task and a single camera image, but it was the first to demonstrate that robot behavior can be mapped directly from images without a human-designed perception-planning-control pipeline. That makes it **the paper that became the starting point for all later robot foundation model research.**

> ### 💡 Key figures in the physical AI startup ecosystem: Sergey Levine, Chelsea Finn, Pieter Abbeel
>
> In 2017 **Pieter Abbeel** co-founded the robot AI startup **Covariant** with his students. It built a general-purpose AI (Covariant Brain) that lets robot arms pick and sort a wide variety of objects in logistics warehouses; in 2024 Amazon licensed Covariant's technology and hired its founders, in effect acquiring it.
>
> **Sergey Levine** (then a postdoc → now a UC Berkeley professor) and **Chelsea Finn** (then a PhD student → now a Stanford professor) co-founded **Physical Intelligence (π)** in 2024. Aiming to build a general-purpose foundation model for robots, it raised a <span class="tex2jax_ignore">&#36;400M</span> Series A led by Jeff Bezos in 2024 at a <span class="tex2jax_ignore">&#36;2B</span> valuation. The company's $\pi_0$ and $\pi_{0.5}$ sit at the frontier of robot foundation model research today.
>
> An advisor-and-student pairing that worked together on one paper in 2016 went on to become founders leading today's physical AI industry.

#### 1-1-2. BC-Z (PMLR 2022, Google)

Google extended Sergey Levine's work to multi-task. Besides images and robot sensor values, the model takes natural language or a human video as additional input, and more than 100 tasks were trained into one model. As a result, it generalized zero-shot to 24 new tasks that were not in training, with an average success rate of 44%.

The natural-language command or human demonstration video that says "which task to do" is summarized, through its own encoder, into a single compressed bundle of numbers the AI can understand (an embedding). That summary is then injected into the middle stages of the image-processing network (CNN).

Specifically, rather than simply appending the information (concat), it uses a technique that changes how the model interprets the image depending on the task (FiLM, feature-wise linear modulation). The model learns to change the very way it looks at the image according to the task.

> ### 💡 FiLM (Feature-wise Linear Modulation)
>
> A technique that uses an input embedding vector to transform a model's intermediate features.
>
> Specifically, FiLM uses a learnable scale $\gamma$ and shift $\beta$. It transforms an intermediate feature as $\gamma \times \text{feature} + \beta$, which decides how much the feature is used.
>
> Unlike concat, which simply appends information, it can modulate the network's internal representation of the input according to the condition, which makes conditioning more effective.

**Task conditioning lets one model perform different tasks depending on the language or video it is given, and in that sense it becomes a core design principle on the road to general-purpose robot foundation models.**

#### 1-1-3. Action Chunking Transformer (arXiv 2023)

Earlier work observed the camera image and robot sensor values at the current moment and predicted only a single action. That required a network inference at every timestep, so inference latency became the bottleneck for real-time control, and prediction errors accumulated step by step (compounding error), pulling the trajectory further and further off course.

To solve this, Tony Z. Zhao, then a PhD student in Professor Chelsea Finn's lab at Stanford, proposed action chunking: predicting the next $k$ steps of actions together, in one go, from the current moment.

Because a single prediction covers several steps, the model makes fewer decisions in the first place, which eases error accumulation; and ensembling overlapping action chunks (temporal ensemble) produces temporally smooth trajectories.

**Action chunking was later adopted as standard in almost all robot foundation model research, including Diffusion Policy and $\pi_0$, and became the basic unit of RFM research.**

> ### 💡 Tony Zhao and Sunday Robotics
>
> Through ACT, **Tony Zhao** became convinced that "with enough data, even low-cost hardware can do fine manipulation." Judging that the time had come for robot research to turn into real products, he left his Stanford PhD in 2024 and co-founded **Sunday Robotics** with **Cheng Chi**, author of the Diffusion Policy paper introduced below.
>
> Sunday Robotics' home robot, Memo, is trained on 10 million household-task recordings collected in more than 500 real homes with a wearable called the Skill Capture Glove. In 2026 the company raised <span class="tex2jax_ignore">&#36;165M</span> in a Series B at a <span class="tex2jax_ignore">&#36;1.15B</span> valuation.

#### 1-1-4. [Era 1] Limitations

The three papers above all predict robot actions by regression. The approach had two structural limitations.

1. **It cannot represent multimodal actions**
    - When a robot performs a task, **there are often several correct trajectories — it could go around to the left or to the right. This is called multimodality.** Regression does not model those trajectories properly; it is trained to minimize average error, so it can end up outputting the average of several trajectories (mode averaging).
    - ACT tried to address this, but only partly eased it and did not solve complex multimodality.
2. **Not enough data to build a foundation model**
    - Training a robot foundation model needs large-scale demonstration data (observation-action pairs) collected by robot teleoperation, but scaling teleoperation data to tens or hundreds of thousands of episodes and beyond is impractical in time and cost.
    - Meanwhile the web holds billions of videos, images and texts, but they contain no robot action values, so they cannot serve as ground truth for regression. The model architectures in the three papers had no pipeline for using such web data.

> ### 💡 Multimodality and the mode-averaging problem
>
> When a robot reaches for a cup on a table, it could go around to the left or to the right. When a single situation has several correct paths like this, it is called multimodality.
>
> The problem arises in regression training.
>
> Regression is trained to "reduce the error against the answer," so when it sees data where both the left path and the right path are correct, it outputs their average — a path straight down the middle. There is an obstacle in the middle, so it collides.
>
> This is the **mode averaging** problem. In a situation that calls for picking one of several correct answers, it picks neither and produces a middling in-between value.
>
> Diffusion Policy, covered later, is the key technique that emerged to solve this problem.
>
> ![The mode-averaging problem](/assets/img/notes/rfm/en/mode-averaging.svg)
> *Both paths are correct, but regression outputs their average and runs into the obstacle*

These two limitations became the motivation for the split into "[Era 2A] Capturing multimodality" and "[Era 2B] Using the architecture and knowledge of web-scale VLMs (vision language models)."

---

### 1-2. [Era 2A] Capturing multimodality

> ### 💡 In one line
>
> **The problem of producing a middling average when one situation has several correct paths was solved by bringing image-generation AI (diffusion) to robots. This was the key turning point that made diffusion the standard for motion generation in almost every robot AI today.**

Era 1's regression approach had the mode-averaging problem: when one observation admits several correct trajectories, it outputs their average. Attempts to capture multimodality (the existence of several correct trajectories for a robot task) split broadly into two branches.

[![Era 2A — Capturing multimodality](/assets/img/notes/rfm/en/era2a.svg)](/assets/img/notes/rfm/en/era2a.svg)
*[Era 2A] Capturing multimodality (click to open full size)*

#### 1-2-1. [Era 2A①] Separating modes explicitly by defining robot actions as a codebook

##### 1-2-1-1. Behavior Transformer (NeurIPS 2022)

Nur Muhammad Shafiullah, in Professor Lerrel Pinto's lab at NYU, set out to solve multimodality with a two-stage structure: "discrete mode selection + continuous correction."

First, the action trajectories in the training data are clustered (K-means) to build a set of $K$ representative actions (a codebook) in advance.

At inference, (1) a transformer discretely selects one of the $K$ modes, and (2) it predicts a continuous residual (offset) from the center of the selected mode to produce the final action.

The part to note is the discrete selection of one mode out of $K$. Regression outputs the average of all modes, whereas discrete classification explicitly selects one. The left and right trajectories each live in their own bin, so no middling in-between value comes out.

This approach solved multimodality explicitly, but it had a limitation: the codebook must be built in advance to fit the data. Choosing $K$ is hard, the codebook has to be rebuilt whenever the data changes, and clustering quality drops in high-dimensional action spaces.

> ### 💡 Action space
>
> A mathematical definition of the full range of actions a robot can take.
>
> For a 7-axis robot arm, for example, the rotation (or torque) of each of the 7 joints is expressed as one number, so the action space is a 7-dimensional continuous space.
>
> Unlike a game character, whose action space is discrete and finite like {up, down, left, right}, a robot's action space is high-dimensional and continuous, because each joint takes a continuous real value.
>
> With many joints, as in a humanoid, it extends to 30 dimensions or more, and generating accurate actions in this high-dimensional continuous space is the central challenge of robot foundation model research.

Later, the same lab released VQ-BeT (ICML 2024), which replaced the clustering method (K-means) with a learnable model (VQ-VAE) so the codebook is learned end to end, easing this limitation. But a discrete codebook inherently divides the action space into a finite number of codes, which limits how continuous and precise a trajectory it can express, so the approach did not become mainstream in later research.

> ### 💡 Lerrel Pinto and ARI
>
> **Lerrel Pinto** was a postdoc with Pieter Abbeel at UC Berkeley and is now an assistant professor at NYU, where he leads a robot learning lab (GRAIL). Building on the lab's work — BeT, VQ-BeT, Robot Utility Models and others — he founded **Assured Robot Intelligence (ARI)**.
>
> ARI aims to deploy general-purpose humanoids in industrial and enterprise settings facing severe labor shortages, and it currently operates in stealth mode.

#### 1-2-2. [Era 2A②] Modeling the multimodal distribution by bringing in diffusion from image generation

##### 1-2-2-1. Diffusion Policy (RSS 2023)

Cheng Chi, then a PhD student in Professor Shuran Song's lab at Columbia University, took an entirely different approach. He brought the diffusion technique (DDPM), which had succeeded at modeling multimodal distributions in image generation, to robot action generation.

The core idea is to learn a forward process that gradually adds noise to robot actions and a reverse (denoising) process that recovers the original action from noise. At inference, it starts from pure random noise drawn from a Gaussian, $a_T\sim \mathcal{N}(0, I)$, and produces the final action chunk through T denoising steps.

> ### 💡 Diffusion (diffusion models)
>
> A generative modeling technique in which an AI is trained, over and over, on two processes: gradually covering a sharp photo with TV static until it becomes an unrecognizable sandstorm of noise (the forward process), and, in reverse, peeling the noise off the sandstorm one layer at a time to restore the original photo (the reverse process).
>
> Once trained, the AI can start from a meaningless sandstorm and still produce a plausible new photo.
>
> It was established in 2020 with DDPM (Denoising Diffusion Probabilistic Models) and is the core technology behind image-generation services such as Stable Diffusion and DALL-E.
>
> Diffusion Policy applies this principle as is, but instead of a photo it generates "the robot's upcoming motion sequence (an action chunk)." **In other words, start from random noise and remove the noise step by step, and out comes a precise robot motion that fits the current scene.**

The current image and sensor observations are injected into the network during denoising via FiLM (see → 1-1-2), so the model generates actions while reflecting both "what situation it is in" and "which stage of denoising it has reached."

> ### 💡 Why does diffusion not suffer from mode averaging?
>
> Regression is built to "produce one answer," so when there are several answers it converges on their average.
>
> Diffusion, by contrast, starts from random noise every time. A different starting point leads to a different destination. Start from noise closer to the left and it is naturally drawn onto the left path; start closer to the right and it goes right.
>
> In other words, diffusion does not average several answers — each time it picks one of them, so mode averaging does not occur.

In ACT, action chunking required an elaborate design: compress the input with a conditional VAE (CVAE) encoder, then unroll future steps autoregressively with a transformer decoder.

In diffusion, by contrast, the "noise → recover the original" process itself does not depend on the shape of the output. Whether it generates a single action $a_t$ or an action chunk $[a_t, ..., a_{t+k}]$, you only change the output shape, with no other architectural change, and action chunking falls out naturally.

The limitation of Diffusion Policy is inference cost. Generating one action chunk takes T denoising steps, which adds latency in real-time control. This was later improved by fast sampling methods such as flow matching and consistency models.

**By achieving three things at once — solving multimodality, integrating action chunking and training stably — Diffusion Policy became the standard for action generation in robot foundation models today.** Major RFM companies' models since then, including $\pi_0$, GR00T N1 and Helix, all adopt diffusion-based (or flow-matching, a variant) action generation.

> ### 💡 Cheng Chi
>
> **Cheng Chi** co-founded **Sunday Robotics** with **Tony Zhao**, the author of ACT introduced earlier, and is currently CTO of Sunday Robotics.

##### 1-2-2-2. RDT-1B (ICLR 2025)

If Diffusion Policy demonstrated multimodality on a single robot and a single task, RDT-1B extended it into a foundation model for large-scale cross-embodiment (training robots with different bodies together in one model).

Led by Huazhe Xu at Tsinghua University, its key change was replacing Diffusion Policy's CNN with the more scalable Diffusion Transformer (DiT).

> ### 💡 DiT (Diffusion Transformer)
>
> An architecture proposed in image generation, and the backbone of the latest image and video generation models such as Stable Diffusion 3 and Sora.
>
> Where the autoregressive transformer used in LLMs and VLMs generates tokens one at a time in sequence, a DiT takes the whole noised input and denoises it in parallel, all at once.
>
> This makes it structurally well suited to generating continuous, temporally correlated robot action chunks, and it is being adopted as the action expert in recent robot foundation models such as CogACT and GR00T N1.

Different robots have different numbers of joints and action dimensions, so training them in one model requires unifying heterogeneous action spaces.

RDT-1B aligns every robot's actions into one unified space and masks the dimensions a robot does not use. This lets data from different robots be trained together in a single diffusion transformer.

> ### 💡 The core mechanism of cross-embodiment: loss masking
>
> Different robots have different numbers of joints. To train them together in one model, you first build a unified answer sheet sized for the robot with the most joints.
>
> ```
> Unified action space (assume max dimension = 14)
> __ = empty dimension (a joint this robot does not have)
>
> 7-axis arm (Franka):       [a1, a2, a3, a4, a5, a6, a7, __, __, __, __, __, __, __]
> 12-axis bimanual (ALOHA):  [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, __, __]
> 14-axis humanoid:          [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14]
> ```
>
> The Franka robot has only 7 joints, so its remaining 7 slots are empty.
>
> Marking those empty slots as "not graded" is loss masking.
>
> As an exam analogy: every student gets the same 14-question paper, but for the 7-axis robot only questions 1–7 are graded and 8–14 are skipped as "not applicable."
>
> While working through the exam, though, the model reads the context of all 14 questions, so the "arm-motion pattern for grasping objects" learned from Franka can transfer indirectly to other robots' arm joints.

#### 1-2-3. [Era 2A] Limitations

Era 2A — especially the diffusion-based work — solved mode averaging structurally and demonstrated that it could scale to cross-embodiment, but it still depended on robot teleoperation data, leaving data scalability unsolved.

To address this, an approach emerged that directly uses the architecture and knowledge of VLMs pre-trained on web-scale data.

---

### 1-3. [Era 2B] Using the architecture and knowledge of web-scale VLMs (vision language models)

> ### 💡 In one line
>
> **An approach that hands robots the visual and language knowledge a large AI like ChatGPT learned from the internet. Told "pick up a good drink for someone who is tired," a robot could now choose an energy drink — but at the cost of motion precision.**

If Era 2A focused on "how to generate actions well," Era 2B starts from an entirely different question.

> *How do we transfer the language understanding, visual reasoning and commonsense knowledge of a VLM (vision language model) pre-trained on web-scale data to robot control?*

A VLM is, at heart, a model that generates discrete tokens in sequence by autoregressive, next-token prediction. So to run continuous robot actions through a VLM pipeline, the most natural approach is to convert actions into discrete tokens.

> ### 💡 Autoregressive / next-token prediction
>
> The core way LLMs and VLMs generate text. They do not process the input all at once and emit the result in one shot; they generate the output one token at a time, in sequence.
>
> Specifically, input text is converted into a token sequence through a vocabulary, and an image into a sequence of visual tokens through a vision encoder. The model then looks at this token sequence and predicts "which token comes next."
>
> For example, given the tokens ["The", "weather", "today", "is"], it predicts the next token "nice," appends it to the input to get ["The", "weather", "today", "is", "nice"] → predicts the next token → ... and repeats.
>
> Generating in sequence by feeding its own previous output back in as input is called autoregressive, and ChatGPT producing text as if typing it out word by word is exactly this process.

[![Era 2B — Using the architecture and knowledge of web-scale VLMs](/assets/img/notes/rfm/en/era2b.svg)](/assets/img/notes/rfm/en/era2b.svg)
*[Era 2B] Using the architecture and knowledge of web-scale VLMs (click to open full size)*

#### 1-3-1. Robotics Transformer 2 (PMLR 2023, Google DeepMind)

Google DeepMind proposed the key idea of using pre-trained VLMs (PaLI-X, PaLM-E) directly for robot control: represent robot actions as text tokens and train them in the same format as the VLM's language output.

Specifically, RT-2 quantizes each dimension of a 7-DoF robot action (change in end-effector position/rotation + gripper pressure) into an integer from 0 to 255, then maps them onto the 256 least-used tokens in the VLM's vocabulary, repurposed for actions.

A single action, for example, becomes the token sequence $[1, 128, 91, 241, 5, 101, 127, 217]$. To the VLM, generating text and generating robot actions become the same next-token prediction, so robot data can be trained together with image-text data collected from the web (co-fine-tuning) without changing the model architecture at all.

> ### 💡 The VLM vocabulary
>
> When an LLM or VLM processes text, it does not work on the raw characters; it converts them into tokens registered in a predefined vocabulary.
>
> For example, the sentence "the robot picks up the cup" becomes a token sequence such as ["the", "robot", "picks", "up", "the", "cup"], and each token has a unique integer ID in the vocabulary (e.g. 4521, 12, 8837, ...). The VLM takes this integer sequence as input and works by predicting the integer ID of the next token (next-token prediction).
>
> Vocabulary size varies by model but is usually tens of thousands of tokens.

> ### 💡 Co-fine-tuning
>
> Google's RT-2 found that fine-tuning on robot data alone makes the VLM's original knowledge disappear through catastrophic forgetting.
>
> To prevent this, it trained on robot data and web-scale language-image data together (co-fine-tuning), adding robot control ability while keeping the original visual-language abilities.

**With this, RT-2 established a new paradigm: the VLA (vision-language-action) model.** Visual-language knowledge learned from web data transferred to robot control, so it could recognize objects it had never been trained on ("pick up a good drink for someone who is tired" → choosing an energy drink) and showed reasoning capability, carrying out multi-step tasks through chain-of-thought reasoning.

> ### 💡 Why must actions be turned into discrete tokens?
>
> A VLM pipeline presupposes a discrete-choice structure: softmax → cross-entropy loss → sampling.
>
> Feeding continuous values in directly would mean changing the model architecture itself, but converting actions into discrete tokens lets you use the VLM's existing architecture as is.
>
> In other words, discrete tokenization was the minimal-change solution for using a VLM.

> ### 💡 Reasoning capability
>
> RT-2 inherited from the VLM reasoning abilities it never learned directly from robot data.
>
> Given the command "pick up a good drink for someone who is tired," for example, it chooses an energy drink using commonsense the VLM learned from the web, even though the robot training data contains no "tired → energy drink" link.
>
> It can also reason about objects' physical properties, for instance choosing a rock when told "pick up something you could use as a hammer."
>
> This is an ability that could never be learned from robot data alone, and it is the key evidence that a VLM's web-scale knowledge transferred to robot control.

#### 1-3-2. OpenVLA (arXiv 2024)

RT-2's paradigm was powerful, but its model and data were closed, making it hard for the research community to reproduce or extend.

Moo Jin Kim, a Stanford PhD student advised by Professors Chelsea Finn and Percy Liang, and Karl Pertsch, a core engineer at Physical Intelligence, reproduced it as open source. It became the research community's baseline and the technical precursor to Physical Intelligence's $\pi_0$.

OpenVLA uses vision encoders (DINOv2, SigLIP) to strengthen spatial features and semantic understanding at the same time, and uses Llama-2 7B as the language backbone to generate discretized action tokens by next-token prediction.

It was trained on the Open X-Embodiment dataset (21 institutions, 22 robot types, over 1 million episodes), supporting cross-embodiment generalization across robots.

#### 1-3-3. [Era 2B] Limitations

It succeeded in transferring VLM knowledge to robots, but the discrete-token structure of the VLM pipeline itself created limitations.

1. **Quantization loss:** Discretizing continuous robot actions into 256 bins loses precision. FAST eased this, but discrete tokens fundamentally cannot represent continuous values perfectly.
2. **The multimodality limit returns:** Action generation by next-token prediction predicts each dimension in sequence, so the multimodality problem that diffusion solved in Era 2A reappears in limited form.
3. **Inference speed:** VLM next-token generation is slow in itself, and when action chunking requires generating more tokens, latency accumulates.

---

### 1-4. [Era 3] Using VLM knowledge + generating continuous actions for precision and multimodality

> ### 💡 In one line
>
> **The current frontier combines Era 2A's precise motion generation (diffusion) with Era 2B's language and visual knowledge (VLMs). Major companies — Physical Intelligence's $\pi_0$, NVIDIA's GR00T N1, Figure AI's Helix — all adopt this architecture.**

Era 2A solved multimodality for continuous actions with diffusion but could not use VLM knowledge; Era 2B used VLM knowledge but sacrificed precision and multimodality by quantizing actions into discrete tokens.

In Era 3 the two branches merge, exploring ways to generate continuous actions while keeping the VLM's visual-language reasoning.

The essence of the merger is simple. At the output end of the VLM pipeline, "language head + classification (cross-entropy loss)" is replaced with "action expert + continuous prediction (flow/diffusion loss)." This removes quantization loss and achieves action chunking and multimodality at the same time.

[![Era 3 — VLM knowledge + continuous actions](/assets/img/notes/rfm/en/era3.svg)](/assets/img/notes/rfm/en/era3.svg)
*[Era 3] Using VLM knowledge + generating continuous actions (click to open full size)*

#### 1-4-1. [Era 3①] The diffusion-based action expert trend

##### 1-4-1-1. CogACT (arXiv 2024, Microsoft Research Asia)

CogACT proposed the most intuitive form of the merger. The VLM is not trained (frozen); its output features are passed as conditioning to a separate DiT (diffusion transformer) action expert, which generates continuous action chunks.

The VLM handles only "cognition" — understanding images and language — and the action expert is dedicated to generating "action." This separation preserves the VLM's pre-trained knowledge, and it was also observed that scaling up only the action expert improves performance.

Specifically, the cognition features the VLM extracts are injected into each layer of the DiT action expert through AdaLN (adaptive layer normalization), so the action-generation process itself is modulated by "what language command was given in what scene."

The limitation is that because the VLM is frozen, it is never trained to selectively extract the information that matters for robot actions.

> ### 💡 AdaLN (Adaptive Layer Normalization)
>
> The way a DiT injects conditioning information. It works on the same principle as FiLM from BC-Z (see → 1-1-2), which transformed intermediate features as $\gamma \times \text{feature} + \beta$, but it is applied in a different place.
>
> FiLM is applied in CNNs, whereas AdaLN is applied inside a transformer.
>
> A transformer normalizes values at each layer to stabilize their distribution, and AdaLN applies a scale $\gamma$ and shift $\beta$, determined by the input condition, right after that normalization.
>
> In other words, it is a structure in which "the way each transformer layer processes information changes depending on the condition."
>
> It was proposed in the original DiT paper (CVPR 2023) and is standard in diffusion-transformer action experts such as CogACT and GR00T N1.

##### 1-4-1-2. $\pi_0$ (arXiv 2024, Physical Intelligence)

If CogACT set out the basic form of the merger — "untrained VLM + separate action expert" — Physical Intelligence's $\pi_0$ took that structure a step further.

CogACT was a separated structure that left the VLM untrained and passed only its output features to a separate DiT. $\pi_0$ instead integrates the VLM and the action expert inside one transformer (the Transfusion approach) and adopts the more efficient flow matching in place of diffusion.

> ### 💡 The Transfusion approach
>
> A typical AI model handles one kind of data: a text model learns only text, an image-generation model only images.
>
> Transfusion is a structure in which one model learns different kinds of data at the same time, each with the training method that suits it.
>
> The concept was originally proposed in image generation: one model learned text by "guess the next word" and images by "remove the noise," simultaneously.
>
> $\pi_0$ applies this idea to robots, learning language by "guess the next word" (cross-entropy loss) and robot motion by "recover the motion from noise" (flow matching loss).
>
> In other words, one model gains both "the ability to understand language" and "the ability to generate continuous robot motion."

Specifically, $\pi_0$ uses PaliGemma, a VLM developed by Google, as its backbone. For robot control, though, it adds tokens for the robot state (joint angles) and actions to the input sequence.

These tokens are a new modality the VLM never saw during pre-training, so to process them separately from the existing image and language tokens, the single transformer holds two sets of weights that share the self-attention computation.

- **Weights 1 (VLM expert):** processes image and language tokens
- **Weights 2 (action expert):** processes robot state and action tokens

> ### 💡 Self-attention
>
> The core operation of a transformer: a mechanism in which every token in a sequence looks at every other and computes, as weights, "which tokens matter to me."
>
> In the sequence "the robot picks up the red cup," for example, the token "picks" gives high weight to "red" and "cup" through self-attention, and lower weight to the less important "the robot."
>
> Through this process, each token's representation takes on contextual information.
>
> In $\pi_0$, action tokens refer to image and language tokens through self-attention, learning "which visual information in the current scene matters for this action."

All input tokens (image, language, robot state, action) are merged (concatenated) into one sequence and go through the same attention computation.

The direction of reference, however, is restricted by an attention mask. After attention, each token is routed to its own expert for processing.

```
Input sequence (concatenated into one):
[img1, img2, ..., lang1, lang2, ... | state | action1, action2, ..., action50]
 ←───── processed by VLM expert ─────→  ←───── processed by action expert ─────→

Processing in each transformer layer:
  (1) Self-attention: all tokens join one attention computation (direction limited by mask)
  (2) MLP branch:  img/lang tokens     → VLM expert
                   state/action tokens → action expert
  → repeated across every layer
```

Unlike CogACT, which compresses the VLM output into a single vector and passes it through AdaLN (see → 1-4-1-1), $\pi_0$'s action expert shares the attention blocks and therefore refers directly to every VLM token, allowing finer visual-action alignment.

> ### 💡 Blockwise causal attention mask ($\pi_0$'s rules for referring to information)
>
> $\pi_0$ processes image, language, robot state and action tokens merged into one sequence, and it sets rules for "who can refer to whose information."
>
> As an analogy: a new hire (action) may freely consult the analysis report a senior researcher (image/language) wrote, but the new hire's notes must not change what the senior's report says.
>
> ```
> Block 1: [image, language]    Block 2: [robot state]    Block 3: [action]
>
> Who can attend to whom:
>
>                     Block 1    Block 2    Block 3
>                    (img/lang)  (state)   (action)
> Block 1 (img/lang):   ✅         ❌         ❌
> Block 2 (state):      ✅         ✅         ❌
> Block 3 (action):     ✅         ✅         ✅
> ```
>
> The key to this design is that image and language tokens are not influenced by action tokens.
>
> As a result, the visual-language knowledge the VLM learned from the web (e.g. "glass cups break easily") is preserved intact, uncontaminated by training on robot data.
>
> Action tokens, conversely, can refer to all the information, so they make fine use of the scene as the VLM understood it to generate precise motion.

> ### 💡 Diffusion vs. flow matching
>
> Both are generative models that start from random noise and produce the desired output (robot motion), but they take different routes to the destination.
>
> Diffusion is like coming down a winding mountain road. It follows a mathematically complex path (DDPM's forward/reverse process), so reaching the destination takes many steps (denoising steps).
>
> Flow matching is like coming straight down a direct road. It learns a simple interpolation that connects noise and the original in a straight line, so it reaches output of the same quality in fewer steps.
>
> The destination (generation quality) is the same, but flow matching gets there faster, which is an advantage where speed matters, as in real-time robot control. That is why $\pi_0$ adopted flow matching instead of diffusion.

$\pi_0$ achieved cross-embodiment generalization across 8 robot embodiments and performed demanding dexterous manipulation tasks such as folding laundry.

However, because it runs the whole model synchronously, inference speed tops out at ~10Hz, which limits whole-body control of humanoids that need fast reactions.

#### 1-4-2. [Era 3②] The two-stage System 2 (cognition/planning) – System 1 (control) trend

CogACT and $\pi_0$ run the VLM and the action expert synchronously in one pipeline. But VLM inference (image encoding + language understanding) is heavy and slow, while a robot's joint control has to be light and fast.

To resolve this speed mismatch, architectures emerged that separate cognition and control asynchronously.

> ### 💡 System 1 / System 2
>
> A dual-process theory of human cognition proposed by Nobel laureate in economics Daniel Kahneman in *Thinking, Fast and Slow* (2011).
>
> System 2 handles slow but conscious, logical thought (e.g. solving a math problem); System 1 handles fast, intuitive, automatic responses (e.g. riding a bicycle).
>
> In AI, the concept has recently been borrowed heavily to strengthen LLM reasoning. OpenAI's o1/o3 models taking "time to think" with chain-of-thought rather than answering immediately is a System 2 approach, while a typical LLM's immediate next-token prediction corresponds to System 1.
>
> Robotics applies it even more directly: architectures emerged that physically separate slow but rich scene understanding (VLM = System 2) from fast joint control (action expert = System 1) and run them asynchronously in parallel.

##### 1-4-2-1. GR00T N1 (arXiv 2025, NVIDIA)

Where $\pi_0$ integrated the VLM and the action expert inside one transformer, NVIDIA's GR00T N1 adopts Daniel Kahneman's System 2/System 1 theory, introduced above, as an explicit architectural design principle, building the two as physically separate networks.

A VLM has rich visual-language reasoning learned from web-scale data, but it is slow. To achieve Era 3's two goals together — "using VLM knowledge" and "generating high-frequency continuous actions" — separating the VLM's slow cognition from fast control is the natural conclusion, and GR00T N1 was the first to realize this structure.

- **System 2 (cognition/planning, ~1Hz):** The VLM processes the image and language to understand "what to do in the current scene," and compresses the result into a task embedding
- **System 1 (control, ~30Hz):** A separate network, a DiT-based action expert, takes the task embedding + robot observations as input and generates, by flow matching, continuous action chunks for "how to actually move the joints" (no image or language needed)

Designing System 2 and System 1 as separate networks meant GR00T N1 faced trade-offs on two fronts: how to pass the VLM's information to the action expert, and how to run each network.

- How information is passed
    - $\pi_0$'s action expert refers directly to every VLM token (e.g. 1,200 visual tokens) by sharing the attention blocks. Given the command "pick up the blue cup next to the red cup," the action expert can attend separately to the visual tokens for "red cup" and for "blue cup" and pin down the exact position.
    - GR00T N1's System 1, by contrast, receives only the task embedding in which the VLM has compressed the whole scene into one vector. In compressing the spatial detail of 1,200 tokens into one vector, fine information such as "exactly where each cup is" can be lost.
    - In other words, $\pi_0$ allows fine token-level visual-action alignment, while GR00T N1 passes on only the global meaning of the scene.
- How the networks run

    ```
    Asynchronous operation over time:
    System 2: [plan1.............] [plan2.............] [plan3...........]
    System 1: [a][a][a][a][a][a][a][a][a][a][a][a][a][a][a][a][a][a]...
               While System 2 updates the plan, System 1 keeps generating actions from the previous plan
    ```

    - In $\pi_0$ the VLM and the action expert are integrated in one transformer, so every action generation has to run the full forward pass, including image encoding. Image encoding is compute-heavy, so ~10Hz is the ceiling.
    - In GR00T N1 the two networks are physically separate, so while System 2 (VLM) updates the task embedding about once a second, System 1 (DiT) generates actions independently, using only the embedding it already received and the robot observations.
    - System 1 does not need to process images, so it is computationally light and capable of high-frequency control at ~30Hz and above.

This physically separated structure also helps with swapping modules and distributed deployment. You can swap in a larger VLM alone, or run the VLM in the cloud and the DiT on the robot's onboard GPU — flexible configurations like these become possible.

For these reasons, GR00T N1 is regarded as the starting point for research on real-time humanoid control.

##### 1-4-2-2. $\pi_{0.5}$ (arXiv 2025, Physical Intelligence)

Physical Intelligence extended $\pi_0$ into a plan-and-execute structure in which the VLM actually generates natural-language subtask plans.

In $\pi_0$, the VLM expert's output was used only for the action expert to refer to through attention (no text generation). In $\pi_{0.5}$, the VLM backbone also performs its original role of generating text: the language head from the VLM's original architecture is kept so it outputs text.

The same VLM backbone, on one side, provides visual-language information to the action expert and, on the other, generates natural-language subtasks such as "1. pick up the plate → 2. put it in the sink → 3. wipe it with a cloth." The action expert takes these subtasks as an additional condition and executes each step's action chunk in sequence.

The advantage of this structure is that the plan becomes visible to people in an explainable form, and it can be corrected. If the robot makes a mistake, the plan can be revised by verbal feedback: "no, not that one — the one on the left."

It also trains the web-scale data used for VLM training (image + text) together with robot data through co-fine-tuning, achieving the first open-world VLA that works in real homes outside the training environments.

However, it keeps $\pi_0$'s synchronous execution structure as is, so inference speed stays at ~10Hz, which limits its use on humanoids.

> ### 💡 GR00T N1 vs. $\pi_{0.5}$ — how the plan is represented
>
> GR00T N1's System 2 output is a vector (an embedding), so people cannot inspect or modify its content. $\pi_{0.5}$'s plan is natural-language text, so people can read and correct it. For transparency and human-robot collaboration, this is an important design difference.

##### 1-4-2-3. Helix 02 (2026, Figure AI)

GR00T N1 achieved ~30Hz control with a two-tier System 2 (cognition/planning) and System 1 (control) structure, but humanoids have one more fundamental problem.

An arm robot (Franka and the like) sits on a fixed base and needs no balance, but a humanoid standing on two feet is always at risk of falling.

Lifting an object shifts the center of mass and reaching out tilts the body, so instant balance correction on the order of milliseconds is needed. At 30Hz (~33ms intervals), this reflexive correction is not fast enough.

To solve this, Figure AI proposed a three-tier asynchronous architecture modeled on the human nervous system's "cerebral cortex (cognition) → motor cortex (control) → spinal reflexes (balance)."

- **System 2 (cognition/planning, ~7–10Hz):** The VLM processes camera images and natural-language commands to understand "what to do in the current scene" and compresses it into a task embedding. (Same role as GR00T N1's System 2)
- **System 1 (whole-body control, ~200Hz):** Takes System 2's task embedding along with robot observations, wrist cameras and tactile sensors, and generates whole-body joint targets. (Specific model architecture not disclosed)
- **System 0 (reflex control, ~1kHz):** A neural network trained on 1,000 hours of human motion data takes only joint sensors, the IMU and contact sensors as input and performs millisecond-level balance, contact and whole-body coordination. On the joint targets System 1 generates, it applies fine adjustments at 1kHz along the lines of "moving like this will make you fall, so correct the ankle torque by this much."

> ### 💡 Why does System 0 not look at images?
>
> Going from System 2 → System 1 → System 0, the input dimension drops sharply. (image (150K) → task embedding (4K) → joint targets (30) → torque corrections (30))
>
> System 0 in particular processes only low-dimensional signals — joint angles, IMU, contact sensors — which makes an extremely fast 1kHz cycle possible.
>
> Skipping image processing as the key to speed is the same design philosophy as GR00T N1's System 1.

---

## Section 2. How VLAs are trained (training algorithms)

### 2-1. Training data

> ### 💡 In one line
>
> **Robot training data sorts into a 2×2: "does a robot appear?" × "is there an action label?" The core dilemma is that the most direct data (robot teleoperation) is expensive, and the most plentiful data (internet video) has no action labels.**

Training a robot foundation model ultimately takes data. How robot training data can be used depends entirely on its nature, so it is worth mapping the data landscape first.

The two most intuitive criteria for classifying robot training data are: **does a robot appear in the data?** and **is there an action label?**

Draw a 2×2 quadrant on these two axes and nearly all the data used in the RFM ecosystem today can be placed in it.

![The four quadrants of RFM training data](/assets/img/notes/rfm/en/data-quadrant.svg)
*The four quadrants of RFM training data — robot present or not × action label present or not*

#### 2-1-1. [Q1] Robot teleoperation data — the most direct, and the most expensive

Data collected while a person remotely operates a robot with a joystick or VR controller.

At every timestep, the robot's state (camera images, joint angles) and the action command the person issued (gripper displacement, joint torque, etc.) are recorded as a pair.

It is the "gold standard" data, the most directly usable for VLA training, but it is very expensive to collect. It requires robot hardware, a skilled operator controlling it in real time, and several minutes to collect a single episode.

The representative large public dataset today is Open X-Embodiment (OXE), built by Google's RT-X project. It combines roughly 1 million or more episodes collected from 22 robot types at 21 institutions, and it demonstrated the hypothesis that "pooling data improves robot performance."

Counting private data too, major companies — Physical Intelligence (to train π0), Google DeepMind (to train RT-2 and Gemini Robotics), NVIDIA (to train GR00T) and others — are collecting large-scale teleoperation data through their own robot fleets or partnerships.

> ### 💡 Robot fleet
>
> A robot fleet means operating many identical or similar robots.
>
> A representative example is Google DeepMind placing dozens of robot arms throughout an office and collecting teleoperation data on all of them at once to gather RT-2's training data.

#### 2-1-2. [Q3] Web-scale data — no robot and no action, but overwhelming scale

Image-text pairs, video and text collected from the internet.

It has no direct connection to robots, but its scale is in another league from Q1. If Q1 is measured in hundreds of thousands to millions of episodes, Q3 is measured in billions of image-text pairs and tens of millions of hours of video.

Because this data contains no robots, it cannot be used to teach a VLA robot actions directly, but it is used in VLM (vision-language model) pre-training to learn "what the world looks like and how language connects to vision."

Most VLAs today take a VLM pre-trained on web-scale data as their backbone and put an action head on top.

RT-2 uses PaLI/PaLM as its backbone, OpenVLA uses Llama-2 and π0 uses PaliGemma. The vision-language knowledge these backbones learned from web-scale data transfers into the robot's visual understanding and its ability to understand language instructions.

#### 2-1-3. [Q2] Derived data — originally without actions, made by extracting them

Quadrant Q2 is derived data generated by using an AI model to extract robot action labels from videos of people working (cooking, cleaning, assembly and so on). The key player here is the AI model for extracting robot action labels: the inverse dynamics model (IDM).

> ### 💡 The inverse dynamics model (IDM), and a preview of LAPA
>
> An IDM is a model that looks at two consecutive frames and infers "what action produced this change?"
>
> Traditional robotics computed actions by solving inverse dynamics equations based on physical parameters such as the robot's mass and link lengths. A learned IDM, by contrast, uses no physics equations; it learns from data "which action produced this change," looking only at the visual change between two consecutive frames.
>
> It needs no knowledge of the robot's specs and can be applied to videos of human hands, so it scales far better. But the extracted action is an abstract representation (a latent action), not an exact physical quantity, so using it for real robot control requires an additional training stage that learns to generate actions suited to the target robot hardware.
>
> **LAPA (Latent Action Pretraining, ICLR 2025)** is the work that extended this idea to foundation-model scale.
>
> Its idea — "even without explicitly defining what an action is, encode the change between frames itself as an abstract representation (a latent action)" — opened the way to using the vast supply of internet video as pre-training data.

The value of Q2 lies in its scale. Q1 (teleoperation) is limited in how far it can scale because of collection cost, but videos of people working exist on the internet in practically unlimited supply. If an IDM can extract action information from these videos, the scale of pre-training data can grow dramatically.

#### 2-1-4. [Q4] Robot video (no action labels) — still of limited use, but growing in value

Data in which a robot is filmed, but which has no action labels because it is not a teleoperation log.

This covers videos of robots exploring autonomously, robot demo videos (YouTube and the like), and cases where only video was recorded, without action logs, while a deployed robot worked.

There are few large public datasets made purely of Q4 today, because most robot datasets are collected by teleoperation and come with action labels (Q1).

But Q4's potential is growing. As with Q2, applying LAPA or an IDM converts it into Q1 for RFM pre-training, and even without action labels it can be used to pre-train world models by next-frame prediction.

---

### 2-2. Pre-training: laying the foundation of robot intelligence, general-purpose or focused on a particular industrial setting

> ### 💡 In one line
>
> **The stage that teaches robot motion to an AI that has inherited internet knowledge. Teaching directly with robot teleoperation data is the baseline, but an approach in which AI infers motion information from internet video on its own is rising fast.**

Pre-training has two goals.

1. **Transfer the knowledge of a web-scale VLM (vision-language model) to the robot**
    - To understand the instruction "pick up the red cup," the model must already know what color "red" is, what a "cup" looks like and what act "pick up" means.
    - This knowledge is learned from billions of image-text pairs on the internet (Q3 data).
2. **Build a general-purpose action space across diverse robots and environments**
    - The goal is not a model that works on one particular robot but an action representation that works across many kinds of robots (articulated arms, humanoids, mobile manipulators and so on).
    - For this, large-scale teleoperation data (Q1) and derived data extracted from human video (Q2) are used.

> ### 💡 Action space
>
> If 1-2-1-1 defined the action space as "the full range of actions a robot can take," from a training perspective the action space determines "the form of the answer the model must output."
>
> For example, a 7-axis robot arm's action space is a 7-dimensional continuous space, and the bimanual robot ALOHA's is a 14-dimensional continuous space.
>
> To train one model on data from different robots, these heterogeneous action spaces have to be unified.
>
> RDT-1B (see → 1-2-2-2. RDT-1B (ICLR 2025)) aligning every robot's actions to the maximum dimension and masking the empty dimensions is the representative solution.
>
> **Building a general-purpose action space in pre-training matters because it becomes the starting point for adapting to the target robot in the later post-training stage.**
>
> The more thoroughly the pre-training stage learns the action spaces of diverse robots, the stronger the foundation for adapting quickly to a new robot with only a small amount of data.

Most VLAs today achieve both goals by taking an already pre-trained VLM as the backbone, attaching an action head on top and training on robot data.

> ### 💡 The VLA pre-training pipeline
>
> **Step 1: Prepare a pre-trained VLM**
>
> The first stage is VLM pre-training, but VLA researchers do not carry it out themselves; they bring in a VLM that has already finished training.
>
> Models such as PaliGemma, Llama-2 and PaLI already have vision-language knowledge from training on billions of examples of Q3 data (web-scale image-text pairs), and VLA researchers start by taking such a model as the backbone.
>
> **Step 2: Teach the VLA actions with robot data**
>
> The second stage is VLA pre-training proper. An action head is added on top of the imported VLM backbone, and the action space is learned from Q1 data (teleoperation).
>
> Optionally, Q2 data (derived data) is used as well to expand the data scale.
>
> In this process the VLM backbone is either fully frozen or fine-tuned at a very low learning rate so its existing knowledge is not damaged.
>
> After these two stages you have a general-purpose VLA that "sees the world, understands language and outputs actions."

The key question for achieving the two goals above comes down to "where, and how, do we get the data to teach the robot?"

The most intuitive way to teach a robot motion is to collect demonstration data of a person operating the robot directly and have the robot imitate it: "in this situation, move like this" (**behavior cloning**).

Mixing web data into this training keeps the visual-language knowledge the VLM originally had from being damaged (**co-fine-tuning**). These two techniques are the standard approach to VLA pre-training today.

But there is a fundamental bottleneck. Collecting data by having people operate robots directly is very expensive, so however hard you try, a few hundred thousand episodes is the limit. The internet, meanwhile, holds hundreds of millions of videos of people working with their hands — cooking, cleaning, assembling. The problem is that these videos have no robot joint values (action labels).

The rapidly rising **learning from observation (LfO)** approach breaks through this limit head-on.

The core idea is simple. If a cooking video shows "an onion on the cutting board" followed by "the onion, sliced," the AI compares the two scenes and infers the motion information on its own: "in between, someone must have cut the onion with a knife." Pre-training on the motion representations extracted this way (latent actions) makes large-scale training possible without robot demonstration data.

The representative work, LAPA, demonstrated the approach's potential by showing that a model pre-trained on human video alone surpasses, in success rate, a model pre-trained on robot data.

Below, we look at these three approaches in order: **behavior cloning, co-fine-tuning/knowledge insulating, and learning from observation.**

#### 2-2-1. Imitation learning & behavior cloning

##### 2-2-1-1. Imitation learning

In traditional robotics, ways of "teaching" a robot have split into two broad branches. One has engineers program the robot's motions directly in code; the other has a person demonstrate and the robot learn by watching.

Robotics called the latter **learning from demonstration (LfD)**, and it was studied very actively in industrial-robot research in the 1980s and '90s.

> ### 💡 The LfD (learning from demonstration) pipeline
>
> The typical LfD pipeline of the time went like this.
>
> A person physically holds the robot arm and demonstrates the desired motion (kinesthetic teaching). The robot's joint angles, velocities, forces and so on are recorded as time-series data. That trajectory is then modeled mathematically into a reproducible form.
>
> Representative tools at this stage were traditional trajectory representations such as dynamic movement primitives (DMP) and GMM/GMR (Gaussian mixture model/regression).

As this LfD tradition met the ML/AI community, it was reframed under a more general term: **imitation learning (IL)**.

##### 2-2-1-2. Behavior cloning

The term **behavior cloning (BC)** was first established in Carnegie Mellon's 1989 autonomous-driving research (ALVINN).

It took camera images as input and predicted the steering angle directly with a neural network — a very simple attempt by today's standards — but the core idea of "cloning human behavior data as is" started here.

BC's training setup is the same as ordinary supervised learning. You collect large amounts of data from an expert (a person) operating the robot — pairs saying "in this situation (state), this action was taken" — and minimize a loss so the model takes the state as input and outputs the action.

**The pre-training of today's major VLA models is also, at heart, this BC.** Only the scale has changed. OpenVLA trained on 970,000 robot episodes, and π₀ performed BC on dexterous-manipulation demonstration data on the scale of thousands to tens of thousands of hours.

The reason BC has served for so long, and still serves, as the default training method is clear. It is simple to implement, requires no elaborate reinforcement learning algorithm design, and can be trained right away as long as you have high-quality demonstration data.

But BC has three fundamental limitations.

1. **Distributional shift**

    ```
    [Error accumulation from distributional shift]

    Training:    expert trajectory   ●──●──●──●──●  (states the expert visited)
    Deployment:  VLA trajectory      ●──●──◎──◎──◎  (◎ = state not in training data)
                                           ↑
                                     first mistake → every later state is unknown territory
                                     → errors accumulate and diverge
    ```

    - It is like a novice driver who has only practiced at driving school panicking on their first trip into the city. Just as the driver cannot cope with situations never met at the school, a model trained with BC cannot output the right action when it meets situations that were not in its training data.
    - In training, BC sees data only in states the expert visited. But if the VLA makes even a slight mistake at deployment, it enters a state the expert never visited.
    - It has never learned the correct action in that state, so it makes another mistake, which pushes the next state somewhere even stranger, and so on: errors compound over time.
    - Formally, in a T-step task BC's error grows quadratically. That is, the longer the task, the failure probability grows with the square.
    - This is fine for short tasks like pick-and-place, but for long-horizon tasks lasting tens of minutes, like "tidy up the kitchen," BC alone becomes very fragile.
    - The standard approach today is to **complement it with RL or DAgger-style techniques in the post-training stage**, described later.
2. **Preserving VLM knowledge**
    - Today's VLAs take a VLM pre-trained at web scale (PaliGemma, Llama-2, etc.) as the backbone, attach an action head on top and train on robot data.
    - But training on robot data alone with BC can damage the vision-language knowledge the VLM originally had. This is called catastrophic forgetting.
    - For example, the VLM may have known the commonsense fact that "glass cups break easily, so pick them up carefully," and fine-tuning on robot data alone can make such knowledge disappear.
    - The standard approach today is to **complement it with co-fine-tuning techniques in the pre-training stage**, described later.
3. **Limits on data scale**
    - Q1 data (teleoperation) is expensive to collect, which puts a fundamental limit on scaling. The internet holds practically unlimited videos of people working, but they have no action labels and cannot be used for BC directly.
    - Work that **complements this with learning-from-observation techniques in the pre-training stage**, described later, is taking hold beyond academia and in industry.

#### 2-2-2. Strategies for preserving VLM knowledge: co-fine-tuning & knowledge insulating

To briefly revisit the catastrophic forgetting problem from BC's second limitation: there was a dilemma — preserve the VLM's knowledge and action learning falls short; train actions well and the VLM's knowledge is damaged.

Ways of resolving this dilemma fall into two broad branches.

One is **co-fine-tuning**, which trains web data and robot data at the same time within the same training loop; the other is the **knowledge insulating** approach, which structurally preserves the VLM backbone and carries out action learning separately.

##### 2-2-2-1. Co-fine-tuning: RT-2 (Google DeepMind)

The model that implemented co-fine-tuning most directly is Google's RT-2 (see → 1-3-1. Robotics Transformer 2 (PMLR 2023, Google DeepMind)).

RT-2 trained on VLM pre-training data (web image-text) and robot teleoperation data at the same time, mixed at roughly a 1:1 ratio. This was possible thanks to RT-2's architectural design.

RT-2 converted robot actions into discrete tokens and incorporated them into the VLM's vocabulary, so web data and robot data became training data of structurally the same format.

> ### 💡 How web data and robot data are trained with the same classification scheme in RT-2
>
> A VLM is originally a model that "looks at an image and generates text." Its training loss is next-token prediction — a cross-entropy loss for guessing "what is the next token?"
>
> RT-2 quantized robot actions (e.g. 7 joint angles) into 256 bins, making them the same form as text tokens.
>
> As a result, web data becomes "image → predict text tokens," and robot data becomes "image + language instruction → predict text tokens + action tokens."
>
> Both use the same next-token prediction loss (cross-entropy), so web data and robot data can be mixed naturally, batch by batch, within one training loop.
>
> No separate loss design or multi-task training framework was needed.

Thanks to this co-training, RT-2 could transfer the visual reasoning it learned on the web directly to robot control.

Representative examples are recognizing and manipulating objects not in its training data (e.g. "pick up the Coca-Cola can") and carrying out instructions that require semantic reasoning, such as "pick out the things that are trash and throw them away."

##### 2-2-2-2. Knowledge insulating

Not every VLA co-trains the way RT-2 does.

In models that treat actions as continuous vectors rather than discrete tokens (diffusion- or flow-matching-based), the text loss on web data and the action loss on robot data take different forms, so RT-2-style co-fine-tuning is structurally difficult.

$\pi_0$ (see → 1-4-1-2. $\pi_0$ (arXiv 2024, Physical Intelligence)) took the most structural approach. It uses the PaliGemma VLM as its backbone but fully separates action generation into a separate flow-matching expert.

In $\pi_0$'s follow-up work on knowledge insulating (arXiv 2025), the VLM part of the overall module structure is either not trained (frozen) or trained at a very low learning rate so its original knowledge does not change, and only the action expert is trained intensively on robot data.

More recently, as the solution to this knowledge-insulating problem, the dual-system structure that separates the VLM (high-level reasoning) and the action expert (real-time motor control) at the architecture level (NVIDIA GR00T N1, Physical Intelligence $\pi_{0.5}$, Figure AI Helix 02) is becoming standard.

This separation makes it possible to keep intensive training of the action expert on robot data from affecting the VLM backbone, so knowledge preservation and action learning can coexist structurally.

#### 2-2-3. Learning from observation (LfO)

If co-training was a strategy for "learning robot actions while preserving VLM knowledge," LfO tackles BC's third limitation, data scale, head-on.

Q1 data (teleoperation) is expensive to collect and limited in how far it can scale, while the internet holds practically unlimited videos of people working. The problem is that these videos have no action labels.

LfO is the approach of "extracting information a robot can learn from, even from video without action labels."

LfO's key tool is the **inverse dynamics model (IDM)**, introduced briefly earlier.

Traditional robotics computed actions by solving inverse dynamics equations based on physical parameters such as the robot's mass and link lengths. A learned IDM, by contrast, uses no physics equations; it learns from data "which action produced this change," looking only at the visual change between two consecutive frames.

It needs no knowledge of the robot's specs and can be applied to videos of human hands, so it scales far better.

But the extracted action is an abstract representation (a latent action), not an exact physical quantity (joint torque, end-effector coordinates, etc.), so using it for real robot control requires an additional training stage that learns to generate actions suited to the target robot hardware.

LAPA is what extended this IDM idea to foundation-model scale.

##### 2-2-3-1. Latent Action Pretraining from Videos (ICLR 2025)

LAPA was carried out by KAIST Professor Minjoon Seo (CEO of Config Intelligence) together with Seonghyeon Ye and Joel Jang (now at NVIDIA GEAR Lab), who were students at the time, as joint research with Microsoft Research and NVIDIA.

It is the first paper to propose pre-training a VLA on abstract representations (latent actions) extracted from internet video without action labels, and it can be called the origin of LfO-based large-scale pre-training.

> ### 💡 Where the LAPA authors went next
>
> Professor **Minjoon Seo** is now an associate professor at the KAIST Kim Jaechul Graduate School of AI, and he co-founded the physical AI startup **Config Intelligence**, where he serves as CEO.
>
> Config Intelligence industrializes the idea LAPA put forward — "make robot training data from human behavior data" — producing and supplying human behavior data for robot learning at scale while developing its own robot foundation model.
>
> LAPA's lead authors, **Seonghyeon Ye** and **Joel Jang**, later joined NVIDIA GEAR Lab (the Project GR00T team led by Jim Fan and Yuke Zhu), where they are driving world model research.
>
> Joel Jang leads the world model team at GEAR Lab and, together with Seonghyeon Ye, led the research series running DreamGen (2025.05) → DreamDojo (2026.02) → DreamZero (2026.02).
>
> DreamZero in particular put forward a new concept, the World Action Model (WAM), which models video and actions together, and Jensen Huang presented it at GTC 2026 under the name **NVIDIA GR00T N2** as the next-generation robot foundation model.
>
> In other words, the latent-action methodology that began with LAPA developed at NVIDIA into the World Action Model, which unifies world models and RFMs.

Human egocentric video data is captured from people, so it is hard to equip with action labels synchronized to a robot. LAPA takes this problem on directly.

> *Even without explicitly defining what a robot action is, can't the change between consecutive frames of ordinary video itself be seen as an action?*

An analogy for the core idea: if a cooking video shows "an onion on the cutting board" followed by "the onion, sliced," then in between there must have been the motion "cut the onion with a knife."

LAPA has the AI compare the earlier and later scenes of a video and infer on its own that "some motion must have happened in between."

The motion representation inferred this way is called a latent action. It is not an exact physical quantity like a robot's joint angles or gripper coordinates, but it carries the essential information: "there was a change of about this size in this direction."

LAPA matters because of the data. Latent actions can be extracted even from ordinary video shot without robots, so the practically unlimited video on the internet (Q3) becomes usable as pre-training data.

Large-scale training becomes possible without expensive robot teleoperation data.

The paper's most striking result is that LAPA, pre-trained on human video alone, outperformed OpenVLA, pre-trained on robot data (Bridge V2), in average success rate.

This shows that manipulation knowledge learned from human video can transfer effectively to robots despite the large difference between robot and human bodies, and points to the potential for large-scale use of human egocentric video data.

##### 2-2-3-2. DreamDojo (arXiv 2026, NVIDIA)

DreamDojo belongs to the same line of work as LAPA — "extract latent actions from video without action labels and pre-train on them" — but approaches it from the world-model side.

It was released by an NVIDIA research team and used 44,000 hours of human egocentric video for pre-training.

> ### 💡 Where DreamDojo sits
>
> DreamDojo is research led by LAPA's lead authors **Seonghyeon Ye and Joel Jang** after they joined NVIDIA GEAR Lab, and it used 44,000 hours of first-person human video for world model pre-training.
>
> If LAPA built a VLA from latent actions, DreamDojo built a world model with the same methodology, making it the intermediate step in the path toward DreamZero (GR00T N2), where RFMs and world models are unified.

It differs from LAPA in two key ways.

First, LAPA represented the change between frames as discrete robot actions in codebook form (abstract latent actions), whereas DreamDojo extended them to continuous robot actions.

Discrete codes "pick one of 256," which limits their expressiveness, but continuous robot actions are real-valued vectors and can capture even the subtle differences of fine manipulation.

Second, if LAPA showed the potential of abstract latent actions for building a VLA, DreamDojo applied the same abstract latent-action methodology to a world model, building a model that simulates "how would the world change if this action were taken?"

That amounts to verifying that latent actions work not only for VLAs but also for world models, and it became the cornerstone for the later unification, in DreamZero, into a World Action Model (WAM) that models video and actions together.

---

### 2-3. Post-training: adapting to the target environment and improving on its own (self-improving)

> ### 💡 In one line
>
> **The stage that fits a model trained for general purposes to the actual target robot and has it improve on its own through real-world experience. By combining human correction with reinforcement learning, self-improving robots whose performance keeps rising after deployment are becoming real.**

If pre-training was the stage that "lays the foundation of general-purpose robot intelligence — seeing the world, understanding language and outputting actions," post-training is the stage that adapts that foundation to a specific robot and task and improves performance through real experience after deployment.

For example, pre-training might have used Franka robot arm data while the actual deployment target is the bimanual ALOHA robot. Joint structure, sensor configuration and the way actions are represented can differ, so it will not work as is.

The distributional shift problem in behavior cloning (see → 2-2-1-2) — which arises when the states the VLA's own actions produce in the real environment differ from the training data — also has to be solved in the post-training stage.

The goals of post-training come down to two.

1. **Adaptation to the target robot's action spec:** The key is to keep the general knowledge built in pre-training (vision-language understanding, manipulation patterns, etc.) while adjusting only the output layer to the target robot
2. **Self-improvement after deployment:** Continuously improving the VLA model using the experience data (successes, failures, human interventions, etc.) the robot accumulates while working in the real environment

The data used in post-training is mainly Q1 (small-scale teleoperation data from the target robot) and experience data collected after deployment.

Where pre-training used hundreds of thousands to millions of episodes, post-training is far smaller, on the scale of tens to thousands of episodes.

#### 2-3-1. Behavior cloning (BC)

The most basic post-training method is to train a VLA pre-trained on large-scale data with a small amount of target-robot data, by behavior cloning.

Technically it uses the same BC as pre-training, but the data and the goal differ. If pre-training "combines data from many robots to build a general-purpose action space," post-training "fits the action space of a specific robot precisely, using that robot's data."

In practice, the most important design decision is "which part of the model to train."

Post-training the whole model fits the target robot well, but training everything on a small amount of data can damage the general knowledge built in pre-training (catastrophic forgetting).

Fine-tuning only the action head, on the other hand, preserves knowledge but may not adapt enough.

OpenVLA (see → 1-3-2. OpenVLA (arXiv 2024)) solved this with LoRA (low-rank adaptation). Instead of changing the weights of the whole model, it adds small low-rank matrices to each layer and trains only those.

The original weights stay fixed, so pre-trained knowledge is preserved, while the low-rank matrices handle the fine adjustment to the target robot.

#### 2-3-2. DAgger (Dataset Aggregation) (AISTATS 2011)

DAgger is an algorithm published by lead author Stéphane Ross, then in Carnegie Mellon University's Machine Learning Department, with his advisor J. Andrew Bagnell.

The core is simple: solve behavior cloning's distributional shift by "keeping the expert involved inside the training loop."

> ### 💡 J. Andrew Bagnell and Aurora Innovation
>
> Carnegie Mellon University Professor **J. Andrew Bagnell** has long researched robot learning and structured prediction at the CMU Robotics Institute. He later co-founded the self-driving startup **Aurora Innovation** with Chris Urmson, former leader of Google's self-driving project (now Waymo).
>
> Aurora listed on Nasdaq through a SPAC in 2021 and focuses on autonomous driving (L4) for heavy trucks.

To revisit BC's problem: training data is collected only in states the expert visited, but at deployment the VLA visits states different from the expert's. DAgger closes this gap with a simple intuition: "then let's get expert labels in the states the VLA model actually visits, too."

With the aim of "aggregating the dataset," each round collects additional expert labels on the state distribution the VLA model actually produces, merges them with the earlier data and retrains. This steadily narrows the gap between the states the VLA model visits and the state distribution of the training data.

But its practical limits are large. Every round requires actually running the robot, and each time an expert (a person) has to be there adding labels. With hundreds of robots deployed, having a person step in every time is impossible in practice.

The follow-up HG-DAgger (Human-Gated DAgger, ICRA 2019) proposes that instead of the expert labeling every state, the expert monitors the VLA model as it runs and takes over control only when it is about to do something dangerous or wrong, demonstrating directly.

It is a shift from "a teacher who is always teaching" to "a teacher who watches and steps in only when needed." Only the (state, expert_action) pairs from intervention segments are collected, which greatly reduces labeling cost.

#### 2-3-3. Advantage-based RL (RECAP): Physical Intelligence $\pi_{0.6}^*$

##### 2-3-3-1. $\pi_{0.6}^*$ (arXiv 2025, Physical Intelligence)

The model Physical Intelligence released after $\pi_0$ and $\pi_{0.5}$, and the first large-scale demonstration in the VLA field of "can a deployed model improve itself through real experience?"

$\pi_{0.6}^*$'s core technique, RECAP (RL with Experience and Corrections via Advantage-conditioned Policies), is a modern extension of the DAgger family while also taking an RL framework.

It works as follows.

1. **Coaching with correction**
    - When the robot makes a mistake while running autonomously, a person steps in by teleoperation and corrects it. This is the same structure as DAgger, covered above (see → 2-3-2. DAgger (Dataset Aggregation) (AISTATS 2011)).
    - Because it gets expert demonstrations in the states the robot actually produces, it directly solves distributional shift, BC's fundamental limitation.
    - But human correction alone has limits. Clear mistakes can be corrected, but it is hard for a person to teach, case by case, how to go faster or refine the details of fine motions.
2. **Practicing with reinforcement**
    - In segments without human intervention, it judges episode by episode whether "this attempt was a success or a failure," reinforcing successful behavior patterns and suppressing failed ones.
    - In the process, each trajectory is given an advantage value, and this advantage is fed in as a conditioning token during training.
    - At inference, it always conditions on "high advantage," steering the model to generate the behavior of successful patterns.

> ### 💡 The core of practicing with reinforcement: the value function and advantage
>
> In tidying the objects on a table, a robot performs dozens of motions.
>
> When it ultimately fails, the key is to find out "which motion was the problem?" In RL this is called the **credit assignment** problem. Two tools are used to solve it.
>
> - **The value function** is a model that predicts "how favorable the current situation is." In Go terms, it is like an AI commentator calculating in real time what the win probability is from the current position.
> - **Advantage** shows how much that win probability changed before and after a particular motion. If the probability of success drops from 80% to 30% after the robot lets a cup slip while grasping it, for example, that motion is given a low advantage score (-50). A motion that lifts the cup stably, conversely, gets a high score.
>
> During training, this score is attached to each motion like a tag, so the model learns "in this situation, this was the motion that scored high."
>
> At inference (real operation), it always runs in "top score" mode, steering it to reproduce only the motions of successful patterns.

Trained with RECAP, $\pi_{0.6}^*$ performed, in deployment environments, laundry folding (50 new kinds of clothing, new home environments), box assembly and labeling (59 boxes for chocolate packaging, in a real factory) and making espresso (running continuously from 5:30 a.m. to 11:30 p.m.).

On the hardest tasks, both throughput and success rate more than doubled, and error recovery that was impossible with BC alone emerged naturally: putting down the extra and starting over when boxes came up stacked, or unfolding a garment it had failed to fold and folding it again.

#### 2-3-4. Reward-model-based RL: DYNA-1

If RECAP, the signature algorithm of Physical Intelligence's $\pi_{0.6}^*$, implements self-improvement by scoring advantage on success or failure and feeding it into training, another RL approach trains a reward model and places it inside the post-training loop.

##### 2-3-4-1. DYNA-1 (2025, Dyna Robotics)

> ### 💡 Dyna Robotics
>
> Dyna Robotics is a physical AI startup co-founded by Lindon Gao (CEO), who sold Caper AI for <span class="tex2jax_ignore">&#36;350M</span>, and Jason Ma, a former Google DeepMind researcher.
>
> The name "Dyna" comes from the Dyna architecture Richard Sutton proposed in 1991. Sutton's Dyna is the prototype of model-based RL: "learn an environment model from real experience, generate simulated experience (imagination) inside that model, and train the VLA model further on it."
>
> After closing a <span class="tex2jax_ignore">&#36;23.5M</span> seed round co-led by CRV and First Round Capital in March 2025, it closed a <span class="tex2jax_ignore">&#36;120M</span> Series A led by RoboStrategy, CRV and First Round Capital in September of the same year, with participation from Salesforce Ventures, NVentures (NVIDIA), the Amazon Industrial Innovation Fund, Samsung Next and others.

DYNA-1 is a closed model, so the full details of its architecture and training methodology are not public.

However, putting together co-founder Jason Ma's research and the company's public tech blog, its core approach is presumed to be a structure that keeps an in-house foundation reward model permanently at the center of the post-training loop and repeatedly improves the VLA model with the reward model's feedback — that is, RM-in-the-loop training.

> ### 💡 What is a foundation reward model?
>
> In robot RL, the reward function is the function that judges "is the robot doing well right now?"
>
> Traditionally, a person had to design this reward function by hand for every task. An engineer would code a rule directly, such as "+1 if the napkin's four corners come within a certain distance of each other."
>
> When the task changes, it has to be designed again from scratch, and the more complex the task, the harder the rules are to define.
>
> Just as a "foundation model" in the LLM field performs many language tasks with one model, a foundation reward model means a reward model that can evaluate the progress of many robot tasks, generally, with one model.

A foundation reward model is implemented by using a VLM (vision language model) trained on large-scale human video from the internet to output, as the reward, how close the robot's current state is to the task-completion state (the goal).

It gives a higher reward the closer the temporal distance to the goal, so a high reward means the task is progressing smoothly, and a low one means it is not.

A pre-trained VLM holds prior knowledge (world knowledge) about the visual stages of tasks — "the motion of picking up an object," "the process of folding a napkin" — so what it learned from videos of people working can transfer to evaluating robot task progress. That is its distinguishing feature.

The key to DYNA-1 is a closed loop: placing the foundation reward model in the post-training loop so the VLA model learns in the direction that raises the reward, and automatically feeding failure cases back into improving the reward model (online adaptation) so training keeps getting better.

> ### 💡 The strategic difference between Physical Intelligence and Dyna Robotics
>
> $\pi_{0.6}^*$ and DYNA-1 share the same philosophy — "self-improvement from experience after deployment" — but aim in fundamentally different directions.
>
> Physical Intelligence aims for a generalist that works across diverse embodiments — mobile manipulators, bimanual robots, fixed robots and so on. One model performs many tasks, such as folding laundry, assembling boxes, making espresso, making beds and clearing dishes, and the same model works when the embodiment changes.
>
> Dyna Robotics, by contrast, fixes the hardware — a low-cost stationary bimanual robot — and concentrates on maximizing commercial performance per task (99%+ success rates, 24-hour continuous operation, meeting commercial quality standards).
>
> The former is "one model, many robots and tasks"; the latter is "one robot, top performance per task."
>
> That said, DYNA-1's public results so far concentrate mainly on light dexterity tasks for a stationary bimanual robot, such as folding napkins, folding laundry and filling cups. Validation in industrial settings that demand high payloads or high task complexity — manufacturing assembly, handling heavy objects, complex tool use — is hard to find on its blog.
>
> Dyna presents "general-purpose robot" and "physical AGI" as its ultimate goals, but its current business model is closer to a structure that fixes the use case and maximizes performance on that task with RL.
>
> How quickly this approach can expand to diverse industrial environments and more complex tasks will be the key question going forward.
