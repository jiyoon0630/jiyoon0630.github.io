---
layout: paper
lang: en
ref: world-model-trends
kind: tech-review
title: "World Models Today: Three Trends Spreading Through Data, Training and Evaluation"
date: 2026-06-18 12:00:00 -0700
tags: [World-Model, Robot-Foundation-Model, VLA, Video-Generation, Evaluation, Tech-Review]
summary: "Where world models came from and where they are going — generating training data, coupling into VLA training, and breaking the evaluation bottleneck. Three branches through which one technology is spreading across the whole robot development pipeline."
---

> **In short** — A world model learns physics implicitly from data instead of having a person design it explicitly. Recent work converges on a single point: **the world model is spreading into every stage of the robot development pipeline — data, training and evaluation.** This piece covers what a world model is, where it came from, and those three trends through their representative research.

---

## Introduction: The Rise of World Models, and the Move from RFM to World Action Model

This piece follows on from **[the development of RFMs (robot foundation models)](/notes/rfm-vla-development-en/)**, which I wrote up earlier, and turns to **how world models are developing** — a topic that has risen quickly in physical AI.

Writing up RFMs, I traced how VLA (vision language action) models developed: models that take a camera image (vision) and a natural-language instruction (language) and directly output the robot's motion (action).

And as the last thread there, I briefly noted that VLAs are moving past simply emitting actions toward integration with world models that can generate and predict the physical environment itself — that is, toward the world action model.

Starting exactly at that point, let me take up properly what a world model is, where it came from, and where it is heading.

**The term "world model" became a core robotics keyword over the past year or two, as big tech and the major startups — NVIDIA, Google DeepMind, Physical Intelligence — all began using it at once.**

**NVIDIA announced Cosmos at CES 2025 as a world foundation model and positioned it as core infrastructure for physical AI; Google DeepMind's Genie series generated interactive virtual environments from video alone; and RFM companies including Physical Intelligence began combining world models into their RFM/VLA models.**

So the world model is not one lab's experiment but a direction the whole industry is betting on at the same time. That said, the term spread so fast that its definition is used slightly differently by each company and each application, and the confusion is real.

In some places, "world model" means technology that generates a manipulable virtual world like a game. In others, it means a simulator producing synthetic data for robot training. In still others, it means a model coupled to a policy that predicts the future.

The applications differ while the root is one and the same, but on the surface they look like entirely different technologies — which is exactly where most people get confused.

So before the development history proper, I want to settle what a world model precisely is, and how it differs from the simulation tools we are most familiar with.

And as I noted while working through RFMs, the heart of RFM development comes down to "how do you obtain the data, how do you train on it, and how do you evaluate the result."

**The world model is a technology being used across all three of those stages — data, training and evaluation** — so understanding its trajectory should help in seeing the larger picture of physical AI.

---

## Section 1. Background: What a World Model Is

### 1-1. World model vs. physics-based simulator

The most accurate way to understand a world model is to compare it directly with the technology that most resembles it and yet differs from it fundamentally: the physics-based simulator.

- **Physics-based simulator**: simulators such as MuJoCo and Isaac Sim construct the virtual environment **explicitly**.
    - Inside the simulation live **assets a person has defined in advance**. The table, the particular robot, the object being manipulated, the structure of the room — all designed beforehand as 3D models.
    - And it needs a **physics engine that computes how those assets interact**. When the robot moves its arm, how surrounding objects are pushed and collide is reflected in the environment by explicitly computing physical laws such as mass, friction and contact.
- **World-model-based simulator**: a world model handles physics **implicitly**. Rather than a person designing the physical laws in detail, the approach is to **learn them from data**.
    - The AI model watches frame-to-frame change across a large volume of video and **directly learns "if this action happens in this situation, the next scene changes like this."**
    - Dynamics — objects falling, liquid flowing, doors opening — are already implicitly contained in internet video, and the core premise is that a sufficiently large generative model can learn them without explicit physics supervision.

![World model vs. physics-based simulator](/assets/img/notes/world-model/world-model-vs-simulator.jpg)
*World model contrasted with a physics-based simulator (source: [arXiv:2507.00917](https://arxiv.org/abs/2507.00917))*

This difference matters because of the **"sim2real gap."**

**Because a physics-based simulator approximates real physics, a policy trained in simulation structurally degrades when deployed on a real robot, through the mismatch between simulation and reality.**

This is known as the dominant failure mode of simulation-based robot learning. Dynamics such as cable routing, manipulating deformable objects, or friction across varied surfaces are very hard to reproduce exactly with a physics engine.

Because a world model is trained on real data, it is intrinsically closer to real dynamics, and it is drawing attention as **an alternative that can narrow the sim2real gap.**

**To summarize: if the physics-based simulator is the approach where "a person designs the world explicitly," the world model is the approach where "an AI learns the world implicitly from data."**

### 1-2. Preliminaries: background worth knowing

Before the development history proper, let me set out the background and key terms that come up repeatedly.

#### 1-2-1. The three layers of NVIDIA's simulation stack (Omniverse · Isaac Sim · PhysX)

Omniverse, Isaac Sim and PhysX are often mentioned together in discussions of robotics simulation, but the three are not competing tools at the same layer — they sit at different levels of abstraction.

These terms get mixed up often in practice, so their relationship is worth laying out, using NVIDIA's ecosystem as the reference.

![NVIDIA simulation stack](/assets/img/notes/world-model/nvidia-sim-stack.jpg)
*NVIDIA's simulation stack (source: [NVIDIA Developer Blog](https://developer.nvidia.com/blog/design-your-robot-on-hardware-in-the-loop-with-nvidia-jetson/))*

1. **Omniverse — the base platform (a development OS)**
    - **A collection of libraries and microservices for developing digital twin and physical AI simulation applications.**
    - It provides USD (Universal Scene Description) scene composition and RTX-based photorealistic rendering — the base platform holding up everything above it.

    > ### 💡 Digital twin (as defined in NVIDIA's ecosystem)
    >
    > "Digital twin" is a widely used word, but NVIDIA's ecosystem uses it in two senses, so the two need separating.
    >
    > - **The original sense:** a physically accurate virtual replica of a real environment (a factory, a warehouse) built inside Omniverse on OpenUSD. That is, an **explicit replica whose assets and physics a person defined**, belonging to the "simulator" family from 1-1.
    > - **The extended usage:** NVIDIA's Cosmos paper calls a world foundation model "a digital twin of the world" and a VLA model "a digital twin of itself." Here the digital twin is not an explicit replica but **the implicitly learned world model itself**.
    >
    > Since the same "digital twin" points at both the explicit simulator and the world model, it has to be read from context.

2. **Isaac Sim — the simulation application**
    - **A robotics simulation platform built on top of Omniverse.**
    - It provides high-fidelity physics (the PhysX engine) and photorealistic rendering, focused on synthetic data generation, testing and validation.
    - It is the application layer where robots are trained and tested inside a digital twin.
3. **PhysX / MuJoCo / Newton — the physics engines**
    - **The engines at the very bottom that compute the actual physics — rigid- and soft-body dynamics.**
    - MuJoCo is light and accurate on contact physics, which made it the academic standard; it can be integrated as the engine inside Isaac Sim, or used on its own.

The relationship between the three layers:

```
NVIDIA simulation stack (top → bottom)

▸ [Base platform] Omniverse
    digital twin construction · USD scenes · RTX rendering

  ▸ [Simulation] Isaac Sim
      robotics simulation app on top of Omniverse
      synthetic data generation · training · testing

    ▸ [Physics engine] PhysX / MuJoCo / Newton ...
        rigid- and soft-body dynamics (swappable)

※ MuJoCo can serve as Isaac Sim's backend and also stands alone as an
  independent physics engine (light, accurate contact physics)
```

The point worth marking here is that all three layers belong to the **physics-based simulator** family.

Since assets and physics are defined explicitly by a person, all three sit on the simulator side of the contrast with the world model (implicit).

#### 1-2-2. Glossary of key terms

Definitions for the world model terms that recur through this piece.

- **Neural trajectory**
    - **Synthetic robot video data generated by a world model.**
    - "Neural" means a neural network produced it; "trajectory" means the video of the robot's task trajectory.
    - Training data made with a world model rather than through actual teleoperation.
- **Rollout**
    - **The whole sequence of (observation, action) produced as a VLA or world model runs a task from start to finish — the complete record of one attempt (one episode).**
    - One performed on a real robot is a real-world rollout; one performed inside the imagination a world model generates is an imagined (synthetic) rollout.
- **World foundation model (WFM)**
    - **A general-purpose world model not confined to a particular domain.**
    - In the same sense as a foundation model for LLMs: train one general model, then fine-tune it for downstream use (a specific robot or environment). NVIDIA Cosmos is the representative case.
- **World action model (WAM)**
    - **A model that predicts future world state (video) and actions jointly.** (= VLA + world model)
    - Unlike a conventional VLA, which emits only actions, the key is that video generation and action generation are handled in one model at once.
    - The keyword standing for the integration of world model and VLA. NVIDIA's latest model DreamZero (Feb 2026) is the representative case.

### 1-3. Three world model trends

Looking over the development of world models, it converges on one large movement: **the world model is spreading into every stage of the robot development pipeline.**

Developing an RFM that generates robot actions (= a robot policy model) divides broadly into three stages: **(1) obtaining data, (2) training, (3) evaluation.**

![Robot policy development pipeline](/assets/img/notes/world-model/en/policy-pipeline.svg)
*The three stages of the robot policy development pipeline, and the directions in which the world model is spreading into each*

1. **[Trend 1. Data] Making training data with a world model (Section 3)**
    - The world model generates synthetic video and rollouts to supply VLA training data.
2. **[Trend 2. Training] World generation couples into the VLA architecture (Section 4)**
    - The world model's video generation ability couples directly into the VLA's own training and inference structure, with the world action model at its peak.
3. **[Trend 3. Evaluation] The world model breaks the evaluation bottleneck (Section 5)**
    - Still early, but an attempt to automate the chronic bottleneck in robot policy evaluation with a world model.

These three are not independent; the right way to read them is as one technology — the world model — spreading across every stage of robot development.

Section 2 covers the technical foundations that made these trends possible; Sections 3 through 5 take each trend through its representative research.

> ### 💡 Robot policy model
>
> A decision model that takes what the robot currently sees and senses (camera images, joint sensors) and outputs the action to take next (joint movement). The term comes from reinforcement learning.

The push to put world models inside the robot development pipeline is broadly visible — from big tech robotics labs (NVIDIA GEAR Lab, Meta, Google DeepMind) through the major startups in robot learning (Physical Intelligence, Generalist AI) to newer startups.

---

## Section 2. The Technical Pillars Holding Up World Models

World models rose recently not because of a breakthrough in any single piece of research, but because three lines of technology, developed in different fields, converged in one place.

This section sets out the three technical pillars that hold up world model development.

One is **the lineage of the concept itself (RL → games → robots)**; the other two are the **video generation** and **inverse dynamics model (IDM)** techniques that have continuously underpinned that lineage.

### 2-1. How world models developed: "reinforcement learning → games → robots"

The origin of the term is, unexpectedly, not robotics but reinforcement learning.

The 2018 paper "World Models" by RL researchers David Ha and Jürgen Schmidhuber first established the term.

The core idea: **rather than having the agent interact directly with the real environment, build an internal model that has learned the environment's dynamics (a "world model"), train the policy inside it as if dreaming (in imagination), then transfer to the real environment.**

What is interesting is that the stage where the paper first demonstrated the idea was game environments — car racing and a 2D shooter. So world models were entangled with games from the start.

> ### 💡 Basic reinforcement learning terms
>
> Because world models started in RL, RL vocabulary appears throughout this piece.
>
> - **Agent:** the entity that judges and acts on its own. A robot is the representative agent.
> - **Environment:** the world the agent acts upon. The workspace the robot sits in, a game screen, and so on.
> - **Policy:** the decision model that determines "what action to take in the current situation." The VLA model that outputs the robot's action is precisely the policy.

Two lines of application branch off from this RL root.

#### 2-1-1. The game branch — Google DeepMind Genie
Genie, released by Google DeepMind in 2024, trained on unlabeled internet game video alone and generated an action-controllable virtual environment from a single image prompt.

Genie 2 then extended this to 3D, producing playable environments that respond to keyboard and mouse input.

Genie's core is **latent action**. **Even with no action labels whatsoever on the video, it infers for itself "what manipulation produced this change" from frame-to-frame differences, and learns a control interface.**

This latent action concept is exactly the same mechanism as in the robot branch covered later (LAPA, DreamGen), and it is the link joining two entirely different applications — games and robots.

> ### 💡 Latent action
>
> Ordinary internet video carries no action labels: which button was pressed in this scene, how far the robot joint moved.
>
> **Latent action is what the model defines for itself, in abstract (latent) form, from nothing but the change between two consecutive frames — "some action must have happened in between."**
>
> Think of it as encoding the action that caused the change into an internal representation, without knowing the actual button or joint values.
>
> This concept is what makes it possible to use vast quantities of unlabeled video for training, and the specific mechanism is covered in detail with the LAPA paper later.

#### 2-1-2. The robot branch — NVIDIA Cosmos
The motivation for world models in robotics is clear: **the sim2real gap of conventional physics-based simulators is too large, so let us learn the real world from data and reproduce it.**

NVIDIA has led this direction since announcing Cosmos as a world foundation model (WFM) at CES 2025. Cosmos is a general-purpose world model for physical AI — a foundation model premised on per-domain fine-tuning.

> ### 💡 The Cosmos lineup (Predict / Transfer / Reason)
>
> Through generation 2.5, Cosmos was not one model but a family with different roles. Predict is the one directly relevant to world models, but the others are worth knowing too.
>
> - **Cosmos Transfer:** converts simulation video or structural inputs (depth, segmentation) into photorealistic video. A tool for sim2real.
> - **Cosmos Predict:** the core world model, generating future world state (video) from text, image or video.
> - **Cosmos Reason:** a vision-language model for physical AI that understands and reasons about video and physical situations.
>
> These three began as separate models and were unified into one in the latest Cosmos 3.

To summarize: **world models began in reinforcement learning and split into two applications, games (Genie) and robots (Cosmos), but only the application differs — the root runs through a single concept, "learn the dynamics of the world from data."**

### 2-2. Video generation develops, and is absorbed into world models

No account of world model development can leave out video generation.

**A world model ultimately generates, as video, "how the next scene unfolds," so progress in video generation translates directly into world model performance.**

And indeed recent world models actively absorb the latest video generation techniques as their backbone.

As diffusion models developed, they extended past image generation into video, and video quality improved dramatically. In the process, video generation architectures split broadly into two structures.

#### 2-2-1. DiT-only
**Based on the Diffusion Transformer (DiT), this generates every frame of the video at once, in parallel.**

Because all frames reference each other simultaneously through full-sequence attention, temporal consistency and motion quality are excellent.

> ### 💡 Diffusion Transformer (DiT) and full-sequence attention
>
> - **Diffusion Transformer (DiT):** a structure for diffusion models used in video and image generation, adopting a Transformer as the backbone in place of the U-Net used previously. Transformers have the property — proven in LLMs — that performance improves stably (scalably) as data and model size grow, which made them the standard structure for large video generation models. Since Open Sora, effectively every high-quality video generation model has been in the DiT family.
> - **Attention:** the core Transformer operation, computing how much each element of the output "references" which part of the input. In video, it decides "which other frames the frame being generated now will reference."
> - **Full-sequence attention (= bidirectional):** every frame of the video references every other simultaneously, past and future alike. Frame 1 references frame 10 and frame 10 references frame 1, so the whole span is bidirectionally entangled. This buys consistency and smooth motion across the video, but because future frames influence the past, "intervening mid-generation" is fundamentally impossible.

**OpenAI Sora, Google Veo, Alibaba Wan** — the high-quality video generation models we know best — belong to this structure.

There is a limit, though. **Because all frames reference even the future simultaneously, the entire sequence is determined in advance once generation begins.**

**That is, the user cannot intervene partway to change what happens next, and computation grows with the square of sequence length.** It suits making a fixed-length "finished video," but not real-time interaction.

#### 2-2-2. AR+DiT
**This generates video sequentially in chunks, applying causal attention so that each chunk references only past frames.**

Using a KV-cache, long video can be extended efficiently.

> ### 💡 Autoregressive (AR), causal attention, KV-cache
>
> All three terms come from one interlocking principle: **"generate sequentially, looking only at the past."**
>
> - **Autoregressive (AR):** take what was generated before as input and continue generating from it. The same principle as an LLM emitting words one at a time; in video, a chunk is made and then the next chunk is generated from it.
> - **Causal attention (= unidirectional):** the opposite of full-sequence (bidirectional) attention — each frame references only past frames and cannot see the future. It enforces causal order, "cause comes before effect," which matches how time actually runs, so what comes next can change with the input (action) arriving at each moment.
> - **KV-cache:** AR generation has to reference the whole past each time, so left alone the computation explodes as length grows. The KV-cache stores and reuses the reference information (key and value) already computed for past frames, which lets long video be extended efficiently. The same technique LLMs use to continue long text quickly.

The representative turning point for this structure was Diffusion Forcing (NeurIPS 2024), which combined the strengths of autoregressive generation (next-token prediction) and full-sequence diffusion.

It developed further through Self-Forcing, Rolling Forcing and others, making stable generation of long video possible.

> ### 💡 Why AR+DiT matters for world models
>
> Games or robots, there is one decisive condition for a world model to be an "interactive world": **the future must not influence the past — that is, temporal causality.**
>
> When a user or a robot enters an action at each moment, the next scene has to change accordingly; a DiT-only (bidirectional) structure that references all frames at once has already determined the whole video, so no such intervention is possible.
>
> **So the AR+DiT structure, referencing only the past, suits world models better, and the core work of "turning a video generation model into a world model" is converting a bidirectional structure into a causal one.**

#### 2-2-3. Convergence on the video world model
After OpenAI Sora put forward the view of "video generation models as world simulators," the movement to use video generation as the backbone of a world model took hold in earnest.

Google Genie 2/3 and Cosmos all sit on this movement, and a world model carrying this video generation ability is called a **video world model**.

The core point is that combining the ability to generate high-quality video (the DiT family) with the ability to interact in real time (the AR/causal family) is the path to a good video world model.

NVIDIA DreamZero, covered later, makes heavy use of a pre-trained video diffusion model (Alibaba's Wan) for exactly this reason.

### 2-3. The inverse dynamics model (IDM) and latent action

Even once a world model can generate video (the future scene), moving an actual robot requires the actions corresponding to that video.

**What bridges video and action is the inverse dynamics model (IDM)**, and the development of the IDM is the third technical pillar of world models.

> ### 💡 Inverse dynamics model (IDM)
>
> An IDM looks at two consecutive frames and infers in reverse "what action produced this change."
>
> **Classical robotics computed the action by solving inverse dynamics equations from physical parameters such as the robot's mass and link lengths, whereas a learned IDM learns the action from data using only the visual change between two frames, with no physics equations.**
>
> It needs no knowledge of the robot's specifications and can even be applied to video of human hands, which makes it highly scalable.

The representative work extending the IDM idea to foundation model scale is **LAPA (Latent Action Pretraining from Videos, ICLR 2025)**.

**LAPA was led by Professor Minjoon Seo of KAIST (CEO of Config Intelligence) together with Seonghyeon Ye and Joel Jang (formerly of NVIDIA GEAR Lab), and was the first to present a methodology for extracting latent actions from unlabeled internet video to pre-train a VLA.**

> ### 💡 The core idea of latent action
>
> If the conventional IDM was "a tool for inferring the action between two frames," LAPA inverted it and declared: "we will define the change between frames itself as a new space called latent action."
>
> **That is, even without defining robot actions explicitly, it encodes the frame change in ordinary video as a single abstract action.**
>
> This is what makes it possible to use internet video with no action labels at all, effectively without limit, for pre-training.

What is worth noting is that this latent action is essentially the same mechanism as the latent action of the game branch (Genie) seen in 2-1.

**Where Genie learned a control interface from game video as latent action, LAPA learned robot actions from robot and human task video as latent action.**

Once again the two branches, games and robots, turn out to share the same technical root.

And this latent action / IDM lineage leads directly into the trends explained in Sections 3 and 4.

- **When a video world model generates a future scene, an IDM or latent action can recover the action from it, producing training data.**
- **Go further and perform video generation and action inference in one model at once, and the world model and the VLA become a unified world action model.**

So with "the ability to generate video" (2-2) joined to "the ability to extract actions from video" (2-3), the technical foundation was laid for the world model to move past being a mere video generator and spread into every stage of the robot development pipeline — data, training and evaluation.

From Section 3 onward, let me take the three trends that unfold on this foundation in turn.

---

## Section 3. [Trend 1. Data] Making Training Data with a World Model

The first way world models enter robot development is the most intuitive: **using the video a world model produces as training data.**

**The largest bottleneck RFM companies share is the cost of acquiring data with real robots, so being able to replace data with generation is worth a great deal.**

What the work in this section has in common is that **the world model and the VLA are separate, and the world model's output (video) is first turned into "data," which then trains the VLA separately (offline).** This is distinct from Section 4, where world generation couples directly into the VLA architecture.

Within this classification, let me take three ways a world model can produce data.

The world model's involvement deepens in order: from simple appearance conversion, to generating video and then recovering actions, to putting an existing VLA inside imagination and running it.

### 3-1. Making simulation data photoreal: NVIDIA Cosmos Transfer

![NVIDIA Cosmos Transfer](/assets/img/notes/world-model/cosmos-transfer.jpg)
*Visual conversion of simulation data (source: [arXiv:2503.14492](https://arxiv.org/abs/2503.14492))*

The simplest approach **changes only the "look" of video an existing simulator produced, making it realistic.**

A physics-based simulator (Isaac Sim and the like) computes physics accurately, but its rendered video differs greatly from real camera footage. That visual difference accounts for a large part of the sim2real gap.

**Cosmos Transfer attacks that point. It takes simulation video or structural inputs (depth, segmentation maps) and converts the same scene into photorealistic video.**

What matters here is that the world model does not produce the actions. **The actions are the simulator's originals, untouched, and the world model (Cosmos Transfer) is limited to making the video's appearance realistic.**

**That is, it is the most conservative form of data generation: "take the (action, video) pairs the simulator produced and raise only the visual realism of the video, improving training data quality."**

### 3-2. Generate video, then recover actions with an IDM: NVIDIA DreamGen (CoRL 2025)

![DreamGen data generation flow](/assets/img/notes/world-model/en/dreamgen-flow.svg)
*DreamGen — generating video first, then recovering actions with an IDM to produce training data*

The next approach goes a step further: **the world model generates the video itself, and actions are recovered from that video to produce training data.**

What joins the generated video to actions is the IDM from 2-3. The representative work is **DreamGen (CoRL 2025) from NVIDIA GEAR Lab.**

The core flow:

1. Train a video world model (using NVIDIA Cosmos Predict) on a small amount of data to match the target robot.
2. Give it only an initial frame and a language instruction, and generate large volumes of synthetic robot video — neural trajectories. (→ 1-2-2, glossary)
3. Since the generated video has no action labels, recover pseudo-actions with an IDM or a latent action model.
4. Train the VLA on the resulting (video, pseudo-action) pairs.

What is worth noting is that **the world model generating the video and the IDM recovering the actions are separate external modules, and their output is, throughout, "data" for training a separate VLA.**

Once training is done, what moves the real robot is the trained VLA; the world model hands over the data and drops out at deployment.

> ### 💡 What DreamGen showed
>
> **DreamGen's significance is in demonstrating that "generalization data for new motions and environments can be made without teleoperation."**
>
> After adapting a video world model using real data for a single task (pick-and-place), **it generated synthetic data for new motions no person had ever demonstrated, and got the robot to perform them.**
>
> A case showing that the RFM's largest bottleneck — the cost of acquiring data — can be routed around with a world model.

### 3-3. Policy-in-the-loop: run an existing VLA in imagination to make data

The third approach involves the world model most deeply.

**Put an already-trained VLA inside the imagination a world model generates, have it perform the task as if for real, and use the result (the rollout) as data.** (→ 1-2-2, glossary) This is called **policy-in-the-loop**.

> ### 💡 Policy-in-the-loop and closed-loop control
>
> A real robot operates by **closed-loop control**: **observe (see) → emit an action (move) → observe again → emit the next action, repeated every moment.** (Open-loop observes everything once and then performs the actions.)
>
> **Policy-in-the-loop reproduces exactly this closed-loop control inside the imagination a world model generates.**
>
> **The VLA emits an action, the world model generates the future observation that results, and that observation goes back to the VLA for the next action.** The only difference is that the observation being "seen again" is video the world model generated rather than real camera footage.
>
> Where 3-2 (DreamGen) had the world model generating video alone, here **the world model generates the imagination and the existing VLA takes part directly in generating actions.**

The representative work here is Ctrl-World and its successor VLAW, from **Professor Chelsea Finn (Stanford, Physical Intelligence)**.

#### 3-3-1. Ctrl-World (ICLR 2026)

![Ctrl-World](/assets/img/notes/world-model/en/ctrl-world.svg)
*Ctrl-World — construct the world model, freeze it, then improve the policy inside imagination*

Ctrl-World takes a video diffusion model pre-trained on general video (Stable Video Diffusion) as its backbone and **first builds a controllable world model by training it on real robot data so that it can interact with a VLA.**

Concretely, it trains on the full DROID dataset (about 95k trajectories, 564 scenes — the roughly 76k successful trajectories the paper usually cites plus about 19k failures, so **successes and failures both included**), producing a world model that predicts future observations multi-view given an action and generates consistent video for over 20 seconds.

On top of that world model, the flow runs:

1. An existing VLA (for example π0.5-DROID) looks at the current observation and outputs an action.
2. Ctrl-World takes that action and generates the future observation of executing it.
3. The generated observation goes back to the VLA for the next action.
4. Repeat to complete a whole task (a rollout) inside imagination. (→ 1-2-2, glossary)
5. Select only the successful rollouts and further train (SFT) the VLA on their (observation, action) data.

> ### 💡 SFT (supervised fine-tuning) and closed-loop improvement
>
> SFT is AI's most basic training method — **"further training a model on correct data."**
>
> Ctrl-World treats rollouts that succeeded in imagination as a kind of correct data and trains the VLA on them by SFT.
>
> **That is, a cycle forms: execute (rollout in imagination) → select the successes → retrain on that data → execute again with a better VLA.** Feeding results back into training this way is called **closed-loop improvement**.
>
> By this route Ctrl-World raised the existing VLA's success rate by 44.7 percentage points.
>
> The key point is that Ctrl-World itself does not produce actions. **The actions come from the existing VLA throughout, and the world model's role is to predict the results of those actions as video, synthesizing "successful rollouts as data."**
>
> Once training ends, what moves the real robot is the improved VLA; the world model drops out at deployment.
>
> So the essence of this work is not an evaluation tool but data generation — training and improving a VLA with high-fidelity synthetic rollouts.

#### 3-3-2. VLAW (arXiv 2602.12063)
![VLAW](/assets/img/notes/world-model/en/vlaw-flow.svg)
*VLAW — an iterative co-improvement loop that alternately improves the VLA and the world model*

VLAW is Ctrl-World's successor (Chelsea Finn), proposing iterative co-improvement that raises the VLA and the world model alternately together. The motivation comes straight out of Ctrl-World's limitation.

**A world model trained once on demonstration data and then frozen, as in Ctrl-World, produces normal task video well but does not sufficiently capture contact-rich situations with frequent collisions or deformable objects — and above all, the dynamics of the moment failure happens.**

The coverage of such situations is simply thin in the training data. As a result, the synthetic rollouts a frozen world model produces turn out to lack the fidelity needed to actually push the policy higher. (→ 1-2-2, glossary)

VLAW breaks through this with the idea of **continuously improving the world model with real rollouts instead of building it once and freezing it.**

**That is, where Ctrl-World "improved only the VLA with a frozen world model," VLAW reinforces the world model itself with real data and raises it alongside the VLA.**

The flow:

1. Run the existing VLA on a real robot a small amount to collect real-world rollouts — gathering not only successes but **failure cases** too.
2. Fine-tune the world model on those real-world rollouts, **raising its fidelity so that it reproduces the complex dynamics and failure modes encountered during policy execution.**
3. With the higher-fidelity world model, generate large volumes of synthetic rollouts in imagination, and auto-label their success or failure with a **vision-language reward model**.
4. Train the VLA on that large synthetic dataset and improve its performance.
5. Collect real-world rollouts again with the improved VLA, repeating world model → VLA improvement.

Through this virtuous cycle, VLAW raised absolute success rate by 39.2% over the base policy, of which the contribution from training on synthetic rollouts is reported at about 11.6 percentage points.

> ### 💡 The difference between Ctrl-World and VLAW
>
> Both share the larger frame of "improve the VLA with rollouts in imagination," but they split on **whether the world model is touched.**
>
> - **Ctrl-World:** trains the world model once on real data, **freezes** it, and improves only the VLA inside it. (One direction.)
> - **VLAW:** **keeps improving the world model itself** with real rollouts (failures included); the better world model makes better synthetic data that improves the VLA, and the better VLA in turn produces richer rollouts that improve the world model. (A two-way cycle.)

> ### 💡 Can a world model used for training also be used for evaluation?
>
> Ctrl-World's and VLAW's rollouts look usable for evaluation too, in that they show success and failure. But evaluating the same VLA with the world model used to train it breaks the independence of the validation.
>
> That VLAW trains on failure cases to raise fidelity touches the same limitation: **"failure modes the world model cannot produce are missed in evaluation."**

To summarize, the three approaches in Section 3 share the property of **"training a VLA using what the world model produced as data,"** with the world model's involvement deepening in order: simple appearance conversion (Cosmos Transfer) → generate video then recover actions (DreamGen) → run an existing VLA in imagination (Ctrl-World, VLAW).

---

## Section 4. [Trend 2. Training] World Generation Couples into VLA Training

The three approaches in Section 3 all shared one property: the world model hands over data and then drops out at deployment.

**What moved the real robot was, throughout, the trained VLA, and the world model remained an external tool synthesizing training data.**

The trend in this section sits at exactly the opposite point: **the world model's ability to "generate the next scene" (world generation) couples directly into the VLA itself, running alongside it (online) not only in training but at deployment and inference.**

Unlike Section 3, the techniques in Section 4 differ in that "actions still come from the VLA, but world generation is coupled into it as a permanent component of the inference pipeline."

That is, **"does the world model drop out at deployment (Section 3) or keep turning alongside (Section 4)"** is the axis dividing the two sections.

Let me take two branches by coupling method.

1. **Decoupled:** keep the world model as a module separate from the VLA and connect its output as conditioning (4-1)
2. **Coupled:** merge video generation and action generation into one model (4-2)

### 4-1. Decoupled — world model and VLA kept apart, joined by conditioning: π0.7 (Physical Intelligence, 2026)

![π0.7 decoupled structure](/assets/img/notes/world-model/en/pi07-decoupled.svg)
*π0.7 — a decoupled structure that keeps the world model as a separate module and conditions the VLA on its output*

The loosest form of coupling **keeps the world model as a module separate from the VLA but injects its output as conditioning for the VLA at inference time.** **Physical Intelligence's π0.7 (arXiv 2604.15483)** is the representative case.

π0.7's main body is a 5B-scale VLA, with a separate lightweight world model based on the BAGEL image generation model.

The flow:

1. The VLA looks at the current situation and outputs the next subtask as a language instruction.
2. The BAGEL-based world model takes that instruction and generates, as a subgoal image, how the scene should look in the near future.
3. The VLA's action expert takes the current observation together with that subgoal image as conditioning and outputs an action.

The result is that **giving the subgoal image improves performance on complex tasks**, and on some tasks it reportedly fails without one.

**Giving a subgoal also speeds up training meaningfully**, which is explained by **action prediction becoming closer to an IDM problem between "current frame → desired future frame."**

Through this arrangement π0.7 is reported to be a generalist while reaching specialist-level performance on some tasks.

The key point here is that **this world model generates a subgoal at every moment at deployment too, running alongside the VLA.** **The modules are decoupled but permanently coupled at inference (online)**, which is where this parts from Section 3.

> ### 💡 Ctrl-World (3-3) and π0.7 — the same "separate world model module," so what differs?
>
> **Both keep a world model separate from the VLA.** **The difference is what the world model does at deployment.**
>
> - **Ctrl-World**: **the world model makes imagined rollouts (data) during training and drops out at deployment.** What moves the real robot is the trained VLA alone.
> - **π0.7**: **the world model generates a subgoal image at every moment at deployment too, and that image keeps feeding into the action expert.** Remove the world model and the behavior itself changes.
>
> So "is the world model used only during training as a data generator (Section 3)" vs. "is it a permanent component of the inference pipeline (Section 4)" is the criterion dividing the two sections.

### 4-2. Coupled — world generation merged into one VLA model

Where the decoupled approach kept the world model as an external module, the coupled approach **trains and runs video generation and action generation together inside one model.** Below are two models, in order of deepening coupling.

#### 4-2-1. NVIDIA DreamZero (World Action Model / GR00T N2, 2026): joint denoising

![DreamZero joint denoising](/assets/img/notes/world-model/en/dreamzero-joint-denoising.svg)
*DreamZero — a coupled structure denoising video and actions together in one model*

The first model is **NVIDIA DreamZero (arXiv 2026)**. **It is the representative model putting the world action model (WAM) naming front and center, led by NVIDIA GEAR Lab (Seonghyeon Ye, Joel and others).**

As the paper's title makes plain ("World Action Models are Zero-shot Policies"), it presents **zero-shot generalization to unseen tasks, environments and even robots without per-task training** as its core result.

DreamZero takes a pre-trained video diffusion backbone (Alibaba Wan 2.1-I2V-14B) as its starting point as-is and **denoises future video and actions together (joint denoising) inside one Diffusion Transformer.**

**The key is that there is no separate IDM module.**

Where DreamGen in 3-2 was a two-stage structure of "video generation model + separate external IDM," **DreamZero treats action as one more modality inside the same denoising process as video, performing video generation and action recovery end-to-end with no separation.**

The flow:

1. Fine-tune the pre-trained video diffusion model (Wan 2.1-I2V-14B) on robot data so that a single Diffusion Transformer generates future video and actions together.
2. At inference, take the current observation (video), the language instruction and the robot state as input.
3. The DiT generates, autoregressively with flow matching, the next chunk's future video tokens and action tokens jointly in one denoising process.
4. Execute the generated action chunk on the real robot and, using the video observation received as a result, replace the generated frames with real frames to stop error accumulating.
5. Repeat, performing the robot task closed-loop.

On results, what stands out most is **zero-shot generalization.**

Where conventional VLAs were strong on semantic generalization but weak on unseen motions in new environments, **DreamZero reports roughly twice the generalization of SOTA VLA models on tasks and environments absent from training.**

For example, on tasks entirely absent from training (untying shoelaces, ironing, shaking hands) it averaged 39.5%, far ahead of a comparable VLA (under 1% trained from scratch; 16.3% even for a large pre-trained VLA).

Further, it reached this **with the DROID dataset alone, without large-scale cross-embodiment robot training**, and scored 1750 on RoboArena — real-world distributed evaluation — ahead of Physical Intelligence's π0.5 (1622). These are read as signals of the WAM's potential.

Beyond that, the core claim of this family is that "video generation quality is the main lever on policy performance," and it **explicitly advances the view that the ability to make good video translates directly into a good policy.** NVIDIA has said it will release GR00T N2 on the basis of this DreamZero WAM. (Model files are not yet public.)

> ### 💡 Zero-shot generalization
>
> **Zero-shot means performing a task never seen in training or demonstration, immediately, with no additional training or data for that task.**
>
> DreamZero manages this because its starting point, the video foundation model, has already learned broadly from web video the knowledge of "language instruction → how the scene changes."
>
> That lets it transfer to unseen motions, objects and environments from narrow robot data (DROID) alone.
>
> This contrasts with conventional VLAs, which were strong on semantic generalization — understanding the name of a new object — but weak on unseen physical motion in new environments.

> ### 💡 Joint denoising
>
> A diffusion model produces its result by progressively removing noise (denoising). DreamZero denoises video tokens and action tokens simultaneously inside the same model, drawing out "the next scene" and "the action that makes that scene" together in one generation process.

> ### 💡 DROID and RoboArena
>
> **DROID is both a training dataset and a standard robot platform**, and **RoboArena is the real-world evaluation benchmark that runs on that DROID platform.** Both recur on the data and evaluation sides, so they are worth pairing.
>
> - **DROID (Distributed Robot Interaction Dataset)**: a large-scale real-world manipulation dataset collected by 13 institutions on identical hardware (Franka Panda 7-DoF arm + Robotiq gripper). It comprises about 76k teleoperation trajectories, 564 scenes and 86 tasks (~350 hours) — the full public release including failure trajectories is about 95.6k — with multi-view RGB-D video and natural-language instructions. It is the de facto standard dataset and platform in the field, so models are compared against each other through "DROID checkpoints" under identical conditions (π0.5-DROID, GR00T N1.6-DROID, DreamZero-DROID and so on).
> - **RoboArena**: a distributed real-world evaluation benchmark running on the DROID robot platform (led by Chelsea Finn, Sergey Levine and others). Rather than evaluating on fixed tasks or in a centralized competition, **evaluators at multiple institutions choose tasks and environments freely but compare two policies double-blind and pairwise**, and these preference data are aggregated into an **Elo-style ranking**. Think of it as the robot version of LMArena, which ranks LLMs by human preference.

#### 4-2-2. NVIDIA Cosmos 3 (GTC Taipei, June 2026): an omnimodal single model

![NVIDIA Cosmos 3](/assets/img/notes/world-model/cosmos3-omnimodal.jpg)
*The omnimodal single-model structure of Cosmos 3 (source: [NVIDIA Cosmos Lab](https://research.nvidia.com/labs/cosmos-lab/cosmos3/))*

The second model is **NVIDIA Cosmos 3 (arXiv 2026).**

Where Cosmos's Predict, Transfer and Reason from 2-1 were a separate family of models, **Cosmos 3 unifies them into a single model and adds action (a policy model) on top, handling and generating language, image, video, audio and action in one model.**

The structure is a two-tower **Mixture-of-Transformers (MoT)**. **The two "towers" are not two separate models, however: one model performs both roles — reasoning and generation — according to the kind of token coming in.**

Technically, within one model each layer holds separate parameters for reasoning (the reasoner, autoregressive) and for generation (the generator, diffusion), connected by shared self-attention.

The flow:

1. Language, image, video, audio and action inputs are tokenized by per-modality encoders and received as one sequence.
2. That sequence passes through a single MoT model in one go; each token is processed by reasoning (AR) or generation (diffusion) parameters according to its kind, and they meet in shared self-attention.
3. The attention mask is asymmetric, so only generation tokens are conditioned on the reasoning context (reasoner→generator, one-way), and understanding and generation happen together within one forward pass. In policy mode it predicts future video and actions jointly.
4. The same model switches modes depending on which inputs and outputs are bound together — Reason (understanding), Predict (future video generation), Transfer (photoreal conversion), Policy (action + video generated together = WAM mode).

Two variants have been announced.

Nano totals 16B (8B per tower), at workstation scale; Super totals 64B (32B per tower), at data-center scale. (A 2B-class Edge tier is also planned.)

The significance of Cosmos 3 is that it stands for **data generation (Transfer, Predict), training (Policy) and evaluation/simulation (Reason, world sim) converging into one model.**

It can be read as the trend of world models spreading into every stage of the robot development pipeline, realized inside a single model.

> ### 💡 MoT (Mixture-of-Transformers) and two-tower
>
> MoT gives each modality and role its own dedicated weights while sharing attention so they exchange information.
>
> It differs from MoE, with which it is often confused — **MoE routes among experts of the same kind, whereas MoT holds dedicated parameters per role.**
>
> The key point is that Cosmos 3's reasoner and generator are not two separate models but two parameter sets inside one model.
>
> Each transformer layer holds two weight sets, one for reasoning (AR) and one for generation (diffusion), and the two token streams meet only in shared self-attention.
>
> Because the attention mask is asymmetric (AR is causal and self-contained; diffusion is full and references the AR context), information flows one way, reasoner→generator — and all of this happens within a single forward pass.
>
> **So it is not the dual-system arrangement where a slow VLM (System 2) hands results to a fast action head (System 1), but an omnimodal structure with understanding and generation unified in one model.**

> ### 💡 The asymmetric attention mask (causal vs. full)
>
> An attention mask is the rule setting "**which tokens each token may reference (attend to)**" within the sequence — the same concept as the causal/full attention applied to video frames in 2-2.
>
> Cosmos 3 merges reasoning tokens and generation tokens into one sequence but designs this reference rule asymmetrically.
>
> By analogy: the execution team (generation) may freely consult the analyst's analysis (reasoning), but the execution team's output cannot change the analyst's analysis.
>
> ```
> Block 1 = reasoning tokens (AR)
> Block 2 = generation tokens (diffusion)
>
> Who may reference (attend to) whom:
>
>                        Block 1      Block 2
>                      (reasoning)  (generation)
> Block 1 (reasoning):    ✅           ❌
> Block 2 (generation):   ✅           ✅
>
> ※ Within blocks: Block 1 is causal (past only), Block 2 is full (mutual)
> ```
>
> The point of this design is that reasoning tokens stay self-contained, unaffected by the noisy tokens on the generation side.
>
> So information flows one way only, reasoner→generator. And yet this is not two models run separately handing results across — it is implemented by the mask rule alone, inside the same model and the same forward pass.

> ### 💡 Dual-system (System 2 → System 1)
>
> Dual-system is a common structure in robotics, named after human cognition's "slow thinking / fast thinking" (System 2 / System 1).
>
> - **System 2**: the slow, deliberate part. Usually a VLM understands the scene and language and plans the next behavior, running at a low rate (NVIDIA GR00T N1's system 2 runs at ~10Hz).
> - **System 1**: the fast-reacting part. An action head generates the actual joint movement, running at a high rate (GR00T N1's system 1 runs at ~120Hz).
>
> The key is that these are **separate networks**. System 2 passes its understanding and plan to System 1, which takes it and executes quickly — a modular, separated structure. **Figure AI's Helix and NVIDIA GR00T** are the representative cases.
>
> Cosmos 3 differs. Reasoning and generation are not split into separate networks exchanging results; they are unified inside one model and one forward pass.

To summarize, the techniques in Section 4 can be read as **a spectrum along which world generation couples ever more deeply into the VLA.**

Section 5 looks at how world models are entering the last stage: evaluation.

---

## Section 5. [Trend 3. Evaluation] The World Model Breaks the Evaluation Bottleneck

Where Sections 3 and 4 covered world models used for data and for training, the last trend is evaluation.

This trend is earlier-stage than the other two, and alongside its promise there are clearly unsolved fundamental problems.

### 5-1. The era of human evaluation — what the bottleneck actually is

**The gold standard for evaluating a robot policy is a person watching a rollout run to completion on a real robot and scoring it success or failure.** The problem is that this is brutally expensive.

In one case, evaluating a single OpenVLA model against baselines alone reportedly took over 2,500 rollouts and over 100 hours of human labor (resetting scenes, running the policy, scoring success) across four robot setups at three institutions.

Attempts to systematize and automate this bottleneck already exist.

RoboArena (4-2-2) is distributed real-world evaluation in which evaluators at multiple institutions compare policies double-blind and pairwise on the DROID platform to build an Elo-style ranking; and AutoEval (arXiv 2025) from Sergey Levine (UC Berkeley, Physical Intelligence) minimizes human involvement by automating policy execution, success scoring and scene resets, running real-robot evaluation around the clock.

Even so, **real-world evaluation is intrinsically tied to hardware and the lab, and it is slow.** AutoEval is limited to a specific setup and a single embodiment, and the more generalist the policy, the more varied the evaluation environments it needs — so the bottleneck arguably gets worse.

### 5-2. Routing around it with reinforcement learning — evaluating without rollout video

Faced with expensive evaluation, the RL camp has responded with **detours that "evaluate and improve a policy without watching rollout video one by one."** Three representative branches:

1. **Intervention-based — DAgger / HG-DAgger**: **a person watches the policy (= the VLA) and intervenes to correct it the moment it is about to fail**, so how often the person had to intervene (intervention frequency) is itself a signal of policy quality.
2. **Advantage-based — π*0.6 (Physical Intelligence, RECAP)**: a value function **estimates how much each action contributes to task success (the advantage), scoring good actions "positive" and bad ones "negative."** Instead of watching rollout video, actions are evaluated by the learned value, and at inference the policy is conditioned to emit only "positive" actions, yielding a policy better than the training data.
3. **Foundation reward model-based — Dyna**: instead of a person hand-writing reward rules per task, **one VLM-based reward model scores progress (temporal distance to the goal) generically across varied robot tasks.** Dyna Robotics' DYNA-1 appears to place such a foundation reward model permanently at the center of the post-training loop (RM-in-the-loop) so the VLA improves iteratively toward higher reward, and to have built a closed loop that automatically feeds failure cases back into improving the reward model (online adaptation). (DYNA-1 is a closed model, so the details are not public.)

These detours are fast and easy to automate. They share a limitation, though.

**Because they compress the policy's behavior into a scalar — value, advantage, reward — or a success/failure label, they struggle to be as intuitive and accurate as watching the actual rollout video.**

A reward model can be wrong or reward-hacked, an advantage is an estimate throughout, and intervention frequency is coarse and cannot show "where and why it failed."

**In the end, the most trustworthy evaluation signal still appears to be the rollout video itself.**

> ### 💡 Advantage conditioning (π*0.6's evaluation signal)
>
> Physical Intelligence's π*0.6 advantage conditioning is the representative method for **"scoring actions without rollout video."**
>
> First a value function estimates how close the current state is to task success; if an action raises that value the advantage is positive (a good action), if it lowers it the advantage is negative (a bad action).
>
> RECAP attaches this advantage to the policy as a condition in text — "positive"/"negative" — rather than as a complicated number, trains on that, and at deployment requests only "positive," drawing out behavior better than the training data.
>
> A flow-matching VLA does not give action probabilities (log-probs), which makes standard RL such as PPO hard to apply directly; this detour solves that problem.

### 5-3. The fundamental problems of world-model-based evaluation, and the directions emerging

Hence **world-model-based evaluation: use "rollout video," but produce it at scale without real hardware.**

**Run the policy (= the VLA) inside a trained world model, generate rollout video, and score that video — and the evaluation becomes scalable and reproducible.**

For this to be a trustworthy evaluation, though, fundamental problems remain.

#### 5-3-1. The world model as evaluator: WorldEval

The representative work is **WorldEval (arXiv 2025).**

The core difficulty of world-model-based evaluation is generating video that faithfully reflects the policy's actions. **Feed the action the policy (= the VLA) emitted straight into the world model and, surprisingly, video that follows the action does not come out well — because robot actions are semantically ambiguous.**

WorldEval solves this with **Policy2Vec**, which converts the policy's output into a latent action and makes video from that. **The policy's output becomes a latent action, and a video generation model produces robot video following it.**

The generated rollout is then watched by a video-capable VLM (Gemini, for example), which automatically judges success or failure (auto-labeling). In effect the human scorer is replaced by a VLM, automating evaluation at scale, around the clock.

The flow:

1. The policy (= the VLA) being evaluated looks at the current observation and outputs an action.
2. The world model generates the future observation video following that action (latent action).
3. The generated observation goes back to the policy (= the VLA), completing a whole task (a rollout) inside imagination.
4. A video-capable VLM (Gemini, for example) watches the completed rollout video and automatically judges success or failure.

The resulting scores can rank policies (= VLAs) and models under experiment with no human involvement, and also serve as a safety detector filtering dangerous behavior in advance. **WorldEval reports that these scores correlate strongly with actual real-world performance.**

**GWM-Robotics from the US startup Runway** (the robotics branch of its general world model GWM-1) sits on the same trend, reporting that it evaluated eight policies without hardware and that simulated and real scores showed 95% similarity.

#### 5-3-2. Two fundamental problems: train/eval independence, and fidelity

World-model-based evaluation has two fundamental problems.

- **Train/eval independence (leakage)**: as explained in 3-3, **using the world model that was used for training to evaluate the very policy (= the VLA) trained with it breaks the independence of the validation.**
    - If the policy (= the VLA) has overfit to the world model's biases, it can look successful inside imagination and fail in reality — and above all, failure modes the world model cannot produce are never seen in evaluation at all.
    - Rigorous evaluation therefore appears to require a world model independent of policy training, or real-world validation.
- **Fidelity (reproducing failure):** if an evaluation world model only renders successful scenes plausibly and **cannot reproduce failure, a policy that would fail looks successful and the evaluation becomes false.**
    - dWorldEval (arXiv 2026) takes this head-on: it puts human-collected failure data into training so the model faithfully reproduces failure under the same action rather than self-correcting or hallucinating, and generates a progress score (0–1) at every step to automatically grade "how far the task got before it stopped."

    > ### 💡 The trap an evaluation world model falls into — it cannot produce failure
    >
    > To sort good policies from bad, **the world model has to show a failing policy (= VLA) as failing video.** But a world model trained for video quality tends to do the opposite.
    >
    > - **Self-correction**: the policy failed the grasp, but in trying to make "natural video" the model fixes it into a successful grasp.
    > - **Hallucination**: it invents an object absent from the scene, drawing a success that was in fact impossible.
    > - **Action mismatch**: it makes video at odds with the input action, failing to reflect the policy's actual behavior.
    >
    > All three make a failing policy look like a success.

In working through these two problems, **an explainable evaluation direction is also emerging early — going past plain success/failure to extract the rollout's 3D trace and explain "where and how it went wrong."**

To summarize, the evaluation trend is still early-stage, but it is clearly an area that will draw focus, given that it is becoming the bottleneck of the robot development pipeline.

From the era of human scoring (5-1), through RL's detour of routing around rollout video with scalars (5-2), the direction moves toward producing rollout video at scale without hardware using a world model (5-3) — with train/eval leakage as the fundamental problem, and auto-labeling, fidelity and explainability rising alongside as the remedies.
