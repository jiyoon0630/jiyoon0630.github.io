---
layout: paper
lang: en
ref: sail-faster-than-demonstration
kind: paper-review
title: "SAIL: Faster-than-Demonstration Execution of Imitation Learning Policies"
date: 2026-04-20 12:00:00 -0700
paper_date: 2025-06-13
venue: "CoRL 2025 · arXiv:2506.11948"
tags: [Imitation-Learning, Diffusion-Policy, Robot-Control, Action-Chunking, Paper-Review]
authors: "Nadun Ranawaka Arachchige, Zhenyang Chen, Wonsuhk Jung, Woo Chul Shin, Rohan Bansal, Pierre Barroso, Yu Hang He, Yingyan Celine Lin, Benjamin Joffe, Shreyas Kousik, Danfei Xu"
affiliations: "Georgia Institute of Technology"
summary: "What keeps imitation-learning policies at demonstration speed is not the policy but the execution stack — reached-pose targets with high-gain tracking, error-gated conditioning (EAG), adaptive speed and latency-aware scheduling run them several times faster than the demos, with no extra data or RL."
paper_url: "https://arxiv.org/abs/2506.11948"
---

> **Core claim** — What keeps an imitation-learning policy from moving faster than its demonstrations lies not in the policy but in the **execution stack**. Shorten only the time interval between target poses, and the way the low-level controller tracks them changes, pushing the policy's inputs OOD; the effects spread into chunk discontinuities, failures in precision phases, and latency. Break this chain at the policy, the controller and the scheduler together, and an already-trained policy can be executed several times faster than the demonstrations, with no additional data or RL.

---

## Introduction

Generative behavior cloning (BC) policies such as Diffusion Policy and ACT learn demanding manipulation, such as deformable-object manipulation and non-prehensile motion, from tens to hundreds of human demonstrations. But these policies inherit from the demonstrations not only the motion but also the **speed**. If a person demonstrates slowly, the policy is slow. In industrial automation, cycle time is throughput, so this slowness becomes a cost the moment it leaves the lab.

SAIL (Speed Adaptation for Imitation Learning) formalizes "executing a learned visuomotor policy faster than its demonstrations" as a problem in its own right. To give the conclusion first, the solution is not a new model. It is a **full-stack design** that breaks, at the policy, the controller and the scheduler together, the chain of problems that speeding up sets off. The expansion of the acronym follows the main text; the caption of the paper's Fig. 1 also spells it out as Speed-Adaptive Imitation Learning.

This piece follows the paper's argument, but builds up the necessary control concepts as they come up, so that readers with a thin background in robot control (an LLM or diffusion background) can also be convinced of the **reason** behind each design decision.

---

## 1. The problem — why imitation-learning policies are stuck at demonstration speed

### 1.1 Slow demonstrations make slow policies

Demonstrations are slow for two reasons.

**⓵ People are slow** — a person teleoperating a robot arm with a VR controller moves carefully.

**⓶ The teleop controller is deliberately soft** — low-level controllers for teleoperation are often set to low gains for the sake of feel. The robot follows the person's commands a beat late, and smoothly.

BC absorbs both. A natural fix then comes to mind. The policy's output is "the next few target poses"; the time interval between targets is not in the output but is set by the system. So why not **just shorten the interval on playback**? The paper's answer is "that doesn't work," and the reason is the content of the whole paper.

### 1.2 The gap left by existing approaches

The paper's related work sums up existing attempts related to acceleration as follows.

| Line of work | How it speeds up | Limitation (as the paper notes) |
|---|---|---|
| IL that surpasses the demonstrator (T-REX, etc.) | Ranking rewards, IRL | Requires environment interaction |
| Online self-supervised speed-up (Sakaino et al.) | Task-specific rewards + online learning | Only 1.1–1.3× speed-up, and loses success rate |
| AWE, SPHINX | A by-product of waypoint extraction, etc. | Speed is not the goal |
| DMP / RMP | The representation itself supports time modulation | The field has moved to simple generative policies like DP |
| Full-stack IL systems (UMI, Mobile ALOHA) | Joint optimization of controller and learning | Do not address deliberately making demonstration and execution speeds differ |

In short, prior work either required interaction, did not aim at speed-up, or did not address the problem of decoupling demonstration speed from execution speed at all.

### 1.3 So the question the paper asks

> Can a policy $\pi$ trained at demonstration interval $\delta^*$ be executed at a faster, time-varying interval $\delta_t = c_t\,\delta^*$ ($c_t \le 1$) while substantially raising successful throughput? With no additional demonstrations, rewards or online learning.

$\delta^*$ is the time interval between consecutive target poses in the demonstrations, and $c_t$ is the speed factor. Both are defined precisely in §2.1.

Four walls stand in front of this question.

**⛔ Wall 1 — The executor changes (controller shift)** Send the same target poses at shorter intervals and the way the low-level controller tracks them changes, so the robot passes through poses it never passed through in the demonstrations. The policy's inputs are pushed OOD.

**⛔ Wall 2 — Chunks disagree with each other (prediction inconsistency)** Generative policies sample each action chunk independently, so the trajectory jumps at replanning boundaries. The faster the speed, the more fatal this jitter becomes.

**⛔ Wall 3 — Not every phase can be done fast (precision phases)** In phases like grasping or insertion, speed itself is a cause of failure even with a perfect controller.

**⛔ Wall 4 — Actions run out (latency)** Sensing–inference latency cannot be reduced. The faster the execution, the sooner the actions to execute run out before a new chunk arrives.

The four walls are not independent. In particular, the entanglement that **the means of getting over Wall 1 raises Wall 2** is the axis of this paper's design, and it surfaces at the end of §3.1.

---

## 2. Background — the minimum of control needed to understand Wall 1

To see what Wall 1 really is, you need to know what happens "between the policy emitting an action and the robot actually moving." If you are familiar with robot control, feel free to skip to §3.

### 2.1 The two-layer policy–controller structure: the policy says only "where to"

```
  sensors: wrist/front RGB (o_t), robot state (x_t)
        |
        v
  +------------------------------+
  | policy pi (Diffusion Policy) |   <- runs every H_e steps
  +------------------------------+
        |  a_t = [x^d_t, ..., x^d_{t+H-1}]   (poses only, no time)
        v
  +------------------------------+
  | scheduler: 1 pose per delta  |   <- time enters here
  +------------------------------+
        |  x^d_t  every delta seconds
        v
  +------------------------------+
  | controller K  (100-500 Hz)   |
  +------------------------------+
        |  u_t (joint torque)
        v
      robot
```

The policy emits an action chunk.

$$\mathbf{a}_t = \big[x^d_t,\ x^d_{t+1},\ \dots,\ x^d_{t+H-1}\big]\ \sim\ \pi(\cdot \mid o_t,\ x_t)$$

- $o_t$ — camera observation (wrist and front RGB)
- $x_t$ — robot state (end-effector pose, etc.)
- $x^d_i$ — desired SE(3) end-effector pose and gripper command
- $H$ — prediction horizon (32 in the sim setup)

The low-level controller turns this target into torques.

$$u_t = K\big(x^d_t,\ x_t,\ \delta\big)$$

- $K$ — the low-level controller. It runs at a much higher rate than the policy (on the Franka, the scheduler at 100 Hz and torque commands at 500 Hz)
- $u_t$ — joint torques
- $\delta$ — the time interval between consecutive target poses. In the demonstrations, $\delta^* = 0.05$ s (20 Hz)

The policy is deployed receding-horizon. It executes only the first $H_e$ steps of a chunk of $H$ and infers again. The sim setup predicts 32, executes 8 steps and then re-infers, and executes 4 more steps while inference runs, to simulate latency.

The decisive fact is that **there is no time information inside a chunk**. Time is assigned by the scheduler through $\delta$. So speeding up simply means shrinking $\delta$.

$$\delta_t = c_t\,\delta^*,\qquad c_t \in (0,\,1]$$

- $\delta_t$ — execution interval at time $t$
- $c_t$ — speed factor. **Smaller is faster.** $c_t = 0.5$ is 2× speed, $0.2$ is 5×. Because of this inverse notation, the equation in §3.3 looks backwards at first

### 2.2 The low-level controller: gains and tracking lag

The controller on the paper's real robot (Franka) is OSC (operational space control) (Appendix Eq. 7).

$$\tau = J^\top M\big(K_p\,e_p + K_v\,e_v\big)$$

- $\tau$ — joint torques
- $J$ — Jacobian (end-effector space ↔ joint space)
- $M$ — mass matrix expressed in end-effector space
- $e_p,\ e_v$ — pose error and velocity error relative to the target
- $K_p,\ K_v$ — stiffness gain, damping gain

With low gains the robot follows the target late and softly, like a spring; with high gains it follows stiffly and immediately.

How tracking lag ties into speed becomes clear with a one-degree-of-freedom toy model. *This model is a supplementary explanation not in the paper.* Consider PD control that uses only the pose error and damping, with no velocity reference.

$$m\ddot{x} = K_p\,(r - x) - K_v\,\dot{x}$$

- $m$ — mass
- $r$ — target trajectory
- $x$ — actual position

If the target moves at constant velocity, $r(t) = vt$, then at steady state ($\ddot x = 0,\ \dot x = v$) the tracking error is:

$$e_{ss} = r - x = \frac{K_v}{K_p}\,v$$

- $v$ — target velocity
- $e_{ss}$ — steady-state tracking error. Proportional to speed, inversely proportional to stiffness

```
 pose
  ^                        /  x^d (commanded)
  |                      /     /  x (reached)
  |                    /     /
  |                  /     /
  |                /     /
  |              /     /
  |            /     /
  +-----------|-----|-------------------> t
              |<--->|  lag grows with speed
```

Compress the time interval by a factor of $c$ and velocity becomes $1/c$ times, acceleration $1/c^2$ times. At 5× speed, the lag in constant-velocity segments can grow 5×, and the inertial error in acceleration and deceleration segments up to 25× in scale. **Send the same command sequence faster, and the poses the robot actually passes through change.** And that this lag does not merely shift things along the time axis but **changes the path itself** comes up again in §4.

### 2.3 The policy is unchanged, but the input distribution shifts

Written as distributions, the phenomenon is:

$$d_{\text{exec}}\big(o,\,x \mid \pi,\ K,\ c\,\delta^*\big)\ \neq\ d_{\text{demo}}\big(o,\,x \mid \text{human},\ K_{\text{teleop}},\ \delta^*\big)$$

- $d_{\text{demo}}$ — the (observation, state) distribution of the training data, produced by a human with the teleop controller $K_{\text{teleop}}$ at interval $\delta^*$
- $d_{\text{exec}}$ — the (observation, state) distribution the policy actually encounters during accelerated execution

It has the same shape as covariate shift, the classic ailment of imitation learning, but a different cause. Here the input distribution shifts not because the policy is wrong but because **the executor (controller and time interval) has changed**. Once in an OOD state, the policy's output wavers, the wavering output pushes the state further, and compounding error begins. This is exactly what Wall 1 is.

---

## 3. Method — four walls, four components

SAIL matches one component to each wall.

| Wall | SAIL component | Layer | Paper section |
|---|---|---|---|
| ⛔1 Controller shift | Controller-invariant target (reached pose) + high-gain tracking | System | §4.2 |
| ⛔2 Chunk inconsistency | Error-Adaptive Guidance (EAG) | Policy | §4.1 |
| ⛔3 Precision phases | Adaptive speed modulation | Policy | §4.3 |
| ⛔4 Latency | Latency-aware scheduling + speed bound | System | §4.4 |

The paper introduces EAG in §4.1 first. But you have to understand the controller side first to see why EAG is needed, so this piece reorders things in causal order.

### 3.1 Wall 1 → learn the reached pose, and follow it stiffly (paper §4.2)

The existing pipeline trains on the teleop commanded pose $x^d$ as the action label and executes with the same $K_{\text{teleop}}$ used during collection. At 1× speed this combination has no problem. The same controller as in training receives the commands at the same interval, so even the mismatch between command and actual arrival is reproduced as is. But the moment $\delta$ shrinks, the lag profile of §2.2 changes and that reproduction breaks.

SAIL changes two things **as a pair**.

**⓵ What to predict — the reached pose $x$ instead of the commanded pose $x^d$**

The demonstration data records both the teleop commands and the poses the robot actually reached, at 20 Hz. SAIL uses the latter as the action label.

$$\text{Before:}\quad \mathbf{a}_t = \big[x^d_t,\ \dots,\ x^d_{t+H-1}\big]\qquad\longrightarrow\qquad \text{SAIL:}\quad \mathbf{a}_t = \big[x_t,\ \dots,\ x_{t+H-1}\big]$$

- $x^d$ — the commanded pose the person sent via VR. A "leading" value that presupposes the teleop controller's lag
- $x$ — the pose the robot actually reached. The physically realized trajectory

**⓶ How to execute — a high-fidelity tracking controller $K_e$ instead of $K_{\text{teleop}}$**

$$u_t = K_e\big(x_t,\ x^{\text{cur}}_t,\ \delta_t\big)$$

- $x_t$ — the reached pose predicted by the policy (the tracking target)
- $x^{\text{cur}}_t$ — the current actual pose
- $K_e$ — high-gain OSC. On the real robot, a spline is fit to the predicted trajectory and its derivative is also fed in as a velocity target (Appendix D.1)

The real-robot gains change as follows (Table J.5).

| | Teleop collection | SAIL execution |
|---|---|---|
| $K_p^{\text{pos}}$ / $K_v^{\text{pos}}$ | 150 / 24.5 | 300 / 34.6 |
| $K_p^{\text{rot}}$ / $K_v^{\text{rot}}$ | 250 / 31.6 | 400 / 40.0 |

All four pairs satisfy $K_v = 2\sqrt{K_p}$. It is a setting that raises only stiffness while keeping critical damping. Also, supplying a velocity target corresponds to replacing the damping term of the §2.2 model with $K_v(\dot r - \dot x)$ to add a velocity reference, which makes the constant-velocity lag term $\frac{K_v}{K_p}v$ vanish. These last two sentences are my interpretation.

A question arises here. What the person intended was the commanded pose; is it right to imitate the **result** instead of the intent? Wouldn't you need to learn the intent to reproduce the intent?

> ### 💡 The commanded pose is a latent tied to "the controller as a decoder"
>
> The meaning of the commanded pose $x^d$ is not fixed on its own. It acquires meaning only **as a pair with the decoder**: "feed this command to $K_{\text{teleop}}$ at interval $\delta^*$ and the robot passes through here." During teleop, a person watches the robot's sluggish tracking and corrects by leading the command, and that correction is baked into $x^d$.
>
> By analogy with latent diffusion: $x^d$ is a latent fitted to a particular VAE decoder, and $x$ is the decoded pixels. Change the decoder ($K$, $\delta$) and the same latent unpacks into a different image, but a target in pixel space is independent of the decoder. This is why SAIL calls the reached pose a "controller-invariant target."
>
> There is one price. To generate pixels directly, you need an output device that prints those pixels exactly. Likewise, making the reached pose the target requires a controller that follows it with almost no error, and that is the high-gain $K_e$. **The two changes must go together as a pair.**

The paper validates this design without any policy learning (Appendix G.1, Fig. G.4). It is an experiment that only **replays** Can task demonstrations with varying speed and gain. At the original speed ($c = 1$), replaying commanded poses is better, because they match the original controller. When speed and gain go up, commanded poses fail by overshooting, and only the combination of reached poses with sufficient gain keeps a high success rate. Because it isolates just the "target × controller" combination, it is a clean basis for the design.

But this solution has a price. A stiff controller follows its target faithfully, so **if the target itself jitters, it faithfully follows the jitter too.** In Appendix G.2 (Fig. G.5), injecting noise into the reference trajectory makes the success rate of the high-gain controller collapse much faster than the low-gain one's. This is the entanglement foreshadowed in §1.3. The moment Wall 1 is cleared, a requirement appears that the reference trajectory the policy emits be smooth, and Wall 2 gets higher.

### 3.2 Wall 2 → use the condition only when it can be trusted: Error-Adaptive Guidance (paper §4.1)

**The jitter comes from chunk boundaries.** DP generates a chunk by iterative denoising (paper's notation).

$$\mathbf{a}_{n} = \mathbf{a}_{n+1} - \gamma\,\epsilon_\theta\big(o_t,\ x_t,\ \mathbf{a}_{n+1},\ n\big) + \mathcal{N}(0,\ \sigma_n^2 I)$$

- $n$ — denoising step (10 DDIM steps at inference)
- $\epsilon_\theta$ — noise-prediction network (1D UNet)
- $\gamma$ — step size
- $\sigma_n$ — injected noise scale

Each chunk is sampled independently from a multimodal distribution. So two consecutive chunks can pick different modes: for example, one chunk goes around an object to the left and the next to the right. At replanning time the executed trajectory jumps from one mode to another, and that becomes jerk (Fig. 3).

```
  x
  ^            chunk k   (mode A)
  |         .------------------->
  |        /
  |-------*
  |        \
  |         '------------------->
  |        ^   chunk k+1 (mode B)
  |        replan: executed path jumps A -> B  =>  jerk
  +------------------------------------------> t
```

Speeding up worsens this in two ways. Replanning happens more often in the same wall-clock time, and once tracking error puts the robot in an OOD state, the prediction itself becomes less stable.

**First attempt — condition on the tail of the previous plan.** The new chunk is conditioned on the not-yet-executed tail of the previous plan so that it continues from it. In CFG form (Eq. 1):

$$\epsilon^{\text{guided}}_\theta(\mathbf{a}^f, \mathbf{a}^c \mid o_t, x_t) = \epsilon_\theta(\mathbf{a}^f, \varnothing \mid o_t, x_t) + w\Big(\epsilon_\theta(\mathbf{a}^f, \mathbf{a}^c \mid o_t, x_t) - \epsilon_\theta(\mathbf{a}^f, \varnothing \mid o_t, x_t)\Big)$$

- $\mathbf{a}^f$ — the chunk to be newly generated
- $\mathbf{a}^c = \mathbf{a}_{H_e : H_e + H_f}$ — the condition. The segment right after the $H_e$ steps already executed from the previous plan (length $H_f$, 4 in the sim setup)
- $\varnothing$ — null token (zero vector)
- $w$ — guidance weight

Lining up the sim setup (infer after executing 8 steps, execute 4 more steps during inference, condition length 4), $\mathbf{a}^c$ is exactly **the segment that will be executed during inference latency**. Training follows standard CFG, blanking the condition to null with some probability so that one network learns both the conditional and unconditional scores. Table J.2 lists $p_{\text{cond}} = 0.3$ and null token = zero, but it does not state whether this 0.3 is the probability of including the condition or of blanking it.

One point that is easy to trip over when reading the paper directly is worth flagging too.

> ### ⚠️ Fact check — the same thing gets different names and symbols in different sections
>
> The technique is called Error-**Adaptive** Guidance in the paper's §4.1 and Error-**Aware** Guidance in the paper's §5. The length of the condition segment is likewise written $H_f$ in the paper's §4.1 and $H^c$ in the paper's §4.4 and Appendix E. All are minor notational inconsistencies referring to the same thing. In §3.4 this piece follows Appendix E's symbol $H^c$.

**Problem — the condition itself can be wrong.** If tracking error during acceleration was large, the robot is not where $\mathbf{a}^c$ assumes it is. The (current state $x_t$, condition $\mathbf{a}^c$) pair is then a combination never seen in training, and its conditional score cannot be trusted. Clinging hard to a wrong condition actually amplifies divergence. The paper tests this hypothesis in three steps in Appendix H.

| Hypothesis | Experiment | Result |
|---|---|---|
| H-EAG1: guidance works only when the condition lies within the unconditional distribution | Compare using one of 64 unconditional samples as the condition vs. using it shifted in time and space (Fig. H.6) | Within the distribution, generation follows the condition; when shifted, it cannot |
| H-EAG2: the faster the execution, the more OOD the condition | Reset the sim to isolate a single replanning, and measure OOD scores with KDE, kNN and MMD over 200 runs (Fig. H.7) | As $c$ goes from 1 to 0.25, the OOD-side tail grows longer |
| H-EAG3: tracking error is a cheap proxy for OOD | Correlation analysis between tracking error and OOD score (Fig. H.8) | The larger the error, the higher the OOD score |

Resetting the sim before each trial in H-EAG2 to observe a single replanning is a careful design to separate accumulated error from the effect of speed. H-EAG3 is needed because running KDE, kNN and MMD at every inference is too expensive.

**The EAG rule.** If the tracking error is at or below a threshold, use the condition; if it exceeds it, turn it off.

$$w_t = \begin{cases} w_0, & e^{\text{pos}}_t \le \rho_{\text{pos}}\ \text{and}\ e^{\text{ori}}_t \le \rho_{\text{ori}} \\ 0, & \text{otherwise} \end{cases}$$

$$e^{\text{pos}} = \big\lVert a^{\text{pos}} - x^{\text{pos}} \big\rVert,\qquad e^{\text{ori}} = \arccos\!\Big(\frac{\operatorname{tr}(R_\Delta) - 1}{2}\Big),\qquad R_\Delta = R(a^{\text{ori}})^\top R(x^{\text{ori}})$$

- $w_0$ — the weight when the condition is used (set per task)
- $a^{\text{pos}},\ a^{\text{ori}}$ — position and orientation of the target pose currently being tracked
- $x^{\text{pos}},\ x^{\text{ori}}$ — current actual end-effector position and orientation
- $R(\cdot)$ — a function converting an orientation representation into an SO(3) rotation matrix. $e^{\text{ori}}$ is the geodesic angle between the two rotations
- $\rho_{\text{pos}},\ \rho_{\text{ori}}$ — thresholds (per Table J.3, mostly 0.02 / 0.05, only Stack 0.01 / 0.03)

There is a threshold sweep as well (Square, $c = 0.33$, Fig. H.9). Setting $\rho_{\text{pos}}$ to 0.01 / 0.02 / 0.04 gave a fraction of inferences with guidance on of 0.27 / 0.47 / 0.78, and success rate dropped at the loosest, 0.04. This is direct evidence that **using the condition unconditionally is harmful**.

Readers familiar with CFG in image generation will catch on two things here. Why only switch $w$ on and off instead of lowering it continuously with the error? And doesn't CFG usually exaggerate the condition with $w > 1$?

> ### 💡 EAG is not guidance-strength tuning but an on/off gate on the condition input
>
> The actual settings give the answer. The optimal weights in Table J.3 are, per task, one of just two values, $w_0 \in \lbrace 0,\ 1\rbrace$. Plug $w = 1$ into Eq. 1 and you get the pure conditional model.
>
> $$\epsilon^{\text{guided}}_\theta = \epsilon_\theta(\mathbf{a}^f, \varnothing) + 1\cdot\big(\epsilon_\theta(\mathbf{a}^f, \mathbf{a}^c) - \epsilon_\theta(\mathbf{a}^f, \varnothing)\big) = \epsilon_\theta(\mathbf{a}^f, \mathbf{a}^c)$$
>
> - $\epsilon_\theta(\mathbf{a}^f, \mathbf{a}^c)$ — conditional score (observation conditioning omitted from the notation)
> - $\epsilon_\theta(\mathbf{a}^f, \varnothing)$ — unconditional score
>
> There is no extrapolation at all. Appendix Alg. 2 does not compute any mixing of the two scores either. If the error exceeds the threshold, it attaches $\varnothing$ to the observation, otherwise $\mathbf{a}^c$, and runs inference once.
>
> So what EAG really is, is **a switch that judges the condition's reliability by tracking error and picks either the conditional or the unconditional model**. In image generation, $w > 1$ is a knob that exaggerates prompt fidelity at the cost of diversity. Here, by contrast, if the condition is even slightly off, exaggerated adherence becomes error amplification, so there is no reason to exaggerate.
>
> Carried over to LLMs, it is the same as a rule in continuation that, when the prefix is judged to be out of line with reality, throws away the prefix and generates again. It is a far simpler device than the name "guidance" suggests, and because it is simple, it fits in a real-time loop.

There is contemporaneous work that solved the same problem, attaching a new chunk to the segment executed during inference latency, under the opposite assumption. §6 compares them.

### 3.3 Wall 3 → slow down only in precision phases (paper §4.3)

Even past Walls 1 and 2, a problem remains. Precision phases like grasping or alignment fail if done fast, no matter how good the controller. SAIL judges at every step whether "this is a precision phase" with a binary flag $k_t$ and switches between two speeds (Eq. 10).

$$c_t = k_t\,c^{\text{slow}} + (1 - k_t)\,c^{\text{fast}},\qquad k_t \in \lbrace 0,\ 1\rbrace$$

- $k_t$ — precision (critical) phase flag. $k_t = 1$ means a precision phase
- $c^{\text{slow}},\ c^{\text{fast}}$ — task-specific preset speed factors

The inverse notation foreshadowed in §2.1 appears here. The speed factor is faster when smaller, so in the equation $c^{\text{slow}} > c^{\text{fast}}$. For example, sim Square has $c^{\text{slow}} = 1.0$, so precision phases are not sped up at all (Table J.4).

There are two ways to produce $k_t$.

| Method | When labeled | Principle |
|---|---|---|
| ⓐ Motion complexity | Offline, before training | Approximating demonstration trajectories with waypoints via AWE makes waypoints denser in complex segments. Waypoints are clustered in 3D space with DBSCAN, segments inside clusters are labeled $k=1$, and one bit is added to the action vector so the policy learns to predict $k$ along with the action (Alg. 1) |
| ⓑ Gripper events | During execution | If the predicted chunk contains a gripper open/close change, set $k=1$ |

The intuition behind ⓐ shows clearly in Fig. F.3. Waypoints bunch up where the end effector lingered in the demonstrations, that is, where the person slowed down to be careful and changed direction often. In effect, it **recycles human hesitation as a signal of precision phases**. As its reason for choosing DBSCAN, the paper cites that it captures clusters of arbitrary shape and naturally filters out noise points.

```
 speed
  ^
  |  _________             _________         ______
  |  fast     |           |  fast   |       | fast
  |           |___________|         |_______|
  |              slow                  slow
  |             (grasp)               (place)
  +--------------------------------------------------> t
```

### 3.4 Wall 4 → absorb latency into the schedule, and draw a line that cannot be crossed (paper §4.4)

The last wall is physical. Between the time an observation is requested, $t_o$, and the time a new chunk arrives, $t_a$, there is an irreducible latency $\delta^{\text{delay}} = t_a - t_o$. In slow execution chunks are left over, so there is no problem. Execute fast and there is nothing left to do before the new chunk arrives. The paper calls this action exhaustion and points to it as the main reason accelerated DP stuttered in the real world.

**Scheduling.** During the latency, keep executing the previous plan. When the new chunk arrives, drop the previous plan's actions scheduled after $t_a$ and switch to the new chunk.

```
 wall clock ------------------------------------------------------->
               t_o                  t_a
               |<---- delta_delay -->|
 inference     [=== policy call k ===]
 execution  ...... chunk k-1 ........|..... chunk k  (step = delta_t) ...
                                     ^
                                     chunk k-1 actions after t_a: dropped
```

It is the same structure as pipelining in inference serving, i.e., double buffering.

**Speed bound.** For a chunk to cover the latency, the $H^p$ predicted steps minus the $H^c$ conditioning steps must last longer than the latency (Appendix E, Eq. 8).

$$H^p\,\delta^{\text{lb}} > \delta^{\text{delay}} + H^c\,\delta^{\text{lb}}\quad\Longrightarrow\quad \delta^{\text{lb}} > \frac{\delta^{\text{delay}}}{H^p - H^c}$$

$$\delta_t = \max\big(c_t\,\delta^*,\ \delta^{\text{lb}}\big)$$

- $H^p$ — prediction horizon ($H$ in §2.1)
- $H^c$ — EAG condition length ($H_f$ in §3.2)
- $\delta^{\text{delay}}$ — sensing–inference latency
- $\delta^{\text{lb}}$ — the minimum step interval that guarantees continuous execution. If the interval $c_t\delta^*$ that adaptive speed wants is shorter than this, it is clipped to this lower bound

Example numbers to get a feel. The paper does not report actual latency values, so the following are assumptions.

| Assumed latency $\delta^{\text{delay}}$ | $\delta^{\text{lb}}$ ($H^p = 32$, $H^c = 4$) | Max speed-up vs $\delta^* = 50$ ms |
|---|---|---|
| 0.1 s (small UNet DP level) | > 3.6 ms | about 14× |
| 0.3 s (heavy model level) | > 10.7 ms | about 4.7× |

With a small UNet DP this bound rarely binds, but with a VLA whose inference is heavy, it becomes a real constraint. Increasing $H^p$ raises the bound, but as the paper points out, accurately predicting a long horizon is itself hard.

---

## 4. Why it works

The paper's own explanation goes like this. Speeding up causes a dynamics shift and a state–action distribution shift at the same time, and because the two are entangled, they cannot be fixed at a single layer. Draw the causal links between the components and the meaning of this becomes visible.

```
  c_t < 1  (execute faster)
     |
     v
 [1] tracking lag grows -----------> reached-pose target + high-gain OSC
     |                                    |
     |                                    |  side effect: stiff tracker
     |                                    v  amplifies reference noise
 [2] chunk inconsistency <------------ smooth reference now required
     |        fix: EAG (condition on previous tail, gated by tracking error)
     v
 [3] precision phases still fail ----> adaptive speed c_t
     |
     v
 [4] actions run out under latency --> latency-aware scheduling + delta_lb
```

- The numbers match the wall numbers in §1.3
- The most important link is the arrow by which the fix for [1] worsens [2]. The components are not a bundle of independent tricks but a structure in which each guards against the others' side effects, and this is what "full-stack" really means

Still, a fundamental question remains. The policy has only seen 20 Hz demonstrations and has never seen 5× motion. So how do its inputs stay in-distribution even when run fast?

> ### 💡 Speed is a time reparameterization — SAIL makes the "path" the robot passes through independent of speed
>
> A trajectory decomposes into a geometric path and a time law.
>
> $$x(t) = \mathbf{p}\big(s(t)\big)$$
>
> - $\mathbf{p}(s)$ — the path. Which poses are passed through, in what order
> - $s(t)$ — the time law. How fast the path is traversed
>
> What the policy actually learns is $\mathbf{p}$, because there is no time in its output and $s(t)$ is set by $\delta$ (§2.1). If tracking were perfect, shrinking $\delta$ would just make the robot traverse the same path faster. Then, at each moment along the path, the robot pose and the (quasi-static) scene the policy sees are the same as in the demonstrations, and the inputs stay in-distribution.
>
> The combination of commanded poses with low gain breaks this separation. As foreshadowed in §2.2, tracking lag is not just a shift along the time axis. Following the target late around a corner makes the robot cut the corner or overshoot, so **the path itself changes with speed.** The "collided with the neighboring cup" and "missed the handle" of the real-world failure cases (Fig. 8) are exactly this path deformation. The combination of reached poses with high gain restores the separation by making the realized path independent of speed again.
>
> This view is the same idea as DMP's time scaling, mentioned in the paper's related work, and the path–time separation (path + time scaling) of trajectory planning. DMP guaranteed it through **the policy representation**; SAIL recovers it through **the execution stack** while leaving DP's representation as it is.
>
> At the same time, this view also tells where SAIL will break. The assumption that "matching the path is enough" holds only when the world is quasi-static. **Physics that depends on the time law**, such as an object's momentum (a can flying off under acceleration), contact forces (wiping a board) and motion blur, does not return to in-distribution even if the path matches. This is exactly where the limitations in §7 lie.

---

## 5. Experiments

### 5.1 Setup and metrics

| | Simulation | Real world |
|---|---|---|
| Tasks | Robomimic Lift, Can, Square (200 human demos), MimicGen Stack, Mug Cleanup (500 generated demos) | 4 on Franka: Stacking Cups, Wiping Board, Baking, Folding Cloth (50 demos) / 3 on bimanual UR5: Plate Fruits, Pack Chicken (100), Bimanual Serve (75) |
| Demo rate | 20 Hz | 20 Hz |
| Policy | DP (ResNet18 encoder + 1D UNet, 10 DDIM steps) | Same |
| Baselines | DP, DP-Fast, BID-Fast, AWE, Aggregated Actions | DP-Fast (both set to 5× speed) |
| Evaluation | Table 1, Table K.6 | 10 trials per task (Table 2) |

| Baseline | Description |
|---|---|
| DP | Baseline executed at demonstration speed as is |
| DP-Fast | Naive speed-up running DP at a fixed speed factor with the low-gain controller |
| BID-Fast | DP-Fast plus BID. Resampling that picks, among several candidate chunks, the one that continues well from the previous chunk |
| AWE | Waypoint-based action labels. Speed-up is a by-product |
| Aggregated Actions | Merges consecutive delta actions in the same direction to reduce the number of steps |

Before reading the sim numbers, there is one setting to know.

> ### ⚠️ Fact check — the sim numbers come from an environment with torque limits removed
>
> So that acceleration would not be blocked by torque limits, the paper removed the Franka joint torque limits in Robosuite (Appendix C.1). For some baselines this was actually harmful, so they were reported with the limits kept. The sim speed-up numbers come from on top of this idealization.

**Primary metric TPR (throughput-with-regret, Eq. 2)**

$$\text{TPR} = \frac{1}{N}\sum_{i=1}^{N}\left(\frac{S_i}{t_i} - \frac{1 - S_i}{t^{\max}}\right)$$

- $N$ — number of trials
- $S_i \in \lbrace 0, 1\rbrace$ — whether trial $i$ succeeded
- $t_i$ — time taken by trial $i$
- $t^{\max}$ — time limit per trial (exceeding it is a failure)

A success earns a larger reward the faster it is ($1/t_i$), and a failure takes a flat penalty of $1/t^{\max}$. Since $t_i < t^{\max}$, **the value of one success is always larger than the penalty of one failure.** In other words, TPR is a metric that can prefer losing a little success rate to become much faster, and this property becomes important in §5.3.

Secondary metrics are SR (success rate), ATR (average time of successful trials) and SOD (speedup-over-demo = mean demonstration length / ATR), with SPARC, LDLJ, CON and WED as trajectory-quality metrics.

### 5.2 Simulation — 2–4× while keeping success rate

**Table 1** (cells are SR / TPR)

| | Lift | Can | Square | Stack | Mug |
|---|---|---|---|---|---|
| DP (1×) | 1.00 / 0.46 | 0.97 / 0.18 | 0.83 / 0.10 | 1.00 / 0.19 | 0.68 / 0.03 |
| DP-Fast | 0.95 / 1.02 | 0.87 / 0.37 | 0.55 / 0.15 | 0.98 / 0.44 | 0.56 / 0.05 |
| BID-Fast | 0.86 / 0.91 | 0.79 / 0.34 | 0.49 / 0.12 | 0.99 / 0.47 | 0.62 / 0.06 |
| **SAIL** | 1.00 / **1.68** | 0.92 / **0.51** | 0.86 / 0.13 | 0.98 / **0.66** | 0.72 / **0.08** |
| SOD (DP-Fast → SAIL) | 1.59 → 3.98 | 2.48 → 3.20 | 2.20 → 1.18 | 2.28 → 3.47 | 1.74 → 2.09 |

- **DP-Fast** — gets 1.6–2.5× faster but success rate collapses (Square 0.83 → 0.55)
- **BID-Fast** — post-hoc consistency correction alone cannot withstand the speed-up (Square 0.49). This matches the paper's criticism that post-hoc correction cannot deal with the OOD inputs created by controller shift itself
- **SAIL** — keeps success rate at DP level (only Can 0.97 → 0.92) while achieving SOD of 2.1–4.0×. The abstract's "up to 4× in sim" is Lift's SOD of 3.98. TPR relative to DP is 3.7× on Lift, 2.8× on Can and 3.5× on Stack
- **Fig. 6** — TPR on Lift and Can keeps rising up to $c = 0.1$ (10× speed). But the paper itself adds the caveat "at least in an ideal simulated environment"

The exception is Square. SAIL's SOD is 1.18, effectively no speed-up, and its TPR is lower than DP-Fast's (0.15). The reason becomes clear together with the ablation.

### 5.3 Ablation — where do the gains come from

**Table K.6** (cells are SR / TPR)

| | SAIL | −HG | −AS | −C | Commanded pose |
|---|---|---|---|---|---|
| Lift | 1.00 / 1.68 | 0.67 / 0.25 | 0.97 / 1.57 | 0.98 / 1.58 | 0.89 / 1.80 |
| Can | 0.92 / 0.51 | 0.83 / 0.18 | 0.95 / 0.60 | 0.89 / 0.50 | 0.63 / 0.37 |
| Square | 0.86 / 0.13 | 0.59 / 0.06 | 0.64 / 0.25 | 0.79 / 0.13 | 0.31 / 0.04 |
| Stack | 0.98 / 0.66 | 0.90 / 0.15 | 0.94 / 0.61 | 0.95 / 0.62 | 0.82 / 0.43 |
| Mug | 0.72 / 0.08 | 0.53 / 0.01 | 0.44 / 0.07 | 0.68 / 0.08 | 0.54 / 0.03 |

- −HG — high-gain controller removed
- −AS — adaptive speed removed
- −C — consistency (conditioning) removed
- Commanded pose — trained on commanded poses instead of reached poses

**⓵ High gain is the foundation.** −HG is worst on every task, and its SOD of 0.71–1.33 is even slower than the demonstrations. A soft controller cannot keep up with targets sent fast, and ends up slower as a result.

**⓶ Reached poses protect success rate.** Switching to commanded poses drops Square from 0.86 → 0.31 and Can from 0.92 → 0.63. The paper's "55% drop on Square" is in percentage points, not a relative ratio; as a relative ratio it is 64%. The average TPR decrease of 0.08 matches the table. Lift is the only exception (TPR 1.80).

**⓷ The consistency component contributes the least.** −C lowers SR by 0.02–0.07 and the TPR difference is small too. EAG, the most "algorithmic" contribution of this paper, has the smallest share in the sim ablation, and most of the gain comes from the control stack (my assessment). But EAG's reason for existing is the link at the end of §3.1: high gain amplifies jitter. So the paper sees EAG's effect as showing up more in smoothness metrics (Table G.1) and in the real-world bimanual tasks.

But Can and Stack, where the difference from −C is especially small, have a separate reason.

> ### ⚠️ Fact check — on Can and Stack, EAG is turned off by design
>
> The optimal weights in Table J.3 are $w_0 = 0$ for both Can and Stack, i.e., always unconditional generation. The "about 3× throughput over DP on Can and Stack" the paper emphasizes is a number obtained without EAG. This also seems to be why the difference from −C is small on these two tasks.

**⓸ Adaptive speed is a trade.** Comparing against −AS, the paper concludes this component is essential. But what metric does that conclusion stand on?

> ### ⚠️ Fact check — adaptive speed is a trade that "buys success rate and sells speed," and the primary metric TPR does not always approve that trade
>
> The paper's basis for calling adaptive speed essential is SR (Square 0.86 → 0.64, Mug 0.72 → 0.44). But the primary metric TPR in the same table shows a different picture.
>
> | Task | SAIL (SR / TPR / SOD) | −AS (SR / TPR / SOD) |
> |---|---|---|
> | Square | 0.86 / 0.13 / 1.18 | 0.64 / **0.25** / 3.01 |
> | Can | 0.92 / 0.51 / 3.20 | **0.95** / **0.60** / 3.60 |
>
> - **Square** — SAIL used $c^{\text{slow}} = 1.0$ (no speed-up) in precision phases, protecting SR but barely speeding up at all. Its TPR is half of −AS's, and falls short even of DP-Fast in Table 1 (0.15). This is the Square exception of §5.2
> - **Can** — −AS is ahead on both SR and TPR
>
> As seen in §5.1, TPR is a metric in which one success weighs more than one failure, so there are cases where SR gained at the cost of slowing down is not rewarded in TPR. It is more accurate to read this not as a flaw but as **a trade-off in which $c^{\text{slow}}$ must be chosen according to the operating goal**. Meanwhile, the main text's "up to a 31% drop" is not reproduced from the table (Square −22pp, Mug −28pp).

### 5.4 Real world — commanded speed and realized speed differ

Both methods were set to 5× speed ($c = 0.2$).

**Table 2**

| Task | DP-Fast SR / SOD | SAIL SR / SOD | TPR (DP-Fast → SAIL) |
|---|---|---|---|
| Stacking Cups | 0.10 / 1.85 | 0.40 / 1.76 | −2.28 → −0.12 |
| Wiping Board | 0.90 / 2.34 | 0.70 / 3.26 | 3.48 → 3.18 |
| Baking | 0.90 / 2.26 | 1.00 / 2.54 | 3.06 → 4.20 |
| Folding Cloth | 0.10 / 2.08 | 0.30 / 2.22 | −2.28 → −0.78 |
| Plate Fruits | 0.60 / 1.66 | 0.80 / 2.67 | 2.22 → 5.46 |
| Pack Chicken | 0.40 / 1.25 | 0.90 / 2.30 | 0.51 → 5.22 |
| Bimanual Serve | 0.40 / 1.43 | 0.70 / 2.39 | 1.00 → 5.40 |

- **The realized speed-up is 1.8–3.3×.** The setting is 5×, but adaptive speed slows down precision phases and the scheduling lower bound also binds
- **SAIL raised both SR and TPR on 6 of 7 tasks.** The big differences are in precise grasping and placing (Pack Chicken 0.4 → 0.9) and bimanual synchronization (Bimanual Serve, TPR 5.4×, SR about 1.8×)
- **DP-Fast's failure types (Fig. 8) map directly onto the four walls of §1.3.** Collisions and misses from insufficient tracking fidelity are Wall 1, dropping fruit from jitter is Wall 2, imprecise grasps are Wall 3, and stalling during inference latency is Wall 4

Then where in this table does the abstract's real-world "up to 3.2×" come from?

> ### ⚠️ Fact check — the real-world "up to 3.2×" is the value from the only task where SAIL lost to DP-Fast
>
> The real-world 3.2× in the abstract and conclusion is Wiping Board's SOD of 3.26. But this is the only task where SAIL trailed DP-Fast on both SR (0.70 vs 0.90) and TPR (3.18 vs 3.48). The paper, in the main text, attributes it to the high-gain controller failing to adapt to the changed robot–object dynamics while maintaining contact, and acknowledges it as a limitation, yet this value was used as the headline number. The maximum among tasks where SAIL had a higher success rate than DP-Fast is 2.67× on Plate Fruits.
>
> The evaluation design needs to be taken into account too. With DP-Fast as the only baseline, there is no way to know how much success rate was lost relative to 1× DP, and at 10 trials per task, a single trial is 10pp.

### 5.5 Transfer to another policy — ACT

SAIL was attached to ACT, using ACT's own temporal ensembling in place of EAG (Table K.7). Relative to 1× ACT, TPR goes up on all four tasks (e.g., Lift 0.40 → 1.50), but SR drops across the board (e.g., Can 0.77 → 0.56). This is consistent with the paper's claim that the components other than EAG apply to other generative policies too. But the cost of the missing consistency component shows as well.

---

## 6. Positioning — among neighboring work

| Lineage | Representative work | Relation to SAIL |
|---|---|---|
| Generative BC | Diffusion Policy, ACT | The base policies. They inherit the problem of temporal consistency between consecutive predictions |
| Post-hoc consistency correction | BID, ACT's temporal ensembling | Smoothing or selection after generation. Cannot handle OOD inputs caused by controller shift (compared via BID-Fast) |
| Time-modulated representations | DMP, RMP, Neural Dynamic Policies | Support speed modulation in principle. SAIL obtains the same property through the execution stack (§4) |
| IL that surpasses the demonstrator | T-REX, D-REX, Sakaino et al. | Require interaction and reward design. SAIL is purely offline |
| By-product speed-up | AWE, SPHINX | Speed is not the goal. SAIL absorbed AWE as a precision-phase detection tool |
| Full-stack IL systems | UMI, Mobile ALOHA | Precedents for joint controller–learning design. Do not address decoupling demonstration and execution speed |

The closest cousins are BID, which deals with the same chunk-boundary problem, and AWE, which SAIL absorbed as a tool. Then where, among these lineages, is SAIL's own place?

> ### 📌 A purely offline setting that makes speed-up the primary goal
>
> The paper claims to have formalized "execution faster than demonstration" as a new problem. It differs in three ways.
>
> - **No additional data, rewards or online learning.** It speeds up the demonstrations and trained policy you already have, through the execution stack
> - **Speed-up is the goal, not a by-product.** Even $c_t$, which varies speed over time, is part of the design
> - **Policy and controller are designed together.** The choice of learning target (reached pose) is paired with the choice of controller (high gain)

There is the contemporaneous work foreshadowed in §3.2. The problem of attaching a new chunk to the segment executed during inference latency is not SAIL's alone. Under what assumption did the other work solve it?

> ### 🔗 Real-Time Chunking (RTC, 2025-06) — the same chunk boundary, the opposite assumption
>
> RTC from Physical Intelligence and UC Berkeley (Black, Galliker, Levine, arXiv:2506.07339, NeurIPS 2025) came out four days before SAIL and is not cited in the SAIL paper. RTC generates the next chunk while executing the previous one, freezing the actions whose execution is committed because of latency and inpainting the rest. It uses soft masking to carry continuity beyond the frozen segment, and it works at inference time only, with no retraining.
>
> | | SAIL (EAG) | RTC |
> |---|---|---|
> | What it solves | Chunk-boundary discontinuity under speed-up and latency | Chunk-boundary discontinuity under latency |
> | How it stitches | Condition input at training time (previous tail) + CFG | Inpainting at inference time (frozen segment + soft masking) |
> | Retraining | Required (trained with condition dropout) | Not required |
> | Trust in the previous plan | Judged by tracking error, and **can be discarded** | The segment committed to execution is frozen as ground truth |
>
> Both deal with the same object in that they use "the segment executed during inference latency" as the condition. The key difference is the last row. RTC treats the actions to be executed as established fact, while SAIL takes issue with the point that, under speed-up, **the commanded action and where the robot actually went can differ**. In the regime where tracking is good, the two approaches do the same thing; in the regime where tracking degrades, SAIL's gate becomes meaningful.
>
> A follow-up by the same group of authors, *Training-time action conditioning for efficient real-time chunking* (Black, Ren, Equi, Levine), uses action conditioning at training time, as its title says, and is structurally closer to SAIL's conditioning training.

A follow-up that cites SAIL is SpeedAug (arXiv:2512.00062, *Policy Acceleration via Tempo-Enriched Policy and RL Fine-Tuning*). Going by its title, it is a direction that fills the place SAIL left purely offline with RL fine-tuning.

---

## 7. Limitations

What the paper states itself, together with what is worth noting from reading it.

**Limitations the paper acknowledges**

- **Robot–object dynamics shift is unsolved.** In sim Can, momentum under acceleration throws the can out of the workspace, and in real-world Wiping Board, sustained contact fails. This is where the "quasi-static world" assumption from §4 breaks
- **There is the finite-data limit of offline IL.** Distribution shift that algorithms alone cannot solve remains, and the paper suggests explicitly including a dynamics model or using simulation data
- **EAG thresholds differ per task** (Appendix H.3)

**Further points to note**

- **The per-task tuning surface is wide** — $w_0 \in \lbrace 0,1\rbrace$, $\rho$, $c^{\text{slow}}/c^{\text{fast}}$ and sim $K_p$ (1000–3000) are all per-task values. It is closer to a recipe tuned for each task than a drop-in system
- **There are only two speed levels** — since $k_t$ is 0/1, it is far from continuous time optimization based on path curvature or contact (TOPP-style)
- **The time spacing of the observation history is unclear** — the policy stacks 4 observation frames (Table J.2). If the displacement between frames grows under speed-up, that itself may be OOD, but how observation sampling was matched is not stated
- **There is no force information** — since the demonstrations carry no force or contact information, speeding up contact-rich tasks is disadvantaged in principle. This is consistent with the Wiping Board result
- **The evaluation scale is small** — the real world has 10 trials per task, one baseline, and no 1× reference. The sim has torque limits removed
- **The cost of transfer differs with model scale** — the experiments use a small UNet DP. With a VLA whose inference is heavy, the latency bound of §3.4 becomes a real constraint, and EAG requires retraining to accept the condition input. On the other hand, reached-pose labeling and high-gain tracking look transferable regardless of the model

---

## 8. Closing — what this paper suggests

SAIL's real contribution lies less in the EAG algorithm than in the problem definition **"speed is a property of the stack, not of the policy."** As the ablation shows, most of the gain came from the control side, and the policy-side components served to tame the side effects that control created.

And this paper reads unusually well to someone with an LLM or diffusion background.

- **Action labels are paired with the executor.** The view that the commanded pose is a latent tied to the controller as a decoder is the same problem as why the definition of the action space (commanded vs reached, delta vs absolute) matters in robot foundation models that mix heterogeneous robot data. Carried over to LLMs, it is in the same vein as the meaning of the same tokens changing when the decoder changes
- **It is the idea of using CFG as a reliability gate.** It uses CFG from image generation as an on/off switch based on condition reliability. Carried over to agentic systems, it is the same structure as stale-plan invalidation, which removes the previous plan from the context and replans when a signal arrives that it has diverged from reality
- **The policy handles the path, the stack handles the time law.** This division of labor is a design principle that stays valid even as models grow into VLAs or world models, and the slower the inference of a large model, the larger the stack's share becomes

Finally, there is one practical implication. The speed in a robot demo video is not the model's performance alone. Control gains, action labels, the scheduler and inference latency determine it together. This is why, when watching a video, you should also ask "is this real time, or played back at some speed-up, and what are the commanded and realized speed-ups?"

---

## Appendix — glossary

| Term | Definition |
|---|---|
| **Commanded pose** | The target pose $x^d$ a person sent during teleop. A value that presupposes the collection controller's lag |
| **Reached pose** | The pose $x$ the robot actually reached. SAIL's action label and a controller-invariant target |
| **Speed factor $c_t$** | The multiplier in the execution interval $\delta_t = c_t\delta^*$. Smaller is faster (0.2 = 5× speed) |
| **Tracking lag** | The error from a controller following its target late. Proportional to speed, inversely proportional to stiffness |
| **High-gain OSC** | Operational space control with raised stiffness. Follows fast targets accurately but also amplifies jitter in the target |
| **EAG (Error-Adaptive Guidance)** | A gate that uses the previous plan's tail as the condition only when tracking error is at or below a threshold |
| **Adaptive speed modulation** | Speed control that switches between $c^{\text{slow}}$ and $c^{\text{fast}}$ according to the precision-phase flag $k_t$ |
| **Action exhaustion** | The robot stalling because the actions to execute run out during inference latency |
| **TPR (throughput-with-regret)** | A throughput metric that rewards fast successes and gives failures a flat penalty |
| **SOD (speedup-over-demo)** | Mean demonstration length / mean time of successful trials. The realized speed-up |

**Original** — [arXiv:2506.11948](https://arxiv.org/abs/2506.11948) · **Project page** — [nadunranawaka1.github.io/sail-policy](https://nadunranawaka1.github.io/sail-policy)
