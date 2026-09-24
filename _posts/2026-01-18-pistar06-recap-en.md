---
layout: paper
lang: en
ref: pistar06-recap
kind: paper-review
title: "π*₀.₆: a VLA That Learns From Experience (RECAP)"
date: 2026-01-18 12:00:00 -0800
paper_date: 2025-11-18
venue: "arXiv preprint · arXiv:2511.14759"
tags: [VLA, Reinforcement-Learning, Offline-RL, Advantage-Conditioning, Robot-Foundation-Model, Paper-Review]
authors: "Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, Danny Driess, Michael Equi, Adnan Esmail, Yunhao Fang, Chelsea Finn, Catherine Glossop, Thomas Godden, Ivan Goryachev, Lachy Groom, Hunter Hancock, Karol Hausman, Gashon Hussein, Brian Ichter, Szymon Jakubczak, Rowan Jen, Tim Jones, Ben Katz, Liyiming Ke, Chandra Kuchi, Marinda Lamb, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Yao Lu, Vishnu Mano, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Charvi Sharma, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, Will Stoeckle, Alex Swerdlow, James Tanner, Marcel Torne, Quan Vuong, Anna Walling, Haohuan Wang, Blake Williams, Sukwon Yoo, Lili Yu, Ury Zhilinsky, Zhiyuan Zhou"
affiliations: "Physical Intelligence"
summary: "Judge with a value function whether each action beat the data's average, feed that verdict in as a single text condition, and train with conditional SFT (CFG-style policy improvement) — one recipe absorbs human demos, autonomous rollouts and corrections, and more than doubles a flow VLA's throughput from deployment experience alone."
paper_url: "https://arxiv.org/abs/2511.14759"
---

> **Core claim** — Improving a huge flow-matching VLA with RL does not need policy gradients. Use a value function to judge whether each action in the data was "better than usual," put that verdict in as a single text condition, and do conditional SFT. This one recipe absorbs human demonstrations, autonomous rollouts and human corrections alike, and from deployment experience alone it more than doubles throughput on the hardest tasks.

---

## Introduction

The standard VLA recipe is large-scale pretraining on mixed web and robot data, followed by SFT on teleoperation demonstrations of the target task. A policy made with this recipe only imitates the demonstrations, so in principle it cannot become better than them.

People are different. Clumsy at first, they try again and again, learn from their mistakes, and eventually become more skilled than whoever taught them. Physical Intelligence's π*₀.₆ paper proposes **RECAP** (RL with Experience and Corrections via Advantage-conditioned Policies) as a way to have a VLA do this "practice." Deploy the robot for real to collect autonomous attempts and human corrections, grade each action as good or bad with a value function, and retrain the VLA conditioned on that verdict. The resulting model makes espresso for 13 hours straight, folds laundry for more than 2 hours without stopping in homes it has never seen, and assembles packaging boxes in a real factory.

The paper's key trick is one already familiar to anyone who has worked with diffusion — **classifier-free guidance**. This piece follows the paper's argument, but builds up the necessary concepts as they come up, so that readers with a thin background in reinforcement learning can also be convinced of the reason behind each design decision.

---

## 1. The problem — the ceiling of imitation learning, and three walls

### 1.1 Two things demonstrations cannot give

An imitation-learning policy suffers compounding error, and at best it is only as good as the demonstration data. There are two things demonstrations cannot provide.

**⓵ Correcting its own mistakes** — the failure states a deployed policy actually falls into are not in the demonstrations. A demonstrator can only show the trajectories that come out when they themselves are in control.

**⓶ Speed and robustness beyond the demonstrator** — motion faster and smoother than teleoperation does not appear no matter how many demonstrations are collected, because the target of imitation learning is the demonstration distribution itself.

Drawn conceptually:

```
  performance
     ^
     |                               .----------  RL from experience
     |                          .---'
     |  - - - - - - - - - - - -/- - - - - - - - -  demo quality (ceiling)
     |           .------------+-----------------  imitation (SFT)
     |      .---'
     +---------------------------------------------> data / iterations
```

- Imitation learning (SFT) saturates below demonstration quality even as data grows
- To break through the ceiling, the policy must learn from **its own experience** — that is RL

Yet the paper's introduction states that there are three hard problems in implementing this principle as a general, scalable robot learning system. I name the three as walls and track them to the end of the piece.

### 1.2 Three walls

**⛔ Wall 1 — Putting RL on a huge, expressive VLA**

π₀.₆'s actions are produced by an 860M-parameter flow-matching action expert. Most RL algorithms require the policy's log-likelihood $\log\pi_\theta(a\mid o)$ or differentiable sampling, and flow models do not provide this cheaply. The paper's related work sorts existing attempts into two groups. The line that applies PPO directly to VLAs is hard to scale efficiently to the real world. The line that puts RL on top of a VLA (residual policies, action-head fine-tuning, action selection and refinement, policies in diffusion noise space) mostly uses discrete actions or simple Gaussian distributions. **There is no case of RL on an entire flow VLA end-to-end.**

**⛔ Wall 2 — Data from all different sources**

The data that actually accumulates is mixed like this.

```
  data source          behavior policy            label
  -------------------  -------------------------  --------------
  demos (pretrain)     many human teleoperators   success / fail
  demos (task l)       human teleoperator         success / fail
  autonomous, iter k   pi_l^(k-1)                 success / fail
  corrections          pi_l^(k-1) + human         success / fail
```

The policies that produced the data (behavior policies) are all different. On-policy methods such as PPO or REINFORCE must throw away past data every time the policy changes. On a physical robot, each episode is an expensive resource that eats human time and equipment uptime, so throwing it away is a luxury.

**⛔ Wall 3 — Rewards in the real world**

There is no function that judges success automatically, as a simulator does. A person has to judge the outcome of each episode, and that judgment can be ambiguous or stochastic. Moreover, success rate is not the only goal that matters in the field. **How fast** it gets done, that is, throughput per hour, has to go up too.

### 1.3 So the question the paper asks

> With **one general RL recipe** that includes human reward feedback and interventions, can the robustness and throughput of a huge flow VLA be raised **at the same time**, using only deployment experience? And can that reward signal be put in not only in post-training but **from the pretraining stage onward**?

What the three walls are, and how RECAP gets over each, is the narrative of this paper.

---

## 2. Background — the minimum of RL needed to understand the three walls

If you are familiar with RL, feel free to skip to Section 3.

### 2.1 Value functions and the n-step advantage

This paper uses no discount ($\gamma=1$, finite horizon). The observation $o_t$ is treated as the Markov state (a common simplification in robot RL, as the paper notes in a footnote).

$$V^{\pi}(o_t)=\mathbb{E}\Big[\sum_{t'=t}^{T}r_{t'}\Big],\qquad A^{\pi}(o_t,a_t)=\mathbb{E}_{\rho_\pi}\Big[\sum_{t'=t}^{t+N-1}r_{t'}+V^{\pi}(o_{t+N})\Big]-V^{\pi}(o_t)$$

- $o_t$ — observation. Camera images and robot joint state
- $r_t$ — reward at step $t$, $T$ — the last step of the episode
- $V^\pi(o_t)$ — expected cumulative reward when starting from $o_t$ and following policy $\pi$
- $A^\pi(o_t,a_t)$ — how much better action $a_t$ was than "the average $\pi$." The first $N$ steps add up actual rewards, and after that $V$ stands in (n-step estimate)
- $\rho_\pi$ — the trajectory distribution produced by the policy and the environment dynamics

### 2.2 Policy extraction — three routes from a value function to a policy

Getting a value function does not by itself make the policy better. A step is needed that turns the verdict "this action is better than average" into a change in the policy's parameters, and this is called **policy extraction**. There are broadly three routes.

| Route | Representative | What it requires of the policy | Problem in this setting |
|---|---|---|---|
| Policy gradient | PPO, REINFORCE, SAC (reparameterization) | Log-likelihood or differentiable sampling | Log-likelihood is expensive for flow models (Wall 1). On-policy throws data away (Wall 2) |
| Weighted regression | AWR, CRR, IQL-style extraction | Supervised learning multiplied by advantage weights | Effectively discards low-advantage data → close to filtered imitation |
| Conditioning | UDRL, RCP, Decision Transformer, CFGRL | Conditional supervised learning | The path RECAP takes |

Let's see with equations why the first route runs into Wall 1. For a model that produces action $a=z_1$ from noise $z_0$ through the flow ODE $\dot z_s=v_\theta(z_s,s\mid o)$, the exact log-likelihood is given by the instantaneous change-of-variables formula.

$$\log\pi_\theta(a\mid o)=\log p_0(z_0)-\int_0^1\nabla_z\cdot v_\theta(z_s,s\mid o)\,ds$$

- $p_0$ — noise prior $\mathcal N(0,I)$
- $v_\theta$ — the learned velocity field, $z_s$ — the intermediate state at time $s$
- $\nabla_z\cdot v_\theta$ — divergence of the velocity field (trace of the Jacobian)

To get the log-likelihood of one sample, you have to integrate the Jacobian trace over time while solving the ODE. For a large VLA, computing the probability ratio $\pi_\theta/\pi_{\text{ref}}$ that PPO requires exactly for every sample is practically impossible. So work that tries to use PPO on flow models relies on ELBO-style approximations (DPPO, FPO). This paper's PPO baseline works that way too (we come back to it in Section 5).

### 2.3 Regularized RL — the closed-form solution for "a little better than the reference policy"

To take several gradient steps on the same data, the standard is to regularize so as to stay near the policy $\pi_{\text{ref}}$ that produced the data.

$$\mathcal J(\pi,\pi_{\text{ref}})=\mathbb E_{\tau\sim\rho_\pi}\Big[\sum_t r_t\Big]-\beta\,\mathbb E_{o}\Big[D\big(\pi(\cdot\mid o)\,\Vert\,\pi_{\text{ref}}(\cdot\mid o)\big)\Big]$$

- $\pi_{\text{ref}}$ — the reference policy. Usually the behavior policy that collected the training data
- $D$ — a divergence between the two distributions, $\beta$ — regularization strength (Lagrange multiplier)

If $D$ is KL, a well-known closed-form solution comes out. This is the theoretical basis of AWR.

$$\hat\pi(a\mid o)\ \propto\ \pi_{\text{ref}}(a\mid o)\,\exp\big(A^{\pi_{\text{ref}}}(o,a)/\beta\big)$$

What RECAP builds on is a less well-known sister result close to this one.

$$\hat\pi(a\mid o)\ \propto\ \pi_{\text{ref}}(a\mid o)\ p\big(I\mid A^{\pi_{\text{ref}}}(o,a)\big)^{\beta},\qquad p(I\mid A)=\frac{g\big(A^{\pi_{\text{ref}}}(o,a)\big)}{\int g\big(A^{\pi_{\text{ref}}}(o,a')\big)\,da'}$$

- $I$ — the event "this action is an improvement over $\pi_{\text{ref}}$"
- $p(I\mid A)$ — the probability that action $a$ is an improvement
- $g$ — any monotonically increasing function
- $\beta$ — sharpness of the reweighting

A $\hat\pi$ defined this way guarantees:

$$\mathcal J(\hat\pi)\ \ge\ \mathcal J(\pi_{\text{ref}})$$

The intuition is simple. Reweight $\pi_{\text{ref}}$'s action distribution by "the probability of being an improvement," and it is guaranteed to get better. But the `exp(A/β)` form would be enough too, so why use this form? The reason emerges in one line of Bayes' rule in Section 3.3.

---

## 3. Method — RECAP

### 3.0 Overall structure

RECAP is the repetition of three subroutines.

- **⓵ Data collection** — run the VLA on the task and attach a success/failure label to each episode. When needed, a person intervenes to correct
- **⓶ Value function training** — with all the data collected so far, train a multi-task value function that detects failures and judges the time remaining to completion
- **⓷ Advantage-conditioned training** — build an optimality indicator from the advantages given by the value function, put it into the VLA's input, and train

What changes at each stage is only the data fed to each subroutine.

```
 PRE-TRAINING   (tens of thousands of hours of demos, many robots)
   (2) V_pre   <- Eq.1 on D_demo
   (3) pi_pre  <- Eq.3 on D_demo, advantages from V_pre
                      |
                      v
 POST-TRAINING  (per task l)
   pi_l^0  <- SFT from pi_pre on demos of l   (I fixed to True)
   for k = 1..K:
     (1) collect with pi_l^(k-1): autonomous + human corrections -> D_l
     (2) V_l^k   <- finetune from V_pre   on ALL of D_l
     (3) pi_l^k  <- finetune from pi_pre  on ALL of D_l
```

- **Pretraining** runs only ⓶⓷ on tens of thousands of hours of multi-robot, multi-task demonstrations. This means the reward signal comes in from pretraining onward
- **Post-training** fine-tunes again at every iteration **from the pretrained checkpoint**, not from the previous iteration's model. This avoids drift over many iterations
- According to the paper, a single iteration often improved things substantially

Now let's look at each component, starting with the reward.

### 3.1 Reward — carrying speed too with a single success label (Wall 3)

All a person has to do is attach one success/failure label per episode. The paper builds the reward from this label as follows.

$$r_t=-1\ \ (t<T),\qquad r_T=0\ \ \text{(success)},\qquad r_T=-C_{\text{fail}}\ \ \text{(failure)}$$

- $T$ — the last step of the episode
- $C_{\text{fail}}$ — a large constant that pushes the value of failed episodes low enough

Under this reward, the return from step $t$ onward is:

$$R_t=\sum_{t'=t}^{T}r_{t'}=-(T-t)\ \ \text{(success)},\qquad R_t=-(T-t)-C_{\text{fail}}\ \ \text{(failure)}$$

- $R_t$ — the actual cumulative reward from step $t$ to the end of the episode

In a successful episode, the return is **the negative of the number of remaining steps**. So the value function comes to predict "how many steps are left until success." Because typical lengths vary widely by task, it is normalized into the $(-1,0)$ interval by dividing by each task's maximum episode length.

One thing is confusing here. The information a person gives is 1 bit per episode, as sparse as an outcome reward in LLM RL. Yet a $-1$ is attached at every step. Is this a sparse reward or a dense one?

> ### 💡 The information source is sparse, the signal is dense — the time penalty creates throughput
>
> The two must be distinguished. **Labeling cost** is sparse: a person gives only success or failure per episode. The **learning signal**, by contrast, is dense: the $-1$ per step is free information no person has to supply, so it creates a gradient along the time axis without increasing labeling cost.
>
> Carried over to LLMs, it has the same shape as "correctness outcome reward + length penalty." GRPO's pure outcome reward has no pressure to "solve it faster," but this reward bundles success and speed into one scalar.
>
> | | Pure outcome reward | RECAP's reward |
> |---|---|---|
> | Information a person gives | Success or failure per episode | Success or failure per episode (same) |
> | What the value function learns | Probability of success | Time remaining to success (failure is a large negative) |
> | Pressure toward speed | None | Present |
>
> Because of this design, RECAP's improvement shows up more in **throughput** than in success rate. Section 5 confirms it.

### 3.2 A distributional value function — a Monte Carlo critic of the data-mixture policy (Walls 2 and 3)

The value function does not regress the return as a scalar; it classifies it as a distribution over $B=201$ bins.

$$\min_\phi\ \mathbb E_{\tau\in\mathcal D}\Big[\sum_{o_t\in\tau}H\big(R^B_t(\tau),\ p_\phi(V\mid o_t,\ell)\big)\Big]$$

$$V^{\pi_{\text{ref}}}(o_t,\ell)=\sum_{b}p_\phi(V=b\mid o_t,\ell)\,v(b)$$

- $R^B_t(\tau)$ — the actual return after step $t$ of trajectory $\tau$, discretized into bins
- $p_\phi(V\mid o_t,\ell)$ — the bin distribution of the value given the observation and the language command
- $\ell$ — language input. The task prompt and metadata that adjusts how the task is performed
- $H$ — cross-entropy, $v(b)$ — the value that bin $b$ represents
- $\mathcal D$ — all the data collected so far

The value function uses the same architecture as the VLA, but with a smaller 670M VLM backbone initialized from Gemma 3. To prevent overfitting, it is co-trained with a small amount of multimodal web data. Because it is small, computing advantages on the fly during VLA training adds almost no cost.

In the paper's visualization of the value function (Fig. 4), the value drops the moment the robot arm crumples a folded shirt and rises again during recovery. In failed episodes, the value plunges the moment an object is knocked over. This value function works as **a mistake detector and a progress speedometer**.

Beyond citing Bellemare et al. (2017), the paper does not separately explain why it made the value function distributional. The practical reasons seem to be that, in multi-task regression where the scale differs widely by task, cross-entropy classification is more stable than MSE, and that it meshes naturally with a VLM's token classification head (my assessment).

But there is something odd about this equation. The paper itself calls this estimator "on-policy," saying it is less optimal than classical off-policy Q estimators but simple and reliable. Yet, as the table in Section 1.2 showed, the data is typical off-policy data mixing human demonstrations, several generations of policies, and human corrections. Does it make sense to handle off-policy data with an on-policy estimator?

> ### 💡 What V estimates is the value not of "the current policy" but of "the data-mixture policy"
>
> The key is **which policy's value is being estimated**. Run Monte Carlo regression on the actual returns in dataset $\mathcal D$, and the result is $V^{\pi_{\text{ref}}}$, the value of $\pi_{\text{ref}}$, the mixture of the policies that produced $\mathcal D$. Here "on-policy" does not mean "uses only new data" but **"evaluates the very policy that produced the data."**
>
> That also fixes the meaning of the advantage $A^{\pi_{\text{ref}}}$: **"was it better than the average action in this data."** The guarantee $\mathcal J(\hat\pi)\ge\mathcal J(\pi_{\text{ref}})$ from Section 2.3 is likewise exactly **a one-step improvement** over this mixture policy.
>
> | | Off-policy Q-learning | RECAP's MC $V^{\pi_{\text{ref}}}$ |
> |---|---|---|
> | What is estimated | Q of the improved policy (bootstrapped) | V of the data-mixture policy |
> | Size of one improvement | Multi-step improvement possible | One step over the mixture policy |
> | Main risk | Overestimation of OOD actions, divergence | Almost none (regression on actual returns) |
>
> It cannot go far in one go, but iteration makes up the distance. Once the rollouts of the $k$-th policy enter $\mathcal D$, $\pi_{\text{ref}}$ itself goes up a step, and it improves one step again on top of that. Reading it as approximate policy iteration is accurate.

Advantages are computed differently at each stage.

**Post-training — 50-step lookahead.** It uses the observation 50 steps later in the same trajectory. If the episode does not end within 50 steps, it simplifies as follows (in pre-normalization units).

$$A(o_t,a_t)=\sum_{t'=t}^{t+49}r_{t'}+V(o_{t+50})-V(o_t)=V(o_{t+50})-\big(V(o_t)+50\big)$$

- $V(o_t)+50$ — the value the data's average policy would have reached 50 steps later
- So the advantage measures **"did it advance further than the data average over 50 steps."** This is the path by which speed enters the advantage directly

**Pretraining — to the end of the episode ($N=T$).** It is a high-variance estimate that subtracts $V(o_t)$ from the return, but it can be computed on the fly with a single value-function call. The paper reports that it worked well empirically on large-scale multi-task data.

### 3.3 Policy extraction — put the advantage in as a "condition" (Walls 1 and 2)

Time to answer the question deferred in Section 2.3. Apply Bayes' rule to the improvement probability $p(I\mid o,a)$.

$$p(I\mid o,a)=\frac{\pi_{\text{ref}}(a\mid I,o)\,p(I\mid o)}{\pi_{\text{ref}}(a\mid o)}\quad\Longrightarrow\quad p(I\mid o,a)\ \propto_a\ \frac{\pi_{\text{ref}}(a\mid I,o)}{\pi_{\text{ref}}(a\mid o)}$$

- $\pi_{\text{ref}}(a\mid I,o)$ — the conditional distribution of only the "actions that were improvements" in the data
- $\pi_{\text{ref}}(a\mid o)$ — the action distribution of all the data
- $p(I\mid o)$ — independent of $a$, so it is absorbed into the normalizing constant

Substitute this into the equation of Section 2.3 and attach the language condition, and you get the paper's Eq. 2.

$$\hat\pi(a\mid o,\ell)\ \propto\ \pi_{\text{ref}}(a\mid o,\ell)\left(\frac{\pi_{\text{ref}}(a\mid I,o,\ell)}{\pi_{\text{ref}}(a\mid o,\ell)}\right)^{\beta}$$

With $\beta=1$ the equation becomes dramatically simpler.

$$\beta=1\quad\Longrightarrow\quad\hat\pi(a\mid o,\ell)=\pi_{\text{ref}}(a\mid I,o,\ell)$$

This is why this form is used instead of `exp(A/β)`. **The need to compute the improvement probability explicitly disappears.** Just learn two distributions by imitation learning, the conditional $\pi_{\text{ref}}(a\mid I,o,\ell)$ and the unconditional $\pi_{\text{ref}}(a\mid o,\ell)$, and the improved policy is contained in them.

The improvement indicator is binarized with a task-specific threshold.

$$I_t=\mathbf 1\big(A^{\pi_{\text{ref}}}(o_t,a_t,\ell)>\epsilon_\ell\big)$$

- $\epsilon_\ell$ — the improvement threshold for task $\ell$

The conditional and unconditional are learned together, and their ratio is raised to a power to sharpen the distribution. Readers who have worked with diffusion will feel déjà vu here — where have we seen this structure?

> ### 💡 This is classifier-free guidance itself
>
> According to the appendix, when inferring with $\beta>1$, the score that flow sampling follows is:
>
> $$\nabla_a\log\pi_\theta(a\mid o)+\beta\big(\nabla_a\log\pi_\theta(a\mid I,o)-\nabla_a\log\pi_\theta(a\mid o)\big)$$
>
> This is literally the same equation as CFG in image generation.
>
> $$\nabla_x\log p(x)+w\big(\nabla_x\log p(x\mid c)-\nabla_x\log p(x)\big)$$
>
> Here $x$ is the image to generate, $c$ the condition (text prompt), and $w$ the guidance scale.
>
> | CFG (image generation) | RECAP |
> |---|---|
> | Condition $c$ (text prompt) | Improvement indicator $I$ |
> | Implicit classifier $p(c\mid x)$ | Improvement probability $p(I\mid o,a)$ |
> | Guidance scale $w$ | $\beta$ |
> | Condition dropout during training | $I$ omitted with 30% probability during training |
> | $w$ too high → oversaturation, artifacts | $\beta$ too high → actions pushed to the support boundary, aggressive motion |
>
> This correspondence is the core proposition of CFGRL (Frans et al., 2025): **diffusion guidance is a controllable policy-improvement operator**. The knob that raises prompt fidelity becomes the knob that raises action optimality. RECAP carried this principle over to large VLAs.

RECAP does one thing differently from CFGRL, though. CFGRL fixed $\epsilon=0$ (an improvement if the advantage is positive) and adjusted $\beta$ at inference. RECAP **controls sharpness at training time with the threshold $\epsilon_\ell$**, and by default infers with $\beta=1$. There are two reasons.

- **⓵ Aggressive motion** — a high $\beta$ pushes the action distribution to the edges of the learned support
- **⓶ No effect on the autoregressive part** — CFG works only on the flow part and cannot affect the model's autoregressive outputs (subtask text, discrete action tokens)

Only when needed does it add moderate guidance, around $\beta\in[1.5,\ 2.5]$.

> ### ⚠️ Fact check — the description of the threshold $\epsilon_\ell$ disagrees between the main text and the appendix
>
> The main text (V-D) says $\epsilon_\ell$ is set per task to "the 30th percentile of the **values** predicted by the value function." The appendix (F) says it is set so that "about 30% of the demonstration data has **positive advantage**." Setting the threshold at the 30th percentile makes about 70% positive, so the direction is reversed, and the targets differ too: values vs. advantages.
>
> The appendix is more specific and also matches the intent of the method ("only the top portion is positive"), so reading it as **roughly the top 30% being positive** is reasonable. According to the appendix, post-training sets it so that about 40% is positive, and for the T-shirt and shorts tasks, where the demonstration-based policy has a high success rate but is slow, it was raised so that only about 10% is positive. In other words, $\epsilon_\ell$ is effectively a **quantile hyperparameter** tuned per task.

The training loss is the negative log-likelihood of two terms (Eq. 3).

$$\min_\theta\ \mathbb E_{\mathcal D_{\pi_{\text{ref}}}}\Big[-\log\pi_\theta(a_t\mid o_t,\ell)-\alpha\log\pi_\theta(a_t\mid I_t,o_t,\ell)\Big]$$

- First term — the unconditional model. Imitates the data-mixture policy $\pi_{\text{ref}}$ as is
- Second term — the conditional model. Imitates together with $I_t$
- $\alpha$ — the balance between the two terms. In practice $\alpha$ is not tuned; it is replaced by dropout that removes $I_t$ from the input with 30% probability. One model comes to represent both distributions
- $\mathcal D_{\pi_{\text{ref}}}$ — **all** the data collected so far

**Human correction segments are forced to $I_t=\text{True}$ regardless of advantage.** The assumption is that an expert's corrective actions are always good actions.

But this loss trains, via log-likelihood, on everything: failed trajectories, slow demonstrations, even the mistakes before correction. Won't it end up imitating bad actions too?

> ### 💡 Failure data is not thrown away; it becomes a "control group"
>
> Bad actions are learned together with the label $I=\text{negative}$. The model learns what bad actions look like, but at inference it samples only with $I=\text{positive}$, so it does not draw actions from that side of the distribution.
>
> | | Filtered BC | AWR | Advantage conditioning |
> |---|---|---|---|
> | Training loss | NLL on successful trajectories only | NLL weighted by exponentiated advantage | NLL on everything + $I$ in the input |
> | Bad data | Discarded | Weight near 0 | Learned with a negative label |
> | Unit of good/bad | Trajectory | State–action | State–action |
> | Inference | Sample as is | Sample as is | $I=\text{positive}$ (CFG if needed) |
>
> Strictly speaking, when $\beta=1$ the target distribution RECAP aims at, $\pi_{\text{ref}}(a\mid I,o)$, is itself "the distribution of positive actions only." **Looking only at the target distribution, it is in the same family as AWR and filtered BC.** The difference lies not in the target but in the learning signal. Negative data also supervises the shared backbone and the unconditional and negative branches, so it is not discarded, and the contrast between conditional and unconditional becomes the material for $\beta>1$ guidance. The point where the paper criticizes AWR as "filtered imitation that throws data away" is exactly the former.

### 3.4 Fitting it into π₀.₆ — without the flow log-likelihood (Wall 1)

RECAP's base model π₀.₆ is a VLA that improves on π₀.₅. Compared with π₀.₅, it adds pretraining data from other robot platforms, and grows the base VLM to Gemma 3 4B and the action expert to 860M parameters. The architecture has three essentials.

- **Knowledge Insulation (KI)** — continuous actions (flow) and discrete tokens (including actions tokenized with FAST) are trained end-to-end together, but a stop-gradient keeps the flow action expert's gradient from flowing into the backbone
- **Subtask prediction** — the model first generates the next subtask as text $\hat\ell$ ("pick up the coffee cup," etc.), and actions are generated after it, conditioned on $\hat\ell$. At inference, subtask prediction runs at a lower frequency than action generation
- **Output** — an action chunk $a_{t:t+H}$ made of 50Hz joint angles and gripper commands

Because $\hat\ell$ is predicted first, the total log-likelihood decomposes into three terms.

$$\log\pi_\theta(a_{t:t+H},a^{\ell}_{t:t+H},\hat\ell\mid o_t,\ell)=\log\pi_\theta(\hat\ell\mid o_t,\ell)+\log\pi_\theta(a^{\ell}_{t:t+H}\mid o_t,\ell,\hat\ell)+\log\pi_\theta(a_{t:t+H}\mid o_t,\ell,\hat\ell)$$

- $\hat\ell$ — the predicted subtask text (autoregressive)
- $a^{\ell}_{t:t+H}$ — FAST discrete action tokens (autoregressive, KI's auxiliary objective)
- $a_{t:t+H}$ — the continuous action chunk (flow matching). The action expert does not take the discrete tokens as input, so the two are predicted independently

What π*₀.₆ adds here is **a single text input**. If $I_t$ is true, it inserts `Advantage: positive`; if false, `Advantage: negative`.

```
 [imgs][q][task + metadata][subtask l_hat][Advantage: +/-][FAST] --> action expert --> a_{t:t+H}
                                          ^
                                          I_t: after l_hat, before actions
```

- The indicator is placed **after** the subtask $\hat\ell$ and **before** the actions, so only the actions' log-likelihood is affected
- "What to do" (the subtask) is left alone, and only "how to do it" (the actions) is put under the optimality condition

The exact log-likelihood of continuous actions still cannot be computed, so the flow matching loss stands in as a lower bound on the log-likelihood (Eq. 4).

$$\log\pi_\theta(a_{t:t+H},a^{\ell}_{t:t+H}\mid I_t,o_t,\ell,\hat\ell)\ \ge\ \mathbb E_{\eta,\omega}\Big[\log p_\theta(a^{\ell}_{t:t+H}\mid I_t,o_t,\ell,\hat\ell)-\alpha_\eta\big\lVert\omega-a_{t:t+H}-f_\theta(a^{\eta,\omega}_{t:t+H},I_t,o_t,\ell,\hat\ell)\big\rVert^2\Big]$$

$$a^{\eta,\omega}_{t:t+H}=\eta\,a_{t:t+H}+(1-\eta)\,\omega,\qquad\omega\sim\mathcal N(0,I)$$

- $\eta\in[0,1]$ — flow matching time, $\omega$ — Gaussian noise
- $a^{\eta,\omega}$ — the noised action
- $f_\theta$ — the velocity output by the action expert
- $\alpha_\eta$ — loss weight per noise level
- On the basis of the correspondence between flow matching and diffusion, and the result that the diffusion loss can be interpreted as an ELBO, the paper treats this sum as an approximate lower bound on the full action log-likelihood

In the end, the flow's log-likelihood is never computed. Then how was Wall 1 overcome?

> ### 💡 The moment the RL objective becomes a conditional SFT loss, Wall 1 disappears
>
> RECAP's policy extraction is "conditional maximum likelihood." For continuous actions, **the flow matching loss already used in pretraining can be used as is**. There is no need to evaluate the log-likelihood at arbitrary actions, to compute probability ratios, or to impose a trust region.
>
> | | PPO (the paper's baseline) | RECAP |
> |---|---|---|
> | What it requires of the policy | Probability ratio of current/reference policy | Conditional generation loss |
> | Handling flow | Log-likelihood approximated with a 1-step diffusion ELBO (FPO-style) | Standard flow matching loss |
> | Stabilization | SPO-style trust region needed | Not needed |
> | Data | Essentially on-policy | All accumulated data |
>
> Then where did the regularization go? The $\hat\pi\propto\pi_{\text{ref}}\cdot(\text{ratio})^{\beta}$ of Section 3.3 is a **reweighting** of the data distribution, so the support of $\hat\pi$ cannot go outside $\pi_{\text{ref}}$. Behavior regularization is built **into the structure of the target distribution**, not added as a penalty term in the loss.
>
> And thanks to this design, pretraining and post-training use **exactly the same loss and infrastructure**. This is why the claim "put reward into every stage" is possible.

### 3.5 Data collection — a division of labor between human corrections and autonomous experience

The post-training flow goes like this.

- **⓵ Initial SFT** — fine-tune on task demonstrations with $I_t=\text{True}$ fixed. The paper says this gave slightly better results. The result of this stage is the iteration-0 policy $\pi^0_\ell$
- **⓶ Collection** — some episodes run fully autonomously, and in others a teleoperator watches and intervenes when needed (**human-gated DAgger**). Regardless of whether there was a correction, the whole episode (autonomous segments + correction segments) goes into the data
- **⓷ Retraining** — with all the accumulated data, first retrain the value function, then retrain the policy with the updated $I_t$

But if a person corrects in real time, isn't DAgger alone enough? Why are a value function and RL needed at all?

> ### 💡 Corrections handle big mistakes and exploration; RL handles fine quality
>
> The paper holds that corrections alone cannot fix every problem. An intervention during autonomous execution is an event that breaks the flow, and even a skilled operator cannot keep the quality of corrections consistent or improve subtle aspects such as overall speed. So corrections serve to fix big mistakes and get past exploration walls, which differs from the "optimal supervision" that DAgger theory assumes.
>
> | | Human corrections (human-gated DAgger) | Autonomous experience + value function |
> |---|---|---|
> | What it fixes well | Fatal mistakes, situations where exploration is stuck | Speed, smoothness of motion, fine quality |
> | Limits | Breaks the flow, inconsistent quality, cannot make fine improvements | Cannot find solutions the policy has never tried |
> | Handling in RECAP | Forced to $I_t=\text{True}$ | Conditioned on advantages assigned by the value function |
>
> The T-shirt and shorts task shows this division of labor best. As the policy's speed approached that of expert data collectors, providing corrections itself became hard, so this task was improved with autonomous data alone, without corrections. **The moment the policy reaches human level, the value of human corrections drops, and beyond that only RL can go.**

---

## 4. Why it works

Gathering back the three walls named in Section 1:

| Wall | RECAP's device | Why it breaks through |
|---|---|---|
| **Wall 1** Huge flow VLA | Conditional SFT with the advantage put in as a text condition | No need for log-likelihood, probability ratios or trust regions. Uses the same loss and infrastructure as pretraining as is (Sections 3.3, 3.4) |
| **Wall 2** Heterogeneous data | MC value function of the mixture policy + conditional training on all data + corrections forced positive | Does not ask "who made this data," only "was it better than that data's average." Uses failure data as a control group too (Sections 3.2, 3.3) |
| **Wall 3** Real-world reward | Success label + $-1$ per step → time-to-go value function, multi-task pretraining | Labeling cost is 1 bit per episode, the learning signal comes every step. The value function captures mistakes and speed of progress (Sections 3.1, 3.2) |

Three more things support it.

**⓵ Theoretical backing** — a single extraction is guaranteed to improve over the data-mixture policy (Section 2.3), and iterating raises that mixture policy itself (Section 3.2). The whole algorithm has the structure of approximate policy iteration.

**⓶ RL pretraining** — even tens of thousands of hours of human demonstrations vary in quality. Assign advantages to each demonstration at the pretraining stage, and the model enters post-training having already learned, across many tasks, "the difference between good and bad execution." The paper reports that a model combining π*₀.₆, pretrained with offline RL, with high-quality SFT is better than standard SFT without offline RL pretraining, and makes a good starting point for subsequent RL on robot data.

**⓷ The source of the throughput improvement** — demonstration SFT cannot exceed demonstration speed. The 50-step advantage, by contrast, picks as positive "the segments that advanced faster than the data average," so average speed rises with each iteration. Narrowing the positive fraction to 10% on the T-shirt task was a choice that deliberately amplified this effect.

So what is really new in RECAP?

> ### 📌 RECAP's contribution is not its components but their combination and scale
>
> As the paper itself acknowledges, each of RECAP's components — the MC value function, advantage conditioning, CFG-style improvement, human-gated DAgger — exists in prior work. What is new is the **combination** and the **scale** at which it was shown to work.
>
> - It runs **one loss** from pretraining all the way through real-world iteration
> - It does RL on an **entire** flow-based large VLA **end-to-end**
> - It absorbs **heterogeneous data** — human demonstrations, autonomous rollouts, human corrections — **in one recipe**
> - As a result, it showed for the first time that a general RL recipe can substantially raise a VLA's robustness and **throughput** from real-world deployment experience alone

---

## 5. Experiments

### 5.1 Setup and tasks

The iterative improvement experiments are run on a static bimanual system. Two 6-DoF arms with parallel grippers are controlled at 50Hz joint positions, and the observations are joint and gripper states plus images from 3 cameras (1 base camera between the arms, 2 wrist cameras). The pretraining data comes from many kinds of robots.

| Task | Content | Success criterion | Limit |
|---|---|---|---|
| Laundry (T-shirts, shorts) | Take out of the basket, spread, fold, and stack at the top right | Fold and stack one item | 200 s |
| Laundry (diverse) | Trained on 11 types of clothing. Metrics measured only on the hardest, button-up shirts | Fold and put on the stack | 500 s |
| Laundry (failure removal) | One orange T-shirt in a fixed initial arrangement | Fold precisely with the collar facing up (strict) | 200 s |
| Cafe (double espresso) | Portafilter → grinding → tamping → mounting → cup → extraction → serving | Complete all steps without a fatal mistake | 200 s |
| Box assembly | Flat cardboard → assembly → label → crate loading (real factory) | From flat to loaded | 600 s |

There are two metrics.

- **Throughput** — successes per hour. Bundles success rate and speed into one practical number
- **Success rate** — the fraction of success labels made by evaluators judging several quality criteria and aggregating them

### 5.2 Models compared

The names are similar and easy to confuse, so let's sort them out first.

| Name | Pretraining | Task adaptation | Role |
|---|---|---|---|
| π₀.₅ | Supervised | — | Previous generation, no RL |
| π₀.₆ | Supervised | — | No advantage input |
| π*₀.₆ (RL pretrained) | Offline RL | — | RECAP at the pretraining stage |
| π*₀.₆ offline RL + SFT | Offline RL | SFT on task demos ($I$=True fixed) | Iteration 0, i.e., the data-collection policy |
| **π*₀.₆ (ours)** | Offline RL | + RECAP iterations with autonomous experience and corrections | Final model |
| AWR | Supervised (π₀.₆) | Same robot data, advantage-weighted regression | Extraction-method comparison |
| PPO | Supervised (π₀.₆) | Same robot data, DPPO/FPO-style + SPO constraint | Extraction-method comparison (foreshadowed in Section 2.2) |

### 5.3 Results

The exact bar values are only in the figures (Figs. 7–12), and the main text gives almost no numbers. Below are the results the main text reports in sentences.

| Question | Setup | Reported result |
|---|---|---|
| **Q1** Overall effect | 5 models × all tasks (Figs. 7, 8) | More than 2× throughput on diverse laundry and espresso, failure rate roughly halved. Success rate above 90% on every task except diverse laundry |
| **Q2** Effect of iteration | T-shirt: autonomous data only, 300 trajectories per iteration (4 robots) / Box: 600 autonomous + 360 intervention per iteration (Figs. 9, 10) | T-shirt: +50% throughput in 2 iterations, success rate past 90% in iteration 1 / Box: throughput dropped in iteration 1, then doubled in iteration 2, folding and labeling success rate about 90% |
| **Q3** Extraction-method comparison | T-shirt, same data (Fig. 11) | Neither AWR nor PPO gets much past the offline RL + SFT model. PPO was stabilized with a small trust region of 0.01 but performs poorly; AWR has a decent success rate but is slow, so its throughput is low |
| **Q4** Failure removal | Strict criterion (collar up), 600 trajectories per iteration × 2 (Fig. 12) | 97% success rate, fast speed |

**As foreshadowed in Section 3.1, the improvement stands out more in throughput than in success rate.** The evidence comes from three places.

- On the easy T-shirt task, success rate is already near the maximum at the SFT stage, yet throughput still rises substantially in the final model
- In the T-shirt iteration experiment, after iteration 1 pushes success rate above 90%, iteration 2 mainly raises throughput
- AWR has a decent success rate but low throughput. The difference between extraction methods shows in drawing out speed improvements

It is also worth pinning down **relative to which baseline** Q1's "2×" is. The main text states explicitly that this gain is the difference from the offline RL + SFT model to the final model, that is, **the effect of adding experience data collected on the robot**. Since it is not relative to π₀.₆, it measures the pure effect of "RL from experience," not mixed with the effect of pretraining improvements. If anything, it is a conservative comparison.

Box assembly failures are mostly timeouts, and failures remain especially at the final crate-loading step.

Outside the quantitative evaluation, it also shows demonstrations at a practical-use level. It made espresso for 13 hours straight, folded never-before-seen laundry in a new home for more than 2 hours without interruption, and assembled packaging boxes in a real factory.

### 5.4 Data scale

The data scale revealed in Appendix F is mostly **1–2 iterations, a few hundred episodes per task**.

| Task | Iterations | Autonomous episodes | Correction episodes |
|---|---|---|---|
| T-shirts, shorts | 2 | 300 per iteration | 0 |
| Diverse laundry | Not stated | 450 | 287 |
| Failure removal | 2 | about 1000 | 280 + 378 |
| Box assembly | 2 | 600 per iteration | 360 per iteration |
| Cafe | 1 | 414 | 429 |

Checked against the main text's claims, this table disagrees in one place.

> ### ⚠️ Fact check — the claim of removing failures "without intervention data"
>
> In the conclusion to Q4, the main text says a specific failure mode was removed purely by RL, without intervention data or additional demonstrations. But Appendix F writes that in the failure-removal experiment both autonomous data and correction data **were collected**, and even gives the number of correction episodes (280 + 378), as in the table above. It is possible that the correction data was collected but not used for training, but the paper does not state this. "Failure removal by pure RL" should be read as a claim that needs confirmation.
>
> In the same appendix, the 600 per iteration for Box is described as "autonomous trials" in the main text and as "demonstrations" in the appendix (minor).

There is one more setup point to note about the Q3 comparison.

> ### ⚠️ Fact check — the extraction-method comparison (Q3) starts from different points
>
> AWR and PPO start from **π₀.₆ (supervised pretraining)**, while the final RECAP model starts from **π*₀.₆ (RL pretraining)**. So the gap in Fig. 11 mixes the difference in extraction method with the difference in pretraining. The comparison was also made only on one task, the easiest, T-shirts. The paper fairly states that "the baselines are, if anything, advantaged because they use the better data RECAP collected," but does not mention this confound.

---

## 6. Positioning — among neighboring work

RECAP sits at the intersection of several lineages.

| Lineage | Representative work | Relation to RECAP |
|---|---|---|
| Interactive imitation learning | DAgger, HG-DAgger, BC-Z, RaC | Adopts human-gated DAgger but combines it with autonomous experience and RL |
| Real-world robot RL | QT-Opt, SERL, MT-Opt, RoboCat | Extended to large VLAs and long-horizon, precise manipulation |
| VLA + PPO line | VLA-RL, SimpleVLA-RL, πRL | Criticized as hard to scale efficiently to the real world |
| RL on top of a VLA | Residual policies, ConRFT, PA-RL, V-GPS, DSRL, RLDG | Discrete or simple Gaussian actions. RECAP trains the entire flow VLA end-to-end |
| Value-function-based end-to-end VLA RL | CO-RFT, GRAPE, VLAC, Self-Improving EFM | Discrete-action models, on-policy policy gradients, relatively simple tasks |
| Reward- or value-conditioned policies | UDRL, RCP, Decision Transformer, RvS, CFGRL | Direct methodological ancestors |

Unpacking the value-function-based end-to-end line a little more: CO-RFT applies Cal-QL to grasping demonstrations but has no online improvement stage. GRAPE applies DPO with human preferences over VLA rollouts. VLAC and Self-Improving Embodied Foundation Models use a value function of the **time remaining to completion** form, like RECAP, but use that value to run PPO and REINFORCE respectively. Starting from the same time-to-go value function, the fork is **whether to go to policy gradients or to conditioning**.

The closest cousin is CFGRL, the method's direct root. Then what did RECAP take from CFGRL, and what did it change?

> ### 🔗 CFGRL (Frans et al., 2025) — the theory as is, the sharpness knob moved to training time
>
> RECAP's improvement-probability theorem (Section 2.3) and the CFG correspondence (Section 3.3) come from CFGRL. The difference is what the principle was applied to, and how.
>
> | | CFGRL | RECAP |
> |---|---|---|
> | Improvement condition | $\epsilon=0$ (an improvement if the advantage is positive) | Task-specific quantile threshold $\epsilon_\ell$ |
> | Controlling sharpness | $\beta$ at inference (guidance) | $\epsilon_\ell$ at training, $\beta=1$ by default at inference |
> | Target model | Diffusion policy | Large VLA mixing autoregressive + flow |
> | Scope | Guidance as a policy-improvement operator | From pretraining to real-world iteration, with human corrections integrated |
>
> The reasons RECAP moved the knob from inference time to training time are the two in Section 3.3 — aggressive motion, and no effect on the autoregressive part. The structural fact that a large VLA has both autoregressive and flow outputs changed the design.

For reference, [PLD](/notes/pld-self-improving-vla-en/), covered on this blog, is cited in this paper's related work as part of the line that "trains a residual policy on top of a VLA with RL and distills it back."

---

## 7. Limitations

What the paper states itself, together with what is worth noting from reading it.

**Limitations the paper acknowledges**

- **It is not fully autonomous** — reward labels, interventions and episode resets are all handled by people. The paper anticipates that VLAs may open new paths to automation, such as inferring reset procedures with a high-level policy
- **Exploration is naive** — it is greedy exploration relying on the policy's stochasticity and human interventions. This is fine when the initial imitation policy is already reasonable, but there is a lot of room for improvement
- **It is iterated offline** — it iterates by collecting data in batches and retraining; it is not concurrent online RL in which the policy and value function update in real time. The paper says this choice was for convenience
- **MC value estimation** — less optimal than off-policy Q estimators; extending it is left as future work

**Further points to note**

- **The assumption that corrections are always positive** — while the paper itself says correction quality is inconsistent, it unconditionally forces corrective actions to $I_t=\text{True}$. There is tension between the two
- **Per-task tuning of the threshold** — $\epsilon_\ell$ is set differently per task and stage, at 30%, 40% and 10%. This somewhat weakens the claim of a "general recipe"
- **What is evaluated are per-task specialists** — the method summary says the final generalist is trained from scratch, but the experimental results are for models fine-tuned per task (inferred from context). The iteration experiments are on a single platform, and diverse laundry was measured only on one item type, button-up shirts
- **The reward is human judgment** — evaluator labels aggregating several quality criteria are the source of reward. The introduction names "ambiguous rewards" as a hard problem, but the effect of label noise on the results was not measured
- **How numbers are reported** — the key results are presented only as bar charts, so when citing them one has to rely on the numbers the main text fixes in sentences (2×, half, 90%, 97%, etc.)

---

## 8. Closing — what this paper suggests

RECAP's real contribution is not a new RL algorithm. It is that it found **one form that reduces RL to conditional SFT** and ran the same loss all the way from pretraining to real-world deployment. As a result, the machinery that would otherwise have to be built separately for RL (probability ratios, trust regions, a dedicated critic update loop) almost disappears, and the large-scale SFT pipeline that is already running becomes the RL pipeline as is.

And this paper reads unusually well to someone with an LLM or diffusion background.

- **CFG = a policy-improvement operator.** Raising the guidance scale in diffusion to increase prompt fidelity and raising $\beta$ here to increase action optimality are the same operation. Even the side effects correspond: oversaturation artifacts and aggressive motion.
- **RL with a single text token.** Putting the advantage in as an `Advantage: positive` token follows the same design philosophy as reward-token conditioning in LLMs (Quark, SteerLM, the Decision Transformer line). **Infrastructure reuse** is a bigger advantage than the algorithm.
- **Quality-aware pretraining.** Assigning advantages to tens of thousands of hours of demonstrations for pretraining resembles the idea of attaching quality tags to LLM pretraining data. It explicitly tells the model that "not all demonstrations are of the same quality."
- **Throughput as a first-class metric.** The time-to-go value function bundles "did it succeed" and "how fast" into one scalar. What matters in real deployment is throughput per hour, so a practical strength of this paper is that its reward design and evaluation metric are aligned with the deployment goal.
- **The value function = a runtime progress monitor.** A value function that drops the moment a mistake happens could also serve, in agentic systems, as a trigger for detecting failure and requesting intervention.

Just as people reach mastery through practice, a VLA can break through the demonstrator's ceiling through deployment experience. RECAP implemented that practice in the simplest form — writing "positive" next to the actions that went well.

---

## Appendix — glossary

| Term | Definition |
|---|---|
| **RECAP** | RL with Experience and Corrections via Advantage-conditioned Policies. The repetition of data collection → value function training → advantage-conditioned training |
| **Policy extraction** | The step of obtaining an improved policy from a learned value function. There are three routes: policy gradient, weighted regression and conditioning |
| **Advantage conditioning** | An extraction method that turns the advantage into a binary indicator, puts it into the policy's input, and does conditional supervised learning on all the data |
| **$\pi_{\text{ref}}$** | The mixture of policies that produced the data. Human demonstrators and past policies are mixed in |
| **Improvement indicator $I_t$** | True if the advantage exceeds the task-specific threshold $\epsilon_\ell$. Input as text |
| **$\epsilon_\ell$** | The improvement threshold. In practice a quantile that sets the positive fraction (30%, 40%, 10%, etc.) |
| **$\beta$** | Guidance strength at inference. 1 means conditional sampling; above 1 means CFG |
| **Time-to-go value function** | A value function that predicts the negative of the number of steps remaining until success. Failure is a large negative |
| **Human-gated DAgger** | A correction scheme in which a person judges the risk and intervenes in autonomous execution |
| **Knowledge Insulation (KI)** | A training method that trains discrete tokens and flow actions together while blocking the action expert's gradient from flowing into the backbone |
| **Throughput** | Successes per hour. A metric that reflects success rate and speed together |

**Original** — [arXiv:2511.14759](https://arxiv.org/abs/2511.14759) · **Project page** — [pi.website/blog/pistar06](https://pi.website/blog/pistar06)
