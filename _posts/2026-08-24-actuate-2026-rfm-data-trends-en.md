---
layout: paper
lang: en
ref: actuate-2026-rfm-data-trends
kind: tech-review
title: "Actuate 2026: RFM and Data Trends"
date: 2026-08-24 12:00:00 -0700
venue: "Actuate 2026 · San Francisco"
tags: [Robot-Foundation-Model, Data, Scaling-Law, World-Model, Conference, Tech-Review]
summary: "Notes from Actuate 2026 — the standardization of data collection and storage formats, the shared recipe of large-scale pre-training plus a small amount of post-training, and the first architectural moves outside the VLA."
---

> **In short** — Notes from Actuate 2026, the robotics developer conference held in San Francisco in August 2026. The major RFM companies are converging along three lines: **standardizing how data is collected and stored**, **a shared training recipe of "large-scale pre-training + a small amount of post-training"**, and **extending the architecture beyond the VLA**.

---

## Introduction: What Actuate 2026 Is

Actuate 2026 was a "developer conference for the people who build robots," hosted by Foxglove — a company that builds robotics data visualization and management tooling — and held in San Francisco over two days, 18–19 August 2026.

Rather than academic paper presentations or product marketing, it is a venue where the engineers and technical leaders actually deploying robots in the field share what has moved.

Sponsors included NVIDIA alongside data companies such as Avala, Encord, Scale and Lightwheel, and robotics companies such as Foundry Robotics, Flexion and Genesis. With over 1,000 attendees, it has become a major networking event in the robotics ecosystem.

Notably, major RFM companies including Physical Intelligence, Generalist AI, Genesis AI and Dyna Robotics took part as speakers and presented technical material they were disclosing for the first time at this event. Panel sessions took up the field's live disputes — the "data war" (human data vs. UMI vs. teleop) and RFM architecture (VLA vs. WAM).

> ### 💡 Major RFM companies in attendance
>
> | **Company** | **Founded / HQ** | **Founders** | **Funding** | **Profile** |
> | --- | --- | --- | --- | --- |
> | **Physical Intelligence** | 2024 / San Francisco, Silicon Valley | • Karol Hausman (CEO, PhD USC, ex-Google DeepMind)<br>• Sergey Levine (Chief Scientist, professor at UC Berkeley, PhD Stanford)<br>• Chelsea Finn (Research Lead, professor at Stanford, PhD at UC Berkeley advised by Pieter Abbeel and Sergey Levine)<br>• and seven others including Brian Ichter and Lachy Groom | ~$1.07B raised to date<br>• Seed $70M (Mar '24)<br>• Series A $400M (Nov '24, valuation $2.4B)<br>• Series B $600M (Nov '25, valuation $5.6B)<br>• New round under discussion (reported $11B+, unconfirmed) | Building a general-purpose foundation model (the π series) applicable to any robot (cross-embodiment). Built on UC Berkeley and Stanford researchers, it leads current RFM research trends including scaling, reinforcement learning, and combination with world models |
> | **Generalist AI** | 2024 / San Mateo, Silicon Valley | • Pete Florence (CEO, PhD at MIT advised by Russ Tedrake, previously led PaLM-E and RT-2 at Google DeepMind)<br>• Andy Zeng (Chief Scientist, PhD Princeton, ex-DeepMind)<br>• Andrew Barry (CTO, developed Atlas, Spot and Stretch at Boston Dynamics) | ~$500M raised to date<br>• Series A $140M (valuation $440M)<br>• Series B $400M (Jun '26, valuation $2B)<br>• New round under discussion (reported $3B, unconfirmed) | Building a general-purpose foundation model that is not tied to specific hardware. Demonstrated robotics' first scaling law with GEN-0/1 and reached one-shot learning in GEN-1.5 purely by expanding pre-training scale |
> | **Dyna Robotics** | 2024 / Redwood City, Silicon Valley | • Lindon Gao (CEO, serial founder who sold Caper AI for $350M)<br>• York Yang (CTO, serial founder who sold Caper AI for $350M)<br>• Jason Ma (Chief Scientist, PhD UPenn, ex-DeepMind) | ~$143.5M raised to date<br>• Seed $23.5M (Mar '25)<br>• Series A $120M (Sep '25, valuation $600M+, with LG Technology Ventures participating) | Building a general-purpose manipulation model deployable in commercial settings. Demonstrated a scaling law at the scale of one million hours of human video along with zero-shot transfer, and built the large-scale training infrastructure that supports it |
> | **Sunday** | 2024 / Mountain View, Silicon Valley | • Tony Zhao (CEO, PhD at Stanford advised by Chelsea Finn, author of ALOHA and ACT)<br>• Cheng Chi (CTO, PhD at Stanford and Columbia advised by Shuran Song, author of Diffusion Policy and UMI) | ~$200M raised to date<br>• Series A $35M (Nov '25, valuation undisclosed)<br>• Series B $165M (Mar '26, valuation $1.15B) | Building a fully autonomous home robot. The researchers who personally produced Diffusion Policy, ACT, ALOHA and UMI — the work current RFMs rest on — develop the full stack from data collection through model and hardware |
> | **Genesis AI** | 2024 / Palo Alto, Silicon Valley · Paris, France | • Zhou Xian (CEO, PhD CMU)<br>• Théophile Gervet (PhD CMU, ex-Mistral AI) | ~$105M raised to date<br>• Seed $105M (Jul '25)<br>• New round under discussion (reported $3B, unconfirmed) | Aiming at large-scale deployment of general-purpose robots. Uses its own simulation engine as a reinforcement learning environment and designs a robot hand with the same form as a human hand, integrating data, model and hardware into a full stack |
> | **1X Technologies** | 2014 / Palo Alto, Silicon Valley | • Bernt Øivind Børnich (CEO) | ~$126M raised to date<br>• Series A2 $23.5M (Mar '23, valuation undisclosed)<br>• Series B $100M (Jan '24, valuation undisclosed)<br>• New round under discussion (reported $10B+, unconfirmed) | Aiming at mass deployment of the home humanoid NEO. Develops robot hardware and a world model in house while opening its data collection rigs, simulation tools and control APIs to outside developers, extending into a platform |
> | **Eka Robotics** | 2025 / Cambridge, Massachusetts | • Pulkit Agrawal (CEO, professor at MIT, PhD UC Berkeley)<br>• Tuomas Haarnoja (ex-Google DeepMind) | Funding undisclosed<br>• Came out of stealth Apr '26 | Building robots that are general and fast at the same time. Trains a VFA model — which treats force as the language of physical interaction — with simulation reinforcement learning to overcome the sim-to-real gap |
> | **Foundry Robotics** | 2025 / San Francisco, Silicon Valley | • Adarsh Kulkarni (CEO) | ~$24.5M raised to date<br>• Seed $5.5M (Jan '26)<br>• Seed $19M (Apr '26) | Aiming at an AI-driven redesign of manufacturing. Decomposes assembly processes into verifiable primitives for training and sells them as modular robot cells reconfigurable in software |
> | **Veeda AI** | 2026 / Toronto, Canada · Mountain View, Silicon Valley | • Sanja Fidler (CEO, professor at University of Toronto, PhD University of Ljubljana, former VP of AI Research at NVIDIA)<br>• Huan Ling (Chief Scientist, ex-NVIDIA)<br>• Zan Gojcic (CTO, ex-NVIDIA) | ~$90M+ raised to date<br>• Seed $90M+ (Aug '26) | Aiming to extend interactive learning for physical AI. Building an interactive loop that evaluates and improves a policy's success inside a simulation environment realized with a generative world model |

## Key Takeaways

**1. Data — collection methods diverge, but standardization for scaling is moving fast.**
To gather data without robots, the UMI gripper, glove and human egocentric approaches coexist, and each company builds its collection device in the same shape as its own robot end-effector to reduce the conversion burden. Data strategy and hardware strategy are therefore not separate decisions, and the collection interfaces (ALOHA, UMI, GELLO) and storage format (MCAP) have already become de facto standards.

**2. Training — "large-scale pre-training + a small amount of post-training" is the shared recipe, but performance is reinforcement learning's job.**
With the discovery of a scaling law in pre-training, Generalist AI performs a new task with one minute of data and a single training run, and Dyna Robotics with thirteen minutes of training — post-training requirements have shrunk by three orders of magnitude or more. Scaling, however, produces generalization and not deployment-grade performance (reliability + speed), and reinforcement learning is being adopted as the common way to close that gap.

**3. Architecture — the expansion beyond the VLA has begun.**
Physical Intelligence's π0.7 places a world model in front of the VLA, generating the next goal state as an image or video and injecting it as a prompt for the policy — a conditioning structure. Veeda AI places a world-model-based simulator behind the VLA to form a closed loop that evaluates and improves the policy. Google DeepMind places an agentic layer in front of the VLA to handle high-level reasoning, multi-robot collaboration and safety.

What everyone emphasized in common is that **this is now the stage for validation in the field rather than in the lab, and proof through partnerships looks like the next point of competition.**

---

## 1. The Race to Standardize for Data Scaling Is On

**An RFM's performance comes down to how much good data you can gather, and how good it is.**

From what the major RFM (Robot Foundation Model) companies presented at the conference, two things were clear: **(1) the ways of collecting data keep expanding, while the collection interfaces (hardware), storage formats and labeling schemes are standardizing, and (2) a general-purpose RFM training recipe is settling into shape.**

What stands out here is that the initiative in the AI industry is concentrating in the companies that own pre-trained models. Not because they hold the model weights, but **because they define the conditions on the data that determine model performance.**
Which data to gather and how much, what structure and standard to normalize it into, and in what mix and proportion to feed it into training — all of these judgments can only be made from a position where the training outcome can be observed. **A data specification does not exist independently of a model; it is derived backwards from running training and watching performance change.**

So **the idea of producing good data without ever having trained a model does not hold.** It amounts to collecting without any criterion for judging what good data is.

### 1-1. Data scaling: robot teleoperation → UMI gripper-based human video → human egocentric video

Robot learning data has long depended on robot teleoperation — a person remotely operating an actual robot while the session is recorded.

The limits of this are clear. Cheng Chi, co-founder of Sunday, explained that it takes one person per robot, so multiplying the number of task types by ten multiplies headcount and cost by ten as well; and each paper or project means installing robots and cameras anew and maintaining that arrangement, which is a heavy operational burden.

**Four collection methods are currently in use across the industry to gather large and varied data.**

| **Data type** | **How it is collected** | **What is recorded** | **What the action really is** | **Representative companies (pre-training)** |
| --- | --- | --- | --- | --- |
| **Robot teleoperation** | A person operates an actual robot with a controller | • Video: head, wrist<br>• Observations: joint action values | Measured, genuine actions (not predicted) | Physical Intelligence |
| **UMI gripper** | Holding a device shaped like a robot gripper | • Video: head, wrist<br>• Observations: IMU acceleration, gripper opening width | Predicted actions (gripper pose recovered from video and IMU → delta pose reconstructed as the action = retargeting) | Sunday, Generalist AI |
| **Glove** | Wearing a sensor-equipped glove | • Video: head, wrist<br>• Observations: finger joint angles, IMU acceleration, palm and finger tactile signals | Measured + predicted actions<br>• Finger joints measured by sensor<br>• Wrist action recovered as pose → reconstructed as the action (= retargeting) | Sunday, Genesis AI |
| **Human egocentric** | First-person video | • Video: head (and sometimes wrist) | Predicted actions (hand pose estimated → reconstructed as the action = retargeting) | Dyna, Genesis AI, Physical Intelligence |

Going down the table, collection scale grows while directly measured information shrinks.

Theophile Gervet, co-founder of Genesis AI, emphasized the two axes of collection scale and action accuracy, and made the case for the glove and five-finger hand the company developed itself.

Teleoperation sits at maximum action accuracy and minimum scale; human video sits at the opposite corner.

> ### 💡 Action retargeting
>
> **The three methods other than robot teleoperation all require converting human motion — holding a UMI gripper, wearing a glove, or the bare hand — into the robot's embodiment.** That conversion is **action retargeting**, and it involves the following.
>
> 1. **Scale normalization:** correcting for differences in arm length and height between collectors
> 2. **Joint mapping:** establishing the correspondence between human joints and robot joints
> 3. **IK (inverse kinematics):** computing how many degrees each joint must move to place the fingertip at a target position, while accounting for joint limits, collisions between the robot's own arms, and positions the arm cannot reach
>
> In practice these are often solved together as one piece of technology, and each company chooses either to reduce the conversion burden in hardware or to absorb it in software.
>
> 1. **Reducing it in hardware**
>     1. The UMI gripper, devised by Sunday co-founder Cheng Chi during his PhD, builds the handheld device in the same shape as the robot gripper, which lets scale normalization (1) and joint mapping (2) be skipped. This is possible precisely because what is recorded is a device, not a human body. (IK is still needed.)
>     2. Genesis AI takes the opposite route, designing the robot's hand to have the same form and contact physics as a human hand and collecting data with a glove that matches it, which lightens the burden of scale normalization (1) and joint mapping (2) — human finger joints map almost directly onto the robot's. (IK is still needed.)
> 2. **Absorbing it in software**
>     1. Sunday has since moved from UMI to the glove approach, building a Skill Capture Glove with the same geometry and sensor configuration as the Memo robot's hand. Even with the hands aligned, however, collectors differ in height and arm length so the whole body is not aligned, and this is handled through a conversion pipeline called Skill Transform.

> ### 💡 Five-finger hands and the glove approach
>
> Robot end-effectors are gradually expanding from two-finger grippers to five-finger hands.
>
> **Tying a garbage bag, screwing in a light bulb, twisting open a bottle cap — precise, delicate motions like these are hard to reach with two fingers, and Google DeepMind disclosed that it made them possible with Sharpa Hands, which have 22 degrees of freedom.** Human-level dexterity is still some way off, but the industry regards the rate of progress as fast.
>
> More fingers also means tactile and force information at the contact points. Think about how a person manipulates things: a glass or a raspberry is handled delicately, a heavy box is supported with force. **The glove approach adds palm and finger tactile sensors so contact force can be measured directly. Sunday and Genesis AI are the representative companies here, and both stated that they improved manipulation performance using contact-rich information.**
>
> Companies developing five-finger hands end up adopting the glove as their collection method. Both Sunday and Genesis AI, described above, record finger joint angles (the action) together with palm and finger force data via the glove.
>
> That said, a five-finger hand is not the only answer for every task. For screw-tightening, Foundry Robotics replaces the end-effector itself with a screwdriver rather than having a two-finger gripper pick one up, solving the problem in hardware so that the model can concentrate only on the insertion motion. This can be read as reducing complexity at the task design stage to maximize model performance between over-design and under-design.

### 1-2. Attempts at data standardization

#### 1-2-1. Collection interfaces — UMI, ALOHA, GELLO

Releasing collection hardware as open source so that other labs and companies can replicate it directly has become a de facto standard. Three examples stand out.

![ALOHA](/assets/img/notes/actuate-2026/aloha.jpg)
*ALOHA — low-cost bimanual teleoperation hardware (Tony Zhao et al., 2023)*

**ALOHA (A Low-cost Open-source Hardware system, 2023)** is **low-cost hardware for robot teleoperation**.

When a person moves the controller directly by hand, the robot arm follows identically, and both the commanded and actual values are recorded in the process.

It came out of research Sunday co-founder Tony Zhao presented during his PhD at Stanford under Chelsea Finn, and it made bimanual manipulation data collectable without expensive industrial robots.

Because it is open source, many robot hardware vendors have commercialized it. **Trossen Robotics in particular sells the ALOHA configuration fully assembled, and it is the route many RFM companies including Physical Intelligence have adopted.**

![GELLO](/assets/img/notes/actuate-2026/gello.jpg)
*GELLO — a low-cost controller built to match the robot arm you already own (Philipp Wu · Yide Shentu et al., 2024)*

**GELLO (General, Low-Cost, and Intuitive Teleoperation Framework, 2024)** is an approach that **fixes ALOHA's dependence on specific hardware**.

ALOHA specifies the controller and the working robot arm as one bundled set, so a designated model of arm has to be acquired. GELLO only requires that **a controller matching the arm you already own be built separately with 3D printing and cheap motors, so it applies to any robot.**

It is research XDOF co-founders Philipp Wu and Yide Shentu presented while at UC Berkeley, and the design and software are open source, so controllers can be built per robot model.

**XDOF's strategy is to secure large-scale general-purpose data on the back of research that standardizes how to set up teleoperation across a variety of robot arms.**

![UMI](/assets/img/notes/actuate-2026/umi.jpg)
*UMI — a handheld gripper that collects data without a robot (Cheng Chi et al., 2024)*

**UMI (Universal Manipulation Interface, 2024)** takes the robot out of the loop entirely. It is research Sunday co-founder Cheng Chi presented during his PhD at Stanford and Columbia under Shuran Song.

The work had two main goals: **establishing a standard sensor configuration for robot installation, and making robot data collectable without an expensive robot.**

Using a low-cost 3D-printed gripper, 1,400 data samples were collected around the Bay Area, and the resulting model demonstrated generalization that worked anywhere on the Stanford campus.

UMI scales well in that it applies as-is to any robot arm, regardless of model, as long as the end-effector is a two-finger gripper.
**Generalist AI and many other companies that prioritize data scaling have adopted UMI-family approaches.**

#### 1-2-2. Labeling — subtasks, subgoal images, metadata, and their use

**What information to attach to collected data** is also settling into a standard. Physical Intelligence disclosed the most concrete scheme in this area through π0.7.

The π0.7 model takes a detailed multimodal prompt made of four elements.

1. **Task instruction:** a natural-language description of the task to perform.
2. **Subtask instruction:** step-level instructions that convey the detailed stages a task instruction alone cannot express. A single instruction to make an espresso cannot distinguish individual motions such as inserting the portafilter, waiting for extraction and carrying the liquid; given step-by-step instructions, the model can recognize which phase it is in and generate behavior to match.
3. **Subgoal images:** presenting the future state to be reached one or two seconds ahead as an image. The point is that physical motion which is hard to describe precisely in language can be specified directly as a target scene. This is where it meets the world model.
4. **Metadata: the element that changed the perspective on data quality and speed.** Normally, using low-quality data for training stalls or degrades performance; the presentation reported that labeling data as low-quality in advance and feeding that label to the model alongside it actually helps performance.

#### 1-2-3. Storage format — Foxglove MCAP

On the question of what format to store collected data in, **MCAP — developed and open-sourced by Foxglove, the conference's host — has become the de facto standard.**

**MCAP is a container format for robotics time-series logs, storing video, joint values and tactile data arriving at different rates from multiple sensors in a single file, aligned on a time axis.**

Its distinguishing features are random access, which reads a specific moment directly, and the ability to tune the unit in which data is grouped (chunking) to the use case.

Actual use cases:

1. **Dyna Robotics:** moved wholesale from a combination of H5 and per-frame JPEG to MCAP. The reasons given were its wide use in autonomous driving and its support for random access and flexible chunking. They did note that they modified some features — compression, encoding, chunking — themselves to make video-action training more efficient.
2. **1X Technologies** — supports downloading collected data in the standard MCAP format from its developer platform, with external storage integration planned.

---

## 2. Improving Performance: Generalization vs. Performance

### 2-1. The RFM recipe for generalization: large-scale pre-training + a small amount of robot teleoperation post-training

For building a general-purpose RFM, the presentations from the major RFM companies point to a shared recipe: **"do large-scale pre-training from a scalable source, and use robot data only in small amounts at the post-training stage."**

| **Company · model** | Pre-training data | Post-training data | Notes |
| --- | --- | --- | --- |
| **Dyna Robotics<br>Dyna-2 (Aug 2026)** | **Human egocentric:** 1M hours (some with hand pose annotation) | **Robot teleoperation:** small (minutes to hours) | • **Presented a scaling law in which performance improves predictably as human egocentric video is scaled from 1K to 1M hours**<br>• Showed that transfer from human to robot is possible by learning object interactions even with no robot data included at all |
| **Generalist AI<br>GEN-1.5 (Aug 2026)** | **UMI gripper:** 500K hours | **UMI gripper:** small, roughly 5 minutes to 1 hour | • **Announced that one-shot, few-shot and zero-shot are achievable through scaling alone, using no robot data in pre-training at all**<br>• For deployment-grade performance, however, about an hour of robot data per task is needed |
| **Sunday<br>ACT-2 (Jul 2026)** | **Glove:** 500 environments, 10M videos | **Robot teleoperation:** can learn new behavior from as little as a single video | • **Demonstrated that scaling pre-training data narrows the performance gap on tasks, objects and environments unseen during training**<br>• Showed experimentally that at the same data scale, quality still splits performance sharply |
| **Genesis AI<br>GENE 26.5 (May 2026)** | **Human egocentric + glove:** 10M hours (the largest scale) | **Glove** (hours per task) + **simulation** (1M) | • **Disclosed a training recipe that carries the LLM structure of pre-training → post-training → reinforcement learning (RL) over to robotics, using human egocentric, glove and simulation data**<br>• Uses its own open-source simulation engine as the RL environment |
| **Physical Intelligence<br>π0.6 (Nov 2025)<br>π0.7 (Apr 2026)** | **Human egocentric + robot teleoperation + open-source robot data** | **Robot teleoperation** (small) + **live robot execution data** (RL) | • **The only one that pre-trains on a mixture of varied data, and found performance gains through multimodal prompting**<br>• In particular, demonstrated zero-shot generalization by learning through reinforcement learning on live robot execution data rather than in simulation |

Two things are visible in this table.

1. **Physical Intelligence is the only one using robot data at the pre-training stage.**
    1. Mixing several data sources increases total volume, but data of differing quality, speed and execution strategy arrives together, leaving the model unclear about what to learn from. **Physical Intelligence's adoption of multimodal prompting can be read as a clever idea for resolving that ambiguity.**
    2. The other four, by contrast — **Dyna Robotics, Generalist AI, Sunday and Genesis AI — are full-stack RFM companies that consolidate their data sources to one or two, and by building collection devices (UMI, glove) in the same shape as their robot end-effectors, they use hardware design itself as a means of controlling variables.**
2. **The amount of robot data used in post-training has shrunk dramatically.**
    1. Generalist AI reached 66.5% with one minute of data and a single training run, and 59% with no training at all, simply by feeding in a 3–12 second demonstration. Sunday taught a new behavior from a single robot teleoperation video, and Dyna Robotics performed bottle-cap opening after thirteen minutes of training.
    2. Against the hundreds of hours of robot teleoperation data once required to teach a single new task, this is a reduction of three orders of magnitude or more.
    3. This structure only holds, however, on the premise that increasing pre-training data actually and predictably improves performance. (The scaling law.)

#### 2-1-1. Generalist AI: from Pre-Gen through GEN-0, GEN-1 to GEN-1.5

Across four model releases over roughly a year, Generalist AI showed step by step that **the more pre-training is scaled, the less data and training a new task requires.**

1. **Pre-Gen (September 2025):** demonstrated in-context learning that replicated Lego tile arrangements. Combinatorially 99,000 configurations were possible, but the range the model could actually perform was limited to three-tile arrangements.
2. **GEN-0 (November 2025):** confirmed a predictable scaling law in which performance improves in a consistent pattern as data scale grows. This is the point at which the development direction was validated.
3. **GEN-1 (April 2026):** demonstrated success rates above 99% on specific tasks and early signs of improvisational intelligence. Per-task robot data requirements at this point are reported at around one hour.
4. **GEN-1.5 (19 August 2026):** the latest model, pre-trained continuously for more than eight months, bringing one-shot, few-shot and zero-shot capability together.

In particular, **GEN-1.5's one-shot in-context learning lets the robot perform a task immediately, with no separate training, once a single demonstration is placed in the model's context window.**

The demonstration data placed in context can come from a person's own hands or from wearing a UMI gripper, from robot teleoperation or from a previous rollout of the task, or from a simulation environment — all are usable.

At an average success rate of 59% across ten tasks it is not at a deployable level, but it is regarded as a meaningful result when considered against the technical progress that followed the discovery of chain-of-thought in ChatGPT.

> ### 💡 Generalist AI's emergent capability
>
> **Emergent capability** refers to a capability appearing on its own as scale grows, even though the model was never designed or trained to have it.
>
> In the LLM field, this describes the cases where reasoning or translation abilities that had not been there before were suddenly observed once model size passed a certain point.
>
> Generalist AI notes that few-shot training on a tiny amount of data — one to five minutes — changes less than 0.15% of the RFM's weights, and reads this as evidence that fine-tuning does not create new representations but finely rearranges knowledge the model already holds.
>
> **What is notable is that although the tasks used in evaluation were not in the pre-training data, the model was already in a state where it could perform tasks it had never learned, and fine-tuning did no more than draw that capability out.**
>
> That this capability appeared purely from scaling pre-training data, with no separate design, is what makes it an emergent capability.

#### 2-1-2. Dyna Robotics: a scaling law from 1M hours of human egocentric video alone

In the Dyna-2 presentation, Dyna Robotics answered quantitatively the question of whether a scaling law holds for robot performance when pre-training uses only human egocentric video and no robot data.

**Measuring performance while scaling human egocentric video from 1K to 1M hours, they reported a relationship in which performance improves in a consistent pattern as data grows — regarded as the first demonstration that human egocentric video improves robot performance.**

Dyna ran an experiment comparing three data types: human egocentric video with hand-annotated action labels only; that plus pure video with no action labels; and pure video alone.

The central finding is that **simply increasing video without action labels improves robot performance**, and they emphasized that what transfers from human egocentric video is not how to move but how objects respond.

How a person's arm moved cannot be used directly, since the body structure differs from the robot's — but how an object slides when pushed and how it deforms when gripped applies to the robot in exactly the same way. That is why video is worth learning from even without action labels.

> ### 💡 Scaling presupposes ML infrastructure
>
> **Dyna Robotics stated that building ML infrastructure capable of training at the 1M-hour scale was the core bottleneck, and repeatedly emphasized "iteration speed is king" in the presentation.**
>
> Without an environment where experiments can be repeated quickly, several times a day, the scaling signal itself cannot be detected.
>
> **At a 1M-hour dataset scale they analyzed that the time spent waiting for data to arrive (latency) was what limited the number of experiments they could run, so they moved the storage format to Foxglove's MCAP and optimized it by modifying features such as compression, encoding and chunking.**
>
> | Item | Before | After |
> | --- | --- | --- |
> | **Storage per camera-minute** | 80.3MB (JPEG) | 25.1MB (~68% reduction) |
> | **Read latency per sample** | 27.0ms | 9.4ms (~2.9× improvement) |
> | **Data collection throughput** | 14,000 episode-hr per week | 440,000 per week (~31×) |
> | **Time to first training batch** | ~48 hours | under 1 minute |

### 2-2. Reinforcement learning for deployment-grade performance

Everything covered so far says that scaling pre-training improves the ability to handle new tasks and environments.

At this conference, however, the objection was also raised that this direction alone **is not enough to deploy in the field**.

Pulkit Agrawal, co-founder of Eka Robotics, **defined performance as what a robot needs in order to be economically useful, and defined it as the combination of reliability and speed.** Unless both are satisfied at once, the technology will not be adopted in the market however impressive it is.

**The common solution each company has chosen at this point is reinforcement learning. Imitating demonstration data makes it hard to exceed the demonstrator, whereas reinforcement learning learns through repeated attempts and failures and can therefore surpass the level of the demonstration.**

**Physical Intelligence chose reinforcement learning in which the robot learns online while performing tasks in the real environment.**

1. When the robot gets stuck, a person intervenes by teleoperation to demonstrate the recovery and terminates the episode early, avoiding wasted time.
2. For each task, a separate model (a value function) that predicts the time remaining to success is trained, which efficiently distinguishes good behavior from bad across a range of tasks.

Applied to a pre-trained VLA model, this doubled throughput (speed) on a box assembly task at a chocolate factory, and achieved thirteen hours of continuous operation with a success rate above 90% on espresso preparation.

**Eka Robotics and Genesis AI, by contrast, chose to run reinforcement learning in simulation.**

The reasoning is that large-scale repeated attempts in the real environment cost too much time and money. Genesis AI presents more than a million simulation environments as its core strategy, though the industry continues to raise the sim-to-real gap as a problem.

---

## 3. The Architecture After the VLA: Interactive Loops and Agentic Systems

Everything so far has been about how to train the VLA model itself better.

Separately from that, this conference put forward two directions for what to attach outside the VLA.

**One places a world model that predicts the physical world into the VLA training pipeline as a simulator for evaluation; the other places an agentic layer responsible for high-level reasoning in front of the VLA model.**

### 3-1. Veeda AI: a world model used as an evaluation simulator

Veeda AI is a world model startup founded in July 2026. Co-founder and CEO Sanja Fidler is a professor at the University of Toronto who served as VP of AI Research at NVIDIA and led the NVIDIA Toronto lab, where she drove research on OmniDreams, a real-time interactive world model.

It is regarded as **the third world model startup to draw serious attention, after Yann LeCun's AMI and Fei-Fei Li's World Labs.** Where those two aim at general-purpose spatial intelligence, Veeda AI is the closest to the RFM space in that it states a clear purpose: improving the interactive loop in order to improve VLA performance.

Veeda AI's framing starts from the view that physical AI today leans too heavily on imitation learning.

Robot teleoperation has a person operate the robot to gather demonstration data and scales it to millions of hours — but the motivation here is that people do not learn by imitation alone; they attempt the same situation thousands of times and find the best behavior through interaction.

![Physical AI Interaction Loop](/assets/img/notes/actuate-2026/interaction-loop.jpg)
*A presentation slide photographed at Actuate 2026 — the Physical AI Interaction Loop*

**What Veeda AI is building is a closed interactive loop in which evaluation and improvement circulate: the VLA model generates an action, the world simulator predicts the result, judges whether the task succeeded, and that outcome feeds back into improving the VLA model.**

Their argument for why current simulation engines cannot play this role is that these depend on graphics rendering, human-written physics solvers and artist-made 3D assets; because they cannot capture the variety and scale of the real world, situations that were never collected and complex effects cannot be reproduced.

Veeda AI therefore announced that it will complete the evaluation-and-improvement cycle by building a world-model-based simulator — named the **Physics World Model** — which trains on the actions a robot takes and the sensor observations (state) and outputs the physical world's response (future sensor observations).

### 3-2. Google DeepMind: extending into an agentic system

**Google DeepMind starts from the view that ordinary robots struggle with new environments, objects and tasks because they are specialized to particular tasks, and proposes a structure that stacks a high-level reasoning layer on top of the VLA.**

Gemini Robotics 2 is not one model but an agentic system made of two.

1. **GR-ER 2 (orchestrator):** handles high-level reasoning — understanding the environment, interpreting natural-language commands, planning task steps and calling the tools it needs.
2. **GR 2 (the VLA model):** takes natural language and visual input and directly controls every joint from toe to fingertip. It supports an on-device version that needs no network connection, and can be fine-tuned for a variety of robot forms.

This can be read as **developing the split between planning (System 2) and control (System 1) that Figure AI and NVIDIA had argued for into an agentic system.** The key results they reported from it:

- **Multi-robot collaboration:** two robots each run the same model stack and reason individually, then coordinate with each other through the agentic layer. Because each robot judges independently and divides the roles, collaborative tasks admit more precise and more accurate judgment.
- **Memory:** GR-ER 2 (the orchestrator) remembers the state of a scene and makes use of it — so it can carry out an instruction to move objects and then return them to their initial state.
- **Safety (harness):** refuses dangerous commands based on allowed and forbidden behavior lists and safety rules, and combines a physical safety layer that stops the robot or runs a safe motion when it comes close enough to a human, so that the action it emits is safe. The associated benchmark has been open-sourced.

> ### ⚠️ The safety–reliability dilemma
>
> Google DeepMind and Dyna Robotics presented opposite responses to the situation where the robot's view is blocked.
>
> Google DeepMind presented stopping the task and alerting a person when the view is obstructed as a safety feature, whereas Dyna Robotics emphasized being able to continue the task even with the view partly blocked as evidence of reliability.
>
> Stopping is safe but leaves the task unfinished; continuing finishes the task but leaves the risk of an accident in an unforeseen situation.
>
> With real deployment environments in mind — homes, industrial sites — how a robot should handle a range of edge cases needs to be thought through in advance.
>
> **Since whether stopping or continuing counts as correct determines the labels on the training data and the evaluation criteria, this is a case showing that task design grounded in clear criteria has to come before data collection.**

---

## Closing

What this conference confirmed is that the major RFM companies are converging in similar directions across data, training and model structure.

**Data is moving toward sources that can be gathered without robots, while each company picks a collection device matched to its own robot hardware; training has settled into a two-stage structure of large-scale pre-training and a small amount of robot teleoperation post-training; and the architecture is expanding downward with a world model to form an evaluation loop and upward with an agentic layer to take on high-level reasoning.**

What the speakers emphasized in common, though, is that this is now the stage for validation in the field rather than in the lab.

Physical Intelligence is running real deployments with partners in warehouse logistics (Ultra Robotics), laundry folding (Weave Robotics) and excavator operation (Actor Labs), and Google DeepMind is working with Boston Dynamics, Agility Robotics and Apptronik to apply its own models to general-purpose humanoids.

Proof through partnerships looks like the next point of competition, and **the three stages of designing data collection, choosing a training method for deployable performance, and defining the evaluation criteria look like where collaboration will have its real contact points.**
