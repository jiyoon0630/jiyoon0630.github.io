---
layout: paper
lang: en
ref: model-hardware-standard
kind: tech-review
title: "Previewing the Model Hardware Standard (MHS)"
date: 2026-09-02 12:00:00 -0700
paper_date: 2026-08-27
venue: "Anthropic News · Research Preview"
tags: [MHS, Lab-Automation, AI-Agent, MCP, Physical-AI, Tech-Review]
authors: "Anthropic — MHS co-developed by Alek Kemeny (Anthropic) and Arco Bast (HHMI Janelia) · Partner case-study authors: Zihao Song (UW), Virginie Ruetten (HHMI Janelia), Sina Barazandeh, Arth Banka, Gün Kaynar, Jiayi Li, Peneeta Wojcik, Carl Kingsford, Jose Lugo-Martinez, Joshua Kangas (CMU), Cristian Ponce (Tetsuwan), the Genentech and QuEra teams"
affiliations: "Anthropic · HHMI Janelia Research Campus · Genentech · University of Washington (Baker/Pinglay Labs) · Carnegie Mellon University · QuEra Computing · Tetsuwan Scientific"
summary: "A driver standard that unifies disparate lab and manufacturing instruments as bundles of readable and writable state. The agent does not control the instrument step by step; it hardens what it learned by exploring into deterministic code, hands that off, and takes on judgment and exception handling. The six partner case studies read through four walls."
paper_url: "https://www.anthropic.com/news/model-hardware-standard-research-preview"
---

> **Core claim** — To let AI agents handle lab and manufacturing equipment, you first need **a driver standard that unifies disparate instruments into "bundles of readable and writable state."** On top of it, the agent does not control the instrument directly at every step. It **hardens what it learned by exploring into deterministic code and hands that to the instrument**, keeping judgment and exception handling for itself.

---

## Introduction

If MCP standardized the connection between LLMs and software tools, the Model Hardware Standard (MHS) aims to do the same for physical equipment. It is a shared specification that lets AI agents discover, understand and safely operate lab and manufacturing equipment such as microscopes, liquid handlers and robot arms. Anthropic opened it on August 27, 2026 as a research preview to some labs and advanced manufacturers, and plans to release it as open source after completing safety evaluations. It is not tied to any particular model (model-agnostic), and any agent harness can access it through a standard protocol such as MCP.

MHS began as a collaboration between Alek Kemeny of Anthropic's Beneficial Deployments team and Arco Bast, a postdoc at HHMI Janelia. The announcement consists of Anthropic's main text and case reports written directly by six partners (Genentech, UW Baker/Pinglay Labs, CMU, HHMI Janelia, QuEra, Tetsuwan). Since it is not a paper, there are no controlled comparison experiments, and the results should be read as case reports. The spec documents have not been released yet, so the architecture described in this piece is reconstructed from the announcement and the case studies.

This piece follows the announcement's argument, filling in concepts from lab automation and instrument control where they are needed, for a reader with an LLM and agent background.

---

## 1. The problem — instruments do not talk to each other

### 1.1 What the fragmentation looks like

The CMU case shows the problem most clearly. The four instruments needed for a serial-dilution dose-response experiment are scattered across three PCs, each controlled differently.

| Instrument | How it is controlled |
|---|---|
| Robot arm | no proper API. You must drop an XML job file into a folder watched by a scheduler |
| Liquid handler | supports only legacy Windows ActiveX/COM scripting; no modern SDK |
| Plate reader | no programming interface at all. GUI only |

HHMI Janelia's two-photon microscope rig is similar. The detectors run in MATLAB, the cameras in Python and the electrophysiology equipment in C#. The two stages need to know each other's position to know the sample's absolute position, but a value one program holds is unknown to another. So an extra DAQ board is sometimes attached just to relay signals physically, and starting a single experiment means launching seven programs in a fixed order. Get the order wrong and the whole session is lost.

As a result, integrating instruments takes weeks to months. Automation pays off only when a protocol is repeated an enormous number of times. The UW author's contrast is exact — a factory line runs one protocol ten thousand times, while a lab runs dozens of protocols a year, half of them new.

Put an AI agent on top of this and one more problem appears. Even with the instruments connected, there is no common way to share their data with the agent, and no common way for the agent to operate them safely.

### 1.2 So the question the announcement asks

> If a single standard is laid down on the instrument side, can AI agents take over integration and operation?

Answering this means crossing four walls. This whole piece is the process of dissecting those four walls and breaking through them.

**⛔ Wall 1 — Interface fragmentation** Each instrument has a different language, protocol and control method.

**⛔ Wall 2 — Knowledge that is not in the code** Information essential for operation, such as a robot arm's weight, exists only in paper manuals or people's tacit knowledge.

**⛔ Wall 3 — Safety** Commands move real objects. A failure can crash an objective lens or push laser power so high that it bleaches the sample.

**⛔ Wall 4 — Timescales** Instruments sometimes need to run faster than the agent's online reasoning.

---

## 2. Background — two tools for seeing the walls precisely

### 2.1 The arithmetic of integration cost (Wall 1)

Comparing a dedicated bridge for every pair of instruments with conforming each one to a standard just once gives this.

$$C_{\text{p2p}}(N) = \binom{N}{2} = \frac{N(N-1)}{2}\qquad\longrightarrow\qquad C_{\text{std}}(N) = N$$

- $N$ — number of instruments
- $C_{\text{p2p}}$ — worst-case integration cost when a bespoke bridge must be written for every pair of instruments
- $C_{\text{std}}$ — integration cost when each instrument is conformed to the standard only once

Including the agent side, the combinations of $M$ agent harnesses × $N$ instruments shrink to $M+N$. It is exactly the logic MCP applied to software tools, and the names pair up too (Model Context Protocol ↔ Model Hardware Standard). When the Janelia author writes that "integration cost no longer grows with the number of instruments," that is the effect of this equation.

### 2.2 The layers of timescale (Wall 4)

To understand Wall 4, first note that instrument control stacks loops of several speeds.

| Layer | Typical period | Example |
|---|---|---|
| Servo loop inside the instrument | µs–ms | the PID servo of QuEra's laser lock |
| Deterministic script | seconds | QuEra recovery script, 0.9–14 s |
| Agent judgment | several to tens of seconds per step (my estimate) | choosing parameters, interpreting curves |
| Experimental round | minutes–hours | one CMU dispense 4–5 min, UW PCR plate swap 90 min |

The only place an agent can sit inside the loop, reasoning at every step, is effectively the bottom layer. It is the same structure of problem as the "slow planning + fast control" hierarchy on the robotics side.

---

## 3. Method — one mechanism per wall

MHS's components map one to one onto the four walls.

```
  scientist / engineer  (natural-language goal)
          |
          v
  +--------------------------------+
  |  AI agent (any model/harness)  |   slow: plan, interpret, decide
  +--------------------------------+
       |           |            |
      MCP         CLI      code files (API)       <- wall 4
       |           |            |
       v           v            v
  +--------------------------------+
  |  MHS driver layer              |
  |   - primitives: read / write   |   <- wall 1
  |   - discovery                  |   <- wall 1
  |   - reference file (from tags) |   <- wall 2
  |   - enforced safety limits     |   <- wall 3
  +--------------------------------+
       |           |            |
     [arm]   [liquid handler] [plate reader] ...
   file-drop    COM/ActiveX      GUI only
```

### 3.1 Wall 1 → a standard driver: treat every instrument as "a bundle of state"

The MHS driver's command set is simple. The basic primitives are two: `read` (e.g., read a temperature) and `write` (e.g., set a temperature). Each instrument announces itself in a standard format (discovery), so instruments across a network and agents can find each other without a dedicated "translator" in between. CMU describes this as "a manifest of states and procedures." Bundling the case studies together, one instrument can be viewed as the following 3-tuple (this is my summary; the announcement does not present this form).

$$\mathcal{D}_i = (\mathcal{S}_i,\ \mathcal{P}_i,\ \mathcal{R}_i)$$

- $\mathcal{D}_i$ — the $i$-th instrument
- $\mathcal{S}_i$ — the set of state variables. E.g., temperature, "plate present at position 3," "well filled." Read with `read`; writable variables are set with `write`
- $\mathcal{P}_i$ — the set of procedures. E.g., aspirate, shake
- $\mathcal{R}_i$ — reference information (Section 3.2). What can be measured, what can be adjusted, safety limits

Janelia's implementation is the prototype of this model. Every instrument's variables, control values and sensor values go into **a single dictionary in shared memory**. Shared memory is a region of memory the OS lets multiple programs access simultaneously, so a process written in any language can attach to it and read and write. Analysis is composed as chains of "read from a slot (dictionary entry) → reusable transform → write to another slot." MHS itself grew out of this idea of Arco Bast's — putting the entire rig's state into a standard dictionary in shared memory.

But a question arises here. Can complex instruments such as robot arms and microscopes really be expressed with just two verbs, `read` and `write`?

> ### 💡 Fewer verbs, more nouns (state) — the same design as REST
>
> Just as REST expresses the web with a few verbs like GET/PUT and countless resources, MHS expresses instruments with read/write and a state dictionary. The instrument's complexity is absorbed not by the verbs but by **the state space and the list of procedures**.
>
> The gain from this choice is that the unit of code reuse changes. The Janelia author wrote viewers once per **data type** (image, time series, spectrum), not once per instrument. Transform code that computed the spectrum of a heartbeat measured by a camera was reused as is on neural activity data from a different instrument running in a different language.
>
> Anything can sit underneath the adapter.
>
> - **CMU** — for the folder-watching scheduler, merges the two result files it produces per submitted file into one (usually within a second). The COM methods were worked out from vendor documentation or by a Claude agent exploring the interface. The GUI-only reader is operated on screen, the way a person would.
> - **Tetsuwan** — with no vendor support, attached an SBC (single-board computer) to each instrument as a connector, and replaced the existing scheduler and driver stack entirely with MHS commands.

### 3.2 Wall 2 → from natural-language tags to a reference file

Some instrument characteristics cannot be known from the code alone. The announcement's example is a robot arm's weight. You need to know it to handle the arm safely, but until now this kind of information existed only in paper manuals, somewhere on a user's PC, or in people's tacit knowledge.

MHS drivers have **tags** for writing this information down in natural language. The user can write them directly, or an agent can interview the user to fill them in. From these tags, the driver automatically generates a **reference file**. It holds what the instrument measures, what can be adjusted, and which safety limits are enforced, and with this file alone an agent can handle an instrument it has never seen.

From an LLM background, this is the physical version of an MCP tool description (docstring). And it matters that this mechanism is **a channel for injecting physical knowledge as text**. Put the other way, physics not written here is unknown to the agent. The scene where this limitation actually shows up is the Genentech case in Section 5.3.

### 3.3 Wall 3 → safety: limits enforced by the driver

The reference file contains the safety limits to be enforced. The Janelia author writes that because MHS enforces limits at the instrument level, there is no need to worry that the agent will accidentally push laser power too high and bleach the sample. CMU deliberately created six fault conditions (no plate, rotated plate, reader in use, camera disconnected, instrument unresponsive, emergency stop active), and the system blocked all six **before any instrument moved**.

Then a question arises. If the agent is smart enough, couldn't safety judgment be left to the agent as well?

> ### 💡 Safety is not the agent's judgment but an invariant of the driver
>
> Gathering the case studies together, the safety mechanisms stack in three layers.
>
> | Layer | Nature | Examples |
> |---|---|---|
> | Hard (driver) | limits and preconditions that cannot be crossed whatever the agent commands | Janelia laser power limit, CMU's six faults blocked in advance, camera check of plate presence and orientation before transfer |
> | Soft (agent) | stops and asks for confirmation if something looks risky | QuEra: waiting for approval at the slightest sign of risk sometimes stalled experiments overnight |
> | Human approval | propose → approve → execute | Tetsuwan: proposes a recovery procedure over Slack and executes it once approved |
>
> The LLM-side analogy is the difference between writing "be careful" in a prompt and putting permissions on the tool server. It is a structure with deterministic guards on top of a stochastic policy; even when the agent is wrong, the hard layer prevents accidents.

### 3.4 Wall 4 → three control paths and "hardening into code"

An agent controls instruments through three paths — **MCP, CLI and code files (API)** — and combining them lets a single line of code orchestrate multiple instruments. Normally the agent receives the instruments' operating data, sequences the steps, monitors the results, and adjusts parameters when conditions change.

The key comes next. When a task is long or must run faster than the agent's online reasoning, the agent **chains the driver commands of several instruments into a code file**. The instruments then run on their own without the agent reasoning at each step.

The announcement's laser alignment example shows this flow well. Claude repeatedly adjusted the laser slightly and observed with a camera how the beam moved, working out cause and effect. It then bundled what it learned into a deterministic script, so that from then on alignment finishes with a single command.

```
  phase A: explore (agent in the loop)      phase B: exploit (agent out)
  -------------------------------------     ----------------------------
    act --> observe --> reason                deterministic script
     ^                     |                  runs at device speed
     +---------------------+                  single command
                |                                     ^
                +------- write code file -------------+
```

- **phase A** — the agent repeats operation, observation and reasoning inside the loop to work out how the instrument behaves
- **phase B** — what it worked out is hardened into code and runs at instrument speed without the agent

Then while the agent is not reasoning at every step, what is its role?

> ### 💡 The agent is not the controller but the one who writes the controller
>
> The QuEra case is the extreme of this arrangement (Section 5.2). Four Claude instances with different roles ran the loop. One formed hypotheses, one revised the recovery script, one ran it on the real laser and kept logs, and one read the logs and decided the next change. This cycle repeated hundreds of times overnight. The final output is **a human-readable deterministic script that runs in production without the agent**.
>
> Conversely, PID tuning keeps the agent in the loop, because the parameters drift with temperature and pressure. Whether to harden into code or keep the agent is decided by how much the target changes (its non-stationarity).
>
> Tetsuwan follows the same philosophy. The LLM only proposes, and a deterministic compiler made of meaning-preserving rewrite rules guarantees correctness.
>
> From the agentic-AI side, it is the same choice as code-as-action: writing and running a script instead of calling tools each time. On the robotics side, it belongs to the same lineage as code-as-policy, where an LLM generates control code. Two things are gained — the cost of exploration is paid only once while execution repeats cheaply, and the runtime artifact can be audited by people.

---

## 4. Why it works

**⓵ Integration cost does not scale with the number of instruments.** The arithmetic of Section 2.1 actually shows up. At Janelia, attaching a new camera took minutes, and the beam position the camera captured was fed straight back to a beam-steering mirror to improve alignment precision. Starting an experiment, formerly seven steps, became a single click on a dashboard.

**⓶ Effort compounds in one codebase.** Because the interface is uniform, analysis and visualization code can be reused across instruments. In the Janelia author's words, effort accumulates in a single codebase instead of being split per instrument.

**⓷ Slow judgment and fast execution are separated.** (my interpretation) The LLM's latency and nondeterminism do not spread into runtime reliability problems. The agent enters only at decision points, and deterministic code handles execution.

Then what exactly did the agent do well in these case studies? Was it because it understood the physics better than human experts?

> ### 📌 Where the agent won was not "smarter heuristics" but "doing the expensive measurement every time"
>
> In QuEra's PID tuning, experts tune by trusting an approximate metric, the RMS error the servo reports, because capturing an oscilloscope trace and Fourier-transforming it every time one of 12 parameters changes is not realistic for a person. Claude captured a trace at every adjustment and computed the full spectrum, repeated this hundreds of times, and caught a resonance the RMS heuristic missed. The recovery script, too, came from injecting disturbances over and over to find patterns.
>
> Conversely, it failed where **physical intuition** was needed, such as bubbles or evaporation (Sections 5.3 and 7). In short, the condition under which the agent wins is "an experimental loop that can be repeated quickly and safely + a quantitative metric," and what MHS supplies is precisely the first part.

---

## 5. Partner case studies

### 5.1 Overview

| Partner | Instruments / task | Baseline | Key result |
|---|---|---|---|
| QuEra | Ti:Sapphire laser lock recovery, PID tuning | a dedicated script built over months, expert tuning | recovery blind test 99.3%, 19 hours without unlocking |
| Genentech | liquid handler + robot arm + plate reader, BCA protein assay | expert transfers on the same plate | autonomous search for per-liquid flow rates. Autonomous recovery from tip-pickup and liquid-detection errors; bubbles needed human guidance |
| CMU | three PCs with incompatible interfaces, serial-dilution dose-response | weeks of vendor setup | integration in about 8 hours (including one autonomous rerun), experiment about 3× faster |
| Tetsuwan | ResearchOS + qPCR, compiler tuning | manufacturer datasheet | dispensing-precision predictions about 12% more accurate than the datasheet |
| HHMI Janelia | two-photon mesoscope, seven vendor programs | days to integrate a new instrument | new camera integrated in minutes, adaptive imaging |
| UW Baker/Pinglay | six instruments: remote dashboard, qPCR supervision, robot arm–liquid handler handoff | months to years for past integrations | under a week including driver writing, zero collisions in repeated tests |

### 5.2 QuEra — the most quantitative case

QuEra's neutral-atom quantum computer controls atoms with lasers, and each laser must hold its frequency to a precision of about one part in a trillion. This state is called a "lock." When everyday disturbances such as temperature, vibration or air pressure break the lock, quantum operations start to fail. Rarer kinds of disturbance still require an expert to watch several instruments at once and recover over 5–10 minutes, and that approach cannot hold up as the number of machines grows.

**Relock.** Previously, a team of laser engineers, software engineers, algorithm specialists and testers spent months building a dedicated recovery script. This script was a **linear sequence** that transcribed the expert's manual procedure step by step. With a linear sequence, if a disturbance comes in midway and undoes steps already finished, you have to start over — something that happens with as little as someone opening the lab door and changing the air pressure.

QuEra handed the same problem to Claude through MHS. It gave a goal (write a Python recovery script that runs standalone) and a definition of success (relock on the first attempt and hold for 30 seconds), and induced disturbances such as blocking the beam, cutting instrument power and forcing the frequency off target. On top of that, the four-role loop of Section 3.4 ran overnight.

Here is a simple model of why speed is success rate (my toy model).

$$P_{\text{success}}(T) \approx e^{-\lambda T}$$

- $P_{\text{success}}$ — the probability that one recovery attempt succeeds
- $T$ — the time one recovery attempt takes
- $\lambda$ — the rate of disturbances that undo steps already finished (assuming a Poisson process)

Plugging in the old script's 58% and 150 seconds gives $\lambda \approx 3.6\times10^{-3}\ \text{s}^{-1}$. At the same $\lambda$, a 6-second attempt comes to about 97.8%, the same size as the 96% obtained in the development run. In other words, much of the rise in success rate is explained by **finishing faster than disturbances arrive**. This model, however, assumes that every failure is due to a disturbance and that the disturbance environment is the same, which cannot be confirmed from the announcement.

The source of the speed is a structural change. Claude turned the linear sequence into a decision tree that branches on instrument readings. For example, if the frequency moved only slightly, most of the adjustment knobs have no effect, so it touches only one or two. It skips knobs a person would have to go through to check. In robot-control terms, **it turned an open-loop procedure into a state-feedback policy**.

**PID tuning.** The quality of the lock — how often it breaks — is set by 12 intertwined parameters inside the servo loop. Claude ran 363 experiments unattended over 16 hours, capturing a trace at every adjustment and checking the full spectrum. For independent validation, an expert retuned from scratch without seeing Claude's results, and the two parameter sets were compared with a phase noise analyzer, a specialized instrument that measures absolute phase noise.

| Task | Evaluation | Result |
|---|---|---|
| Relock | the existing dedicated script | 58%, about 150 s per attempt |
| Relock | development run (at the end of the overnight improvement loop) | 96%, about 6 s |
| Relock | blind test (the finished script alone, without the agent, 700 random induced disturbances) | **99.3%** (695/700). Simple disturbances 0.9–5.4 s, hardest 10–14 s |
| Relock | human expert | 5–10 min |
| PID | RMS error (existing tuning → Claude) | 15.7 mV → 1.55 mV |
| PID | phase noise (expert's new tuning vs Claude) | equivalent across the band. Only at a resonance near 220 kHz did the manual tuning have about 1000× more noise |
| PID | 19-hour hold | Claude: 0 unlocks / expert: about 1.6 per hour |

> ### ⚠️ Check what QuEra's headline numbers are compared against before quoting them
>
> **⓵ "99.3%" and "about 6 seconds" are different evaluations.** About 6 seconds is from the development run, whose success rate was 96%; 99.3% is from the blind test run without the agent. In the blind test, hard disturbances took 10–14 seconds. Bundling them into one line as "58% → 99.3%, 150 s → 6 s" is a misquotation.
>
> **⓶ The baseline for "about 10× quieter" is not the expert's new tuning.** The 10× is relative to **the existing tuning**, measured on **RMS error** itself, the very metric Claude was optimizing. Against the expert's fresh tuning, the two are equivalent except at the single 220 kHz resonance. The substantive evidence is on the side of "19 hours without unlocking vs 1.6 per hour."

### 5.3 Genentech — flow-rate optimization, and physics nobody wrote down

A BCA protein assay needs three instruments: a liquid handler for precise transfers, a robot arm to move labware, and a microplate reader to measure absorbance. When given only the standard protocol at first, Claude used the same flow rate for water and for viscous BSA solution, and bubbles formed in the BSA, throwing off the transfer volume.

So it was set to closed-loop optimization. Within a flow-rate range set by an expert, it made test transfers with dyed liquid, read the absorbance, and compared against the expert's transfers on the same plate.

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{j=1}^{n}\big(y_j^{\text{Claude}} - y_j^{\text{expert}}\big)^2}$$

- $y_j^{\text{Claude}}$, $y_j^{\text{expert}}$ — the measured values of Claude's transfer and the expert's transfer in the $j$-th well (the announcement does not give units)
- $n$ — the number of wells compared

The results were about 140 µL/s for water (RMSE 0.016) and about 10 µL/s for BSA (RMSE 0.181), which automation experts confirmed as reasonable. It is worth remembering as a limit on autonomy, though, that the search range itself was set by an expert. During execution, Claude recovered on its own from tip-pickup failures and liquid-detection errors.

Bubbles were different. The "physics not written in the reference" previewed in Section 3.2 shows up right here. When bubbles during mixing caused an error, Claude's default response was to retry in the same well with only the parameters changed, which stirred the liquid more and increased the bubbles. Claude could not work out on its own that the error code came from physical bubbles rather than a software bug. It was corrected only after a researcher told it that "the cause is bubbles; move to a clean well and reduce the number of mixes," and from then on it kept that context until the run ended. Genentech codified this lesson into a reusable liquid-handling **skill** so that reasonable default parameters are chosen for liquids with different properties. It is a loop in which, on failure, a person diagnoses and the diagnosis is fixed in text.

### 5.4 CMU — the worst combination of interfaces

With the three-PC setup seen in Section 1.1, going from raw instrument state to a finished dilution curve took about 8 hours (including one autonomous rerun). Left to a vendor, this usually takes weeks. The agent was Claude Opus 4.8. One dispense takes 4–5 minutes, and if the dispensing error exceeds 5% the curve is unusable.

The workflow itself is the same as before MHS; only the decisions are handed to the agent. The liquid handler makes the dilution series, the camera checks the plate's presence and orientation, only then does the robot arm move the plate to the reader, the reader measures, and the agent looks at the curve and decides whether to adjust the concentration range and run again or accept the result.

In the first run, the signal saturated at the top concentration of 200 µg/mL, and the fit came out at $R^2 < 0.9$. The agent decided on its own to discard this plate, lower the top concentration to 100 µg/mL, and run again on a new plate. The second run gave $R^2 > 0.98$, with no human input throughout. However, a colorimetric dye was used as a stand-in for actual drug candidates. CMU said it plans to release its per-instrument drivers.

### 5.5 Tetsuwan — hardware-independent protocols and compiler tuning

Tetsuwan's ResearchOS turns natural-language protocols into DSL programs through conversation with Claude, and a dedicated compiler lowers them to automation code. With MHS attached, protocols become independent of any particular instrument. If a protocol specifies only centrifugal force, ResearchOS finds a compatible centrifuge on the network and learns its driver interface, and Claude converts to the units that instrument accepts. For an instrument that accepts only rotation speed, the conversion is as follows.

$$\text{rpm} = \sqrt{\frac{\text{RCF}}{1.118\times10^{-5}\cdot r}}$$

- $\text{RCF}$ — relative centrifugal force (in × g)
- $r$ — rotor radius (cm)
- The original writes this conversion as "divide the force by the radius" and the centrifugal force as "15,000 × rpm"; these are slips for the equation above and "× g," respectively

Error handling was extended too. A camera photographs every transfer, and when a CV algorithm detects bubbles and foam, Claude looks at the list of connected instruments, proposes over Slack the procedure "spin the source plate at low speed to remove the bubbles," and executes it once approved.

On the compiler side, closed-loop optimization was run. Across 9,143 dispenses, 300 transfer types and 1,508 measurement conditions, dispensing accuracy and precision were measured with tracer dye, and plate reader data received through MHS was used to fix the compiler's **precision prediction model**. The trained model predicted multi-dispense precision on held-out sessions about 12% more accurately than the manufacturer's datasheet, and about 17% on the most heavily replicated data. What improved was not the dispensing precision itself but **the accuracy of the predictions** the compiler uses when deciding instrument placement and tip reuse.

> ### ⚠️ Anthropic's summary of Tetsuwan diverges from the original blog in two places
>
> **⓵ The sign-test numbers.** The Anthropic announcement says "better in 31 of 45, $p \approx 0.001$," but Tetsuwan's original blog says "33 of 45 held-out sessions, $p \approx 0.003$." Working it out, only the original blog is self-consistent.
>
> $$p = \Pr[X \ge k],\qquad X \sim \text{Binomial}(45,\ 0.5)$$
>
> - $X$ — the number of sessions the model wins, under the null hypothesis that the model and the datasheet are equivalent
> - $k$ — the number of sessions in which the model actually beat the datasheet
> - $k=31$ → one-sided 0.008, two-sided 0.016 (does not match 0.001)
> - $k=33$ → one-sided 0.0012, two-sided 0.0025 (matches the original blog's 0.003)
>
> **⓶ The autonomy of the bubble recovery.** Reading only the Anthropic announcement, it looks as if Claude handled everything up to centrifugation autonomously. According to the original blog, Claude carried out, via MHS, moving the plate to the sealer and sealing it; **moving the sealed plate to the centrifuge off the instrument deck was done by a human operator asked over Slack**. The agent only ran the centrifugation, and human hands were involved on the way back as well.

### 5.6 Janelia and UW — the agent at decision points

**Janelia.** A rig that images cellular activity across the whole zebrafish with two-photon microscopy. Imaging has a tradeoff between speed and coverage: image one plane fast, or several planes slowly. Finding the oscillating cell populations of interest needs coverage, and measuring their activity needs speed, but existing experiments had to pick one setting at the start and stay with it to the end.

The agent harness the Janelia author wrote follows the pattern of Section 3.4 exactly. Its backbone is **a deterministic loop** alternating acquisition and analysis, and the agent comes in **only at decision points** to choose which region to image and which analysis to run. As a result, it found oscillating cell populations a fixed setting would have missed, and the number of repeat experiments and animals went down. On Arco Bast's rig, the prototype of MHS, Claude took over beam alignment and optical tuning, and preparation work that used to take half a day shrank to a single step.

**UW.** Six instruments were connected within a week, including writing the drivers. Previously, integrations like this cost months to years and thousands to millions of dollars. In qPCR, the amplification curve rises in an S shape and must be stopped just before it enters the plateau. The agent watches the curve in real time, asks the researcher at the right moment whether to stop, and, if stopped, moves on to a 4 °C hold step that prevents DNA degradation. In the robot arm handoff, the arm moved about 10 seconds after receiving the liquid handler's completion signal, and in repeated tests the two instruments never once collided.

One point is worth making from a robotics perspective. What the LeRobot-based open-source robot arm does here is a **fixed motion** that moves a plate to a set position, and the coordination is sequential execution that waits for completion events. No learned manipulation policy appears.

---

## 6. Positioning — among neighboring standards

The only thing the announcement itself compares against is MCP. The table below is my attempt to place MHS among existing standards.

| | ROS 2 | SiLA 2 / OPC UA LADS | MCP | MHS |
|---|---|---|---|---|
| Primary target | robots (sensors, actuators) | lab and analytical instruments | software tools and data | lab and manufacturing equipment broadly |
| Primary consumer | robot control software | automation software such as schedulers and LIMS | LLM agents | LLM agents (+ people, code) |
| How it describes things | message types, topics | machine-readable capability and information models | tool schemas + natural-language descriptions | state and procedures + natural-language reference files |
| Physical context and safety for agents | not a design goal | not a design goal | not applicable | first-class |

Existing instrument standards were designed for "machines calling machines." They have schemas but do not carry operating context such as "this arm weighs so many kg, so it must be handled like this." On the premise that the consumer is an LLM, MHS makes three things first-class — natural-language reference, safety limits enforced by the driver, and a path for executing code the agent wrote. Open-source libraries such as PyLabRobot, which offers a hardware-independent liquid-handling interface at the Python level, are also close neighbors.

The announcement does not say whether MHS replaces SiLA 2 or LADS or wraps on top of them. Since the case studies wrap existing interfaces such as COM, file drops and GUIs underneath, it looks like a structure where a SiLA 2 instrument could also be wrapped with a single adapter (my inference).

Then since the name pairs with MCP, is it fine to understand MHS as roughly "a collection of MCP servers for hardware"?

> ### 🔗 MHS is not "a collection of MCP servers"
>
> MCP is only one of the three paths into MHS. MHS's own contribution lies beneath it — the instrument state model (the shared-memory dictionary), reference files, safety limits enforced by the driver, and the code-file execution path.
>
> On the ecosystem side, points of contact with the robot-learning community's toolchain are emerging. Hugging Face is adding MHS support to its robotics library LeRobot, and Raspberry Pi, after testing an MHS driver for its cameras, is integrating it into several products. On the hardware-vendor side, AWS (Strands Robots), Automata, Danaher, Doosan Robotics (testing with a robot arm), MBF Bioscience, QIAGEN, Tecan and Universal Robots are preparing or experimenting with support.

---

## 7. Limitations — how far were the four walls breached?

Returning to the four walls named in Section 1.2, here is where things stand.

| Wall | MHS's mechanism | Current state |
|---|---|---|
| **Wall 1** Interface fragmentation | standard driver + discovery | breached most clearly. But the cost of writing drivers did not disappear; it moved to once per instrument. UW, CMU and Tetsuwan all wrote drivers themselves, and Tetsuwan attached an SBC to every instrument. Native vendor support is the key to adoption |
| **Wall 2** Knowledge not in the code | natural-language tags → reference file | only half breached. Only written knowledge gets through. Bubbles (Genentech), evaporation (Tetsuwan — about 1/3 of the measured CV came from evaporation at the plate edges, and the agent struggled to reason about physical effects like evaporation) and hardware fault diagnosis (QuEra — its understanding of the instrument was "programmatic, not physical") were all filled in by people |
| **Wall 3** Safety | driver-enforced limits + the agent's caution | works in the case studies (CMU's six faults blocked in advance). But the grammar for declaring and verifying limits cannot be checked, since the spec is unreleased. The soft layer's overcaution sometimes stalled experiments overnight (QuEra) |
| **Wall 4** Timescales | chaining code files | solved conceptually. But there are no latency numbers. Whether it meets closed-loop requirements such as the phase-locked optogenetics Janelia is aiming for, and what the latency is between machines outside shared memory, are not described |

Other points worth noting:

- **The boundary of interface requirements** — Anthropic's main text says instruments without a programming interface are not yet supported, but CMU integrated a GUI-only reader by operating its screen. The real boundary is "purely analog or manual equipment with no API, no SDK and no GUI." CMU also noted, though, that on the GUI path the only means of verifying results is the screen.
- **Evaluation design** — every result is a partner's self-report, and the baselines vary (vendor setup time, a dedicated script built over months, human experts). The shortened integration times also mix "the effect of writing drivers with Claude" and "the effect of the MHS spec," which cannot be separated. CMU's COM driver is a prime example.
- **The cost of supplying context** — the QuEra team had to fill in a lot of context themselves about the desired results and how to achieve them.
- **Operating cost and maturity** — compute costs for long-running monitoring add up and must be weighed against researcher time saved. The UW author says outright that it is still a PoC and that complex protocols need further optimization.

---

## 8. Closing — what this announcement suggests

**⓵ From an agentic-AI perspective.** The most portable lesson of MHS is the design that **takes the agent out of the fast loop and hardens the results of exploration into deterministic code**. LLM agents always face the choice between a tool-calling loop and code execution, but in the physical world this choice governs not only latency and cost but also safety and auditability.

**⓶ From a Physical AI / RFM perspective.** Sketching where a VLA fits into this stack gives the following.

```
  LLM agent          plan, interpret, decide        (sec - min)
      |
  MHS                discover, read/write, safety limits
      |
  device procedures  e.g. arm.pick(plate)           <- a VLA could live here
      |
  device firmware    servo loops                    (us - ms)
```

MHS does not solve manipulation. It standardizes the orchestration layer **above** manipulation. A VLA's place is below MHS as a "procedure provider," and LeRobot support could be the link.

**⓷ From a data perspective.** The expectation that standardized instrument interfaces will relieve Physical AI's data bottleneck is only half right. Genentech states explicitly that the very process of checking Claude's decisions against expert judgment becomes a dataset for improving models. But MHS logs are time series of states and readings, recorded at the granularity of the agent's decisions. They are **process-level data** such as "flow rate → transfer error" or "concentration range → $R^2$," not the high-frequency observation-action trajectories used to train VLAs. So they are closer to **a pipe for training world models or surrogate models of the experimental process** than to an RFM data pipe. The failures left over from Wall 2, such as bubbles and evaporation, are also for a model that predicts process physics to fill, not a VLA. Genentech, too, said it will attach a separate model that monitors and optimizes experiments with real-time data.

---

## Appendix — Glossary

| Term | Definition |
|---|---|
| **MHS driver** | the standard software layer that translates between the OS and the instrument. Provides primitives, discovery, reference files and safety limits |
| **Primitive** | the minimal commands every instrument understands: `read` (read state) and `write` (set state) |
| **State dictionary** | a dictionary in shared memory holding every instrument's variables, control values and sensor values. Originated at Janelia |
| **Reference file** | an instrument description auto-generated from natural-language tags: what can be measured and adjusted, and the safety limits |
| **Code-file chaining** | stringing driver commands together in code so they run at instrument speed without the agent's step-by-step reasoning |
| **Hard / soft safety layer** | invariants enforced by the driver / the agent's cautious judgment |
| **BCA assay** | a standard quantification method that measures a sample's total protein concentration by absorbance |
| **qPCR** | a method that amplifies DNA through heating and cooling cycles while measuring the number of copies in real time by fluorescence |
| **Lock (laser)** | the state in which a laser's frequency is held ultra-precisely at a target value |

**Original** — [Previewing the Model Hardware Standard](https://www.anthropic.com/news/model-hardware-standard-research-preview) · **Tetsuwan original blog** — [Integrating Anthropic's Model Hardware Standard](https://tetsuwan.com/blog/mhs) · **QuEra blog** — [Holding the Light](http://www.quera.com/blog-posts/holding-the-light-teaching-an-ai-to-lock-and-tune-our-quantum-computers-lasers) · **Preview sign-up** — [modelhardwarestandard.com](https://www.modelhardwarestandard.com/)
