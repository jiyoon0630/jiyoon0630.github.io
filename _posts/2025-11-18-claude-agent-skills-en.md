---
layout: paper
lang: en
ref: claude-agent-skills
kind: tech-review
title: "Claude Agent Skills: Building AI Agents from Markdown Files"
date: 2025-11-18 12:00:00 -0700
math: false
tags: [Claude, Agent-Skills, Agentic-AI, MCP, LLM, Tech-Review]
summary: "Claude Agent Skills define an agent's behavior with nothing but Markdown files and folders instead of complex MCP servers — how they are structured and work, how they compare with other concepts and frameworks, and what they imply seen as a move toward an AI-native OS."
---

> **In short** — **Claude Agent Skills let you define an AI agent's behavior with nothing but Markdown files and a folder structure, instead of building complex MCP servers, so that domain practitioners can build agents themselves.** Through progressive disclosure, only the Skills that are needed get loaded, which keeps token use efficient, and computations that must be exact are handled by scripts bundled with the Skill. This piece covers how Skills are structured and how they work, how they compare with existing agentic AI concepts and frameworks, and a use case, and closes with what they imply if you read them as Anthropic's move to claim the AI-native OS.

---

## 📚 What are Claude Agent Skills?

- Claude Agent Skills are an AI agent development tool Anthropic released on October 17, 2025. **They dramatically simplify building AI agents (pre-built agents), which used to demand considerable technical expertise and complex infrastructure.**

> ### ⚠️ Update, September 2026
>
> The release date was October 16, 2025 in US time (October 17 in Korea).

- At heart, a Skill is a unit of capability that lets you build an AI agent specialized for an enterprise environment (or for specialist work) with nothing but simple Markdown files and a folder structure — no complex code, no servers.

> ### 💡 Markdown
>
> **"Markdown"** is a simple way of writing documents in which a few plain symbols (#, *, - and so on) added to ordinary text express the document's structure.

- In an enterprise, Skills let **the practitioners who know the organization's own processes, domain knowledge and internal rules best pass that knowledge to an AI agent and design the agent themselves.**
    - For example, with Claude Agent Skills and no AI or development skills, a management team could build a business-insight agent that helps analyze company-wide performance data and shape strategy, and a marketing team could build a content-generation agent that applies the brand guidelines to a high standard.

## ✏️ Why are Claude Agent Skills Needed?

1. **They are shifting agent building from a complex protocol (MCP) to simple Markdown files and folders.**
    - Until now, getting an AI agent to do real work — financial analysis and management reporting, legal review of contracts, content generation — meant building and operating many components (host, server, tools and so on) according to the MCP (Model Context Protocol) standard. It was complex work that took considerable development resources and infrastructure management.
    - **Skills replace server building with simple Markdown files that define the agent's behavior, so practitioners who are not technical experts can build and modify agents from their own domain knowledge.**
2. **They move from loading every tool up front to selective loading based on progressive disclosure, so tokens can be managed efficiently.**
    - The number of tokens an LLM (large language model) can take in at once — its context window — is limited. That has raised several problems, and the area is still developing.
    - With MCP, the agent had to add the description of every tool it could use (Excel, email, ERP and so on) to the context every time, including tools it never actually used, wasting tens of thousands of tokens.
    - As a multi-turn conversation goes on, earlier turns and tool-call results pile up. When the context window reaches its limit, the model forgets important instructions, or automatic compaction loses information and performance drops sharply.
    - **Skills solve this with an innovative approach called progressive disclosure. Rather than handing the model every manual from the start, the agent holds only the list of Skills with short descriptions, and loads and runs a Skill only when the user's request relates to it.**

## ⚡ How Claude Agent Skills work?

- **A Skill consists of a single Markdown file (SKILL.md) holding metadata and task instructions, plus script and data files added when needed.**

    ![When each part of a Skill is loaded, and how many tokens it takes](/assets/img/notes/claude-agent-skills/levels.png)
    *Progressive disclosure: when each part of a Skill is loaded, and how many tokens it takes (source: [Anthropic](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills))*

    - **SKILL.md metadata (YAML)**: a header (front matter) with the Skill's name, description, conditions for use and other basics, which lets the model pick the Skill best suited to the current task out of many. **(always loaded, ~100 tokens)**
    - **SKILL.md body (Markdown):** the actual instructions — a document describing in detail in what order and in what way the model should carry out a particular task. **(loaded only when the Skill runs, <5,000 tokens)**
    - **Bundled files (text files, scripts, data)**: additional resources the Skill needs — Excel formulas, logic, Python code, reference data — included when complex calculation or data processing is required. **(loaded only when needed, unlimited)**

    → **Thanks to this structure, an agent can hold tens or hundreds of Skills while keeping only each Skill's metadata in context rather than every Skill's full content, cutting token use dramatically compared with the previous approach.**

- **Skills work by analyzing the user's request (prompt), comparing it with the metadata in the Skill list, selecting the right Skill, and loading and running it to produce the best result.**

    ![Skills and the Context Window](/assets/img/notes/claude-agent-skills/context-window.png)
    *Skills and the Context Window: how a Skill is loaded into context in response to a request (source: [Anthropic](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills))*

    1. **Prompt**: The user makes a request such as "please fill out this PDF form" and attaches the file. At this step the model works out the user's intent and the type of material provided.
    2. **Skill selection**: The model reviews the metadata of every Skill currently in context (docx, pdf, pptx, xlsx and so on) and automatically selects the one most relevant to the request. For a PDF task, for instance, the "pdf" Skill is selected.
    3. **Skill loading**: The full content of the selected Skill (SKILL.md) is loaded into context. Only now are the Skill's detailed instructions brought in, along with related scripts or reference files (forms.md and the like) when needed.
    4. **Reasoning from the Skill's instructions**: The model does the actual work following the loaded instructions — identifying the PDF form fields, filling them in appropriately from the user's information, and running any calculations or checks.
    5. **Final response**: When the work is done, the model returns the result to the user, with an explanation of what it did where needed.

- **In addition, when a Skill includes executable scripts in SKILL.md, the agent can run them directly in a fully isolated, secure environment (sandbox), securing both accuracy and security.**

    ![Bundling executable scripts](/assets/img/notes/claude-agent-skills/scripts.png)
    *Bundling executable scripts: the agent runs a script bundled with the Skill directly (source: [Anthropic](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills))*

    - For work where the result must always be identical and exact (deterministic) — Excel formula calculations, integration with internal systems — this guarantees 100% reliable results, free of the model's chronic hallucination problem.

> ### 💡 The hallucination problem in AI models
>
> An LLM does not actually perform mathematical calculation; it infers a "plausible" answer from learned patterns, so it often makes mistakes on complex formulas.

> ### ⚠️ Update, September 2026
>
> - Scripts do not go inside SKILL.md; they sit as separate files in the same folder as SKILL.md, which references them.
> - Isolated sandbox execution applies to the code-execution environment of the API and claude.ai. In Claude Code, scripts run in the user's local environment.
> - A script itself always returns the same result, but the model still decides what input to call it with. "Removes hallucination from the calculation step" is therefore more accurate than "guarantees 100%."

## 💥 Why are Claude Agent Skills special?

- **Compared with core agentic AI concepts**

| Comparison | **Description** |
| --- | --- |
| **Skills vs. system prompts** | • A system prompt — guidance such as an agentic AI system's answer style and project direction — applies globally at all times, so it carries guidance unrelated to the task at hand, wastes tokens, and risks performance loss from excessive context buildup<br>• **Skills solve the token-efficiency problem by loading selectively, only when needed** |
| **Skills vs. RAG** | • RAG is designed to inject external knowledge by embedding documents and running semantic search, but it needs additional infrastructure (vector DB, embedding model, retrieval techniques, etc.)<br>• **Skills teach the LLM to reason like an expert through curated knowledge and methods, so combined with RAG's broad retrieval they enable a powerful, complementary agentic workflow** |
| **Skills vs. fine-tuning** | • Fine-tuning specializes a model by permanently changing its weights to learn style, format and domain vocabulary; it is expensive and ill-suited to frequent knowledge updates<br>• **Skills let you update domain knowledge in Markdown files without training the model, and improve it iteratively** |
| **Skills vs. tool use / MCP** | • Tools connected to an agentic AI system through MCP supply the model with information for reasoning<br>• **Skills teach how to analyze and judge the information provided, so they complement MCP** |

- **Compared with agentic AI frameworks**

|  | LangChain / LangGraph | OpenAI AgentKit | Claude Skills |
| --- | --- | --- | --- |
| **Who builds the agent, and how** | **Developers**<br>• Implemented directly in Python/JS code **(code)**<br>• Every workflow change means modifying and redeploying code | **Developers and practitioners**<br>• Agents designed as diagrams by connecting nodes in OpenAI Agent Builder, a no-code/low-code workflow tool **(visual nodes)**<br>• Extended with the OpenAI Agents SDK when needed **(code)** | **Developers and practitioners**<br>• Instructions written in Markdown files in the .claude/skills/ folder **(text)** → **gives up the visual intuitiveness of a diagram in exchange for the expressiveness and flexibility of natural language**<br>• Extended with the Claude Agent SDK when needed **(code)** |
| **Connecting tools** | • External APIs, DBs and internal systems connected directly at code level through Python libraries, LangChain connectors and MCP-based servers/tools<br>• **Token use grows as more tool definitions enter the context** | • External APIs, DBs and internal systems connected through MCP-based servers/tools and OpenAI Connector Registry, a predefined list of tools<br>• **Token use grows as more tool definitions enter the context** | • **No separate MCP setup; editing a Skill file is enough to connect it to the agent**<br>• **Skills are loaded and run selectively from the list, so tokens are used efficiently** |
| **Model / ecosystem compatibility** | • A framework aimed at swapping and combining models (OpenAI, Anthropic, Google, open source, etc.) | • An agent/tool ecosystem optimized for OpenAI models by default (other models usable through MCP)<br>• Workflow building, deployment and monitoring, and permission management are all tied to the OpenAI console, so the operating environment and ecosystem are strongly bound to OpenAI | • **Works in vibe-coding environments, so other coding models such as OpenAI Codex and Gemini Code can use it, not just Claude**<br>• **Anthropic has open-sourced the Claude Skills Cookbook, the Agent Skills docs and example Skills, so it is likely to become the industry standard** |

> ### ⚠️ Update, September 2026
>
> - **The industry-standard prediction came true.** Anthropic released Agent Skills as an open standard on December 18, 2025, and many tools — OpenAI Codex, Gemini CLI, GitHub Copilot, Cursor and others — now support Skills in the same format.
> - At the time of writing, in November 2025, however, Skills worked only in Claude products. "Gemini Code" in the table refers to Gemini CLI.
> - In AgentKit, MCP is a standard for connecting external tools, not a way to connect other AI models.

## 🔎 Claude Agent Skills Use-cases

- **Claude for Financial Services, released officially by Anthropic**
    - On July 15, 2025, Anthropic released Claude for Financial Services, which helps financial professionals such as analysts and portfolio managers with market research and decision-making, to an enthusiastic response from the financial industry. **Just three months later, in October 2025, it emerged that the service is built on a standardized framework called Claude Agent Skills, successfully demonstrating the value of Agent Skills for building enterprise pre-built agents specialized for professional domains.**
    - The Claude Agent Skills framework itself is **open-sourced so anyone can develop new Skills, expanding the ecosystem**. Meanwhile, **core capabilities that companies can apply to their business right away, like Claude for Financial Services, were offered only on the paid (Enterprise) plan.** With this, **Anthropic took a strategy of meeting the enterprise scene's real needs while securing stable revenue.** Beyond that, **companies can build further customized capabilities for their own work on top of these core capabilities.**
    - Claude for Financial Services could connect through MCP to companies' internal data platforms such as Databricks and Snowflake, as well as major financial data providers such as S&P Global, LSEG and Morningstar. Through this, the Claude model could automatically collect the financial data a task required and carry out the professional work from analysis through report writing.
        - **It also offers specialized solutions developed by consulting partners such as Deloitte and PwC — equity research, private credit analysis, regulatory compliance gap analysis and more.**

| **Core Skills** | **Function and purpose** |
| --- | --- |
| **Public/private comparable analysis** | Selects a peer group and generates a report comparing key financial metrics and valuation multiples. |
| **DCF (discounted cash flow) modeling** | Quickly builds a DCF model from current data and the user's assumptions and runs a sensitivity analysis. |
| **Company coverage research** | Drafts an in-depth market and company **coverage report** similar to one an analyst would write. |
| **Company profile documents** | Automatically generates a draft document for fundraising, including a business overview and financial highlights. |
| **Due diligence data** | Quickly collects the company financial and operating data needed in M&A due diligence and organizes it into a structured data pack. |
| **Earnings analysis** | Analyzes the latest earnings materials and summarizes key growth drivers, financial health and management commentary. |

> ### ⚠️ Update, September 2026
>
> To be precise, Claude for Financial Services was not built on Skills from its July launch. The financial Agent Skills in the table above were **added** in the October 2025 update, together with an Excel add-in and new data connectors.

- **(What if?) Redesigning a multi-agent data-analysis agent with Claude Agent Skills**
    - Take as an example an agentic AI system that analyzes unstructured data (document-style reports) and structured data (a DB) together, retrieving and delivering the information users need and supporting decisions. Such a system can be built around a Super Agent that analyzes the user's complex requirements, draws up the best work plan and orchestrates the functions below it. Through the MCP (Model Context Protocol) standard, this Super Agent connects to specialized unit agents — a RAG agent for searching and analyzing unstructured data, a DB agent w/LLM for searching and analyzing structured data, and an agent that can write code for functions such as visualization and precise computation — calling tools flexibly as needed and producing an integrated result.
    - Redesigned on Claude Agent Skills, this structure could **shift from a Super Agent assigning work to sub-agents/units to a single model (e.g. Claude) handling clearly defined Skills directly.** This would sharply cut the excessive token use caused by keeping unused tools or unit agents loaded at all times, and, as I see it, would make it easy to add new capabilities without implementing a complex protocol like MCP.
    - Let's define the detailed functions the RAG agent and DB agent provide as Skills.

> ### 💡 (Example) Defining the RAG agent's and DB agent's functions as Skills
>
> - **(Example) A RAG Skill for searching and analyzing unstructured data**
>     - **In-document search skill**: to return the chunks most relevant to the user's request, prepare reference Markdown files such as a search-strategy guide and a chunking guide, plus script files that perform chunking and search, and implement the RAG logic in SKILL.md.
>     - **Answer generation skill**: to generate an answer that fits the request from the search results, write guides in SKILL.md for simple answers, comparative-analysis answers, insight extraction and so on.
> - **(Example) A DB Skill for searching and analyzing structured data**
>     - **DB discovery skill**: to capture the DB's key information such as table schemas and distinct column values, prepare script files that analyze the schema, detect table relationships and analyze columns, and implement the logic in SKILL.md.
>     - **Schema linking skill**: to select the items most relevant to the user's question using the discovered DB information, define the main functions such as entity extraction and table selection in SKILL.md.
>     - **SQL writing and execution skill**: to generate queries such as table-lookup SQL and value-extraction SQL, prepare reference Markdown files with SQL patterns and frequently used queries, plus script files that validate and execute SQL, and implement the logic in SKILL.md.

## 📢 (Takeaways) Claude Agent Skills Key Takeaways: "AI-Native OS"

- Claude Agent Skills are **a tool that lets practitioners build AI agents themselves, using simple Markdown files that can be expressed as structured text instead of the complex MCP protocol.** The emphasis is on the token efficiency of loading only the Skills that are needed, but in practice they can be said to drive a revenue structure in which the model's reasoning and decision steps become more granular and API calls more frequent.
- In particular, Anthropic lowered the barrier to entry by open-sourcing the Skills standard, while **designing it to work best on Claude and letting people publish the Skills they write freely — a de facto standardization strategy for the agent/agentic AI market that expands the ecosystem and naturally drives Claude API use.**
    - **The technology and its usage are open source, but ultimately, when companies set out to build pre-built agents for professional work, practitioners will build hundreds of Skills on Claude's infrastructure and integrate them into their work, locking them into the Claude ecosystem. Over the long term, this can be seen as a move to claim the AI-native OS.**

> ### 💡 Andrej Karpathy's LLM OS
>
> Andrej Karpathy (former Tesla AI director and OpenAI co-founder) redefined the LLM not as a mere AI model but as a computer's central processing unit (CPU) and the kernel of its operating system (OS). He **stressed that building an LLM OS — a single computing system in which the LLM organically controls and coordinates memory (the context window), storage (DBs) and various tools — is the key path toward AGI.**
> (Source: [https://analyticsindiamag.com/ai-news-updates/andrej-karpathy-says-the-pathway-to-agi-is-through-a-language-model-operating-system/](https://analyticsindiamag.com/ai-news-updates/andrej-karpathy-says-the-pathway-to-agi-is-through-a-language-model-operating-system/))
>
> In other words, a new kind of OS centered on the LLM, unlike Windows or macOS — an idea first discussed two years ago — seems to be becoming real faster through Anthropic.
>
> Until now, a great deal of software has been developed and distributed so that the same features work across different operating systems, or offered as each vendor's own software in cloud environments.
>
> **In the coming era of the LLM OS, I expect LLM OSes to run on large-scale infrastructure, and service and solution companies to emerge that provide agents able to work regardless of which LLM OS they run on.**

- Anthropic has already proven the value of Skills in finance — a domain where data security and accuracy are essential — through Claude for Financial Services, an enterprise vertical solution built on Skills. As it expands into other high-value areas such as law and medicine, **this can be read as a business model that gets companies to go beyond plain API use and subscribe to specialist AI agents themselves.**
- **Recent cases such as OpenAI's AgentKit and Anthropic's Claude Agent Skills suggest that big tech companies are expanding beyond providing LLMs into AI-native OS ecosystems.**
    - Major LLM providers, Anthropic among them, are rapidly turning into AI infrastructure-layer companies and are entering the enterprise market faster than expected.
