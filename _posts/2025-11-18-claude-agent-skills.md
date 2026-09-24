---
layout: paper
lang: ko
ref: claude-agent-skills
kind: tech-review
title: "Claude Agent Skills: Markdown 파일로 만드는 AI Agent"
date: 2025-11-18 12:00:00 -0700
math: false
tags: [Claude, Agent-Skills, Agentic-AI, MCP, LLM, Tech-Review]
summary: "복잡한 MCP 서버 대신 Markdown 파일과 폴더 구조만으로 Agent를 정의하는 Claude Agent Skills — 구조와 작동 방식, 다른 개념·프레임워크와의 비교, 그리고 AI-Native OS 관점에서 본 시사점."
---

> **핵심 정리** — **Claude Agent Skills는 복잡한 MCP 서버 구축 대신 Markdown 파일과 폴더 구조만으로 AI Agent의 행동을 정의하게 하여, 실무자도 직접 Agent를 만들 수 있도록 하는 도구입니다.** 점진적 공개(Progressive Disclosure)로 필요한 Skill만 선택적으로 로드하여 토큰을 효율적으로 사용하고, 정확해야 하는 연산은 Skill에 포함된 스크립트로 처리합니다. 이 글에서는 Skills의 구조와 작동 방식, 기존 Agentic AI 개념·프레임워크와의 비교, 활용 사례를 정리하고, Anthropic이 Skills를 통해 AI-Native OS를 선점하려 한다는 관점에서 시사점을 정리했습니다.

---

## 📚 What are Claude Agent Skills?

- Claude Agent Skills는 Anthropic이 2025년 10월 17일에 새롭게 공개한 AI Agent 개발 도구로, **기존에는 상당한 기술 전문성과 복잡한 인프라 구축이 필요했던 AI Agent(pre-built agent) 개발을 획기적으로 단순화하는 도구입니다.**

> ### ⚠️ 2026년 9월 기준 업데이트
>
> 공개일은 미국 시간 기준 2025년 10월 16일입니다(한국 시간 10월 17일).

- 본질적으로 Skills는 복잡한 코드나 서버 구축 없이도 단순한 Markdown 파일과 폴더 구조만으로 Enterprise 환경에 특화된(또는 전문적인 작업을 수행해야 하는) AI Agent를 만들 수 있게 해주는 기능 단위입니다.

> ### 💡 Markdown
>
> **“Markdown”**이란 일반 텍스트에 간단한 기호(#, *, - 등)를 추가하여 문서 구조를 표현하는 단순한 문서 작성 방식을 말합니다.

- Enterprise 환경에서 Skills는 각 조직의 고유한 업무 프로세스, 도메인 지식, 그리고 내부 규정을 가장 잘 아는 **실무자들이 직접 이를 AI Agent에게 효과적으로 전달하고 설계할 수 있도록 하는 역할**을 합니다.
    - 예를 들어, AI 및 개발 역량없이도 Claude Agent Skills를 통해서, 경영관리팀은 전사 실적 데이터 분석 및 전략 수립을 돕는 경영 인사이트 Agent를 만들 수 있고, 마케팅팀은 브랜드 가이드라인을 높은 수준으로 반영하여 콘텐츠 생성 Agent를 만들 수 있습니다.

## ✏️ Why are Claude Agent Skills Needed?

1. **Agent 구축에 있어 복잡한 프로토콜(MCP)에서 단순한 Markdown 파일 및 폴더 구조로 변화를 이끌고 있습니다.**
    - 기존에 AI Agent가 재무분석/경영관리, 계약서 내 법무 검토나 콘텐츠 생성과 같은 실제 업무를 하게 하려면, MCP(Model Context Protocol)라는 표준에 따라 수많은 구성요소(Host, Server, Tool 등)를 구축하고 운영해야 하므로 상당한 개발 리소스와 인프라 관리가 필요한 복잡한 작업이었습니다.
    - **하지만 Skills는 복잡한 서버 구축 대신 단순한 Markdown 파일로 Agent의 행동을 정의할 수 있게 하여, 기술 전문가가 아닌 실무자도 자신의 업무 지식을 바탕으로 쉽게 Agent를 만들고 수정할 수 있게 되었습니다.**
2. **전체 도구 로딩에서 점진적 공개(Progressive Disclosure) 기반의 선택적 로딩으로 전환하여 토큰을 효율적으로 관리할 수 있습니다.**
    - LLM(Large Language Model)에 한번에 입력할 수 있는 토큰의 양(Context Window)에는 한계가 있어 몇 가지 문제가 언급되었고, 아직도 지속 발전 중에 있습니다.
    - MCP 사용 시 Agent가 사용 가능한 모든 도구(엑셀, 이메일, ERP 등)에 대한 설명을 매번 컨텍스트로 추가해야 했고, 실제로 사용하지 않는 도구들까지 포함되어 수만 개의 토큰이 불필요하게 소모되었습니다.
    - 멀티턴 대화가 진행될수록 이전 대화 내역과 도구 호출 결과가 누적되어 Context Window가 한계에 도달하면, AI 모델은 중요한 지시사항을 잊어버리거나 자동 압축으로 인한 정보 소실 및 급격한 성능 저하가 발생했습니다.
    - **Skills는 이러한 문제를 ‘점진적 공개(Progressive Disclosure)’라는 혁신적인 방식으로 해결합니다. 즉, 처음부터 AI 모델에게 모든 설명서를 전달하는 것이 아니라, Skill 목록과 간략한 설명만 들고 있다가 사용자의 요청이 특정 Skill과 관련이 있을 때만 선택적으로 로드하여 실행합니다.**

## ⚡ How Claude Agent Skills work?

- **Skills의 구성 요소는 메타데이터와 작업 지침을 담은 단일 Markdown 파일(SKILL.md)과 필요 시 추가되는 Scripts/Data 파일들로 이루어집니다.**

    ![Skill 구성 요소별 로드 시점과 토큰 규모](/assets/img/notes/claude-agent-skills/levels.png)
    *Progressive Disclosure: Skill 구성 요소별 로드 시점과 토큰 규모 (출처: [Anthropic](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills))*

    - **SKILL.md Metadata (YAML)**: Skill의 이름, 설명, 적용 조건 등 기본 정보를 담은 헤더(Frontmatter)로, AI 모델이 수많은 Skills 중에서 현재 작업에 가장 적합한 Skill을 자동으로 선택할 수 있게 합니다. **(항상 로드, ~100 토큰)**
    - **SKILL.md Body (Markdown):** 실제 작업 지침서로, AI 모델이 특정 업무를 어떤 순서로, 어떤 방식으로 수행해야 하는지 상세하게 기술한 문서입니다. **(Skill 실행 시에만 로드, <5,000 토큰)**
    - **Bundled files (text files, scripts, data)**: Excel 공식, logic, Python 코드, 참조 데이터 등 Skill 실행에 필요한 추가 리소스들로, 복잡한 계산이나 데이터 처리가 필요한 경우 포함됩니다. **(필요시에만 로드, 무제한)**

    → **이러한 구조 덕분에 Agent는 수십, 수백 개의 Skills를 보유하더라도 모든 Skill의 전체 내용을 컨텍스트에 포함할 필요 없이 각 Skill의 메타데이터만 유지하면 되어, 기존 방식 대비 토큰 사용량을 획기적으로 줄일 수 있습니다.**

- **Skills의 작동 방식은 사용자의 요청(Prompt)을 분석하고, Skill 목록에 있는 메타데이터와 비교하여 적절한 Skill을 선택하고, 해당 Skill을 로드·실행하여 최적의 결과를 도출하게 됩니다.**

    ![Skills and the Context Window](/assets/img/notes/claude-agent-skills/context-window.png)
    *Skills and the Context Window: 사용자 요청에 따라 Skill이 컨텍스트에 로드되는 흐름 (출처: [Anthropic](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills))*

    1. **Prompt 입력**: 사용자가 "이 PDF 양식을 작성해 주세요"와 같은 요청을 하면서 필요한 파일을 첨부합니다. 이 단계에서 AI 모델은 사용자의 의도와 제공된 자료의 유형을 파악합니다.
    2. **Skill 선택**: AI 모델은 현재 컨텍스트에 로드된 모든 Skill 목록의 메타데이터(docx, pdf, pptx, xlsx 등)를 검토하여 사용자 요청과 가장 관련성이 높은 Skill을 자동으로 선택합니다. 예를 들어 PDF 파일 작업이라면 'pdf' Skill이 선택됩니다.
    3. **Skill 로드**: 선택된 Skill의 전체 내용(SKILL.md)을 컨텍스트에 로드합니다. 이 시점에서 비로소 해당 Skill의 상세한 작업 지침과 필요한 경우 관련 스크립트나 참조 파일(forms.md 등)도 함께 불러옵니다.
    4. **Skill 지침 기반 추론**: 로드된 Skill의 지침에 따라 AI 모델이 실제 작업을 수행합니다. PDF 양식 필드를 식별하고, 사용자 정보를 바탕으로 적절한 내용을 채우며, 필요한 계산이나 검증 작업을 진행합니다.
    5. **최종 응답**: 작업이 완료되면 AI는 처리된 결과물을 사용자에게 제공하고, 필요한 경우 수행한 작업에 대한 설명을 함께 전달합니다.

- **또한, Skills는 SKILL.md 파일 내에 실행 가능한 스크립트를 포함시키면, Agent는 이를 완전히 격리된 보안 환경(Sandbox)에서 직접 실행할 수 있어 정확성과 보안을 동시에 확보합니다.**

    ![Bundling executable scripts](/assets/img/notes/claude-agent-skills/scripts.png)
    *Bundling executable scripts: Skill에 포함된 스크립트를 Agent가 직접 실행하는 구조 (출처: [Anthropic](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills))*

    - 엑셀 수식 계산이나 내부 시스템 연동처럼 결과가 항상 동일하고 정확해야 하는(Deterministic) 업무에서 AI 모델의 고질적인 '환각(Hallucination)' 문제 없이 100% 신뢰할 수 있는 결과를 보장합니다.

> ### 💡 AI 모델의 환각(Hallucination) 문제
>
> LLM은 수학 계산을 실제로 수행하는 것이 아니라 학습된 패턴을 바탕으로 "그럴듯한" 답을 추론하기 때문에, 복잡한 수식에서는 종종 오류가 발생합니다.

> ### ⚠️ 2026년 9월 기준 업데이트
>
> - 스크립트는 SKILL.md 파일 안이 아니라, SKILL.md와 같은 폴더에 별도 파일로 두고 SKILL.md에서 참조하는 구조입니다.
> - 격리된 Sandbox 실행은 API와 claude.ai의 코드 실행 환경 기준이며, Claude Code에서는 사용자의 로컬 환경에서 실행됩니다.
> - 스크립트 자체는 항상 같은 결과를 내지만, 어떤 입력으로 스크립트를 호출할지는 여전히 AI 모델이 정합니다. 따라서 "100% 보장"보다는 "연산 단계의 환각을 제거한다"가 정확한 표현입니다.

## 💥 Why are Claude Agent Skills special?

- **Agentic AI Core Concepts과 비교**

| 비교 구분 | **설명** |
| --- | --- |
| **Skills v.s. System Prompts** | • Agentic AI 시스템의 답변 스타일, 프로젝트 방향성 등 지침에 해당되는 System Prompt는 전역적(Global)으로 상시 적용되므로, 특정 업무와 무관한 지침까지 포함하여 불필요한 토큰을 소모하며 과도한 Context 누적으로 인한 성능 저하 우려가 있음<br>• **Skills는 필요한 시점에만 불러오는 선택적 로딩 방식을 통해 토큰 효율성 문제를 해결함** |
| **Skills v.s. RAG** | • RAG는 외부 지식을 주입하기 위해 문서를 임베딩하고, 의미 기반 검색을 수행하도록 설계되었으나 추가적인 인프라(Vector DB, Embedding 모델, 검색 기술 등)를 필요로 함<br>• **Skills는 선별된 지식과 방법론을 통해 LLM 모델이 전문가처럼 사고(Reasoning)하도록 가르치기 때문에 이를 RAG의 방대한 정보 검색 능력과 결합하면 상호 보완적인 강력한 Agentic Workflow 구현 가능** |
| **Skills v.s. Fine-tuning** | • Fine-tuning은 스타일, 형식, 도메인 어휘 학습을 위해 AI 모델의 가중치를 영구적으로 변경하여 특화하는 것으로, 비용이 많이 들고, 빈번한 지식 업데이트에 적합하지 않음<br>• **Skills는 모델 학습 없이 Markdown 파일 내 도메인 지식을 업데이트할 수 있으며 반복 개선이 가능함** |
| **Skills v.s. Tool Use / MCP** | • MCP를 통해 Agentic AI 시스템에 연결된 도구들은 AI 모델에게 추론(reasoning)을 위한 정보를 제공하게 됨<br>• **Skills는 제공된 정보를 바탕으로 어떻게 분석하고 판단할 지 가르치기 때문에 MCP와 상호보완적이라고 할 수 있음** |

- **Agentic AI Frameworks과 비교**

|  | LangChain / LangGraph | OpenAI AgentKit | Claude Skills |
| --- | --- | --- | --- |
| **Agent 구축 주체 및 방식** | **개발자**<br>• Python/JS 코드를 통해 직접 구현 **(코드)**<br>• 워크플로 변경 시마다 코드를 수정/배포해야 함 | **개발자·실무자**<br>• OpenAI Agent Builder라는 노코드/로우코드 Workflow 제작 도구에서 노드를 이어 다이어그램 형식으로 Agent 설계 **(시각 노드)**<br>• 필요 시 OpenAI Agents SDK로 확장 **(코드)** | **개발자·실무자**<br>• .claude/skills/ 폴더 내 Markdown 파일에 지침 작성 **(텍스트)** → **다이어그램의 시각적 직관성을 포기하는 대신, 자연어의 표현력과 유연성을 얻게 됨**<br>• 필요 시 Claude Agent SDK로 확장 **(코드)** |
| **Agent 구축 시 도구 연결** | • Python 라이브러리, LangChain 커넥터 및 MCP 기반 서버/툴을 통해 외부 API, DB, 내부 시스템을 코드 수준에서 직접 연결<br>• **컨텍스트 안에 툴에 대한 정의가 많이 들어가면 토큰 사용량이 커짐** | • MCP 기반 서버/툴 및 OpenAI Connector Registry라는 사전 정의된 툴 목록에서 외부 API, DB, 내부 시스템 연결<br>• **컨텍스트 안에 툴에 대한 정의가 많이 들어가면 토큰 사용량이 커짐** | • **별도의 MCP 세팅 필요 없으며, Skill 파일만 수정하면 Agent에 연결됨**<br>• **Skill 목록 중 선택적으로 로드·실행하여 토큰을 효율적으로 사용할 수 있음** |
| **AI 모델 / 생태계 호환성** | • 여러 모델(OpenAI, Anthropic, Google, 오픈소스 등)을 교체/조합할 수 있는 프레임워크 지향 | • 기본적으로 OpenAI 모델에 최적화된 에이전트/툴 생태계 (MCP를 통해 다른 모델 사용 가능)<br>• Workflow 제작, 배포·모니터링, 권한 관리 등이 모두 OpenAI 콘솔에 묶여 있어 운영 환경과 생태계가 OpenAI에 강하게 종속됨 | • **Vibe Coding 환경에서 동작하므로 Claude 뿐 아니라 OpenAI Codex, Gemini Code 등 다른 코드 모델 사용 가능**<br>• **Anthropic이 Claude Skills Cookbook과 Agent Skills 문서, 예제 Skill들을 오픈소스로 공개하여 업계 표준으로 자리 잡을 가능성 큼** |

> ### ⚠️ 2026년 9월 기준 업데이트
>
> - **업계 표준 예측은 실현되었습니다.** Anthropic은 2025년 12월 18일 Agent Skills를 오픈 표준으로 공개했고, 현재는 OpenAI Codex, Gemini CLI, GitHub Copilot, Cursor 등 여러 도구가 같은 형식의 Skill을 지원합니다.
> - 다만 이 글을 쓴 2025년 11월 시점에는 Skills가 Claude 제품에서만 동작했습니다. 표의 "Gemini Code"는 Gemini CLI를 가리킵니다.
> - AgentKit에서 MCP는 외부 도구를 연결하는 표준이며, 다른 AI 모델을 연결하는 수단은 아닙니다.

## 🔎 Claude Agent Skills Use-cases

- **Anthropic이 공개한 공식 Claude for Financial Services**
    - 2025년 7월 15일, Anthropic은 애널리스트, 포트폴리오 매니저 등 금융 전문가의 시장 조사 및 의사 결정을 돕는 Claude for Financial Services를 공개하며 금융 업계에서 뜨거운 호응을 얻었습니다. **불과 3개월 후 25년 10월, 이 서비스가 Claude Agent Skills라는 표준화된 프레임워크를 기반으로 한다는 사실이 밝혀지면서, 전문 영역에 특화된 Enterprise향 Pre-built Agent를 만드는 데 있어 Agent Skills의 효용성을 성공적으로 입증했습니다.**
    - Claude Agent Skills 프레임워크 자체는 누구나 새로운 Skills를 개발할 수 있도록 **오픈소스로 공개되어 생태계를 확장**하고 있습니다. 또한, **Anthropic이 공개한 Claude for Financial Services와 같이 기업이 즉시 비즈니스에 적용할 수 있는 핵심 기능들은 유료 플랜(Enterprise)으로만 제공**하였습니다. 이를 통해 **Anthropic은 Enterprise scene의 실질적인 니즈를 충족시킴과 동시에 안정적인 수익성을 확보하는 전략을 취했습니다.** 나아가, **기업들은 이 핵심 기능을 기반으로 자체 업무에 맞는 Customize된 기능을 추가 개발**할 수도 있습니다.
    - Claude for Financial Services는 Databricks, Snowflake 등 기업 내부 데이터 플랫폼은 물론, S&P Global, LSEG, Morningstar 등 주요 금융 데이터 제공업체와 MCP를 통해 연동할 수 있었고, 이를 통해 Claude AI 모델은 필요한 기업 업무를 위해 필요한 재무 데이터를 자동으로 수집하고, 분석 및 보고서 작성까지의 전문적인 업무를 수행할 수 있었습니다.
        - **또한, Deloitte, PwC와 같은 컨설팅 기업과의 파트너십을 통해 주식 리서치, 사모 신용 분석, 규제 준수 격차 분석 등 각 파트너사가 전문화한 특화 솔루션을 제공하기도 합니다.**

| **핵심 Skills** | **기능 및 사용 목적** |
| --- | --- |
| **Public/Private 데이터 비교 분석** | Peer Group 기업을 선정하고 주요 재무 지표 및 Valuation Multiples를 비교하는 보고서를 생성합니다. |
| **DCF(Discounted Cash Flow) 모델링** | 현재 데이터와 사용자의 가정을 기반으로 DCF 모델을 빠르게 구축하고 민감도 분석(Sensitivity Analysis)을 수행합니다. |
| **기업 Coverage 리서치** | 애널리스트가 작성하는 것과 유사한 심층적인 시장 및 기업 **Coverage Report** 초안을 작성합니다. |
| **기업 Profile 문서 생성** | 투자 유치를 위한 비즈니스 개요, 재무 하이라이트 등을 포함하는 초안 문서를 자동으로 생성합니다. |
| **Due diligence 데이터 정리** | M&A 실사(Due Diligence) 과정에서 필요한 기업 재무 및 운영 데이터를 신속하게 수집하고 구조화된 데이터 팩으로 정리합니다. |
| **실적 분석** | 최신 기업 실적 발표 자료를 분석하여 주요 성장 동인, 재무 건전성 및 경영진 코멘트를 요약합니다. |

> ### ⚠️ 2026년 9월 기준 업데이트
>
> 정확히는, Claude for Financial Services가 7월 출시 당시부터 Skills 기반이었던 것은 아닙니다. 2025년 10월 업데이트에서 Excel add-in, 신규 데이터 커넥터와 함께 위 표의 금융 업무용 Agent Skills가 **추가**되었습니다.

- **(What if?) Multi-agent 구조의 데이터 분석 Agent를 Claude Agent Skills로 재설계한다면**
    - 비정형 데이터(문서형 보고서)와 정형 데이터(DB)를 통합 분석하여 사용자에게 필요한 정보를 검색·제공하고 의사결정을 지원하는 Agentic AI 시스템을 예로 들어보겠습니다. 이러한 시스템은 사용자의 복잡한 요구사항을 분석하고 최적의 작업 계획을 수립하여 하위 기능을 조율하는 Super Agent를 중심으로 구성할 수 있습니다. 이 Super Agent는 MCP(Model Context Protocol) 표준을 통해 비정형 데이터 검색·분석을 위한 RAG Agent, 정형 데이터 검색·분석을 위한 DB Agent w/LLM, 그리고 시각화 및 정밀 연산 등의 기능을 Coding 할 수 있는 Agent 등 각각 특화된 Unit Agent들과 연결되어, 필요에 따라 유연하게 도구를 호출하고 통합적인 결과를 도출합니다.
    - 이러한 구조를 Claude Agent Skills를 기반으로 재설계한다면, **기존의 Super Agent가 하위 Agent/Unit에게 작업을 할당하는 방식에서 단일 AI 모델(e.g. Claude)이 명확하게 정의된 Skills를 직접 핸들링하는 방식으로 전환**할 수 있습니다. 이를 통해 실제로 사용하지 않는 도구 또는 Unit Agent를 상시 로드하여 소모되던 과도한 토큰 사용량을 획기적으로 줄일 수 있으며, MCP와 같은 복잡한 프로토콜 구현 없이 손쉽게 새로운 기능을 추가할 수 있을 것으로 파악됩니다.
    - RAG Agent, DB Agent가 제공하는 세부 기능들을 Skills를 기반으로 정의해보겠습니다.

> ### 💡 (예시) RAG Agent와 DB Agent의 기능을 Skills로 정의하기
>
> - **(예시) 비정형 데이터 검색·분석을 위한 RAG Skill**
>     - **문서 내 검색 skill**: 사용자 요청과 관련성이 높은 chunk를 반환하기 위해 Search 전략 가이드, Chunking 가이드 등 참조용 Markdown 파일과 Chunking, Search를 수행하는 Script 파일을 준비하고, SKILL.md 파일 내 RAG 로직을 구현합니다.
>     - **답변 생성 skill**: 검색 결과를 참고하여 요청에 맞는 답변 생성하기 위해 SKILL.md 파일 내 단순 답변 생성/비교 분석 답변/인사이트 도출 등 가이드를 작성합니다.
> - **(예시) 정형 데이터 검색·분석을 위한 DB Skill**
>     - **DB 파악 skill**: 테이블 스키마, 컬럼 고유값 등 DB 핵심 정보 파악을 위해 스키마 분석, 테이블 관계 탐지, 컬럼 분석을 수행하는 Script 파일을 준비하고, SKILL.md 파일 내 로직을 구현합니다.
>     - **Schema 연결 skill**: 파악된 DB 정보를 활용하여 사용자 질문과 관련성이 높은 항목 선별을 위해 SKILL.md 파일 내 Entity 추출, 테이블 선택 등 주요 기능을 정의합니다.
>     - **SQL문 작성·실행 skill**: 테이블 조회 SQL, 수치 추출 SQL 등 쿼리 생성을 위해 SQL 패턴, 자주 사용되는 쿼리 등 참조용 Markdown 파일과 SQL 검증, 쿼리 실행을 수행하는 Script 파일을 준비하고, SKILL.md 파일 내 로직을 구현합니다.

## 📢 (시사점) Claude Agent Skills Key Takeaways: “AI-Native OS”

- Claude Agent Skills는 **복잡한 MCP 프로토콜 대신 구조화된 형태의 텍스트로 표현할 수 있는 단순한 Markdown 파일로 AI Agent를 개발할 수 있게 하여 실무자도 직접 Agent를 만들 수 있도록 하는 도구**입니다. 필요한 Skill만 선택적으로 로드하여 토큰 효율성을 높인다는 장점을 강조하고 있지만, 실제로는 AI의 추론과 의사결정 단계가 세분화되어 API 호출 빈도가 증가하는 수익 구조를 유도한다고 할 수 있습니다.
- 특히, Anthropic은 skills 표준을 오픈소스로 공개하여 진입 장벽을 낮추면서도, **Claude에서 가장 최적화되어 작동하도록 설계하였고, 작성된 skills를 자유롭게 공개할 수 있도록 하여 생태계를 확장함과 동시에 자연스럽게 Claude API 사용을 유도하는 사실상의 Agent/Agentic AI 시장의 표준화 전략을 취하고 있습니다.**
    - **기술과 사용법을 오픈소스로 공개하지만, 궁극적으로 기업들이 전문적인 업무를 수행하는 pre-built agent를 구축하고자 할 때 실무자들이 직접 수백 개의 Skills를 Claude 인프라 위에서 제작하고 업무에 통합하게 되면서 Claude 생태계에 종속(Lock-in)되도록 하여, 장기적으로는 AI-Native OS를 선점하려는 움직임으로 볼 수 있습니다.**

> ### 💡 Andrej Karpathy의 LLM OS
>
> Andrej Karpathy(전 Tesla AI Director 및 OpenAI 공동 창업자)는 LLM을 단순한 AI 모델이 아닌, 컴퓨터의 중앙처리장치(CPU)이자 운영체제(OS)의 커널로 재정의했습니다. 그는 **LLM이 메모리(Context Window)와 저장 장치(DB), 그리고 다양한 도구(Tool)들을 유기적으로 제어하고 조율하는 하나의 컴퓨팅 시스템, 즉 LLM OS를 구축하는 것이 AGI로 나아가는 핵심 경로라고 강조**했습니다.
> (출처: [https://analyticsindiamag.com/ai-news-updates/andrej-karpathy-says-the-pathway-to-agi-is-through-a-language-model-operating-system/](https://analyticsindiamag.com/ai-news-updates/andrej-karpathy-says-the-pathway-to-agi-is-through-a-language-model-operating-system/))
>
> 즉, 2년 전부터 언급이 되기 시작한, Windows OS/Mac OS와 달리 LLM이 중심이 되는 새로운 개념의 OS가 Anthropic을 통해서 보다 빠르게 현실화 되어 가는 것 같습니다.
>
> 과거부터 현재까지 다양한 소프트웨어들이 OS 종류에도 같은 기능이 동작될 수 있도록 개발/배포 되었고, 또는 클라우드환경에서는 각자 만의 소프트웨어들이 제공되었습니다.
>
> **앞으로의 새로운 개념인 LLM OS 시대에는, LLM OS는 대규모 인프라 위에서 동작되며 LLM OS 종류에 따른 경계 없이도 동작될 수 있는 Agent를 제공하게 되는 서비스/솔루션 기업들이 존재하게 될 것으로 판단됩니다.**

- Anthropic은 이미 skills 기반의 Enterprise향 Vertical Solution인 Claude for Financial Services를 통해 데이터 보안과 정확도가 필수적인 금융 도메인에서 그 가치를 증명했습니다. 이는 향후 법률, 의료 등 고부가가치 영역으로 확장 시, **기업들이 단순 API 사용을 넘어 전문성 있는 AI Agent 자체를 유료로 구독하게 만드는 비즈니스 모델을 제시한 것으로 해석됩니다.**
- **최근 OpenAI의 AgentKit, Anthropic의 Claude Agent Skills 등의 사례는 Big Tech 기업들이 단순 LLM 제공을 넘어 AI-Native OS 생태계로 확장하고 있음을 시사합니다.**
    - Anthropic을 비롯한 주요 LLM Provider들이 AI 인프라 레이어(AI Infra Layer) 기업으로 빠르게 전환하고 있으며, 예상보다 빠른 속도로 엔터프라이즈 시장까지 진입하고 있습니다.
