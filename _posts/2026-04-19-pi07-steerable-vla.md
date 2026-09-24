---
layout: paper
lang: ko
ref: pi07-steerable-vla
kind: paper-review
title: "π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities (π0.7)"
date: 2026-04-19 17:00:00 -0700
paper_date: 2026-04-16
venue: "arXiv preprint · arXiv:2604.15483"
tags: [VLA, Robot-Foundation-Model, Context-Conditioning, World-Model, Cross-Embodiment, Paper-Review]
authors: "Physical Intelligence — Bo Ai, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Greg Balke, Kevin Black, George Bokinsky, Shihao Cao, Thomas Charbonnier, Vedant Choudhary, Foster Collins, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Maitrayee Dhaka, Jared DiCarlo, Danny Driess, Michael Equi, Adnan Esmail, Yunhao Fang, Chelsea Finn, Catherine Glossop, Thomas Godden, Ivan Goryachev, Lachlan Groom, Haroun Habeeb, Hunter Hancock, Karol Hausman, Gashon Hussein, Victor Hwang, Brian Ichter, Connor Jacobsen, Szymon Jakubczak, Rowan Jen, Tim Jones, Gregg Kammerer, Ben Katz, Liyiming Ke, Mairbek Khadikov, Chandra Kuchi, Marinda Lamb, Devin LeBlanc, Brendon LeCount, Sergey Levine, Xinyu Li, Adrian Li-Bell, Vladislav Lialin, Zhonglin Liang, Wallace Lim, Yao Lu, Enyu Luo, Vishnu Mano, Nandan Marwaha, Aikys Mongush, Liam Murphy, Suraj Nair, Tyler Patterson, Karl Pertsch, Allen Z. Ren, Gavin Schelske, Charvi Sharma, Baifeng Shi, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, Will Stoeckle, Jiaming Tang, Jimmy Tanner, Shalom Tekeste, Marcel Torne, Kyle Vedder, Quan Vuong, Anna Walling, Haohuan Wang, Jason Wang, XuDong Wang, Chris Whalen, Samuel Whitmore, Blake Williams, Charles Xu, Sukwon Yoo, Lili Yu, Wuming Zhang, Zhuoyang Zhang, Ury Zhilinsky"
affiliations: "Physical Intelligence"
summary: "거르지 말고 설명하라 — 각 에피소드에 품질·속도·실수·제어 모드·subgoal 이미지라는 \"어떻게\"를 context로 붙이면, 실패와 RL 롤아웃까지 흡수한 하나의 generalist VLA가 specialist급 숙련과 조합적 일반화를 함께 보인다."
paper_url: "https://arxiv.org/abs/2604.15483"
---

> **핵심 주장** — 이질적인 로봇 데이터를 쓰는 데 필요한 것은 더 엄격한 필터링이 아니라 **더 풍부한 context**다. 각 에피소드에 "무엇을"뿐 아니라 "어떻게"(품질·속도·실수·제어 모드·목표 이미지)를 붙여 조건화하면 실패 에피소드와 RL 롤아웃까지 버리지 않고 한 모델에 흡수할 수 있다. 추론 때 그 context를 지정하면 하나의 generalist가 specialist급 숙련과 조합적 일반화를 함께 보인다.

---

## 들어가며

LLM은 번역을 알고 JSON 출력을 알면, 학습 데이터에 그 조합이 없어도 번역 결과를 JSON으로 내놓는다. 이런 **조합적 일반화(compositional generalization)** 는 foundation model을 generalist라고 부르는 근거다. 그런데 로봇 파운데이션 모델, 즉 VLA는 아직 이 능력을 보여주지 못했다.

π0.7은 Physical Intelligence가 그 "첫 징후"를 주장하며 내놓은 5B 규모의 VLA다. 새 아키텍처를 내세운 논문은 아니다. 논문은 스스로의 기여를 "VLA가 더 다양한 데이터를 쓰게 하는 방법론과 그 실증 분석"으로 규정한다. 핵심은 하나다. 프롬프트에 "무엇을"뿐 아니라 "어떻게"를 넣는다.

이 글은 논문의 논지를 따라가되, LLM·diffusion 배경의 독자를 가정하고 필요한 로봇 학습 개념을 그때그때 쌓는다. diffusion을 아는 독자라면 이 논문의 설계 상당수가 text-to-image에서 이미 본 장치라는 것을 알아챌 것이다. 그 다리를 곳곳에 놓는다.

---

## 1. 문제 — generalist VLA는 왜 아직 generalist가 아닌가

### 1.1 두 가지 증상

논문은 기존 VLA의 한계를 두 증상으로 진단한다.

**⓵ 조합 불가** — 학습한 skill을 재조합해 새 task를 풀지 못한다. 처음 보는 주방 기기를 다루거나 새 도구를 쓰는 일이 여기에 해당한다.

**⓶ specialist 의존** — 학습한 task조차 task별 fine-tuning 없이는 유창하게 수행하지 못한다. 실제로 가장 잘 동작하는 정책은 generalist 사전학습 위에 task별로 따로 다듬은 specialist다(RL로 다듬은 π\*0.6, task별 SFT를 한 π0.6). PI 블로그는 이를 초기 언어모델(BERT)이 도메인마다 fine-tune되던 시절에 비유한다.

### 1.2 처방은 더 넓은 데이터 — 그리고 두 개의 벽

foundation model의 원리대로라면 처방은 분명하다. 더 넓고 다양한 데이터다. 로봇에서 "다양한 데이터"란 다음을 뜻한다.

| 데이터 | 다양성의 원천 |
|---|---|
| 여러 로봇의 시연 | 형태·자유도·제어 방식이 다르다 |
| 여러 조작자의 시연 | 같은 task에도 전략이 다르다 |
| 자율 실행 기록 | 이전 정책의 평가 롤아웃이며, 실패를 포함한다 |
| RL 에이전트 롤아웃 | 학습 과정의 좋은 시도와 나쁜 시도가 섞여 있다 |
| 사람 1인칭 영상, 웹 데이터 | 로봇 행동 라벨이 없다 |

그런데 이것을 그냥 섞으면 두 벽에 부딪힌다.

**⛔ 벽 1 — 다양성은 곧 모호성이다.** 같은 관측, 같은 지시("셔츠를 개라")에 빠르고 깔끔한 에피소드, 느리고 재파지가 잦은 에피소드, 실패한 에피소드가 모두 대응한다. 논문의 표현으로 순진하게 학습한 모델은 이 모드들을 "평균내고", 결과는 준최적이 된다. 그래서 관행은 필터링이다. 고품질·일관된 전략의 시연만 골라 쓴다. 논문은 이 필터링이 노동집약적이고, task마다 기준이 다르며, 가치 있는 정보를 버린다고 지적한다. 다양성을 얻으려고 모은 데이터를 다양성을 깎아서 쓰는 셈이다.

**⛔ 벽 2 — 언어로는 "어떻게"를 다 적을 수 없다.** 모호성을 풀려면 데이터에 설명을 붙이면 된다. 이미지·비디오 생성 모델이 prompt expansion으로 캡션을 풍부하게 해 품질을 올리는 것과 같은 발상이다. 그러나 논문은 로봇에서는 텍스트 캡션만으로 부족하다고 말한다. 성패를 가르는 정보가 ⓐ 에피소드 전반의 품질처럼 미묘하거나, ⓑ 깔끔하게 접힌 티셔츠의 생김새나 냉장고 손잡이를 잡는 방식처럼 말로 표현하기 어렵기 때문이다.

벽 1의 정체는 §2.3에서 해부하고 §3.3에서 뚫는다. 벽 2를 뚫는 장치는 §3.4에서 본다.

### 1.3 그래서 논문이 던지는 질문

> "무엇을" 하는지에 더해 "어떻게" 했는지를 context로 넣으면, 이질적이고 품질이 낮은 데이터까지 버리지 않고 학습할 수 있는가? 그렇게 넓힌 데이터가 fine-tune 없는 specialist급 성능과 조합적 일반화로 이어지는가?

---

## 2. 배경 — 두 벽을 읽기 위한 최소한

### 2.1 Flow 기반 VLA의 학습 목적

VLA는 사전학습된 VLM backbone에서 출발해 로봇 제어로 적응시킨 모델이다. 학습 목적은 다음과 같다.

$$\max_\theta\ \mathbb{E}_{\mathcal{D}}\big[\log \pi_\theta(\mathbf{a}_{t:t+H}\mid \mathbf{o}_{t-T:t},\ \mathcal{C}_t)\big]$$

| 기호 | 의미 |
|---|---|
| $\theta$ | VLA 파라미터 |
| $\mathcal{D}$ | 학습 궤적 데이터셋 |
| $\mathbf{a}_{t:t+H}$ | action chunk. 앞으로 $H$ 스텝의 행동이며 π0.7은 $H=50$. 실제로는 앞의 $\hat H\in\{15,25\}$ 스텝만 실행한다 |
| $\mathbf{o}_{t-T:t}$ | 최근 관측 이력. $\mathbf{o}_t=[\mathbf{I}^1_t,\dots,\mathbf{I}^n_t,\mathbf{q}_t]$는 카메라 이미지 $n$장과 관절 상태 $\mathbf{q}_t$ |
| $\mathcal{C}_t$ | prompt(context). 행동을 조건화하는 부가 정보 전체 |

- action expert는 VLM backbone의 activation을 attend하는 작은 transformer로, flow matching으로 action chunk를 생성한다. flow matching은 closed-form log-likelihood 대신 그 근사 하한을 최적화한다.
- 기존 모델(π0, π0.5, π0.6)의 context는 짧은 task 설명 한 줄, 즉 $\mathcal{C}_t=(\ell_t)$였다.

이 목적식에서 π0.7이 바꾸는 것은 **$\mathcal{C}_t$ 하나뿐**이다. 그 안에 무엇을 넣는지가 §3의 전부다.

### 2.2 Knowledge Insulation — backbone과 action expert의 분업

π0.7은 Knowledge Insulation(KI) 레시피를 따른다. 논문의 서술을 식으로 옮기면 다음과 같다(표기는 필자).

$$\mathcal{L}\ =\ \underbrace{\mathcal{L}_{\text{CE}}\big(\text{FAST tokens}\mid h_{\text{VLM}}\big)}_{\text{backbone}}\ +\ \underbrace{\mathcal{L}_{\text{FM}}\big(\mathbf{a}_{t:t+H}\mid \mathrm{sg}[h_{\text{VLM}}]\big)}_{\text{action expert}}$$

- $h_{\text{VLM}}$ — VLM backbone의 activation
- FAST tokens — action chunk를 이산 토큰으로 압축한 것
- $\mathcal{L}_{\text{CE}}$ — 이산 토큰에 대한 cross-entropy
- $\mathcal{L}_{\text{FM}}$ — 연속 행동에 대한 flow matching 손실
- $\mathrm{sg}[\cdot]$ — stop-gradient. action expert는 backbone activation을 attend하되 gradient를 되돌려 보내지 않는다

오해하기 쉬운 지점이 있다. KI에서 backbone은 얼어 있지 않다. FAST 토큰 cross-entropy로 행동을 계속 배운다. 차단되는 것은 action expert의 연속 flow 손실이 backbone으로 흘러가는 gradient 경로뿐이다. 논문이 드는 이유는 backbone을 비교적 안정적인 이산 cross-entropy로 학습시키려는 것이다. LLM 관점에서 보면 backbone은 next-token 예측을 하는 LLM이고, action expert는 그 표현을 읽어 연속 행동을 뽑아내는 diffusion decoder다.

### 2.3 벽 1의 정체 — 모드 평균화

벽 1을 그림으로 그리면 이렇다.

```
 same (o, l = "fold the shirt")
   episode A : fast, clean         -> a^A
   episode B : slow, 2 regrasps    -> a^B
   episode C : failed grasp        -> a^C

 no m   : learn mixture of A/B/C, sample in data proportion
 with m : a ~ p(a | o, l, m* = {Quality 5, Mistake false, Speed 2000})
```

- 같은 관측과 같은 지시에 세 가지 수행 방식이 섞여 있다
- `m`은 그 수행 방식을 가리키는 변수다 (§3.3의 episode metadata)

diffusion을 아는 독자라면 여기서 의문이 든다. flow matching은 원래 multimodal 분포를 표현하려고 쓰는 생성 모델이다. MSE 회귀처럼 모드의 평균을 내지는 않을 텐데, 왜 "평균화"가 문제인가?

> ### 💡 문제는 '평균'이 아니라 '혼합에서 뽑는 것'이다
>
> 데이터가 만드는 조건부 분포를, 수행 방식을 뜻하는 잠재변수 $m$으로 분해해 보자.
>
> $$p_{\mathcal{D}}(\mathbf{a}\mid\mathbf{o},\ell)\ =\ \sum_{m}\ p_{\mathcal{D}}(m\mid\mathbf{o},\ell)\ p_{\mathcal{D}}(\mathbf{a}\mid\mathbf{o},\ell,m)$$
>
> - $m$ — 전략·품질·속도 같은 수행 방식. 데이터에 기록되지 않은 잠재변수
> - $p_{\mathcal{D}}(m\mid\mathbf{o},\ell)$ — 데이터 안에서 각 방식이 차지하는 비중
> - $p_{\mathcal{D}}(\mathbf{a}\mid\mathbf{o},\ell,m)$ — 방식 $m$일 때의 행동 분포
>
> flow 모델이 완벽하더라도 학습하는 것은 좌변, 즉 혼합분포다. 여기서 세 가지 문제가 생긴다.
>
> - **⓵ 빈도대로 뽑는다** — 데이터의 30%가 굼뜬 시연이면 chunk의 30%가 그 모드에서 나온다. 실패 데이터를 넣으면 실패도 그 비율로 재현된다.
> - **⓶ chunk마다 모드가 바뀐다** — $\hat H$ 스텝마다 새로 샘플하므로, A 전략으로 시작했다가 B 전략으로 갈아타는 비일관적인 궤적이 나올 수 있다.
> - **⓷ 실제로는 보간도 생긴다** — 용량이 유한하고 denoising step이 적으면(π0.7은 5 step) 모드 사이의 어정쩡한 행동이 실제로 나온다.
>
> 근본 원인은 $m$이 관측 $\mathbf{o}$로부터 추론되지 않는 **교란변수**라는 데 있다. 누가 조작했는지, 어느 평가 런이었는지는 카메라 이미지에 보이지 않는다. 그렇다면 해법은 $m$을 관측변수로 바꿔 우변을 직접 학습하고, 추론 때 원하는 $m$을 고르는 것이다. 그것이 §3.3의 episode metadata다.
>
> 논문은 "모드를 평균낸다"고만 쓴다. 위의 분해와 세 가지 경로는 필자의 해석이다.

---

## 3. 방법 — context를 "무엇을 + 어떻게"로

### 3.1 한눈에 보기

π0.7은 context를 다섯 성분으로 넓힌다.

$$\mathcal{C}_t=(\ell_t)\quad\longrightarrow\quad \mathcal{C}_t=\{\ell,\ \hat\ell_t,\ \mathbf{g}_t,\ m,\ c\}$$

| 성분 | 내용 | 학습 시 출처 | 추론 시 출처 | 겨냥하는 벽 |
|---|---|---|---|---|
| $\ell$ | 전체 task ("clean the kitchen") | 사람 주석 | 사용자 | — |
| $\hat\ell_t$ | 다음 subtask ("open the fridge door") | 세그먼트 단위 주석 | high-level policy 또는 사람의 coaching | 벽 2 (일부) |
| $\mathbf{g}_t$ | multi-view subgoal 이미지 | 실제 미래 프레임 + world model 생성 이미지 | world model | 벽 2 |
| $m$ | episode metadata (speed, quality, mistake) | 측정값 + 사람 주석 | "최선" 값으로 고정 + CFG | 벽 1 |
| $c$ | 제어 모드 (`joint` / `ee`) | 데이터의 제어 방식 | task별로 선택 | 벽 1 (제어 방식의 이질성) |

프롬프트는 대략 다음 순서로 이어 붙는다(논문 §V-E 예시의 구조).

```
<multi-view obs><multi-view subgoals> Task: ... Subtask: ... Speed: ... Quality: ... Mistake: ... Control Mode: ... <proprio>
```

학습과 추론의 흐름을 한 장으로 그리면 다음과 같다.

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

- 학습 때는 context의 각 성분을 무작위로 떨어뜨리며 학습한다 (§3.5)
- 추론 때 `l^`은 high-level policy가, `g*`는 world model이 만들고, `m*`은 "최선" 값으로 고정한다 (§3.7)

**학습 데이터 구성** (논문 §VI-A)

| 데이터 | 비고 |
|---|---|
| 로봇 시연 | 정적·이동형, 단일팔·양팔 로봇 / 실험실형·가정형 환경과 실제 가정 |
| 자율 평가 데이터 | 이전 모델들의 정책 평가 롤아웃. π\*0.6의 RL 학습 중 수집된 데이터와 실패 에피소드를 포함한다 |
| 사람 개입 | 정책 롤아웃 도중의 사람 intervention |
| 오픈소스 로봇 데이터, 사람 1인칭 영상 | — |
| 웹 멀티모달 데이터 | 객체 위치·속성 예측, VQA, text-only 예측, 로봇·웹 영상 캡셔닝 |

일반화 평가용 task에서 나온 자율 데이터는 학습에서 제외했다(논문 각주). 평가 누수를 막는 장치다.

### 3.2 Subtask 지시 $\hat\ell$ — 말로 가르치는 통로

전체 task 설명 $\ell$("주방을 치워라") 옆에 다음 단계의 subtask $\hat\ell_t$("냉장고 문을 열어라")를 함께 넣는다. 이 구조 자체는 π0.5에서 가져왔다. 달라진 것은 역할이다. 저수준 정책이 세밀한 언어를 잘 따르게 되면, $\hat\ell$을 누가 채우느냐에 따라 두 가지 사용법이 생긴다.

| $\hat\ell$을 채우는 주체 | 사용법 |
|---|---|
| 학습된 high-level policy | 장기 task를 자율 수행한다 |
| 사람 | 처음 보는 task를 단계별로 불러주는 **coaching** |

coaching 기록은 그 자체로 학습 데이터가 된다. 기록을 모아 π0.7을 high-level policy로 fine-tune한다(표기는 필자).

$$\hat\ell_{k+1}\ \sim\ \pi^{\text{HL}}_\phi\big(\cdot\mid \mathbf{o}_t,\ \ell,\ \hat\ell_{1:k}\big)$$

- $\pi^{\text{HL}}_\phi$ — high-level policy. π0.7과 같은 아키텍처(Gemma3 4B 기반)에서 fine-tune한다
- $\mathbf{o}_t$ — 현재 로봇 관측
- $\hat\ell_{1:k}$ — 지금까지 내린 subtask 지시의 이력
- $\hat\ell_{k+1}$ — 다음 subtask 지시

결과적으로 teleop 같은 행동 수준 데이터 없이 새 task를 자율화할 수 있다. 그 결과는 §5.4에서 본다.

### 3.3 Episode metadata $m$ — 벽 1을 뚫는 장치

§2.3에서 본 대로 벽 1의 원인은 수행 방식 $m$이 관측되지 않는 데 있다. π0.7은 이것을 직접 라벨로 붙인다.

| 항목 | 정의 | 라벨 출처 |
|---|---|---|
| Speed | 에피소드 길이(timestep). 500 스텝 단위로 이산화한다 (1750–2250 → "2000") | 측정값 |
| Quality | 수행 품질 1–5점 (5가 최고) | 사람 주석 |
| Mistake | 해당 **행동 세그먼트**에서 실수가 있었는지 (파지 실패, 잘못된 subtask 수행 등) | 사람의 거친 주석 |

두 가지 디테일이 중요하다.

- **mistake는 세그먼트 단위다.** 전체적으로 성공했지만 중간에 한 번 파지에 실패한 에피소드라면, 실패한 구간만 따로 표시할 수 있다.
- **speed는 품질과 상관된다.** 논문은 빠른 에피소드가 대개 실수도 적다고 적는다.

추론 때는 항상 "최선"을 요청한다.

| 항목 | 추론 시 값 |
|---|---|
| Speed | task별 에피소드 길이의 15 percentile |
| Quality | 5 |
| Mistake | false |

그런데 에피소드 길이는 에피소드가 끝나야 알 수 있는 값이다. 행동을 고르는 시점에는 존재하지 않는 미래의 통계를 어떻게 조건으로 쓸 수 있는가?

> ### 💡 사후에만 아는 값도 조건이 된다 — return-conditioning과 micro-conditioning의 로봇판
>
> 학습 때는 끝난 에피소드에 사후(hindsight) 라벨을 붙이고, 모델은 "이런 결과가 나온 에피소드의 행동은 이렇게 생겼다"를 배운다. 추론 때는 원하는 결과를 먼저 선언한다.
>
> $$\text{train: }\ p_\theta(\mathbf{a}\mid\mathbf{o},\ell,m),\ \ m=f(\tau)\qquad\quad \text{test: }\ \mathbf{a}\sim p_\theta(\mathbf{a}\mid\mathbf{o},\ell,m^\star)$$
>
> - $\tau$ — 에피소드 전체 궤적
> - $f$ — 궤적의 사후 통계 (길이, 품질 점수, 실수 여부)
> - $m^\star$ — 추론 때 요청하는 값
>
> 같은 발상은 다른 분야에 이미 있다.
>
> | 분야 | 사후 조건 | 추론 시 요청 |
> |---|---|---|
> | RL (Decision Transformer 계열의 return-conditioning) | 궤적의 return | 높은 return |
> | 이미지 생성 (SDXL micro-conditioning) | 원본 해상도·crop 좌표 | 고해상도·crop 없음 |
> | π0.7 | 길이·품질·실수 | 빠르게·5점·실수 없음 |
>
> SDXL이 저해상도 학습 이미지를 버리는 대신 "원본 크기"를 조건으로 넣어 활용한 것과 같은 태도다. **버리지 말고, 설명해서 넣는다.**
>
> 최소 길이가 아니라 15 percentile을 요청하는 이유는 논문에 나오지 않는다. 분포 꼬리의 조건값은 학습 support가 얇아 조건 자체가 OOD가 되므로, 빠르면서도 충분히 관측된 구간을 요청하는 것으로 읽힌다(필자 해석).

### 3.4 Subgoal 이미지 $\mathbf{g}$ — 벽 2를 뚫는 장치

"냉장고 문을 열어라"라는 지시는 손잡이를 어떻게 잡을지까지 정해주지 않는다. subgoal 이미지는 가까운 미래에 장면이 어떻게 보여야 하는지를 그림으로 명시한다.

$$\mathbf{g}_t=[G^1_t,\dots,G^n_t]$$

- $G^i_t$ — 카메라 $i$에서 본 가까운 미래의 목표 이미지
- multi-view로 주는 이유 — base view는 환경과 물체의 결과를, wrist view는 팔과 그리퍼의 결과를 명시하기 쉽다

**World model — subgoal을 그리는 모델**

추론 때 subgoal은 별도의 world model $g_\psi$가 생성한다. 입력은 π0.7과 같은 subtask 지시다.

$$\max_\psi\ \mathbb{E}_{\mathcal{D}_g}\Big[\mathcal{L}_{\text{CFM}}\big(\mathbf{g}^\star_t,\ g_\psi(\mathbf{o}_t,\hat\ell_t,m)\big)\Big],\qquad \mathbf{g}^\star_t=\mathbf{o}_{t_{\text{end}}}$$

| 기호 | 의미 |
|---|---|
| $\psi$ | world model 파라미터. BAGEL(이미지 이해·편집·생성이 가능한 14B mixture-of-transformers)에서 초기화 |
| $\mathcal{D}_g$ | subtask 라벨 품질이 특히 좋은 세그먼트의 부분집합 |
| $\mathbf{g}^\star_t$ | 정답 subgoal. 현재 세그먼트의 마지막 프레임 $\mathbf{o}_{t_{\text{end}}}$ (3 view) |
| $m$ | episode metadata. world model도 품질·속도 조건을 받는다 |
| $\mathcal{L}_{\text{CFM}}$ | 표준 conditional flow matching 손실 |

> ⚠️ 원문은 손실을 max로 표기했는데, min의 오기로 보인다.

핵심은 SuSIE의 계보를 따라 웹 사전학습된 이미지 편집 모델에서 출발한다는 점이다. 로봇 데이터에 웹 데이터, 사람 1인칭 영상, 기타 비디오를 섞어 학습시켜, 거기서 얻은 의미적·물리적 개념을 subgoal 이미지라는 형태로 π0.7에 전달한다. 부록 C에 따르면 BAGEL 구조를 따라 입력을 ViT(의미 이해)와 VAE(세부 정보) 두 경로로 인코딩하며, 라벨의 시간 분할 품질이 subgoal 품질을 크게 좌우한다.

논문은 이 모델을 일관되게 "lightweight world model"이라고 부른다. 실제로 얼마나 가벼운가?

> ### ⚠️ '경량' world model은 14B다
>
> BAGEL 기반 world model은 이해용 7B와 생성용 7B backbone으로 된 14B 모델이다. subgoal 한 세트를 25 denoising step으로 만드는 데 4×H100 텐서 병렬, 8-bit 양자화, 수정된 SageAttention을 모두 동원하고도 1.25초가 걸린다(부록 D). 단일 H100에서 38–127ms로 도는 π0.7 본체보다 훨씬 무겁다. "경량"은 비디오 생성 모델 대비의 상대적 표현으로 읽는 편이 맞아 보이며(필자 해석), 실시간성은 비동기 실행(§3.7)으로 확보된다.

**π0.7에 subgoal을 넣는 방식**

| 설정 | 값 |
|---|---|
| subgoal 포함 비율 | 배치 샘플의 25% |
| 실제 미래 프레임 선택 | p=0.25로 세그먼트 끝 프레임(world model의 예측 타깃과 일치), p=0.75로 현재부터 0–4초 뒤 사이에서 균등 샘플 |
| 생성 이미지 | world model로 subgoal을 대량 생성해 실제 프레임 대신 넣은 샘플을 추가한다. 실제 이미지와 생성 이미지 사이의 train–test 불일치를 줄이기 위함이다 |
| subtask 제거 | subgoal이 있는 샘플 중 30%에서 $\hat\ell$을 지운다 |

subgoal이 얼마나 기여하는지는 π0.7과 π0.7 (GC)(world model subgoal로 프롬프트한 설정)의 비교로 §5.2와 §5.3에서 본다.

그런데 논문에 따르면 subgoal을 주면 모델이 눈에 띄게 빨리 학습된다. 그렇다면 모든 샘플에 넣지 않고 25%로 제한하는 이유는 무엇인가?

> ### 💡 subgoal은 지름길이라서 아껴 쓴다
>
> 논문의 설명대로 subgoal이 주어지면 행동 예측은 사실상 inverse dynamics 문제가 된다.
>
> $$\mathbf{a}_{t:t+H}\ \approx\ f_{\text{ID}}(\mathbf{o}_t,\ \mathbf{g}_t)$$
>
> - $f_{\text{ID}}$ — 현재 프레임과 목표 프레임 사이의 행동을 추론하는 역동역학 함수
>
> 쉬운 문제는 빨리 풀리고, 바로 그래서 위험하다. subgoal이 항상 있으면 언어와 metadata를 읽고 "다음에 세계가 어떻게 되어야 하는지"를 스스로 추론하는 경로가 학습 신호를 덜 받는다. 정답 힌트를 늘 받는 학생이 문제를 읽지 않게 되는 것과 같다. 25%는 두 경로를 모두 살려두는 배합이다. 논문은 "빨리 학습되므로 25%만 넣는다"까지만 쓰며, shortcut 해석은 필자의 추론이다.
>
> subgoal이 있는 샘플의 30%에서 $\hat\ell$을 지우는 것은 반대 방향의 설계다. 이번에는 이미지만 보고 의도를 읽도록 강제한다. §5.3의 UR5e 셔츠 개기에서 언어 없이 subgoal과 metadata만으로 프롬프트하는 사용법이 이 설계에 기댄다.

### 3.5 Control mode와 dropout 설계

행동의 좌표계도 context에 넣는다. 관절 제어와 말단(end-effector) 제어 데이터를 함께 학습하고, 텍스트 식별자 $c\in\{\texttt{joint},\texttt{ee}\}$로 구분한다. 추론 때는 task에 맞게 고른다. end-effector 명령은 수치 역기구학(IK)으로 관절 목표로 바꿔 PD 제어기에 넘긴다.

모든 성분은 학습 때 무작위로 떨어뜨린다. control mode만은 예외다.

| 성분 | 학습 시 처리 |
|---|---|
| subgoal 이미지 | 배치 샘플의 25%에만 포함 |
| subtask $\hat\ell$ | subgoal이 있는 샘플 중 30%에서 제거 |
| metadata 전체 | 15% 확률로 통째로 제거 |
| metadata 항목별 | speed·quality·mistake 각각 5% 확률로 제거 |
| control mode | 제거하지 않음 |
| history 프레임 | 30% 확률로 통째로 제거 |
| 후면 카메라 이미지 | 30% 확률로 제거 |

control mode를 떨어뜨리지 않는 이유는 논문에 나오지 않는다. 행동의 좌표계 자체가 바뀌는 정보이므로, 모르는 채로 샘플하면 의미 없는 행동이 나오기 때문일 것이다(필자 해석).

dropout의 효과는 두 가지다.

- **⓵ 유연성** — 추론 때 어떤 부분집합으로든 프롬프트할 수 있다. subgoal 없이 언어만으로, 또는 언어 없이 subgoal만으로도 동작한다.
- **⓶ 조건부와 무조건부의 동시 학습** — 한 모델이 $p(\mathbf{a}\mid\mathbf{o},\mathcal{C})$와 일부 성분을 뺀 $p(\mathbf{a}\mid\mathbf{o},\mathcal{C}^{\text{uncond}})$를 함께 배운다. text-to-image의 null-prompt dropout과 같은 장치이며, 이것이 §3.7의 CFG를 가능하게 한다.

### 3.6 아키텍처 — π0.6-MEM 위에 context를 얹다

π0.7은 π0.6의 VLA 구조와 MEM 메모리 시스템 위에 multimodal context 조건화를 더한 모델이다. 총 약 5B 파라미터다.

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

- **backbone** — Gemma3 4B VLM(400M vision encoder 포함)에서 초기화한다
- **관측** — 최대 4대 카메라(정면, 양 손목, 선택적으로 후면), 카메라당 최대 6개의 history 프레임(1초 간격). MEM 방식 인코더가 시공간 압축으로 이력을 단일 프레임 분량의 토큰으로 줄인다. 입력 해상도는 448×448이다
- **subgoal** — 최대 3장(후면 제외)이며 관측과 같은 인코더로 처리한다
- **attention** — block-causal 구조다. 관측 블록과 subgoal 블록은 각자 내부에서 양방향이고, subgoal은 관측을 참조하며, 텍스트는 causal이다
- **proprio** — π0.6처럼 이산 텍스트 토큰으로 넣지 않고 선형 투영으로 임베딩한다. history state마다 토큰 하나다
- **action expert** — 860M transformer. 50개 action 토큰이 서로 양방향으로 보고 backbone activation을 참조하며, flow 시간 정보는 adaptive RMSNorm으로 주입한다
- **training-time RTC** — 학습 때 0–12 스텝(50Hz 기준 최대 240ms)의 추론 지연을 시뮬레이션한다. 추론이 도는 동안 이미 실행이 확정된 행동 prefix에 이어지는 chunk를 만들도록 배우는 것이다. test-time RTC와 달리 추론 오버헤드가 없다(부록 D)

### 3.7 런타임 — 세 모델의 비동기 조합

추론 때는 학습을 마친 세 모델이 속도가 다른 루프로 맞물린다.

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

- **high-level policy** — 다음 subtask $\hat\ell$을 만든다. coaching할 때는 사람이 이 역할을 대신한다
- **world model** — $\hat\ell$이 바뀌거나 마지막 생성 후 $\Delta=4$초가 지나면 subgoal을 새로 그린다
- **π0.7** — 50 스텝 chunk를 5 denoising step으로 만들고 15 또는 25 스텝만 실행한다. 최소 구성에서 38ms, MEM 인코더와 subgoal을 모두 켠 최악의 경우 127ms다(단일 H100)
- **비동기** — subtask와 subgoal 생성은 별도 스레드에서 돌고, VLA는 그 시점에 준비된 가장 최신 값을 쓴다
- **항상 들어가는 성분** — control mode와 metadata는 모든 task에서 프롬프트에 들어간다

metadata에는 한 단계가 더 있다. 행동 denoising의 매 스텝에 CFG를 건다.

$$\tilde\nabla_{\mathbf{a}}\ =\ \nabla_{\mathbf{a}}\log\pi_\theta(\mathbf{a}\mid\mathbf{o}_t,\mathcal{C}_t)\ +\ \beta\Big(\nabla_{\mathbf{a}}\log\pi_\theta(\mathbf{a}\mid\mathbf{o}_t,\mathcal{C}_t)\ -\ \nabla_{\mathbf{a}}\log\pi_\theta(\mathbf{a}\mid\mathbf{o}_t,\mathcal{C}^{\text{uncond}}_t)\Big)$$

| 기호 | 의미 |
|---|---|
| $\mathbf{a}$ | action chunk $\mathbf{a}_{t:t+H}$의 약식 표기 |
| $\mathcal{C}_t$ | metadata를 포함한 전체 context |
| $\mathcal{C}^{\text{uncond}}_t$ | "무조건부" context. π0.7에서는 metadata를 뺀 context |
| $\beta$ | CFG 가중치. 논문은 $\beta\in\{1.3,\ 1.7,\ 2.2\}$를 쓴다 |
| $\tilde\nabla_{\mathbf{a}}$ | denoising에 실제로 쓰는 가이드된 score |

그런데 quality 5와 mistake false는 이미 조건으로 들어가 있다. 왜 CFG로 한 번 더 밀어야 하는가?

> ### 💡 metadata CFG는 암묵적 '품질 판별기' 쪽으로 분포를 날카롭게 만든다
>
> $\mathcal{C}_t=\mathcal{C}^{\text{uncond}}_t\cup\{m^\star\}$로 두고 Bayes 규칙을 쓰면, 괄호 안의 차이는 $\nabla_{\mathbf{a}}\log p_\theta(m^\star\mid\mathbf{a},\mathbf{o},\mathcal{C}^{\text{uncond}})$가 된다. 따라서 위의 가이드는 다음 분포에서 샘플하는 것과 같다.
>
> $$\tilde\pi(\mathbf{a}\mid\mathbf{o},\mathcal{C})\ \propto\ \pi_\theta(\mathbf{a}\mid\mathbf{o},\mathcal{C}^{\text{uncond}})\cdot p_\theta(m^\star\mid\mathbf{a},\mathbf{o},\mathcal{C}^{\text{uncond}})^{\,1+\beta}$$
>
> - $m^\star$ — 요청한 metadata (quality 5, mistake false, 빠른 speed)
> - $p_\theta(m^\star\mid\mathbf{a},\mathbf{o},\cdot)$ — "이 행동 chunk가 그런 에피소드에서 나왔을 확률". 따로 학습하지 않았지만 조건부·무조건부 모델의 비율로 암묵적으로 정의되는 분류기
> - $1+\beta$ — 분포를 분류기 쪽으로 얼마나 날카롭게 할지 정하는 지수
>
> 조건부 모델만으로는 조건이 약하게 반영된다. text-to-image에서 CFG 없이 샘플하면 프롬프트 충실도가 떨어지는 것과 같은 현상이다. 지수 $1+\beta$가 분포를 판별기 쪽으로 몬다. 그리고 여기서 판별기가 가려내는 것은 **품질**이다. 결과적으로 metadata CFG는 critic 없이 수행하는 정책 개선처럼 작동한다. advantage가 높은 행동에 가중치를 주는 정책 개선과 같은 모양이다(필자 해석).
>
> 구현에서는 조건 분기와 무조건 분기를 attention tree로 한 시퀀스에 packing해 forward 한 번으로 처리한다(부록 B).

---

## 4. 왜 작동하는가

논문이 드는 이유는 네 가지로 정리된다.

**⓵ 모호성 해소** — context가 품질과 전략을 구분하므로 저품질 데이터가 성능을 해치지 않고 흡수된다.

**⓶ 상태 커버리지** — 준최적 데이터는 task 안에서 방문하는 상태와 시나리오를 넓힌다. 논문은 이것이 강건성을 높이고, 때로 RL specialist를 넘어서게 하는 이유라고 설명한다.

**⓷ Specialist 증류** — π\*0.6가 RL 학습 중 수집한 데이터가 좋은 시도와 나쁜 시도를 섞은 채로 들어간다. metadata가 그중 좋은 행동을 불러내는 손잡이가 되어, generalist가 specialist의 능력을 상속한다.

**⓸ 웹 지식의 인터페이스** — subgoal 이미지는 웹 사전학습된 world model이 가진 의미적·물리적 지식을 policy로 옮기는 통로다.

이것을 한 문장으로 줄이면, π0.7은 **교란변수였던 "어떻게"를 관측변수로 바꾼다.** §2.3의 분해에서 좌변 대신 우변을 학습하게 된 것이다. 여기에 부수효과가 하나 있다. metadata는 task에 무관한 어휘다. "quality 5"는 에스프레소에서나 셔츠 개기에서나 같은 방향을 가리킨다. 그래서 "어떻게" 축은 task 사이에서 공유되고, "무엇을" 축은 수행 방식의 잡음이 걷힌 채로 학습된다. 조합에 유리한 구조다(필자 해석).

그렇다면 조합적 일반화를 만든 것은 context인가, 데이터 다양성인가? §5.5에서 자세히 볼 두 실험을 나란히 놓으면 역할이 갈린다.

> ### 📌 context는 필요조건이고, 조합의 연료는 task 다양성이다
>
> | 실험 (Fig. 18) | 조작 | 결과 | 드러나는 역할 |
> |---|---|---|---|
> | 왼쪽 | 품질이 낮은 데이터를 점점 더 넣는다 | metadata가 있으면 계속 향상, 없으면 저하될 수 있음 | context는 다양한 데이터를 **쓸 수 있게** 만든다 |
> | 오른쪽 | 같은 양을 빼되, 가장 다양한 20% vs 무작위 20% | 가장 다양한 20%를 뺄 때만 unseen task 성능이 크게 하락 | 조합 능력의 **연료**는 task 다양성이다 |
>
> π0.7의 기여는 새 능력을 직접 만드는 장치가 아니라, 필터링 때문에 버려지던 다양성을 학습 가능하게 만드는 장치다. 논문이 스스로를 "아키텍처가 아니라 방법론"이라고 부르는 이유다.

논문의 Discussion도 같은 선상에 있다. 일반화는 결국 이미 본 행동들의 재조합(remix)이며, 그것이야말로 조합적 일반화의 본질이라는 입장이다.

§1.2에서 명명한 두 벽이 어떻게 회수되었는지 정리하면 다음과 같다.

| 벽 | 처방 | 근거 |
|---|---|---|
| ⛔ 벽 1 — 다양성은 곧 모호성 | episode metadata $m$ (+ control mode $c$, subtask $\hat\ell$), dropout, metadata CFG | Fig. 7의 metadata 제거 ablation, Fig. 18 왼쪽 |
| ⛔ 벽 2 — 언어의 한계 | world model이 생성한 subgoal 이미지 $\mathbf{g}$ | Fig. 10·11·12(오른쪽)·15에서 π0.7 (GC)의 향상 |

---

## 5. 실험

평가는 여러 로봇에서 이루어졌다. 이동형 양팔 로봇(6-DoF 팔 2개), 경량 6-DoF 팔을 쓴 정적 양팔 로봇, Robotiq 그리퍼를 단 UR5e 양팔 시스템, 그리고 단일팔 로봇이다. UR5e는 20Hz, 나머지는 50Hz로 돈다. 결과는 대부분 막대그래프로 제시되고, 본문에 수치가 명시된 것은 사람 비교 실험과 Discussion의 성공률 구간 정도다. 아래는 그림 번호와 함께 정리한다.

| 축 | task | 비교 대상 | 결과 요지 | 그림 |
|---|---|---|---|---|
| 숙련 | 세탁(티셔츠·반바지 / 버튼셔츠), 에스프레소, 상자 조립 | π\*0.6 RL specialist | 성공률 대등, 버튼셔츠 세탁·상자 조립에서 throughput 우위 | Fig. 6 상단 |
| 숙련 | 땅콩버터 샌드위치, 셔츠 뒤집기, 문 통과 주행, 주키니 썰기, 채소 껍질 벗기기, 쓰레기봉투 교체 | π0.6 SFT specialist | 근접 | Fig. 6 하단 |
| 메모리 | 머그 3개 교환, 물건 찾기, 커피 푸기, 창문 닦기 | π0.6-MEM specialist | 대등 이상 | Fig. 8 |
| 지시 따르기 | 미학습 부엌 4곳 + 침실 2곳, 14개 시나리오 | π0.5, π0.6 | 전반적으로 큰 폭 우위 | Fig. 9 |
| 참조형 지시 | "수프 먹을 때 쓸 물건", "가장 큰 접시 위의 과일" | π0.5, π0.6 | 복잡한 지시에서 우위, GC로 추가 향상 | Fig. 10 |
| 데이터 편향 역행 | Reverse Bussing, Reverse Fridge to Microwave | π0.5, π0.6 | 우위, 후자는 GC가 결정적 | Fig. 11 |
| Cross-embodiment | 6개 task (§5.3) | π0.5, π0.6 | 로봇 형태 차이가 클수록 π0.7 우위 | Fig. 12 |
| 조합 (단기) | French press 누르기, 밥솥에 쌀 푸기, 사무용품 닦기, 관절 물체 돌리기 | π0.5, π0.6 | 프롬프트만으로 수행, GC와 비슷 | Fig. 17 |
| 조합 (장기) | 에어프라이어 넣기·비우기, 베이글 굽기 | π0.5, π0.6 (coaching) | 이전 모델은 coaching 지시도 따라가지 못함 | Fig. 15 |
| coaching → 자율 | 5개 task | coaching 에피소드 | 자율 수행 ≈ coaching 수행 | Fig. 16 |

### 5.1 Specialist 수준의 숙련 — 학습한 task를 얼마나 잘하나

첫 질문은 이미 학습 데이터에 있는 dexterous task를 fine-tune 없이 specialist만큼 빠르고 강건하게 수행하느냐다. §1.1의 증상 ⓶에 대한 직접적인 답이다.

- **RL specialist와의 비교 (Fig. 6 상단)** — π\*0.6 평가에 쓰였던 세탁(티셔츠·반바지 / 가장 어려운 품목인 버튼셔츠), 에스프레소, 상자 조립에서 성공률은 대등하다. 버튼셔츠 세탁과 상자 조립에서는 시간당 성공 횟수(throughput)가 specialist보다 높다.
- **SFT specialist와의 비교 (Fig. 6 하단)** — "Robot Olympics" task를 포함한 여러 dexterous task에서 π0.6 기반 SFT specialist에 근접한다.
- **메모리 task (Fig. 8)** — 과거 관측을 기억해야 하는 task에서도 메모리를 fine-tune한 π0.6-MEM specialist와 대등하거나 낫다.
- **ablation (Fig. 7)** — metadata를 빼거나 자율 평가 데이터를 빼면 모든 task에서 성능이 떨어지고, 격차는 throughput에서 가장 크다. 속도는 speed metadata와 RL 롤아웃이 실어 나르는 정보이므로 설계 의도와 들어맞는 결과다.

그런데 이것을 "generalist가 specialist를 이겼다"로 읽어도 되는가?

> ### ⚠️ "out-of-the-box"는 "task 데이터 없음"이 아니다
>
> 이 표현은 task별 post-training 단계가 없다는 뜻이다. Fig. 6 상단의 네 task는 π\*0.6가 RL로 학습하며 쌓은 롤아웃이 π0.7의 학습 데이터에 들어 있고, 논문도 이를 "증류"라고 명시한다. 따라서 이 결과는 "specialist를 이겼다"보다 "specialist 여러 개를 하나의 generalist로 손실 없이 접었다"로 읽는 것이 정확하다. 반면 일반화 평가용 task의 자율 데이터는 학습에서 제외했으므로 §5.2–§5.4의 결과에는 이 문제가 없다.

### 5.2 지시 따르기 — 언어를 실제로 읽는가

- **열린 지시 (Fig. 9)** — 미학습 부엌 4곳과 침실 2곳에서, 3–6개의 열린 지시를 순서대로 따르는 14개 시나리오를 평가했다. 정리, 가구 조작, 엎지른 것 치우기 등이다. 지시 이행률에서 π0.5·π0.6을 전반적으로 크게 앞선다.
- **참조형 지시 (Fig. 10)** — 사무용 책상 정리 task에서 표준 지시("숟가락을 집어라")는 모든 모델이 성공한다. 복잡한 참조 지시("수프 먹을 때 쓸 물건을 집어라", "가장 큰 접시 위의 과일을 집어라")에서 π0.7이 앞서고, subgoal을 주면(GC) 더 오른다.
- **데이터 편향 역행 (Fig. 11)** — 어떤 장면에서 늘 같은 행동만 한 데이터로 학습하면, 모델은 그 장면에서 언어를 무시하고 습관을 따른다. "Reverse Bussing"(쓰레기는 식기통에, 식기는 쓰레기통에)과 "Reverse Fridge to Microwave"(데이터에는 냉장고에서 전자레인지로 옮기는 방향만 있다)에서 π0.7은 편향을 거슬러 지시를 따른다.

특히 "Reverse Fridge to Microwave"에서는 GC가 결정적이었다. 논문은 웹 규모 이미지 생성 사전학습 덕분에 world model이 텍스트 지시로부터 subgoal을 잘 만들어내기 때문이라고 설명한다. 데이터 습관과 반대되는 장면을 world model이 그려주면, policy는 그 그림을 따라가면 된다. §3.4에서 예고한 subgoal의 기여가 가장 선명하게 드러나는 장면이다.

### 5.3 Cross-embodiment — 해본 적 없는 로봇으로 기술 옮기기

어떤 task의 데이터가 전혀 없는 로봇에서 그 task를 할 수 있는가. 논문은 소스와 타깃 로봇의 형태 차이를 점점 키우며 실험한다(Fig. 12).

| task | 데이터를 모은 로봇 | 평가한 로봇 | 결과 |
|---|---|---|---|
| Table Setting | 여러 로봇(이동형·정적·단일팔) | 정적 양팔 | 모든 모델 양호 |
| Bag In Backpack, Organize Tupperware | UR5e 양팔 (크고 무거움) | 소형 정적 양팔 | π0.5 크게 하락, π0.6·π0.7 양호 |
| Shirt Bagging | 소형 정적 양팔 | UR5e 단일팔 | π0.7 뚜렷한 우위 |
| Towel / Shirt Folding | 소형 정적 양팔 | UR5e 양팔 | π0.7 성공, GC로 추가 향상 |

UR5e는 소스 로봇보다 훨씬 길고 무거우며 형태가 다르고, 테이블 한쪽 끝이 아니라 양옆에 배치되어 있다. 그래서 조작 전략 자체를 바꿔야 한다. 흥미로운 것은 π0.7이 소스 로봇의 행동을 흉내 내는 대신 타깃 로봇에 맞는 전략을 새로 쓴다는 점이다(Fig. 13).

| 상황 | 소스 로봇에서 사람 조작자의 전략 | UR5e에서 π0.7의 전략 |
|---|---|---|
| 셔츠를 가방에 넣기 | 한 팔로 가방을 벌리고 다른 팔로 넣는다 | 긴 팔로 한 번에 집어서 넣는다 |
| 셔츠 파지 | end-effector를 기울여 천을 테이블에 눌러 잡는다 | 팔 배치에 맞는 수직 파지를 쓴다 |

UR5e 셔츠 개기는 §3.4에서 예고한 대로 언어 없이 subgoal과 metadata만으로 프롬프트한다. 논문은 world model이 소스와 타깃 로봇 사이의 시각적 유비를 만들어, 타깃 로봇에 적합한 파지와 천의 배치를 subgoal로 제시한다고 설명한다.

결과를 가늠하기 위해 사람과도 비교했다. 조작 경력 상위 2%의 숙련 조작자 10명(모든 로봇 합산 평균 약 375시간)이 UR5e 셔츠 개기를 연습 없이 각 3회 시도했다. 사람은 task progress 90.9%, 성공률 80.6%를 기록했고, π0.7은 85.6%, 80%를 기록했다. 이 결과를 인용할 때는 조건 두 가지를 함께 적어야 한다.

> ### ⚠️ 사람과 맞선 것은 π0.7 (GC)이고, UR5e는 처음 보는 로봇이 아니다
>
> 사람 비교(Fig. 22, 부록 F)의 상대는 world model subgoal을 쓴 π0.7 (GC)다. 본문은 그냥 "π0.7"이라고 쓴다. 또 UR5e 양팔 데이터는 학습에 존재한다(Bag In Backpack·Organize Tupperware의 소스 로봇). 처음인 것은 "UR5e × 빨래 개기"라는 조합이다. joint-space 제어를 택한 근거도 π0.5·π0.6의 joint vs end-effector 비교(Fig. 20)이며, π0.7 자체의 비교는 보고되지 않았다.

논문이 드는 실무적 함의는 분명하다. teleop하기 쉬운 저가 경량 팔로 모은 기술을, teleop이 어렵고 시연 수집이 비싼 고하중 산업용 팔로 옮길 수 있다는 가능성이다.

### 5.4 조합적 task 일반화 — 처음 보는 task

논문이 로봇 파운데이션 모델의 "grand challenge"라고 부르는 축이다. 이전 모델들도 처음 보는 이름의 물체를 집는 식의 의미 수준 일반화는 보였지만, 새 task 수행은 어려웠다.

**단기 task는 프롬프트만으로 (Fig. 17)** — French press 누르기, 밥솥에 쌀 푸기, 헤드폰·자 같은 사무용품 닦기, 기어 세트나 탁상 선풍기 같은 관절 물체 돌리기를 해당 task의 로봇 데이터 없이 수행한다. 언어로만 프롬프트한 π0.7과 subgoal을 쓴 π0.7 (GC)가 비슷한 수준이다.

**장기 task는 coaching으로 (Fig. 14, 15)** — 에어프라이어로 고구마 굽기, 에어프라이어 비우기, 토스터로 베이글 굽기는 여러 단계가 최대 5분간 이어지는 task라 프롬프트 한 줄로는 되지 않는다. PI 블로그에 따르면 "고구마를 에어프라이어에 넣어라"라는 zero-shot 프롬프트만 주면 몇 번 헛손질한 끝에 일부만 수행한다. 여기서 §3.2에서 예고한 coaching이 등장한다. 사람이 "고구마를 집어라", "에어프라이어를 열어라"처럼 단계별로 불러주면 π0.7은 이를 따라 task를 해낸다. 이전 모델은 이 지시 자체를 따라가지 못해 매우 낮은 성능에 머문다.

**coaching에서 자율로 (Fig. 16)** — coaching 기록으로 high-level policy를 학습시키면, 다섯 task에서 사람 없이도 coaching 수준에 근접한 자율 수행이 나온다. 이 과정 어디에도 teleop 같은 행동 수준 데이터는 없다.

에어프라이어에 대한 지식은 어디서 왔을까. 논문 본문은 비슷한 기기가 사람 데이터와 외부 데이터셋에 다른 맥락으로 등장했다고만 적는다. PI 블로그는 더 구체적이다. 가장 가까운 데이터는 가정에서 로봇이 에어프라이어를 *닫는* 에피소드 2건과 오픈소스 DROID 데이터셋의 Franka 데이터였고, 실험 속 동작과는 생김새가 상당히 달랐다고 한다. 흩어진 조각을 재조합한 사례로 읽을 수 있다.

### 5.5 이질적 데이터에서 실제로 배우는가

마지막은 통제된 ablation이다(Fig. 18). §4에서 역할을 나눠 본 두 실험이 여기서 나온다.

**왼쪽 — 품질이 섞인 데이터를 늘리면?** 세탁(티셔츠·반바지) 데이터를 품질·속도 기준으로 상위 30%, 50%, 80%, 전체의 네 버킷으로 나누고, metadata 유무를 조합해 모델 8개를 처음부터 학습했다. 버킷이 클수록 데이터는 늘지만 평균 품질은 떨어진다. metadata가 없는 모델은 오히려 나빠질 수 있었지만, metadata가 있는 모델은 데이터가 늘수록 계속 좋아졌다.

**오른쪽 — 다양성을 빼면?** task 다양성이 가장 높은 20%를 뺀 모델과, 같은 양을 무작위로 뺀 모델을 §5.4의 단기 unseen task에서 비교했다. 앞의 모델만 크게 떨어졌다.

그런데 왼쪽 실험의 결론은 어디까지 일반화할 수 있는가?

> ### ⚠️ Fig. 18 왼쪽은 단일 seen task 실험이다
>
> 캡션은 두 패널을 묶어 "generalization performance의 스케일링"이라고 부르지만, 왼쪽은 이미 학습한 task 하나의 성능이다. "metadata가 데이터 확장을 가능하게 한다"는 결론은 이 한 task에서 확인된 것이며, 일반적인 스케일링 법칙으로 읽으면 과대해석이다. 논문 스스로도 거대한 데이터셋은 "다양성"을 깨끗하게 떼어내기 어려워 이런 질문에 확정적으로 답하기 힘들다고 인정한다.

---

## 6. 위치잡기 — 이웃 연구들 속에서

논문의 related work를 π0.7이 차지하는 자리 중심으로 정리하면 다음과 같다.

| 계보 | 대표 연구 (논문 인용) | π0.7의 자리 |
|---|---|---|
| VLM에서 초기화한 generalist 정책 | RT-2, OpenVLA, π0, π0.5, Gemini Robotics | 같은 뼈대. π0.6-MEM 위에 구축 |
| VLA 구성요소 | 메모리(MEM), 장기 계획용 계층 구조(Hi Robot, π0.5), 목표 이미지 조건화(CoT-VLA) | 셋을 한 모델에 통합 |
| 비표준 데이터로 사전학습 | 웹 데이터(RT-2), 사람 1인칭 영상, 자율 경험(π\*0.6, RLDG, PLD) | 전부 넣되 context로 구분 |
| task·embodiment 일반화 | 사람 영상 기반 표현학습, 2D point track, Open X-Embodiment, 휴대형 수집 장치(UMI) | 올바른 프롬프트로 로봇·사람·인터넷 데이터를 함께 활용 |
| 목표 이미지 프롬프트 | 사용자 제공 목표, 별도 모델이 생성한 목표(SuSIE, VLP), chain-of-thought식 생성(CoT-VLA) | 외부 world model이 생성한 subgoal을 선택적 조건으로 |

가장 가까운 사촌은 생성 subgoal 계열이다.

| | 별도 모델이 목표 생성 (SuSIE 등) | chain-of-thought식 생성 (CoT-VLA) | π0.7 |
|---|---|---|---|
| subgoal을 만드는 곳 | 정책 밖의 생성 모델 | 정책 모델 내부 | 정책 밖의 14B world model, 비동기 |
| 정책에서 subgoal의 지위 | 정책을 조건화하는 목표 | 행동 생성 전에 먼저 만드는 중간 산출물 | 언어·metadata·제어 모드와 나란한 선택적 조건 |

π0.7은 SuSIE를 따라 웹 사전학습 이미지 편집 모델로 world model을 초기화하고, subgoal 재생성 주기 4초도 SuSIE에 맞췄다(부록 C). 논문은 자신의 기여를 이 계열과 상보적인 것으로 규정한다. 새 아키텍처가 아니라 VLA가 더 다양한 데이터를 쓰게 하는 방법론이며, 빨래 개기의 zero-shot 로봇 간 전이나 에어프라이어 같은 새 물체 상호작용은 선행 연구의 정량적 개선을 크게 넘어선다고 주장한다.

그런데 품질을 조건으로 넣어 혼합 품질 데이터를 활용한다는 발상 자체는 π0.7이 처음인가? π0.7이 증류한 데이터를 만든 π\*0.6가 이미 비슷한 일을 했다.

> ### 🔗 π\*0.6 (RECAP) — "어떻게"를 조건화하는 같은 조직의 직전 연구
>
> π\*0.6의 학습 방법 RECAP도 정책을 "이 행동이 얼마나 좋았는가"에 조건화한다. 다만 신호의 출처와 목적이 다르다.
>
> | | π\*0.6 (RECAP) | π0.7 |
> |---|---|---|
> | 목적 | task별 RL 성능 개선 | generalist가 이질적 데이터를 흡수 |
> | "어떻게" 신호 | 학습된 value function으로 추정한 advantage를 이진 지표로 | 측정한 에피소드 길이 + 사람이 주석한 품질·실수 (3개 속성) |
> | 추론 시 요청 | 높은 advantage | quality 5, mistake false, 빠른 speed + CFG |
> | 둘의 관계 | RECAP 학습 중 생성된 롤아웃 → | π0.7 학습 데이터로 증류 |
>
> π0.7의 metadata는 RECAP의 "학습된 1비트 조건"을 사람이 읽을 수 있는 다차원 라벨로 일반화한 것으로 볼 수 있다(필자 해석). 그리고 두 방법은 PI의 파이프라인 안에서 직렬로 이어진다. RL로 specialist를 만들고, 그 경험 전체를 라벨과 함께 generalist에 접어 넣는다. PI 블로그도 RECAP 학습 중 생성된 경험을 strategy metadata와 함께 π0.7로 증류했다고 설명한다.

---

## 7. 한계

논문이 스스로 밝힌 것과, 읽으며 추가로 짚어둘 만한 것을 함께 정리한다.

**논문이 인정한 한계**

- **일반화 성공률의 격차** — 학습한 task는 흔히 90%를 넘지만, 미학습 task와 미학습 task–로봇 조합은 60–80% 구간이다.
- **seen/unseen 경계의 모호함** — 데이터가 방대해서 관련 skill이 다른 라벨로, 혹은 다른 task의 일부로 이미 들어 있었을 수 있다. LLM 일반화 논쟁과 같은 구도다. 논문은 기존 부분의 재조합이야말로 조합적 일반화의 본질이라고 응수한다.
- **장기 신규 task** — 프롬프트 한 줄로는 되지 않고 coaching을 거쳐야 한다.

**추가로 짚을 지점**

- **"Steerable"의 실증 범위** — 실험에서 metadata는 항상 "최선" 고정값과 CFG로만 쓰였다. speed 값을 바꾸면 실제 에피소드 길이가 그에 맞게 변하는지 같은 조향 정확도(controllability) 평가는 보고되지 않았다. 실시간 조향이 실증된 축은 언어(coaching)와 subgoal 이미지다.
- **병목의 이동** — 필터링 비용이 주석 비용으로 옮겨간다. quality와 mistake는 사람 주석이고, world model은 시간 분할이 정확한 라벨에 민감하다(부록 C).
- **GC 기여의 분리가 불완전** — π0.7과 π0.7 (GC)는 같은 모델이고, 추론 때 subgoal을 주느냐만 다르다. 그런데 이 모델은 학습 때 world model이 생성한 이미지를 context로 봤다. subgoal 없이 돌린 π0.7의 성능에도 world model의 지식이 간접적으로 들어가 있을 수 있다.
- **비교군** — 모두 PI 자체 모델(π0.5, π0.6, specialist)이다. 외부 VLA 기준선이 없고, 본문과 그림 캡션에는 시행 횟수와 신뢰구간이 표기되어 있지 않다.
- **런타임 비용** — 5B VLA, 4B high-level policy, 14B world model(4×H100)로 구성된다. world model은 선택 사항이지만, 가장 어려운 일반화(편향 역행, UR5e 셔츠 개기)가 GC에 기대고 있다.

---

## 8. 마치며 — 이 논문이 시사하는 것

π0.7의 진짜 기여는 특정 모듈이 아니다. **데이터를 거르는 대신 설명하게 만든 것**이다. 실패 에피소드, 느린 시연, RL 롤아웃, 사람 영상처럼 지금까지 버려지거나 따로 쓰이던 데이터가 하나의 context 문법 안으로 들어온다.

그리고 이 논문은 생성 모델 배경의 독자에게 유난히 익숙하게 읽힌다.

- **text-to-image가 이미 걸어간 길이다.** ⓐ 거르지 말고 속성으로 조건화한다(SDXL micro-conditioning). ⓑ 캡션을 풍부하게 한다(prompt expansion, DALL·E 3의 recaptioning). ⓒ 조건 dropout과 CFG로 추론 때 품질 쪽으로 민다. 로봇에서 새로운 점은 "캡션"이 텍스트만으로는 부족해 이미지(subgoal)와 수치(metadata)로 확장됐다는 것이다.
- **LLM의 궤적과 겹친다.** task별 fine-tune에서 prompting으로의 이행, 그리고 RL로 만든 전문 모델의 경험을 학습 데이터로 되먹여 범용 모델에 접는 흐름이 그대로 보인다.
- **agentic 시스템의 구조다.** 런타임은 planner(high-level policy), 도구(시각 사양을 그리는 world model), executor(VLA)로 나뉜다. coaching 기록으로 high-level policy를 fine-tune하는 것은 사람이 쓴 계획 trace로 planner를 SFT하는 것과 같다.

가장 큰 함의는 마지막 항목에 있다. 새 task를 가르치는 단위가 **행동 시연에서 지시문으로** 한 층 올라갔다. PI 블로그는 한 걸음 더 나아가, 프롬프트를 정확히 따르는 모델이라면 foundation model의 의미적 추론을 물리적 행동으로 grounding하는 통로가 될 수 있다고 전망한다. 로봇에게 행동 데이터를 더 모아주는 대신 설명하고, 지시하고, 가르치는 방식으로 학습의 무게중심이 옮겨갈 수 있다는 뜻이다.

---

## 부록 — 용어 정리

| 용어 | 정의 |
|---|---|
| **context $\mathcal{C}_t$** | 행동을 조건화하는 부가 정보 전체. π0.7에서는 $\{\ell,\hat\ell,\mathbf{g},m,c\}$ |
| **episode metadata** | 에피소드 길이(speed), 품질 점수(quality), 세그먼트별 실수 여부(mistake) 라벨 |
| **subgoal 이미지** | 가까운 미래에 장면이 어떻게 보여야 하는지를 보여주는 multi-view 목표 이미지 |
| **world model** | subtask 지시로부터 subgoal 이미지를 생성하는 BAGEL 기반 14B 모델 |
| **π0.7 (GC)** | 추론 때 world model이 생성한 subgoal로 프롬프트한 설정 |
| **high-level policy** | 관측·task·subtask 이력으로부터 다음 subtask 지시를 생성하는 모델 |
| **language coaching** | 사람이 새 task를 단계별 언어 지시로 이끄는 것. 그 기록이 high-level policy의 학습 데이터가 된다 |
| **Knowledge Insulation** | backbone은 이산 FAST 토큰으로 학습하고, action expert의 gradient는 backbone으로 흘리지 않는 학습 레시피 |
| **metadata CFG** | metadata를 뺀 무조건부 예측과의 차이로 행동 분포를 품질 쪽으로 날카롭게 만드는 추론 기법 |
| **training-time RTC** | 추론 지연을 학습 때 시뮬레이션해, 이미 확정된 행동 prefix에 이어지는 chunk를 만들게 하는 기법 |

**원문** — [arXiv:2604.15483](https://arxiv.org/abs/2604.15483) · **프로젝트 페이지** — [pi.website/pi07](https://pi.website/pi07) · **PI 블로그** — [A Steerable Model with Emergent Capabilities](https://www.pi.website/blog/pi07)
