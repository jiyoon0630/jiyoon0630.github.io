---
layout: paper
lang: ko
ref: cosmos-3-omnimodal-world-model
kind: paper-review
title: "Cosmos 3: Omnimodal World Models for Physical AI"
date: 2026-06-20 12:00:00 -0700
paper_date: 2026-06-01
venue: "NVIDIA Technical Report · arXiv:2606.02800"
tags: [World-Model, WAM, VLA, Omnimodal, Mixture-of-Transformers, Robot-Foundation-Model, Paper-Review]
authors: "NVIDIA (기여자 294명, 논문 부록 G)"
affiliations: "NVIDIA (Cosmos Lab)"
summary: "VLM을 이해·생성 두 타워로 나누고 영상·오디오·행동을 하나의 물리 시간축에 묶어, 어떤 토큰을 깨끗하게 주느냐만으로 VLM·영상 생성·FD·ID·정책을 한 모델에서 돌린다. 행동을 mid-training에 넣은 효과는 정책 적응 속도에서 드러난다."
paper_url: "https://arxiv.org/abs/2606.02800"
code_url: "https://github.com/nvidia/cosmos"
---

> **핵심 주장** — Physical AI에 필요한 이해(VLM), 시뮬레이션(영상 생성·forward dynamics), 행동(정책·inverse dynamics)은 별개의 모델일 필요가 없다. VLM을 두 벌로 나눠 한쪽은 이해, 한쪽은 생성에 쓰고, 영상·오디오·행동을 하나의 물리 시간축 위 시퀀스로 묶으면, **어떤 토큰을 깨끗하게 줄지**만 바꿔 한 모델이 이 모든 역할을 한다. 그리고 행동을 mid-training 단계의 일급 모달리티로 넣으면 로봇 정책으로의 적응이 빨라진다.

---

## 들어가며

Cosmos 3는 NVIDIA가 공개한 "옴니모달 world model" 기술 보고서다. 언어·이미지·영상·오디오·행동을 하나의 mixture-of-transformers 안에서 이해하고 생성하며, 입출력 구성만 바꿔 VLM, 영상 생성기, world simulator, world-action model 역할을 모두 한다고 주장한다.

보고서는 100쪽이 넘고 데이터 파이프라인, 학습·서빙 인프라, 이미지·오디오 생성 평가까지 다룬다. 이 글은 그중 모델링 논지에 집중한다 — 이질적인 모달리티를 한 모델에 넣을 때 무엇이 문제이고, 어떤 장치로 그걸 넘었으며, 그 통합이 로봇 정책에 실제로 도움이 되는가. 로보틱스 배경이 얕아도 따라올 수 있도록 필요한 개념은 그 자리에서 쌓는다.

---

## 1. 문제 — 모델을 이어 붙이는 Physical AI

### 1.1 조각난 파이프라인

논문은 Physical AI 에이전트가 두 능력을 함께 가져야 한다고 본다. 부분 관측에서 의미와 동역학을 추론하는 **이해**, 그리고 세계가 어떻게 전개되고 에이전트가 어떻게 반응해야 할지를 미리 그려 보는 **생성**이다. 그런데 지금까지 둘은 따로 발전했다. 지각·추론은 VLM이, 세계 시뮬레이션은 영상 생성 모델과 forward dynamics 모델이, 행동 예측은 VLA와 WAM(World-Action Model)이 맡는다.

논문이 드는 예시는 저녁 식사 뒤 식탁을 치우는 가정용 로봇이다. 지금 방식이라면 이 로봇은 모델 세 개를 이어 붙여야 한다.

```
  task: "clean the dining table"

  [ VLM ]                  locate dishes, make a plan
     |
     v
  [ VLA / WAM ]            generate action sequence
     |
     v
  [ FDM / world model ]    simulate and evaluate future states
```

저자들은 이 분리가 근본적인 한계라고 주장한다. 이해에는 세계의 미래 전개와 행동의 결과에 대한 추론이 필요하고, 생성에는 세계와 행동에 대한 압축된 구조적 표현이 필요하다. 두 능력이 서로를 필요로 하는데 모델이 갈라져 있으니 표현을 공유하지 못하고, 계산도 중복된다.

### 1.2 그래서 논문이 던지는 질문

> 모델을 이어 붙이는 대신, Physical AI 에이전트에 필요한 능력 전부를 네이티브로 다루는 **단일 모델**을 설계할 수 있는가?

### 1.3 세 개의 벽

이질적인 것들을 한 transformer에 순진하게 밀어 넣으면 세 개의 벽에 부딪힌다.

**⛔ 벽 1 — 이해와 생성의 간섭.** 이해 경로는 깨끗한 토큰 위에서 causal attention과 next-token 교차엔트로피로 학습된다. 생성 경로는 노이즈 섞인 토큰 위에서 양방향 attention과 velocity 회귀로 학습된다. 같은 가중치에 두 목적을 얹으면 노이즈가 이해 표현을 오염시키고, 사전학습된 VLM의 언어·시각 능력이 무너질 수 있다.

**⛔ 벽 2 — 모달리티마다 다른 시계.** 24fps 영상은 VAE의 4배 시간 압축을 거치면 초당 6 토큰, 오디오는 초당 25 토큰, 로봇 행동은 제어 주기(DROID는 15Hz)를 따른다. 토큰 순번과 물리 시간이 어긋난 채로는 "이 행동이 이 프레임을 만들었다"는 동기화를 학습할 수 없다.

**⛔ 벽 3 — 제각각인 행동 공간, 부족한 행동 데이터.** 차량의 조향, 로봇 팔의 관절, 사람의 손가락, 카메라의 움직임은 모두 다른 제어 공간이다. 게다가 행동 라벨이 붙은 데이터는 영상보다 자릿수가 다르게 적다.

벽 1은 §3.2, 벽 2는 §3.3, 벽 3은 §3.4에서 넘는다. 벽을 넘고 나면 질문이 하나 남는다 — 이렇게 통합해서 학습하는 것이 실제로 재사용 가능한 world-action prior를 주는가. 그 답은 §4에서 본다.

---

## 2. 배경

세 벽의 정체와 해법을 읽으려면 세 개념이 필요하다. 행동이 들어간 world model이 푸는 문제(2.1), 생성 목적함수(2.2), 위치 인코딩(2.3)이다. 익숙하면 §3으로 건너뛰어도 좋다.

### 2.1 행동이 들어간 world model의 세 가지 질문

영상 latent 시퀀스를 $v$, 언어 지시를 $l$, 행동을 $a$라 하자. 이 논문에서 행동은 세계 상태를 바꾸는 원인 변수로 정의되고, 연속된 영상 토큰 사이에서 $a_t$는 $v_{t-1}$에서 $v_t$로의 전이를 나타낸다. 행동이 들어간 world model이 답하는 질문은 세 가지다.

$$\text{FD:}\quad p_\theta\big(v_{t:t+H}\mid v_{<t},\ a_{t:t+H},\ l\big)$$

$$\text{ID:}\quad p_\theta\big(a_{t:t+H}\mid v_{t-1:t+H},\ l\big)$$

$$\text{Policy:}\quad p_\theta\big(a_{t:t+H},\ v_{t:t+H}\mid v_{<t},\ l\big)$$

| 기호 | 의미 |
|---|---|
| $v_t$ | 시점 $t$의 영상 latent 토큰 |
| $a_t$ | $v_{t-1}\to v_t$ 전이를 일으킨 행동 |
| $l$ | 언어 지시 |
| $H$ | 한 번에 예측하는 구간(chunk) 길이 |
| $v_{<t}$ | 이미 관측한 과거 영상 |

| 이름 | 주어진 것 | 묻는 것 | 로봇에서의 역할 |
|---|---|---|---|
| FD (forward dynamics) | 과거 영상 + 행동 | 미래 영상 | **시뮬레이터** — 이렇게 움직이면 무슨 일이 생기나 |
| ID (inverse dynamics) | 전후 영상 | 행동 | **라벨러** — 이 변화를 만든 행동은 무엇인가 |
| Policy | 과거 영상 + 지시 | 행동 + 미래 영상 | **행위자** — 행동과 그 결과를 함께 상상 |

그렇다면 시뮬레이터, 라벨러, 정책을 각각 따로 학습해야 할까?

> ### 💡 세 역할은 하나의 결합분포에서 무엇을 묻느냐의 차이다
>
> 세 분포는 모두 결합분포 $p(v, a\mid l)$ 하나의 서로 다른 조건부다. 무엇을 주고 무엇을 묻느냐만 다르다.
>
> 언어 모델에서 이미 익숙한 구도다. BERT, T5의 span corruption, MaskGIT은 결합분포를 여러 마스크 패턴으로 학습해, 한 모델이 임의의 조건부에 답하게 만든다. diffusion에서는 마스크가 "깨끗하게 주는 토큰"이 된다. 주어진 토큰은 깨끗하게 두고, 물어볼 토큰만 노이즈로 덮어 denoise시키면 된다.
>
> §3.1이 이 발상을 그대로 구현한다.

### 2.2 Rectified flow 목적함수

Cosmos 3의 생성 쪽은 모든 모달리티를 rectified flow matching 하나로 학습한다. SD3, Wan과 같은 계열이다.

$$x_\sigma=\sigma\,\epsilon+(1-\sigma)\,x_0,\qquad v^{*}=\epsilon-x_0$$

$$\mathcal{L}=\sum_{m}\lambda_m\,\mathbb{E}\Big[\big\|M_m\odot\big(v_\theta(x_{\sigma_m},\sigma_m,c)-v^{*}\big)\big\|_2^2\Big]$$

| 기호 | 의미 |
|---|---|
| $x_0$ | 깨끗한 타깃 latent — 영상, 오디오, **행동 벡터** |
| $\epsilon\sim\mathcal{N}(0,I)$ | 가우시안 노이즈 |
| $\sigma_m\in[0,1]$ | 모달리티 $m$의 노이즈 수준 |
| $v_\theta$ | 속도를 예측하는 denoiser |
| $c$ | 조건 (텍스트, 깨끗한 조건 토큰) |
| $M_m$ | 깨끗한 조건 토큰을 loss에서 빼는 마스크 |
| $\lambda_m$ | 모달리티별 loss 가중치 |

눈여겨볼 점이 셋 있다.

- **모달리티별 독립 노이즈** — $\sigma$를 모달리티마다 따로 뽑는다. 이미지·오디오·행동은 logit-normal, 영상은 mode sampling을 쓴다.
- **행동도 영상처럼 깎아낸다** — 행동 벡터도 노이즈에서 denoise한다. 정규화된 행동 벡터는 원소당 MSE가 작아서, mid-training에서 행동 loss에 10배 가중치($\lambda_{\text{act}}=10$)를 준다.
- **해상도 적응 shift** — 뽑은 시간 변수를 고노이즈 쪽으로 치우치게 재매개변수화한다.

$$\sigma=\frac{s\,\bar t}{1+(s-1)\,\bar t},\qquad \bar t=1-t$$

- $t$ — 위 분포에서 뽑은 시간 변수
- $s\ge 1$ — shift 계수. 사전학습에서 256p/480p/720p에 각각 1/3/5, mid-training에서 3/5/10

### 2.3 RoPE에서 MRoPE로

RoPE는 위치를 회전으로 인코딩해서, attention 점수가 상대 위치에만 의존하게 만든다.

$$\big\langle R(m)\,q,\ R(n)\,k\big\rangle=q^{\top}R(n-m)\,k$$

- $R(\cdot)$ — 위치에 비례하는 각도로 회전시키는 블록 대각 회전행렬
- $m, n$ — query와 key의 위치 인덱스

MRoPE(Qwen 계열)는 head 차원을 $(t, h, w)$ 세 묶음으로 나눠, 각 묶음을 자기 축의 인덱스로 회전시킨다. 영상 토큰은 프레임 번호와 공간 위치를 동시에 갖게 된다. 그런데 원래 MRoPE에서 시간 좌표 $t$는 **프레임 순번**이다. 이것이 왜 벽 2로 이어지는지는 §3.3에서 본다.

---

## 3. 방법

세 벽과 그것을 넘는 장치의 대응은 다음과 같다.

| 벽 | 장치 | 절 |
|---|---|---|
| ⛔ 1 이해와 생성의 간섭 | MoT 이중 타워 + 단방향 joint attention + reasoner 동결 | §3.2 |
| ⛔ 2 모달리티마다 다른 시계 | 절대 시간축 MRoPE | §3.3 |
| ⛔ 3 이질적 행동 공간, 부족한 데이터 | 상태 변화 기반 pseudo-action + 도메인별 projection + 영상에서 행동 추출 | §3.4 |

그 전에 모든 장치가 올라타는 토대, 즉 시퀀스 구성부터 본다.

### 3.1 한 시퀀스, 모드는 "무엇이 깨끗한가"로 정한다

§2.1에서 예고한 발상이 여기서 구현된다. 입력 시퀀스는 두 부분으로 나뉜다 — 이해를 맡는 AR 서브시퀀스, 그리고 그 뒤에 붙는 diffusion(DM) 서브시퀀스다. 모든 태스크에 같은 배치 규칙을 적용한다.

- ⓵ AR 토큰이 DM 토큰보다 앞에 온다
- ⓶ DM 안에서는 모달리티마다 깨끗한 조건 토큰이 노이즈 토큰보다 앞에 온다
- ⓷ 조건부와 노이즈부 모두 vision → audio → action 순서로 정렬한다

모든 모드가 공유하는 AR 접두부는 다음과 같다.

$$S_{\text{AR}}\triangleq\big[\,l_1,\dots,l_n,\ \langle\text{EOS}\rangle,\ \langle\text{BOG}\rangle\,\big]$$

- $l_1,\dots,l_n$ — 언어 토큰
- $\langle\text{EOS}\rangle$, $\langle\text{BOG}\rangle$ — 문장 끝, 생성 시작 특수 토큰

깨끗한 영상·오디오·행동 토큰을 $v, s, a$, 노이즈 토큰을 $\tilde v, \tilde s, \tilde a$로 쓰면 각 모드는 다음과 같다(Eq. 3–6, Fig. 4).

| 모드 | 시퀀스 구성 | 하는 일 |
|---|---|---|
| Language | $S_{\text{AR}}$ 만, 생성 파라미터 비활성 | 표준 VLM |
| T2I | $[S_{\text{AR}},\ \tilde v_1]$ | 이미지 생성 |
| T2V(+Audio) | $[S_{\text{AR}},\ \tilde v_{1:N},\ \tilde s]$ | 영상(+소리) 생성 |
| I2V / V2V | $[S_{\text{AR}},\ v_{1:P},\ \tilde v_{P+1:N}]$ | 조건 프레임 뒤를 이어 생성 |
| Transfer | $[S_{\text{AR}},\ v^{\text{ctrl}}\_{1:N},\ \tilde v\_{1:N}]$ | edge·depth 등 제어 영상에서 RGB 생성 |
| FD | 깨끗한 $a$, 노이즈 $\tilde v$ | 행동 조건 미래 예측 |
| ID | 깨끗한 $v$, 노이즈 $\tilde a$ | 행동 역추정 |
| Policy | 노이즈 $\tilde v$, 노이즈 $\tilde a$ | 행동과 결과 공동 생성 |

- $N$ — latent 영상 프레임 수, $P$ — 조건 프레임 수 ($P=1$이면 I2V, $P>1$이면 V2V)
- $v^{\text{ctrl}}$ — VAE로 인코딩한 깨끗한 제어 영상 토큰

모든 모드가 같은 아키텍처와 같은 목적함수를 공유한다. 모드를 가르는 것은 어떤 토큰을 깨끗하게 주느냐, 즉 마스크 패턴뿐이다. diffusion 관점에서 보면 I2V는 시간 방향 inpainting이고, FD와 ID는 모달리티 사이 inpainting이다. 언어 모드에서는 생성 쪽 파라미터가 켜지지 않으므로 표준 VLM과 똑같이 동작한다.

### 3.2 벽 1 돌파 — MoT 이중 타워와 단방향 joint attention

각 decoder layer는 파라미터를 두 벌 가진다. AR 서브시퀀스를 처리하는 **reasoner**와 DM 서브시퀀스를 처리하는 **generator**다. LayerNorm, attention projection, FFN이 모두 따로다. 두 경로는 같은 사전학습 VLM 가중치에서 출발하고, AR 토큰은 reasoner로, DM 토큰은 generator로 결정론적으로 라우팅된다. 학습된 router가 없다는 점에서 MoE와 다르다.

두 타워가 만나는 곳은 attention 한 곳뿐이고, 그 방향은 비대칭이다(Eq. 7–8).

$$\mathbf{O}_{\text{AR}}=\text{Attn}_{\text{causal}}\big(\mathbf{Q}_{\text{AR}},\ \mathbf{K}_{\text{AR}},\ \mathbf{V}_{\text{AR}}\big)$$

$$\mathbf{O}_{\text{DM}}=\text{Attn}_{\text{full}}\big(\mathbf{Q}_{\text{DM}},\ [\mathbf{K}_{\text{AR}};\mathbf{K}_{\text{DM}}],\ [\mathbf{V}_{\text{AR}};\mathbf{V}_{\text{DM}}]\big)$$

- $\mathbf{Q}\_{\ast},\mathbf{K}\_{\ast},\mathbf{V}\_{\ast}$ — 각 타워가 **자기 projection**으로 만든 query, key, value
- $[\cdot\,;\cdot]$ — 시퀀스 방향 연결
- $\text{Attn}\_{\text{causal}}$ — 앞선 토큰만 보는 attention, $\text{Attn}\_{\text{full}}$ — 양방향 attention

```
 token sequence
 [ l1 ... ln EOS BOG | v(1:P) | ~v(P+1:N) | ~s | ~a ]
 |<---- AR: reasoner ---->|<------- DM: generator ------->|

 attention mask (row = query, col = key)
              K_AR          K_DM
          +-------------+-------------+
   Q_AR   |   causal    |   blocked   |
          +-------------+-------------+
   Q_DM   |    full     |    full     |
          +-------------+-------------+
```

AR은 VLM에서 물려받은 causal 성질 그대로 AR끼리만 본다. DM은 AR과 DM 전체를 양방향으로 보며, 텍스트 조건과 다른 모든 조건·생성 토큰에 자유롭게 attend한다. 논문은 AR 토큰이 DM 토큰에 의해 갱신되는 일이 없다고 못 박는다. 조건 경로가 노이즈 섞인 생성 과정에 오염되지 않게 하고, 사전학습 VLM의 텍스트 생성 능력을 보존하려는 것이다.

학습 쪽에도 장치가 하나 더 있다. Reasoner를 먼저 학습(사전학습 → Physical AI SFT)한 뒤 그 가중치를 복사해 generator 타워를 초기화하고, **generator 사전학습 동안에는 생성 전용 파라미터만 갱신하며 reasoner는 얼려 둔다.**

여기서 의문이 생긴다. 마스크가 이미 AR을 DM으로부터 막고 있는데, 왜 reasoner를 또 얼리는가?

> ### 💡 마스크는 순전파를, 동결은 역전파를 막는다
>
> 마스크가 보장하는 것은 "AR의 출력이 DM 토큰에 의존하지 않는다"는 순전파 방향뿐이다. 반대 방향은 열려 있다. DM의 attention 출력은 AR의 key·value를 입력으로 쓰므로
>
> $$\frac{\partial\mathcal{L}_{\text{DM}}}{\partial W_K^{\text{AR}}}\neq 0,\qquad \mathbf{K}_{\text{AR}}=W_K^{\text{AR}}\,h_{\text{AR}}$$
>
> - $W_K^{\text{AR}}$ — reasoner 타워의 key projection
> - $h_{\text{AR}}$ — 해당 레이어의 AR 은닉상태
>
> 즉 reasoner를 얼리지 않으면 diffusion loss의 gradient가 reasoner의 K/V projection과 그 아래 모든 AR 표현으로 흘러 들어간다. 동결은 이 두 번째 통로를 닫는다. VLA 쪽의 knowledge insulation(action expert에서 VLM으로 가는 gradient 차단)과 같은 문제를 다른 도구로 푼 셈이다.
>
> 추론에서도 이득이 있다. AR이 노이즈 토큰에 의존하지 않으므로, AR의 KV는 한 번만 계산해 모든 denoising step에서 재사용할 수 있다. LLM 서빙의 prefix caching과 같은 구조다. 논문 서빙 절의 "Reasoner tower caching" 항목이 이것에 해당하는 것으로 보인다(절 제목 기준 추론).

그렇다면 영상이나 행동을 생성할 때 reasoner는 장면을 보고 있는가? "이해 타워가 장면을 해석하고, 생성 타워가 그 해석을 받아 그린다"는 그림이 자연스럽지만, 식을 따라가면 조금 다르다.

> ### 💡 생성 모드에서 reasoner가 보는 것은 텍스트다 — 분리 축은 모달리티가 아니라 기능
>
> 영상 입력은 두 인코더로 갈라진다. 이해용 ViT는 backbone과 함께 학습되고, 생성용 VAE(Wan2.2-TI2V-5B)는 얼린 채 쓴다. 그런데 §3.1의 생성 모드 식(Eq. 3–6)에서 AR 접두부 $S_{\text{AR}}$은 언어 토큰과 특수 토큰뿐이다. 입력 이미지, 제어 영상, 로봇 관측 같은 시각 조건은 모두 **깨끗한 VAE 토큰으로 DM 쪽에** 들어간다. ViT를 거친 영상이 AR에 들어가는 것은 모델을 VLM으로 쓸 때다.
>
> 여기서 두 가지가 따라 나온다.
>
> - **같은 vision이 역할에 따라 다른 타워로 간다.** 이해용은 ViT를 거쳐 reasoner로, 생성용은 VAE를 거쳐 generator로 간다. 원조 Mixture-of-Transformers처럼 모달리티로 나눈 것이 아니라 이해 대 생성이라는 기능으로 나눈 것이다.
> - **생성·정책 모드에서 픽셀 수준의 지각은 generator가 맡는다.** reasoner는 Physical AI SFT를 거친 매우 강한 지시문 인코더에 가깝다.
>
> 정책 모드의 AR 구성은 원문에 식으로 명시돼 있지 않아, 이 해석은 Eq. 3–6에서 추론한 것이다. 이것이 π0 계열 VLA와의 결정적 차이이기도 하다(§6).

### 3.3 벽 2 돌파 — 절대 시간축 MRoPE

§2.3에서 예고한 문제로 돌아가자. 원래 MRoPE에서 시간 좌표 $t$는 프레임 순번이다. 영상만 다룰 때는 충분하지만, 영상·오디오·행동을 서로 다른 샘플링 주기로 동시에 생성해야 하는 Cosmos 3에서는 벽 2가 그대로 드러난다. 순번 한 칸이 모달리티마다 다른 물리 시간이기 때문이다.

좌표 배정부터 보자(Fig. 6).

| 토큰 | $(t, h, w)$ 배정 |
|---|---|
| 언어 (AR) | $t=h=w$, 단조 증가 — 1D RoPE로 환원 |
| ViT 영상 (AR) | Qwen3-VL의 MRoPE와 동일 |
| VAE 영상 (DM) | 세 축 모두 변화, 세그먼트마다 0부터 다시 셈 |
| 오디오, 행동 (DM) | $t$만 증가, $h=w=0$ |

핵심은 $t$의 증분이다. 논문은 초당 시간 스텝 수(TPS)를 정의한다. 영상은 프레임레이트를 VAE 시간 압축비 4로 나눈 값, 오디오는 $48000/1920\approx25$, 행동은 샘플링 주파수 그 자체다. 그리고 증분을 기준 TPS에 맞춘다(Eq. 9).

$$t_k=t_{\text{AR,end}}+g+k\cdot\delta t,\qquad \delta t=\frac{\text{TPS}_{\text{base}}}{\text{TPS}},\qquad \text{TPS}_{\text{base}}=\frac{24}{4}=6$$

- $k$ — 해당 모달리티 세그먼트 안의 토큰 순번
- $t_{\text{AR,end}}$ — 마지막 AR 토큰의 시간 인덱스
- $g$ — AR과 DM 사이의 고정 간격 (아래 설명)
- $\text{TPS}_{\text{base}}$ — 학습 데이터에서 가장 흔한 24fps 영상의 TPS

| 모달리티 | TPS | $\delta t$ | 1초당 $t$ 증가량 |
|---|---|---|---|
| 영상 24fps | 6 | 1.0 | 6 |
| 영상 16fps | 4 | 1.5 | 6 |
| 영상 30fps | 7.5 | 0.8 | 6 |
| 오디오 48kHz, hop 1920 | 25 | 0.24 | 6 |
| 행동 15Hz (DROID) | 15 | 0.4 | 6 |

마지막 열이 요점이다. $\text{TPS}\cdot\delta t=\text{TPS}_{\text{base}}=6$이 항상 성립하므로, **어느 모달리티든 물리적 1초가 시간 인덱스 6단위를 차지한다.** 시계가 통일되니 같은 순간의 행동 토큰과 영상 토큰이 attention 안에서 같은 위상 근처에 놓인다. 영상 길이와 FPS는 프롬프트에도 적어 넣어, 추론 시 원하는 시간 특성을 조건으로 줄 수 있게 한다.

간격 $g$는 경험적으로 들어간 장치다. DM 토큰을 마지막 AR 토큰 바로 다음 위치에서 시작시키자 첫 프레임에 과포화와 체커보드 아티팩트가 생겼고, Super에서 특히 심했다. 논문의 가설은 마지막 언어 토큰과 첫 프레임이 인접 위치라 시간 임베딩이 거의 같아진다는 것이다. RoPE의 저주파 성분은 한 칸 차이로는 위상이 거의 변하지 않으니 그럴 법하다. 해법은 두 서브시퀀스 사이에 $g=15000$의 고정 간격을 넣어, 추가 파라미터 없이 텍스트에서 시각으로 넘어가는 신호를 분명히 하는 것이다.

### 3.4 벽 3 돌파 — "상태 변화"로 재정의한 행동

행동은 최대 세 성분으로 구성된다(Fig. 3).

| 성분 | 의미 | 표현 |
|---|---|---|
| ego pose | 주 관측 프레임(머리 카메라, 차량)의 움직임 | 상대 포즈 9D |
| effector pose | 손목·엔드이펙터의 움직임 | 상대 포즈 9D |
| grasp state | 현재 조작 상태 | 손가락 끝 좌표 15D, 또는 그리퍼 개폐 1D |

ego와 effector는 embodiment 고유의 제어기 세부(PID 파라미터, 구동 인터페이스)를 피하려고, 연속 포즈의 차이로 만든 **pseudo-action**으로 표현한다.

$$\Delta\mathbf{T}_t=\mathbf{T}_{t-1}^{-1}\,\mathbf{T}_t\ \ \longrightarrow\ \ \big[\,\Delta p\in\mathbb{R}^3,\ r\in\mathbb{R}^6\,\big]\in\mathbb{R}^9$$

- $\mathbf{T}_t\in SE(3)$ — 시점 $t$의 포즈 (회전 + 이동)
- $\Delta p$ — 상대 이동량
- $r$ — 6D 회전 표현(회전행렬의 두 열). 오일러각이나 쿼터니언과 달리 연속 표현이라 회귀에 유리하다. 출력 후 SVD로 $SO(3)$에 사영한다

grasp만은 차분이 아니라 시점 $t$의 상태를 그대로 담는다. 이 성분들을 조합하면 embodiment마다 행동 벡터의 차원이 달라진다.

| Embodiment | 구성 | 차원 |
|---|---|---|
| 자율주행, 카메라 모션 | ego 9 | 9 |
| 단일 팔 로봇 | effector 9 + gripper 1 | 10 |
| 양팔 로봇 | (effector 9 + gripper 1) × 2 | 20 |
| 휴머노이드 | ego 9 + (effector 9 + gripper 1) × 2 | 29 |
| 사람 egocentric | head 9 + (wrist 9 + fingertip 15) × 2 | 57 |

차원이 다른 벡터는 도메인마다 따로 둔 선형 projection으로 공유 latent 공간에 올리고 내린다(Eq. 1–2). backbone은 공유한다.

$$\mathbf{z}=\mathbf{W}^{(k)}_{\text{in}}\,\mathbf{x}+\mathbf{b}^{(k)}_{\text{in}},\qquad \hat{\mathbf{x}}=\mathbf{W}^{(k)}_{\text{out}}\,\mathbf{z}+\mathbf{b}^{(k)}_{\text{out}}$$

- $k\in\{1,\dots,K\}$ — embodiment 도메인 ID
- $\mathbf{x}\in\mathbb{R}^{d^{(k)}_{\text{in}}}$ — 정규화된 행동 벡터 (차원별로 대략 $[-1,1]$로 스케일)
- $\mathbf{z}\in\mathbb{R}^{d_{\text{model}}}$ — 공유 공간의 행동 토큰
- $\mathbf{W}^{(k)},\mathbf{b}^{(k)}$ — 도메인 전용 projection, 처음부터 학습

그런데 로봇 정책이 최종적으로 내야 하는 것은 관절 명령이다. 왜 굳이 "상태의 차이"로 행동을 정의했을까?

> ### 💡 행동을 상태 변화로 정의하면 영상 코퍼스가 행동 코퍼스가 된다
>
> 제어기 독립성보다 더 큰 이유는 **데이터**다. 행동을 관절 명령으로 정의하면 행동 데이터는 로봇 텔레옵 로그에서만 나온다. 상태 변화로 정의하면 사람 손 영상(머리·손목 포즈와 21-keypoint 손 포즈)과 일반 영상(추정한 카메라 궤적)이 모두 행동 데이터가 된다.
>
> 실제 행동 mid-training 데이터 구성이 이를 보여준다(Fig. 9).
>
> | 도메인 | 시간 | 비중 |
> |---|---|---|
> | 사람 egocentric (머리 카메라 + 양손) | 41.3K h | 67.4% |
> | 자율주행 (사내 주행 로그) | 10.0K h | 16.3% |
> | 로봇 (공개 데이터셋 6종) | 5.4K h | 8.7% |
> | 카메라 모션 (사전학습 영상에서 추출) | 4.6K h | 7.5% |
>
> 합계는 8.4M 에피소드, 61.3K시간이다. 카메라 모션은 사전학습 영상에서 ViPE와 DepthAnything3로 카메라 포즈를 추정해 뽑았고, 로봇 데이터에는 실패 에피소드도 일부러 넣어 계획에서 벗어난 행동의 결과까지 보게 했다.
>
> 로봇 텔레옵은 전체의 10%도 안 된다. 벽 3의 "데이터 부족" 절반은 이렇게 **행동의 정의를 바꿔서** 넘은 것이다. LLM으로 치면 라벨 없는 텍스트를 pseudo-labeling으로 학습 데이터로 바꾸는 것과 같은 발상이다.

공통 행동 공간은 mid-training용 공용어다. 실제 배포 인터페이스는 post-training에서 새로 붙인다(§3.6).

### 3.5 학습 커리큘럼

```
 Qwen3-VL-8B / 32B
      |
      v
 [Reasoner PT 22.0M] -> [Reasoner SFT 2.2M]
                              |
                              | init generator tower (copy)
                              v
 [Gen PT: img+video+audio] -> [Gen MT: +action +transfer] -> [Post-train]
  reasoner frozen              base: Nano / Super             T2I, I2V, Policy
```

- **Reasoner** — 사전학습 22.0M 샘플(OCR 42.9%, 2D grounding 16.5% 등). projector만 먼저 맞추는 정렬 단계 없이 처음부터 전체를 함께 학습한다. SFT는 2.2M 샘플로, 절반이 video-text이며 자율주행·로봇·스마트 인프라에 집중한다(Tab. 3).
- **Generator 사전학습** — 원본 이미지 7.8B장과 영상 3B개에서 정제한 이미지 767M장, 영상 클립 347.7M개. T2I/T2V/I2V/V2V를 20/56/16/8% 비율로 섞고 오디오를 동반 생성한다. 256p/480p/720p 다해상도를 74K 토큰 packing으로 학습하며, Nano는 GB200 1024장으로 31.05T 토큰, Super는 2048장으로 17.86T 토큰을 봤다. 이 동안 reasoner는 동결.
- **Generator mid-training** — 행동과 transfer가 처음 들어온다. Nano 2.4T, Super 1.9T 토큰. 이 단계의 산출물이 공개된 base 모델 Cosmos3-Nano / Super다.
- **Post-training** — base에서 T2I, I2V, Policy 특화 모델을 각각 독립적으로 뽑는다. 아키텍처는 base와 동일하다.

mid-training의 데이터 혼합은 다음과 같다(Tab. 6).

| 스트림 | 모드 | 비중 |
|---|---|---|
| Image | T2I | 10% |
| Video | T2V, I2V, V2V | 32% |
| Video + Audio | T2(V+A), I2(V+A), V2(V+A) | 8% |
| **Action** | FD, ID, Policy | **25%** |
| General Transfer | edge, blur, depth, segmentation 제어 | 20% |
| Driving Transfer | world-scenario map 제어 | 5% |

행동을 사전학습이 아니라 mid-training에서 넣는 구조는 LLM의 mid-training(코드·수학 비중을 올려 넣는 단계)과 닮았다. 넓은 시각 prior를 먼저 굳히고, 희소하지만 값진 모달리티를 뒤에서 비중 있게 섞는다.

모델 크기는 세 가지다(Tab. 2).

| 변형 | 총 파라미터 | 기반 dense transformer | 초기화 |
|---|---|---|---|
| Edge | 4B | 2B, 28층 | 자체 scratch 학습 (Qwen3-1.7B 유사 구조) |
| Nano | 16B | 8B, 36층 | Qwen3-VL-8B |
| Super | 64B | 32B, 64층 | Qwen3-VL-32B |

"총 파라미터"는 reasoner와 generator 두 타워의 합이다. 보고서 시점에는 Nano와 Super가 공개됐고, Edge는 이후 공개로 예고됐다.

### 3.6 정책으로 특화 — Cosmos3-Nano-Policy-DROID

논문은 이 단계를 파일럿 연구로 규정한다. 플랫폼은 DROID(Franka Panda 7-DoF 팔 + Robotiq 2F-85 그리퍼)이고, 데이터는 7.6만 궤적, 350시간, 86 태스크, 564 장면이다. 360×640 해상도로 쓰고, idle 프레임 필터링과 실패 시연 제거를 적용한다.

- **초기화** — mid-trained Nano에서 이어 학습하되, 행동 인코더·행동 디코딩 MLP·행동 임베딩 토큰은 새로 초기화한다. 여기서 행동 인터페이스가 공통 pseudo-action에서 DROID의 관절 위치 명령으로 바뀐다.
- **입력** — 현재 proprioception과 3-view 관측.
- **출력** — 정책 모드로 행동 chunk와 미래 영상을 함께 생성한다.
- **배포** — 추론 최적화 후 RTX Pro 6000 두 장에서 정책 서버를 돌리고, Franky 기반 관절 위치 제어기가 예측된 32개 행동을 15Hz로 실행한다.

```
  obs(t): 3 views + proprio
      |
      v
  [ Cosmos3-Nano-Policy-DROID ] --> 32 actions @ 15 Hz (~2.1 s) + predicted video
      |
      v
  execute chunk --> new obs --> replan
```

chunk 하나는 약 2.1초다. chunk 안에서는 open-loop, chunk 사이에서는 새 관측으로 재계획하는 closed-loop다.

---

## 4. 왜 작동하는가

§1.3에서 남겨 둔 질문으로 돌아가자 — 통합 학습이 실제로 재사용 가능한 world-action prior를 주는가.

논문의 핵심 실험 설계는 통제 비교다. 행동 데이터를 보지 않은 사전학습 체크포인트에서 시작하는 **PT-init**과, 여러 도메인·모드의 행동을 본 mid-trained 체크포인트에서 시작하는 **MT-init**을 비교한다. 학습 레시피, 모델 크기, 데이터, 연산량은 모두 같게 고정한다.

| 비교 | PT-init | MT-init | 출처 |
|---|---|---|---|
| 로봇 FD (Super, PSNR) | 22.69 dB | 26.04 dB | Tab. 18 |
| LIBERO-10 새 embodiment, 500 iter | 0.0% | 24.6% | Tab. 20 |
| LIBERO-10 새 embodiment, 2000 iter | 95.2% | 97.4% | Tab. 20 |
| RoboLab 정책 | 열세 | 우세 | Tab. 19 |

LIBERO-10 실험은 3인칭 카메라와 손목 카메라를 쓰고, 체크포인트당 500 rollout(검증 태스크당 50회)으로 평가했다.

이 결과가 더 강하게 지지하는 것은 **"출발점이 다르다"** 쪽이다. 500 iter에서는 격차가 극적이지만 2000 iter에서는 2%p까지 좁혀진다. 행동 mid-training의 가치는 천장을 높이는 것보다 적응 속도에서 먼저 드러난다.

도메인 간 전이도 따로 측정했다. 언어 간 전이 연구를 본떠, 카메라·자율주행·로봇 5종·egocentric 사이의 전이 행렬을 만든다(Fig. 28). 두 도메인을 50/50으로 섞어 4000 iter, 단일 도메인을 2000 iter 학습해 평가 도메인의 학습 노출량을 맞춘 뒤, 대각 원소(단일 도메인) 대비 증감을 본다. 예컨대 자율주행 데이터를 섞자 카메라 FD PSNR이 11.96에서 12.82로 올랐다. egocentric으로 예열한 체크포인트가 AgiBot 로봇 적응의 출발점으로 쓰일 수 있는지도 따로 본다(Fig. 29). 다만 양의 전이는 도메인마다 고르지 않아, 교차 도메인 행동 데이터가 언제나 이득이라고 말할 수는 없다.

정책이 함께 예측하는 영상이 실제와 얼마나 맞는지도 확인한다. 예측된 행동 chunk를 RoboLab 시뮬레이터에서 같은 초기 상태로 실행한 영상과, 모델이 상상한 영상을 나란히 놓는다(Fig. 37).

> ### 📌 행동을 "mid-training 모달리티"로 넣고, 그 효과를 통제 비교로 분리했다
>
> 기존 VLA는 VLM 위에 행동을 사후에 붙이고, WAM은 영상 모델을 정책으로 파인튜닝한다. Cosmos 3는 행동을 영상·오디오와 같은 층위의 학습 데이터로 mid-training에 넣었다. 그리고 그 효과를 같은 레시피의 PT-init과 MT-init으로 분리해 보였다. 이 비교가 "omnimodal"이라는 주장 중 행동 부분을 받치는 핵심 증거다.

---

## 5. 실험

### 5.1 전 능력 요약

Tab. 1은 모든 능력을 한 표에 모은다(* post-trained, † closed 모델).

| 능력 | Super | Nano | 비교 대상 |
|---|---|---|---|
| Reasoning · General | 73.7 | 69.6 | Qwen3-VL-32B 72.8 / Gemini 3.1 Pro† 77.5 |
| Reasoning · Robotics | 57.8 | 55.1 | Qwen3-VL-32B 52.6 / Gemini 3.1 Pro† 58.2 |
| Reasoning · Driving | 79.3 | 76.0 | Qwen3-VL-32B 40.7 / Gemini 3.1 Pro† 47.2 |
| Text2Image | 91.36* | 84.61 | Gemini 3 Pro Image† 90.85 / Qwen-Image-2512 84.25 |
| Text2Video | 80.0 | 79.4 | Veo-3.1† 79.1 / Wan2.2-A14B 78.0 |
| Image2Video | 82.8 | 82.7 | Veo-3.1† 82.6 / Wan2.2-A14B 81.3 |
| Audio | 7.31 | 7.34 | Veo-3.1† 7.45 |
| FD · Robot | 26.0* | 25.5* | Ctrl-World 23.0 |
| Policy · Robot | – | 39.7* | π0.5 28.1 |

행동 쪽 개별 결과로는, 카메라 FD에서 MT-init이 RRE 0.142°, RTE 0.026m, ATE 0.99m로 Lingbot-World와 HY-World 1.5를 세 지표 모두에서 앞섰다(Tab. 18).

### 5.2 정책

- **RoboLab (시뮬레이션, Tab. 19)** — 120 태스크 × 10 rollout. 가장 구체적인 지시문 조건에서 평균 39.7%로, π0.5의 28.1%와 DreamZero의 25.2%를 앞섰다. 논문은 모든 지시문 세분도와 난이도에서 우위라고 적는다.
- **RoboArena (실세계, Fig. 26)** — DROID 보유자 누구나 두 정책을 이중맹검 A/B로 비교하고, 쌍별 선호를 집계해 점수를 매긴다. 2026-05-30 기준 1위다.

| 정책 | RoboArena 점수 |
|---|---|
| Cosmos3-Nano-Policy | 1870 |
| Spirit v1.6 | 1785 |
| DreamZero X | 1732 |
| WALL-OSS | 1657 |

- **MolmoSpaces (시뮬레이션, Fig. 27)** — 2026-06-20 기준 1위. RoboLab, RoboArena와 동일한 모델·하이퍼파라미터를 벤치마크별 튜닝 없이 제출했다.

> ### ⚠️ 팩트체크 — 헤드라인 수치의 한정어
>
> ⓵ **"오픈소스 기준"이 붙어야 성립한다.** Tab. 1 캡션은 특화된 오픈소스 baseline을 모든 능력에서 앞선다고 쓴다. closed 모델까지 넣으면 일반 추론(Gemini 3.1 Pro 77.5 vs 73.7), 로봇 추론(58.2 vs 57.8), 오디오(Veo-3.1 7.45 vs 7.34)에서 뒤진다. Artificial Analysis 순위도 open-weight 1위지만, proprietary 포함 시 T2I 4위, I2V 22위다(Fig. 18–19, 2026-05-28 기준).
>
> ⓶ **정책 39.7%는 "specific" 지시문 조건의 값이다.** Tab. 1에는 이 한정어 없이 실려 있다. RoboLab-120 전체 기준 제3자 비교(BFL, 2026-09)는 Cosmos 3 Nano를 36.8%로 놓는다.
>
> ⓷ **Driving 79.3 대 40.7은 in-domain SFT를 감안해야 한다.** reasoner SFT에 사내 주행 로그에서 자동 라벨링한 의사결정 영상 약 110만 개가 들어갔다. 벤치마크와 SFT 분포의 거리를 보지 않고 일반화 능력으로 읽기는 어렵다.
>
> ⓸ **"1위"는 날짜가 찍힌 스냅샷이다.** 논문도 RoboArena에 커뮤니티 평가가 더 쌓일 것이라고 적는다. RoboLab에서는 이후 BFL이 7B 규모의 FLUX 3 Action으로 42.92%를 자체 보고했다(2026-09).

---

## 6. 위치잡기 — 이웃 연구들 속에서

논문 스스로의 좌표축은 §1의 세 모델 부류다 — 지각·추론의 VLM, 시뮬레이션의 영상 생성·FD 모델, 행동의 VLA·WAM. 실험의 비교군도 부류마다 대표를 세운다. VLA로 π0.5, WAM으로 DreamZero, FD 모델로 Ctrl-World다.

아키텍처 면에서 논문이 가장 가깝다고 인정하는 것은 BAGEL(Deng et al., 2025)이다. decoder layer 구조가 비슷하되, 학습 전략, 위치 임베딩, 전체 능력 범위가 다르다고 구분한다. §3.2의 이중 타워와 §3.3의 절대 시간 MRoPE가 그 차이의 실체다.

§3.2에서 예고한 π0 계열과의 차이는 같은 DROID 위의 세 설계를 나란히 놓으면 선명해진다.

> ### 🔗 같은 DROID 위의 세 설계 — VLA, WAM, omnimodal world model
>
> | 축 | π0.5 (VLA) | DreamZero (WAM) | Cosmos 3 |
> |---|---|---|---|
> | 백본 출처 | VLM | 영상 diffusion 모델 | VLM(Qwen3-VL) 두 벌 |
> | 관측 이미지를 보는 곳 | VLM | 영상 백본 | generator (VAE 경로, §3.2 추론) |
> | 미래 영상 예측 | 없음 | 있음 | 있음 |
> | 텍스트 이해·출력 | VLM | 없음 | reasoner |
> | FD / ID 모드 | 없음 | 해당 없음 | 같은 가중치로 가능 |
> | RoboLab (specific) | 28.1 | 25.2 | 39.7 |
>
> π0 계열과 Cosmos 3는 "분리된 가중치 + 비대칭 attention"이라는 뼈대가 같다. 차이는 관측을 누가 보느냐다. π0에서는 VLM이 관측 이미지를 보고 action expert가 그 KV를 읽는다. Cosmos 3의 생성·정책 모드에서는 generator가 VAE latent로 관측을 직접 보고, reasoner는 지시문을 인코딩한다.

diffusion 쪽에서 보면 가장 가까운 조상은 SD3의 MM-DiT다. 모달리티별 가중치와 joint attention이라는 골격이 같다. 다른 점은 둘이다. 텍스트 스트림이 T5 수준 인코더가 아니라 **VLM 전체**이고, attention이 양방향이 아니라 **AR이 DM을 보지 않는 단방향**이다. 또 "모드를 마스크로 정한다"는 발상은 영상과 행동의 diffusion timestep을 분리해 policy·FD·ID를 한 모델에서 뽑는 UWM 계열과 맞닿아 있다.

---

## 7. 한계

**논문과 공식 자료가 인정한 것**

- **생성 품질** — 모델 카드는 시간적 비일관성, 부정확한 물리 상호작용, 특히 장기·고해상도 출력에서의 action-state drift를 명시한다. 명시적 물리 시뮬레이터가 없어 접촉 동역학과 물리 법칙은 근사에 그친다.
- **정책 검증 범위** — 정책은 DROID 단일 embodiment에서의 파일럿이다. Super의 정책 결과는 보고되지 않았다(Tab. 1에서 "–").
- **도메인 간 전이** — 양의 전이가 도메인마다 고르지 않다(§4).

**추가로 짚을 지점**

- **"omnimodal 통합이 정책을 돕는다"의 직접 증거가 좁다.** 통제 비교는 사실상 행동 mid-training 유무(PT-init 대 MT-init)다. 부록 ablation 목록(E.1–E.5)도 제목상으로는 오디오·transfer 데이터나 reasoner 유무가 정책 성공률에 미치는 효과를 직접 떼어 본 항목이 보이지 않는다.
- **배포 비용** — chunk마다 16B generator 전체를 돌린다. 제3자 측정(BFL)으로 real-time factor는 Cosmos 3 Nano 0.150, π0.5 0.032로, 동작 1초당 연산이 약 5배다. 측정 조건이 달라 참고치로만 봐야 한다.
- **실패 데이터의 비대칭** — mid-training에는 실패 에피소드를 넣었지만, DROID post-training은 실패 시연을 제거한 순수 BC다. 정책 단계에 복구 데이터를 만드는 장치는 없다.

---

## 8. 마치며 — 이 논문이 시사하는 것

Cosmos 3의 기여는 특정 벤치마크 1위보다, **이질적인 모달리티를 한 모델에 넣기 위한 장치 세트를 공개된 가중치와 함께 정리했다**는 데 있다. 기능별 이중 타워, 단방향 attention, 절대 시간 MRoPE, 상태 변화 기반 행동 표현 — 각각은 새롭지 않을 수 있어도, 이 조합이 VLM·영상 생성·FD·ID·정책을 한 체크포인트에서 돌린다.

LLM·diffusion 배경을 가진 독자에게는 익숙한 다리가 여럿 보인다.

- **"VLM 두 벌 + 한쪽만 생성 학습"** 은 BAGEL과 MM-DiT의 계보다. AR이 DM을 보지 않는 덕에 LLM 서빙의 prefix caching이 그대로 denoising 루프의 최적화가 된다.
- **"모드 = 마스크"** 는 T5 span corruption, MaskGIT의 연장이다. 추가 헤드 없이 FD(시뮬레이터)와 ID(라벨러)가 정책과 같은 가중치에서 나온다는 점은, 정책 평가와 데이터 라벨링을 한 모델로 해결할 수 있다는 실무적 의미를 가진다.
- **"데이터의 정의가 곧 데이터의 규모"** — 행동을 기하학적 상태 변화로 재정의한 순간, 사람 손 영상과 일반 영상이 행동 데이터가 됐다. 로봇 텔레옵이 10%도 안 되는 행동 데이터 구성은 physical AI의 데이터 전략에 대한 강한 신호다.

무엇보다, 이 논문이 가장 설득력 있게 보인 것은 "통합"이 아니라 "행동을 사전학습 단계로 끌어올리는 것"의 효과다. 오디오와 transfer까지 포함한 옴니모달 통합이 정책에 무엇을 더하는지는 아직 열린 질문으로 남아 있다.

---

## 부록 — 용어 정리

| 용어 | 정의 |
|---|---|
| **AR / DM 서브시퀀스** | 이해를 맡는 autoregressive 토큰 구간과 생성을 맡는 diffusion 토큰 구간. AR이 항상 앞에 온다 |
| **MoT 이중 타워** | 레이어마다 reasoner와 generator 두 파라미터 셋을 두고, 토큰 유형으로 결정론적 라우팅하는 구조 |
| **Dual-stream joint attention** | AR은 AR만 causal하게, DM은 AR과 DM 전체를 양방향으로 보는 비대칭 attention |
| **절대 시간 MRoPE** | 시간 좌표 증분을 $\text{TPS}_{\text{base}}/\text{TPS}$로 맞춰, 모든 모달리티에서 물리적 1초가 같은 위치 폭을 갖게 한 위치 인코딩 |
| **Pseudo-action** | 연속 포즈의 상대 변환($\mathbf{T}_{t-1}^{-1}\mathbf{T}_t$)으로 정의한 제어기 독립적 행동 |
| **FD / ID / Policy 모드** | 영상·행동 중 무엇을 깨끗하게 주느냐로 정해지는 세 가지 행동 관련 생성 모드 |
| **PT-init / MT-init** | 행동을 보지 않은 사전학습 체크포인트 / 행동을 본 mid-trained 체크포인트에서 시작하는 초기화 |

**원문** — [arXiv:2606.02800](https://arxiv.org/abs/2606.02800) · **프로젝트 페이지** — research.nvidia.com/labs/cosmos-lab/cosmos3 · **코드** — github.com/nvidia/cosmos
