---
layout: paper
lang: ko
ref: dreamzero-world-action-model
kind: paper-review
title: "World Action Models are Zero-shot Policies (DreamZero)"
date: 2026-02-21 12:00:00 -0800
paper_date: 2026-02-17
venue: "Preprint · arXiv:2602.15922"
tags: [VLA, World-Model, World-Action-Model, Video-Diffusion, Robot-Foundation-Model, Paper-Review]
authors: "Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi \"Jim\" Fan, Joel Jang"
affiliations: "NVIDIA"
summary: "웹 비디오로 사전학습된 video diffusion 모델이 미래 영상과 action을 함께 생성하게 하면, action 학습은 모방에서 inverse dynamics로 바뀌고 반복 없는 이질적 데이터만으로 새 동작과 새 환경에 일반화합니다."
paper_url: "https://arxiv.org/abs/2602.15922"
code_url: "https://github.com/dreamzero0/dreamzero"
---

> **핵심 주장** — **VLA에게 부족한 것은 반복 시연의 양이 아니라 "세계가 어떻게 움직이는가"에 대한 prior입니다.** 웹 비디오로 사전학습된 video diffusion 모델이 미래 영상과 action을 함께 생성하게 하면, action 학습은 "상태에서 행동을 모방하는 문제"에서 "상상한 미래로부터 행동을 읽어내는 문제(inverse dynamics)"로 바뀝니다. 그 결과 반복 없는 이질적 데이터만으로도 새 동작과 새 환경에 일반화합니다.

---

## 들어가며

VLA(Vision-Language-Action) 모델의 표준 레시피는 사전학습된 VLM에 action head를 붙이고 로봇 시연으로 학습하는 것입니다. RT-2, OpenVLA, π0.5, GR00T가 모두 이 틀 안에 있으며, VLM이 웹에서 배운 의미 지식을 로봇으로 옮겨오는 전략입니다.

DreamZero는 출발점을 바꿨습니다. 기존 VLA가 VLM을 백본으로 삼았다면, DreamZero는 **image-to-video diffusion 모델**(Wan2.1-I2V-14B)을 백본으로 삼고 미래 영상과 action을 하나의 모델에서 함께 생성합니다. 논문은 이 부류를 **World Action Model(WAM)** 이라 부릅니다. 아래에서는 논문의 논지를 따라가면서, 각 설계 결정의 **이유**를 이해하는 데 필요한 개념을 그때그때 함께 정리합니다.

---

## 1. 문제 — VLA는 "무엇을"은 알지만 "어떻게"는 모른다

### 1.1 VLM prior의 공백

VLM은 정적인 image-text 쌍으로 학습되기 때문에, 의미 수준의 지식은 풍부하지만 물체가 어떻게 움직이고 접촉하면 무엇이 변하는지에 대한 **시공간적(spatiotemporal) prior** 는 비어 있습니다. 논문의 예시가 이 차이를 선명하게 보여줍니다.

| 지시 | VLA 결과 | 이유 |
|---|---|---|
| "move coke can to Taylor Swift" | 성공 | 대상 식별은 웹 지식, "move"는 로봇 데이터에 있던 skill |
| "untie the shoelace" | 실패 | "풀기"라는 **동작**이 로봇 데이터에 없으면 VLM prior로 메울 수 없음 |

**VLM prior는 *what* 을 인코딩하지만, 기하·동역학·모터 제어와 정렬된 *how* 는 인코딩하지 않습니다.**

### 1.2 그 공백을 데이터로 메우는 대가

논문의 related work에 따르면 기존 VLA는 이 공백을 데이터 수집으로 메웁니다.

- **⓵ 환경 일반화** — 특정 태스크를 수백 개 환경에서 teleoperation으로 수집합니다 (π0.5).
- **⓶ 태스크 일반화** — 언어 조건부 motion primitive 라이브러리를 키웁니다 (Gemini Robotics). 다만 가능한 물리 상호작용 전체를 episode 단위 태스크 목록으로 덮는 것은 본질적으로 비현실적입니다.

이 전략은 데이터 수집 방식까지 규정합니다. VLA는 **같은 태스크를 비슷한 조건에서 반복한 구조화된 시연**에 의존합니다. 논문의 가설은 다음과 같습니다. action만 예측하는 모델은 noisy한 state–action 쌍에서 동역학을 암묵적으로 추론해야 하므로, 반복 없는 이질적 데이터로는 잘 학습하지 못합니다. 반면 video world model은 데이터의 **모든 연속 프레임 쌍**에서 학습 신호를 얻고, 웹 스케일 비디오로 물리 동역학을 이미 익혀 둔 상태입니다.

두 접근의 정보 흐름을 나란히 놓으면 다음과 같습니다.

```
  VLA :  (o, c) --> [ VLM backbone ] --> [ action head ] --> a
                          ^
                          prior = static image-text

  WAM :  (o, c) --> [ video diffusion backbone ] --+--> future video o'
                          ^                        |
                          prior = web video        +--> a
```

- VLA는 관측에서 action으로 곧장 갑니다
- WAM은 미래 영상을 함께 그리고, action은 그 미래와 정렬되도록 생성됩니다

### 1.3 그래서 논문이 던지는 질문

> 웹 비디오의 물리 동역학 prior를 정책에 그대로 상속시키면, **반복 시연 없이 이질적 데이터만으로** 새로운 동작과 새로운 환경에 일반화할 수 있는가?

비디오 모델을 정책으로 쓰는 발상 자체는 새롭지 않습니다(6절). 논문은 이것을 실제로 작동하는 WAM으로 만들려면 세 개의 벽을 넘어야 한다고 밝혔습니다.

**⛔ 벽 1 — Video–action 정렬.** 미래 영상과 모터 명령이 긴밀히 맞물려야 합니다. 별도의 video head와 action head를 순진하게 붙이면 둘이 어긋납니다.

**⛔ 벽 2 — 아키텍처 선택.** bidirectional(BD)과 autoregressive(AR) 중 무엇이 맞는지 불분명합니다. modality 정렬, 오차 누적, 추론 효율이 모두 이 선택에 걸려 있습니다.

**⛔ 벽 3 — 실시간 추론.** 14B 모델이 고차원 latent에서 반복 denoising을 하면 closed-loop 제어에 너무 느립니다. 순진하게 구현하면 action chunk 하나에 약 5.7초가 걸립니다.

2절에서는 세 벽을 이해하는 데 필요한 개념을 정리하고, 3절에서는 이 벽들을 차례로 넘는 과정을 따라갑니다.

---

## 2. 배경 — 세 벽을 읽기 위한 최소한의 개념

### 2.1 Forward dynamics, inverse dynamics, policy

로봇 학습에서 "모델"은 무엇을 조건으로 무엇을 예측하느냐에 따라 세 종류로 나뉩니다.

| 모델 | 분포 | 묻는 질문 |
|---|---|---|
| Forward dynamics (고전적 world model) | $p(s_{t+1}\mid s_t,a_t)$ | 이렇게 움직이면 세계가 어떻게 되나 |
| Inverse dynamics (IDM) | $p(a_t\mid s_t,s_{t+1})$ | 세계가 이렇게 바뀌었다면 어떤 동작을 했나 |
| Policy | $p(a_t\mid s_t,g)$ | 목표가 이것이면 무엇을 해야 하나 |

- $s_t$: 시점 $t$의 상태, $a_t$ — 행동, $g$ — 목표

DreamZero를 요약하는 문장은 "action 학습을 dense state–action 모방에서 inverse dynamics로 옮긴다"입니다. 그런데 policy든 IDM이든 결국 action을 출력합니다. **따라서 inverse dynamics로 바꾸는 것이 왜 문제를 쉽게 만드는지 짚어볼 필요가 있습니다.**

> ### 💡 VLA는 "미래를 적분한 평균"을 배우고, WAM은 그 적분을 풀어서 배운다
>
> 정책은 가능한 미래 관측 $o'$에 대해 다음처럼 분해할 수 있습니다.
>
> $$\pi(a\mid o,c)\;=\;\int \underbrace{p(a\mid o,o')}_{\text{IDM}}\;\underbrace{p(o'\mid o,c)}_{\text{future prediction}}\;do'$$
>
> - $o$: 현재까지의 관측, $o'$ — 미래 관측, $c$ — 언어 지시, $a$ — action chunk
>
> **VLA는 좌변을 직접 학습합니다.** 같은 $(o,c)$에 유효한 전략이 여럿이면(왼손/오른손, 위에서/옆에서 잡기) 그 다봉성이 전부 action 분포로 접혀 들어옵니다. 모드마다 반복 시연이 필요해지는 이유입니다.
>
> **WAM은 두 인자를 나눠 학습합니다.** 다봉성은 $p(o'\mid o,c)$가 떠안는데, 이 부분은 웹 비디오 사전학습이 이미 잘합니다. 미래가 주어진 뒤의 $p(a\mid o,o')$는 거의 결정적이라 엔트로피가 낮습니다.
>
> diffusion으로 비유하면 text-to-image(조건이 약해 분포가 넓음)와 sketch-conditioned 생성(조건이 강해 분포가 좁음)의 차이입니다. WAM에서는 예측된 미래 영상이 sketch 역할을 합니다.
>
> 이 적분식은 논문에 직접 나오지 않으며, "inverse dynamics로 옮긴다"는 주장의 근거를 명시적으로 풀어 쓴 것입니다. 다만 이 분해가 성립하려면 IDM이 언어 $c$를 필요로 하지 않아야 합니다. 이 가정은 3.1절의 식 (1)에서 다시 다룹니다.

### 2.2 AR video diffusion과 teacher forcing — 벽 2의 배경

비디오 diffusion 모델이 영상을 만드는 방식은 두 가지입니다.

| | Bidirectional (BD) | Autoregressive (AR, chunk-wise) |
|---|---|---|
| 생성 단위 | 고정 길이 클립 전체를 한 번에 | chunk 단위로 순차 생성 |
| 과거 맥락 | 같은 클립 안에서만 | KV cache로 임의 길이 context |
| 학습 | 클립 전체가 같은 noise level | chunk마다 독립 timestep, 이전 chunk는 clean (teacher forcing) |
| 약점 | 길이가 고정이라 subsampling 문제 | 추론 시 자기 출력을 다시 먹으며 오차 누적 |

AR은 LLM 학습과 같은 구조입니다. 학습 때는 정답 prefix(clean한 이전 chunk)를 조건으로 주고, 추론 때는 자기가 생성한 것을 조건으로 씁니다. 따라서 약점도 학습과 추론의 조건이 달라지는 **exposure bias** 로 같습니다. 논문이 인용하는 Self Forcing 같은 연구가 AR video diffusion의 이 train-test gap을 따로 다루는 이유입니다. DreamZero가 이 약점을 어떻게 피하는지는 3.3절에서 다룹니다.

### 2.3 Flow matching 표기, 그리고 step 수가 곧 지연이다 — 벽 3의 배경

DreamZero는 flow matching으로 학습합니다. 이 논문의 표기는 다음과 같습니다.

$$z_t = t\,z_1 + (1-t)\,z_0,\qquad z_0\sim\mathcal{N}(0,I),\qquad v = z_1 - z_0$$

- $z_1$: clean latent, $z_0$ — gaussian noise
- $t\in[0,1]$: timestep. **$t=1$이 clean, $t=0$이 순수 노이즈**입니다 (반대 규약을 쓰는 문헌도 있으니 주의)
- $v$: 네트워크가 회귀하는 목표 velocity

추론은 $t=0$의 노이즈에서 출발해 velocity를 적분하며 $t=1$로 갑니다. step마다 DiT 전체를 한 번 forward하므로, 지연은 대략 **(denoising step 수) × (DiT forward 비용)** 입니다. 14B DiT에 16 step이면 벽 3이 왜 생기는지 바로 보입니다. step 수를 줄이려 할 때 무엇이 깨지는지는 3.4절에서 다룹니다.

---

## 3. 방법 — DreamZero

전체 구조는 다음과 같습니다.

```
  o_0:l    -- VAE (frozen) ---------+
  c        -- text enc (frozen) ----+
  q_l      -- state enc (new) ------+----> AR DiT : Wan2.1-I2V-14B (trainable)
  noisy a  -- action enc (new) -----+            |
                                                 +--> video latent o_l:l+H
                                                 +--> action dec (new) --> a_l:l+H
```

- 입력은 관측 영상, 언어, proprioception, 그리고 denoise할 noisy action입니다
- 단일 AR DiT가 chunk 단위로 미래 영상 latent와 action을 함께 denoise합니다
- 실행 후에는 실제 관측을 KV cache에 넣고 다음 chunk로 넘어갑니다

### 3.1 식 (1): 분해와 단일 모델 — 벽 1을 넘기

논문은 문제를 다음처럼 정식화하였습니다.

$$\pi_\theta(o_{l:l+H},a_{l:l+H}\mid o_{0:l},c,q_l)=\underbrace{\pi_\theta(o_{l:l+H}\mid o_{0:l},c,q_l)}_{\text{video prediction}}\;\underbrace{\pi_\theta(a_{l:l+H}\mid o_{0:l+H},q_l)}_{\text{IDM}}$$

- $o_{0:l}$: 현재까지의 관측 영상, $o_{l:l+H}$ — 예측할 미래 영상
- $a_{l:l+H}$: action chunk, $H$ — 고정 horizon
- $c$: 언어 지시, $q_l$ — proprioception
- $l$: trajectory에서 무작위로 뽑은 시점
- 원문은 이 분포를 $\pi_0$로 표기하지만, π0 모델과 헷갈리지 않도록 여기서는 $\pi_\theta$로 씁니다

**벽 1에 대한 답은 이 분해를 두 모델로 구현하지 않는 것입니다.** 별도의 video model과 IDM을 두는 대신(Li et al., 2026; Pai et al., 2025의 방식), 하나의 모델이 [영상, action]을 공유 objective로 joint denoise합니다. 정렬을 두 head의 사후 결합으로 맞추지 않고, 같은 attention 안에서 함께 생성해 확보한다는 발상입니다. 다만 논문의 근거는 "We believe"이며, 분리형과 비교한 ablation은 없습니다.

구체적 설계는 다음과 같습니다.

| 항목 | 선택 | 이유 |
|---|---|---|
| 추가 파라미터 | state encoder, action encoder/decoder만 | 비디오 모델의 일반화 능력을 보존 |
| multi-view | 여러 뷰를 한 프레임에 이어붙임 | 백본 구조를 건드리지 않음 |
| 학습 범위 | DiT 전체 학습, text/image encoder·VAE는 freeze | LoRA는 결과가 나빴음 (각주 7) |
| action 표현 | relative joint position, idle action 제거 | — |
| timestep | chunk 안에서 video·action이 같은 $t$를 공유 | 학습 초반 수렴이 빠름 |

마지막 행은 눈여겨볼 만합니다. 최근 WAM들(Kim et al., 2026; Li et al., 2025a 등)이 modality별로 timestep을 분리하는 방식이었다면, DreamZero는 의도적으로 공유합니다. 이 선택은 3.4절에서 다시 뒤집힙니다.

그런데 식 (1)의 우변을 다시 보면 이상한 점이 있습니다. **IDM 항에서 언어 $c$가 사라졌습니다.** chain rule을 그대로 적용하면 남아 있어야 합니다.

> ### 💡 식 (1)은 항등식이 아니라, 가정 하나를 품은 분해다
>
> chain rule만 적용하면 두 번째 항은 다음이 됩니다.
>
> $$\pi_\theta(a_{l:l+H}\mid o_{0:l+H},\,c,\,q_l)$$
>
> 식 (1)은 여기서 $c$를 지웠습니다. 즉 $a\perp c\mid(o_{0:l+H},q_l)$, 다시 말해 "미래 영상과 자세가 주어지면 언어는 action에 추가 정보를 주지 않는다"는 조건부 독립 가정이 들어 있습니다. 2.1절의 적분식도 같은 가정 위에 서 있습니다.
>
> | 경우 | 가정의 성립 |
> |---|---|
> | 영상이 동작을 시각적으로 특정함 (대부분의 pick-and-place) | 성립 |
> | 같은 시각 궤적인데 힘 조절이 다름 ("살살 눌러") | 깨질 수 있음 |
> | 접촉 지점이 가려져 영상이 동작을 과소결정함 | 깨질 수 있음 |
>
> 다만 실제 모델은 두 항을 따로 학습하지 않으므로 action token도 attention으로 $c$를 봅니다. **따라서 식 (1)은 구조적 제약이라기보다 설계의 정당화에 가깝다고 판단됩니다.**

### 3.2 AR chunk-wise 학습 — 벽 2를 넘기

논문은 AR의 이점으로 세 가지를 제시하였습니다. KV cache로 추론이 빠르고, 관측 이력을 다음 생성의 guidance로 쓸 수 있으며, BD의 modality 정렬 문제를 피합니다. 이 중 세 번째가 결정적인데, 부록 B의 예시로 보면 다음과 같습니다.

긴 demo의 특정 구간에 "put the black objects into the drawer"라는 캡션이 붙어 있다고 가정합니다.

| 방식 | 무슨 일이 일어나나 | 결과 |
|---|---|---|
| BD, subsampling 없이 | 고정 길이 클립이 태스크 구간의 일부만 덮어, 언어가 아직 영상에 나오지 않은 동작을 묘사하게 됨 | 언어 추종 저하 |
| BD, 캡션 구간에 맞춰 subsampling | closed-loop 학습은 태스크 중간의 임의 지점에서 시작해야 하는데, 중간에서 구간을 맞추면 native FPS가 왜곡됨 | video–action 정렬 저하 |
| AR | 앞부분은 context로 넣고 뒷부분만 생성 | 캡션 대응과 native FPS를 둘 다 유지 |

**BD가 "언어 정렬"과 "시간 해상도" 중 하나를 포기해야 하는 딜레마에 빠지는 방식이었다면, AR은 context를 받는다는 구조 덕분에 이 딜레마 자체를 피합니다.**

chunk 규격(부록 C, AgiBot 기준)은 다음과 같습니다.

```
  |<-- chunk 1 -->|<-- chunk 2 -->|<-- chunk 3 -->|<-- chunk 4 -->|
  | 2 latent fr.  | 2 latent fr.  | 2 latent fr.  | 2 latent fr.  |
  | 48 actions    | 48 actions    | 48 actions    | 48 actions    |
  | 1.6 s         | 1.6 s         | 1.6 s         | 1.6 s         |
  |<------------ max context: 8 latent = 33 raw frames = 6.6 s -->|
```

- video 5 FPS, action 30Hz. chunk당 latent frame $K=2$, action horizon $H=48$이라 chunk당 1.6초입니다
- 논문은 $K=2$가 $K=1$보다 나았다고 보고하였습니다
- DROID는 action 15Hz, $H=24$로 역시 chunk당 1.6초입니다
- 기본 chunk 수는 4개이고, 최대 visual context는 6.6초입니다

**또 하나의 설계는 AR을 video modality에만 적용한다는 점입니다.** 논문은 그 이유를 closed-loop action 예측에서 오는 오차 전파를 피하기 위해서라고 밝혔습니다.

학습 objective는 chunk별 flow matching입니다. 먼저 chunk $k$의 noisy 입력을 만듭니다.

$$z^k_{t_k}=t_k\,z^k_1+(1-t_k)\,z^k_0,\qquad a^k_{t_k}=t_k\,a^k_1+(1-t_k)\,a^k_0$$

- $z^k_1$: chunk $k$의 clean 영상 latent, $a^k_1$ — 정규화된 clean action
- $z^k_0,\ a^k_0\sim\mathcal{N}(0,I)$: 노이즈
- $t_k$: chunk $k$의 timestep. **chunk 안에서는 video와 action이 공유하고, chunk끼리는 독립**입니다

그리고 두 modality의 velocity를 함께 회귀합니다.

$$\mathcal{L}(\theta)=\mathbb{E}\Big[\frac{1}{K}\sum_{k=1}^{K}w(t_k)\,\big\lVert u_\theta([z^k_{t_k},a^k_{t_k}];\,\mathcal{C}_k,c,q_k,t_k)-v^k\big\rVert^2\Big],\qquad v^k=[z^k_1,a^k_1]-[z^k_0,a^k_0]$$

- $u_\theta$: video와 action의 velocity를 함께 내는 DiT
- $\mathcal{C}_k$: 이전 chunk들의 clean context (teacher forcing)
- $q_k$: chunk $k$의 proprioception, $c$ — 언어
- $w(t_k)$: timestep 가중치, $v^k$ — 목표 velocity

학습은 trajectory 단위로 한 번에 하고, attention mask로 현재 noisy chunk가 이전 clean chunk만 보게 합니다. LLM이 가변 길이 시퀀스를 causal mask로 한 번에 학습하는 것과 같은 방식입니다.

> ### ⚠️ 팩트체크 — 학습 objective 주변의 표기 비일관
>
> 세 곳이 서로 맞지 않습니다.
>
> - **context에 action이 들어가는가.** 식 (3)과 Algorithm 1은 $\mathcal{C}_k$를 과거 chunk의 clean 영상 latent와 clean action $a^j_1$의 쌍으로 정의합니다. 반면 본문은 "AR은 video에만"이라고 하고, Fig. 14와 Algorithm 2의 추론 context도 영상뿐입니다. 본문과 Fig. 14 쪽이 의도로 보입니다.
> - **$K$의 이중 사용.** 부록 C에서 $K$는 chunk당 latent frame 수(=2)인데, 식 (3)에서는 chunk에 대한 합의 범위로 쓰입니다. Algorithm 1은 같은 자리에 $M$을 씁니다.
> - **timestep 규약.** Algorithm 2는 clean GT를 주입하는 호출에 $t=0$을 붙이는데, 식 (2)의 규약에서 $t=0$은 순수 노이즈입니다.

### 3.3 추론 — 실제 관측을 KV cache에 되먹이기

```
  [chunk k]      KV cache = {GT_0, ..., GT_k}
                    |
                    v
               joint denoise  --+--> video latent Z_k+1  --> discarded
                                +--> action Y_k+1        --> filter --> robot (async)
  [chunk k+1]  real obs --VAE--> GT_k+1 --forward(update KV)--> KV cache = {GT_0, ..., GT_k+1}
```

- **⓵ 생성** — 노이즈에서 출발해 [영상, action]을 joint denoise합니다
- **⓶ 실행** — clean action만 꺼내 smoothing한 뒤 로봇에서 비동기로 실행합니다. **예측 영상 latent는 버립니다**
- **⓷ 되먹임** — 실제 카메라 관측을 VAE로 encode하고, forward해서 그 KV를 cache에 씁니다

그런데 2.2절에서 본 AR의 약점을 떠올리면 의문이 생깁니다. AR 비디오 생성은 모델이 자기가 그린 프레임을 다시 조건으로 먹기 때문에 작은 오차가 눈덩이처럼 누적됩니다. **DreamZero도 AR인데 이 문제를 겪지 않는다고 하는 이유는, 추론 시 context에 들어가는 것이 다르기 때문입니다.**

> ### 💡 closed-loop에서는 환경이 추론 시에도 teacher다
>
> | 단계 | context에 들어가는 것 | 결과 |
> |---|---|---|
> | 학습 (teacher forcing) | GT 프레임 | — |
> | 순수 비디오 생성의 추론 | 자기가 생성한 프레임 | train-test 불일치 → 오차 누적 |
> | DreamZero의 추론 | **실제 카메라 관측** | 학습 조건과 동일 |
>
> LLM으로 말하면, 문장을 한 덩어리 생성할 때마다 다음 스텝 전에 그것을 "정답 continuation"으로 바꿔치기해 주는 셈입니다. 텍스트에서는 불가능하지만, 로봇에서는 세계가 실제 다음 프레임을 공짜로 렌더링해 주기 때문에 자연스럽습니다. 즉 예측 영상은 상태로 남지 않고 **쓰고 버리는 계획**이 됩니다.
>
> 엄밀히 말하면 "예측 프레임을 GT로 교체"한다기보다, 예측 프레임은 cache에 **애초에 들어가지 않고** GT만 쓰입니다. 논문은 이를 "WAM 고유의 이점"이라 부르고 3절의 세 가지 핵심 설계 중 하나로 내세웠습니다. 다만 이 장치만 떼어낸 ablation은 없습니다.

이 구조 덕분에 DreamZero는 visual history를 기억으로 쓰는 **stateful policy** 가 됩니다. 다만 memory가 있어야만 풀리는 태스크는 평가하지 않았습니다(각주 2).

### 3.4 실시간화 — 벽 3을 넘기

순진한 구현이 chunk당 5.7초 걸리는 원인은 세 가지입니다.

| 병목 | 내용 |
|---|---|
| 반복 denoising | 부드러운 action을 위해 16 step |
| 백본 크기 | 14B DiT |
| 순차 실행 | 추론하는 동안 로봇이 멈춤 |

어차피 예측 영상은 버리므로, **action만 생성하면 빠르지 않을까** 하는 의문이 생길 수 있습니다. 논문은 각주 3에서 이를 부정하였습니다. 14B 규모에서는 이득이 미미한데, 2.3절에서 본 대로 지연은 step 수와 DiT block 수가 지배하기 때문입니다. 또한 두 modality를 함께 학습했으므로 action step만 순진하게 줄이면 품질이 떨어집니다. **결국 줄여야 하는 것은 step 수 자체이며, 이것이 뒤에서 볼 DreamZero-Flash의 동기입니다.**

**비동기 실행.** 첫 단계는 추론과 실행을 분리하는 것입니다. 모션 컨트롤러는 가장 최근 action chunk를 계속 실행하고, 추론은 최신 관측으로 동시에 돕니다. 이렇게 하면 제약이 "로봇이 움직이기 전에 추론이 끝나야 한다"에서 "현재 chunk(1.6초)가 소진되기 전에 끝나야 한다"로 완화됩니다. 논문은 충분한 overlap을 위해 약 200ms 이하를 목표로 잡았습니다.

**스택 전체의 최적화.** 그 위에 최적화를 누적합니다 (Table 1, H100 baseline = 1×).

| 층위 | 기법 | 누적 (GB200) |
|---|---|---|
| — | baseline | 1.1× |
| System | CFG parallelism — conditional/unconditional forward를 GPU 2장에 분산 (step당 지연 −47%) | 1.8× |
| System | DiT caching — 연속 velocity의 cosine similarity가 임계값을 넘으면 재사용 (실효 16→4 step) | 5.4× |
| Implementation | torch.compile + CUDA Graphs | 10.9× |
| Implementation | cuDNN attention, scheduler 연산의 GPU 이전 | 14.8× |
| Implementation | NVFP4 양자화 (QKV·Softmax는 FP8, 비선형 연산은 FP16) | 16.6× |
| Model | **DreamZero-Flash** (1-step) | **38×** |

**결과적으로 5.7초가 150ms가 되어 약 7Hz 제어가 가능해집니다.** DiT caching과 양자화를 제외한 나머지는 수학적으로 baseline과 동등합니다.

**DreamZero-Flash — modality별 noise schedule 분리.** step 수를 줄이면, few-step 추론에서는 "영상은 아직 noisy한데 action은 clean해야 하는" 상황이 됩니다. 그런데 학습 중에는 두 modality가 항상 같은 noise level이었습니다(3.1절의 timestep 공유). 이 train-test 불일치를 학습 분포로 메우는 것이 Flash입니다.

$$t^{\text{video}}_k = 1-\eta,\quad \eta\sim\text{Beta}(\alpha,\beta),\ \alpha>\beta,\qquad t^{\text{action}}_k\sim\mathcal{U}(0,1)$$

- $\eta$: Beta 분포 샘플. $\alpha>\beta$면 1 근처에 몰립니다
- 실제 설정은 $\text{Beta}(7,1)$입니다. $\mathbb{E}[\eta]=0.875$이므로 $\mathbb{E}[t^{\text{video}}_k]=0.125$이고, 2.3절의 규약상 **대부분 고노이즈**입니다 (공유 설정에서는 평균 0.5)
- action timestep은 그대로 균등분포입니다

이렇게 하면 학습 중에 "noisy한 시각 context로부터 clean action을 예측하는" 상황을 자주 겪게 되어, 1-step 추론 regime과 맞아떨어집니다. Flash는 본 학습이 끝난 뒤 마지막 stage로 적용합니다.

| Table 3 (table bussing) | step | task progress | 지연 |
|---|---|---|---|
| DreamZero | 4 | 83% ± 6.1 | 350ms |
| DreamZero | 1 | 52% ± 10.2 | 150ms |
| DreamZero-Flash | 1 | **74%** ± 10.1 | 150ms |

**Flash의 기여는 분리 그 자체가 아니라, 1-step 추론 regime에 맞춰 video noise를 고노이즈 쪽으로 편향시킨 분포 설계입니다.** 3.1절에서 봤듯 modality별 timestep 분리는 이전 WAM들이 이미 쓰던 방식이고, DreamZero 본체는 수렴 속도 때문에 오히려 공유로 되돌렸습니다.

그런데 1-step이면 현재 chunk의 영상 token은 순수 노이즈($t=0$)에서 출발합니다. **그렇다면 action이 어떤 "미래"를 보고 IDM을 하는지가 문제가 됩니다.**

> ### 💡 Flash에서 visual plan은 "그려진 영상"에서 "네트워크 내부 표현"으로 옮겨간다
>
> **Flash에서 계획은 denoise된 영상이 아니라 feature로 존재합니다.** 1-step forward에서 입력된 영상 token 자체에는 픽셀 정보가 없습니다. 다만 DiT 내부에서 그 token의 hidden state는 "이 노이즈를 어떤 미래로 보낼지", 즉 clean latent를 향한 velocity를 계산하고 있습니다. action token은 attention으로 이 hidden state를 봅니다.
>
> 이렇게 보면 Flash는 명시적 visual planning에서 "과거의 clean context + 암묵적 미래"로 action을 뽑는 쪽으로 한 걸음 이동한 셈입니다. 4-step 83%가 1-step Flash 74%로 내려가는 것은 이 해석과 부합합니다. 식 (1)의 IDM 해석이 가장 문자 그대로 성립하는 것은 multi-step 추론일 때입니다.
>
> 논문에 없는 해석이며, 내부 표현을 분석한 실험으로 뒷받침된 것은 아닙니다.

마지막으로 **action chunk smoothing**을 적용합니다. action chunk를 cubic interpolation으로 2배 upsample하고, Savitzky-Golay filter(window 21, 3차)로 고주파 노이즈를 누른 뒤, 원래 해상도로 downsample합니다.

---

## 4. 왜 작동하는가

논문의 자기설명은 네 가지로 정리됩니다.

| 설명 | 내용 | 근거 |
|---|---|---|
| ⓵ prior 상속 | video prediction은 이미 웹 데이터로 최적화되어 있으므로, 로봇 embodiment의 영상 예측과 action 추출만 추가로 배우면 됨 | 3.1절 |
| ⓶ 실패는 영상에서 온다 | 대부분의 실패는 action 추출이 아니라 video 생성 오류에서 나옴. 정책은 틀린 영상 계획도 충실히 실행함 | Fig. 16 |
| ⓷ 다양성이 IDM을 만든다 | video 쪽은 대부분 상속받으므로 병목은 IDM임. 견고한 IDM에는 다양한 맥락의 state–action 대응이 필요하고, 반복 데이터는 이것이 부족함 | Table 4 |
| ⓸ 백본 크기 | 작은 모델은 visual hallucination이 잦고, 그것이 잘못된 action으로 전파됨 | Table 4 |

⓶의 사례가 구체적입니다. "마커를 집어 화이트보드에 선을 그어라"에서 모델은 마커를 반대 손에 건네는 영상을 그렸고, 로봇도 그대로 건넸습니다. "오븐에 크루아상을 구워라"에서는 오븐을 열기 전에 빵부터 집는 영상을 그렸고, 로봇은 빵을 든 채 오븐 앞에서 멈췄습니다. 논문은 여기서 "로봇 능력의 개선은 비디오 생성 능력의 개선으로 환원된다"고 결론 내렸습니다.

여기에 두 가지를 덧붙일 수 있습니다.

- **2.1절의 적분식이 "왜 비반복 데이터가 WAM에게만 통하는가"를 설명합니다.** VLA는 $(o,c)$마다 모드를 덮을 표본이 필요합니다. 반면 WAM의 IDM은 태스크 라벨과 무관하게 **모든 연속 프레임 쌍이 supervision** 이 됩니다. 태스크 라벨의 반복도가 덜 중요해지는 이유입니다.
- **⓶는 약점 보고처럼 보이지만, 사실 식 (1) 분해를 가장 강하게 지지하는 증거입니다.** action이 영상에 종속되어 있지 않다면 영상 오류가 이렇게 충실히 action으로 옮겨가지 않을 것입니다. 동시에 정책의 상한이 비디오 모델의 물리적 그럴듯함에 묶인다는 뜻이기도 합니다.

> ### 📌 로봇 능력 = 비디오 생성 능력 + 얇은 action 판독
>
> **DreamZero의 가장 큰 차별점은 로봇 전용으로 배우는 부분을 "영상에 동기화된 action 판독"으로 얇게 만든 것입니다.** 이 설계에서는 **비디오 백본을 키우는 것이 곧 정책을 키우는 것**이 됩니다. 같은 데이터에서 DreamZero는 5B → 14B로 21% → 50%가 되었지만, 같은 크기로 키운 VLA는 5B든 14B든 0%에 머물렀습니다(5.5절). 로봇 foundation model의 스케일링 축이 "로봇 데이터의 양"에서 "비디오 모델의 품질"로 일부 옮겨갈 수 있다는 신호로 판단됩니다.

---

## 5. 실험

### 5.1 설정

| 항목 | 내용 |
|---|---|
| 사전학습 데이터 | AgiBot G1(mobile bimanual) 약 500시간. 22개 환경, 7,193 episode, episode당 평균 4.4분·약 42개 subtask. Franka는 DROID |
| 수집 방식 | 태스크가 50 episode에 도달하면 목록에서 제거하고 작업자가 새 태스크를 제안하게 해 long-tail을 강제 |
| 사전학습 | embodiment별 별도 학습, 100K step, batch 128 |
| Post-training | 셔츠 개기 33시간, 과일 포장 12시간, 테이블 정리 40시간. 태스크별 50K step |
| 비교군 | GR00T N1.6, π0.5를 각각 ⓐ scratch(VLM 가중치만) ⓑ pretrained(공식 체크포인트를 같은 데이터로 continual training)로 준비. batch와 step 수를 맞춤 |
| 평가 조건 | 기본이 **unseen environment + unseen object**. 평가 장소가 데이터 수집 지역과 다름 |
| seen / unseen 태스크 | "동작 + 물체 종류" 조합으로 정의. 셔츠 색이 바뀌면 seen, 양말 개기는 unseen |
| 지표 | 태스크당 8 rollout(로봇 4대), 체크포인트당 80 rollout. 주 지표는 **task progress**(부분 점수) |

### 5.2 일반화 — seen, unseen, post-training

AgiBot G1 결과(평균 task progress, %)는 다음과 같습니다.

| | GR00T N1.6 scratch | GR00T N1.6 pretrained | π0.5 scratch | π0.5 pretrained | DreamZero |
|---|---|---|---|---|---|
| Seen tasks (Fig. 8) | 0.6 | 8.4 | 0 | 27.4 | **62.2** |
| Unseen tasks (Fig. 9) | 0.7 | 5.0 | 0 | 16.3 | **39.5** |
| Post-training 3종 평균 (Fig. 10) | 9.8 | 53.3 | 0.5 | 79.8 | **90.5** |

- scratch VLA는 같은 이질적 데이터에서 거의 0%입니다. 쉬운 pick-and-place에서도 올바른 물체 쪽으로 손을 뻗을 뿐 제대로 상호작용하지 못합니다
- unseen 10개 태스크 중 강한 것은 "Remove Hat from Mannequin" 85.7, "Shake Hands" 59.2입니다
- post-training은 셔츠 개기 동률(92.5 vs 92.5), 과일 포장 큰 우위(96 vs 71), 테이블 정리 소폭 우위(83 vs 76)입니다. unseen 환경에서 평가하므로, 환경 일반화가 post-training 후에도 유지된다는 근거가 됩니다
- 정성적으로, pretrained VLA는 지시와 무관하게 물체로 손을 뻗어 잡으려는 경향을 보였습니다. 지배적 학습 행동(pick-and-place)에 과적합했다는 해석입니다

DROID-Franka의 unseen 태스크(DROID에 없는 동사 20개)에서는 task progress 49 / 31 / 33, 성공률 22.5 / 12.5 / 7.5입니다(DreamZero / GR00T N1.6 / π0.5 순).

> ### ⚠️ 팩트체크 — "over 2×"는 AgiBot 기준이고, 지표는 부분 점수다
>
> **"2배 이상"은 AgiBot 결과에 대한 서술로 읽어야 합니다.** abstract는 실로봇 실험에서 VLA 대비 2배 이상의 일반화 향상을 주장하였습니다. AgiBot에서는 seen 2.27×(62.2/27.4), unseen 2.42×(39.5/16.3)로 주장과 일치합니다. 반면 DROID unseen의 task progress는 49/33 ≈ 1.5×입니다.
>
> 또한 주요 수치가 모두 task progress라서 완수율과는 다릅니다. DROID에서 DreamZero의 task progress 49와 성공률 22.5가 보여주듯, 둘은 크게 벌어질 수 있습니다.

### 5.3 Cross-embodiment — action 없는 영상으로 배우기

다른 embodiment의 영상에는 action 라벨 없이 **video prediction objective만** 걸고, 사전학습 데이터와 1:1로 섞어 10K step co-train합니다. 3.1절의 분해로 보면 이 데이터는 식 (1)의 첫 항(video prediction)만 강화합니다.

| Table 2 (unseen 9개 태스크) | task progress |
|---|---|
| DreamZero | 38.3% ± 7.6 |
| + Human2Robot (egocentric 영상 12분) | 54.3% ± 10.4 |
| + Robot2Robot (YAM 영상 20분) | 55.4% ± 9.5 |

- 향상 폭이 가장 큰 것은 Robot2Robot입니다. 논문은 YAM과 AgiBot이 둘 다 bimanual parallel gripper라 embodiment gap이 좁기 때문으로 보았습니다
- 형태 차이가 크고 시점이 흔들리는 사람 영상도 거의 같은 폭으로 향상시켰습니다

> ### ⚠️ 팩트체크 — 이 실험이 보여주는 것의 범위
>
> - **전이 영상이 평가 태스크 그 자체입니다.** 9개 unseen 태스크를 태스크당 8개씩 시연한 72개 trajectory입니다. "action 데이터 기준으로 unseen인 태스크를 영상으로 보여줬을 때"의 효과이며, zero-shot 전이는 아닙니다.
> - **추가 학습량의 대조군이 없습니다.** baseline은 추가 10K step을 거치지 않은 체크포인트입니다. 사전학습 데이터만으로 10K step을 더 돌린 대조군은 없습니다.
> - **"over 42%"의 정확한 값.** 상대 향상은 Robot2Robot +44.6%, Human2Robot +41.8%입니다.

### 5.4 Few-shot embodiment 적응

AgiBot으로 사전학습한 체크포인트를 새 로봇(YAM)의 55개 trajectory, 11개 태스크, 약 30분 분량의 play data로 post-train했습니다. 논문은 새 물체(호박, 곰인형, 컵라면, 종이봉투 등)가 등장하는 pick-and-place 변형에서 언어 추종을 유지했다고 보고하였습니다.

논문은 이 효율의 이유로 두 가지를 들었습니다. 두 embodiment의 시각적 유사성, 그리고 더 근본적으로 **IDM 학습이 직접적인 policy 학습보다 본질적으로 표본 효율적일 수 있다**는 점입니다(2.1절). 실패 역시 action 추출이 아니라 영상 예측 오류에서 주로 나왔습니다.

다만 이 결과는 **정량 표 없이 Fig. 12의 정성 결과로만** 제시됩니다. "zero-shot 일반화 유지"도 새 동작이 아니라 물체 수준 일반화에 대한 관찰입니다.

### 5.5 Ablation

모든 ablation은 계산 제약 때문에 50K step, batch 32로 학습하고 PnP Easy 태스크에서만 평가했습니다(Table 4).

| 질문 | 비교 | task progress |
|---|---|---|
| 데이터 다양성 | 반복 데이터(70 태스크) vs 다양한 데이터, 각 500시간 | 33% vs **50%** |
| 모델 크기 | DreamZero 5B vs 14B | 21% vs **50%** |
| 같은 크기의 VLA | 5B / 14B (8B·32B VLM의 앞쪽 절반 block + DiT action module) | 0% / 0% |
| BD vs AR | 14B | 50% vs 50% |

- **다양성** — 같은 시간이라도 다양한 데이터가 낫습니다. 쉬운 pick-and-place에서도 그렇습니다
- **크기** — VLA는 크기를 키워도 이질적 데이터를 소화하지 못하고 물체 근처에서 맴돌기만 합니다. 모델 용량만으로는 해결되지 않는다는 뜻입니다
- **BD vs AR** — task progress는 같지만 AR 쪽 motion이 눈에 띄게 부드럽고, KV cache 덕에 추론이 3–4배 빠릅니다

> ### ⚠️ 팩트체크 — Ablation을 읽을 때 주의할 점
>
> - **HTML과 PDF의 불일치.** arXiv HTML 판 Table 4는 VLA 행을 50%로 표시하지만, 본문과 PDF는 0%입니다. HTML 변환 오류로 보입니다.
> - **"scaling"의 근거는 두 점입니다.** 5B와 14B, 그것도 PnP Easy 한 범주입니다. 논문도 6절에서 scaling law의 부재를 인정하였습니다.
> - **5B 백본의 정체가 불분명합니다.** 논문은 "Wan2.1-I2V-5B-480P"라고 썼지만, Wan2.1의 공식 체크포인트는 1.3B와 14B이고 5B는 Wan2.2-TI2V-5B(압축률이 다른 VAE 사용)에 있습니다. 후자라면 크기 비교에 모델 계열의 차이가 섞입니다.

---

## 6. 위치잡기 — 이웃 연구들 속에서

DreamZero는 여러 계보의 교차점에 있습니다.

| 계보 | 대표 연구 | DreamZero와의 관계 |
|---|---|---|
| VLA | RT-2, OpenVLA, π0/π0.5, GR00T N1, Gemini Robotics | 비교군. VLM prior의 공백이 출발점 |
| 영상 생성 후 test-time에 action 추출 | UniPi (Du et al., 2023), AVDC (Ko et al., 2024), VLP (Du et al., 2024) | 영상 생성과 action 추출이 분리된 파이프라인 |
| 비디오 모델로 합성 데이터 생성 | DreamGen (Jang et al., 2025) | 비디오를 데이터 생성기로 씀. DreamZero는 정책 자체로 씀 |
| Joint video-action, scratch 또는 VLA 기반 | GR-1, GR-2, UVA 등 | 동시 예측의 이점은 보였지만 웹 비디오 prior를 쓰지 않음 |
| **Joint video-action, 사전학습 video diffusion 기반** | Cosmos Policy, Video Generators are Robot Policies, mimic-video, Genie Envisioner, VPP | **가장 가까운 사촌.** 같은 WAM 계열 |

같은 WAM 계열 안에서 논문이 스스로를 구분하는 축은 다음과 같습니다.

| 설계 축 | DreamZero | 논문이 지목한 대안 |
|---|---|---|
| 모델 구성 | 단일 모델 joint denoise | video model + 별도 IDM (Li et al., 2026; Pai et al., 2025) |
| timestep | video·action 공유 (Flash만 분리) | modality별 분리 (Kim et al., 2026; Li et al., 2025a; Liao et al., 2025; Zhu et al., 2025) |
| 데이터 | 비반복·이질적 500시간 | 대부분 반복 시연 중심 |
| 생성 방식 | AR chunk + GT 관측 KV 주입 | BD (ablation 대조군) |

"world model"이라는 이름은 WAM 말고도 여러 갈래에 쓰이므로, V-JEPA 2나 Dreamer 같은 world model과 WAM의 차이도 짚어둘 필요가 있습니다.

> ### 🔗 WAM은 forward model이 아니라 "미래와 그 미래를 만드는 action"의 결합분포다
>
> 부록 A가 이 차이를 정리하였습니다.
>
> | | 모델링 대상 | 추론 시 action 산출 |
> |---|---|---|
> | Latent world model (JEPA / V-JEPA 2, Dreamer) | $p(s_{t+1}\mid s_t,a_t)$ — forward dynamics | goal-conditioned planning이나 search 필요 |
> | 3D point world model (PointWorld) | action 조건부 3D point flow | MPPI 같은 명시적 최적화 필요 |
> | **WAM** | $p(o_{t:t+H},a_{t:t+H}\mid o_{0:t},c)$ — joint | **직접 출력**, test-time 최적화 없음 |
>
> **forward model이 "이 action을 하면 어떻게 되나"에 답하므로 좋은 action을 찾으려면 추론 시 탐색이 필요한 방식이었다면, WAM은 action을 미래와 함께 생성하므로 탐색이 없습니다.** 7Hz 제어가 가능한 것도 이 덕분입니다. 다만 WAM은 여러 action 후보를 가정해 결과를 비교하는 식의 planning은 하지 않습니다.

---

## 7. 한계

**논문이 인정한 한계**

| 항목 | 내용 |
|---|---|
| Scaling law | 모델 크기·데이터·compute에 대한 체계적 scaling curve 부재 |
| In-the-wild 인간 데이터 | 12분 분량의 in-lab 데이터로만 확인 |
| 추론 비용 | 7Hz에 GB200 2장이 필요. VLA는 소비자 GPU에서 20Hz 이상 |
| Long-horizon | System 1 모델이고 visual memory는 약 6초(부록 C 기준 최대 6.6초). System 2 planner나 훨씬 긴 context가 필요 |
| 고정밀 작업 | 열쇠 삽입, 정밀 조립 같은 sub-cm 태스크. 다양성 우선 데이터가 dense demo를 과소대표 |
| Embodiment | 고DOF일수록 IDM 학습에 play data가 더 필요할 것이라는 가설. multi-embodiment 사전학습은 하지 않음 |
| Memory | stateful하지만 memory가 필요한 태스크는 평가하지 않음 |

**추가로 짚을 지점**

- **평가 설계** — in-house 실로봇 평가, 부분 점수 지표, 태스크당 8 rollout입니다. AgiBot 데이터가 아직 공개되지 않아 재현은 DROID 경로로만 가능합니다.
- **핵심 설계의 미검증** — 단일 모델 vs 분리형 모델(3.1절), GT 관측 주입의 기여도(3.3절) 모두 떼어낸 ablation이 없습니다.
- **Embodiment gap의 폭** — few-shot 적응의 두 로봇은 모두 bimanual parallel gripper입니다. 형태가 크게 다른 로봇으로의 적응은 검증되지 않았습니다.
- **수치 해석** — abstract의 "2×"는 5.2절, 전이 실험의 범위는 5.3절의 팩트체크 참조.

---

## 8. 마치며 — 이 논문이 시사하는 것

**DreamZero의 기여는 특정 트릭보다, 로봇 정책의 대부분을 비디오 생성 모델에 맡기고 로봇 전용으로 배우는 부분을 얇은 action 판독으로 줄여도 된다는 것을 실로봇에서 보인 점에 있습니다.** 이 경우 로봇 데이터의 역할도 같은 태스크를 반복해 모드를 채우는 것에서, 다양한 상황의 프레임 쌍으로 IDM을 넓히는 것으로 바뀝니다.

또한 이 논문은 LLM·diffusion 배경이 있으면 특히 잘 읽힙니다.

| 배경 | DreamZero에서 보이는 대응물 |
|---|---|
| Diffusion | action은 영상에 동기화된 **또 하나의 트랙**입니다. 영상-오디오 joint 생성에서 오디오 자리에 action이 들어간 구조입니다. Flash는 few-step 추론에 맞춰 학습 timestep 분포를 설계하는 익숙한 기법을 modality 간 불일치로 확장한 것입니다 |
| LLM | teacher forcing, causal mask, KV cache를 그대로 가져왔습니다. 다른 점은 추론 시에도 환경이 GT를 공급해 exposure bias를 구조적으로 없앴다는 것입니다 |
| Agentic AI | DreamZero는 순수 System 1입니다. 다단계 절차가 필요한 과제에는 planner–executor 분리가 여전히 필요하며, 논문도 Hi Robot 같은 dual-system 구조를 보완책으로 꼽았습니다 |

마지막으로 데이터 전략에 대한 함의가 있습니다. "태스크당 시연 수보다 태스크 다양성"이라는 명제는 이 논문에서 architecture prior와 **결합될 때만** 성립했습니다. 같은 데이터로 scratch VLA는 거의 아무것도 배우지 못했습니다. **데이터 수집 방식과 모델 구조는 따로 평가할 수 없는 한 쌍이라고 할 수 있겠습니다.**

---

## 부록 — 용어 정리

| 용어 | 정의 |
|---|---|
| **WAM (World Action Model)** | 미래 world state와 action을 정렬된 방식으로 함께 예측하는 foundation model. 영상 외에 촉각·힘 등으로 확장될 수 있다는 뜻에서 "Video Action Model" 대신 이 이름을 씀 |
| **IDM (inverse dynamics model)** | 상태 변화로부터 그것을 일으킨 action을 추정하는 모델. $p(a\mid o,o')$ |
| **Teacher forcing** | AR 학습에서 이전 단계의 정답(clean chunk)을 조건으로 주는 방식 |
| **GT 관측 주입** | 추론 시 예측 영상 대신 실제 카메라 관측을 KV cache에 쓰는 것. 학습 조건과 추론 조건을 일치시킴 |
| **DreamZero-Flash** | video timestep을 고노이즈 쪽으로 편향시킨 schedule로 추가 학습해 1-step 추론을 가능하게 한 변형 |
| **Task progress** | 완수 여부가 아니라 진행 단계에 부분 점수를 주는 평가 지표 |
| **Action chunk** | 한 번의 추론으로 생성하는 연속 action 묶음. AgiBot 기준 48 step(1.6초) |

**원문** — [arXiv:2602.15922](https://arxiv.org/abs/2602.15922) · **프로젝트 페이지** — dreamzero0.github.io · **코드** — github.com/dreamzero0/dreamzero
