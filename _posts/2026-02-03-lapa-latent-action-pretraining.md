---
layout: paper
lang: ko
ref: lapa-latent-action-pretraining
kind: paper-review
title: "Latent Action Pretraining from Videos (LAPA)"
date: 2026-02-03 12:00:00 -0800
paper_date: 2024-10-15
venue: "ICLR 2025 · arXiv:2410.11758"
tags: [VLA, Latent-Action, Video-Pretraining, VQ-VAE, Robot-Foundation-Model, Paper-Review]
authors: "Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, Minjoon Seo"
affiliations: "KAIST · University of Washington · Microsoft Research · NVIDIA · Allen Institute for AI"
summary: "두 프레임 사이의 변화를 VQ-VAE로 토큰화해 행동 라벨 없는 비디오에 의사 행동을 붙이고, VLM을 그 토큰으로 사전학습한 뒤 소량의 로봇 라벨로 실제 행동에 연결한다. embodiment가 바뀔 때는 GT 행동 사전학습보다 나을 수 있다."
paper_url: "https://arxiv.org/abs/2410.11758"
---

> **핵심 주장** — 로봇 행동 라벨이 없는 비디오로도 VLA를 사전학습할 수 있다. 두 프레임 사이의 변화를 VQ-VAE로 이산 토큰화해 비디오에 의사 행동을 붙이고, VLM이 그 토큰을 예측하도록 학습한 뒤, 소량의 로봇 라벨로 실제 행동에 연결하면 된다. 그리고 embodiment가 바뀌는 상황에서는 이 방식이 GT 행동으로 한 사전학습보다 나을 수 있다.

---

## 들어가며

VLA의 표준 레시피는 대규모 VLM을 로봇 행동 데이터로 fine-tune하는 것이다. RT-2와 OpenVLA가 이 틀 안에 있고, OpenVLA는 Open X-Embodiment의 약 97만 궤적으로 사전학습되었다. 문제는 이 데이터가 거의 전부 사람의 teleoperation으로 모인다는 것이다. 늘리려면 로봇과 사람과 시간이 모두 필요하다.

반면 인터넷에는 사람이 물건을 집고, 덮고, 쓰러뜨리는 영상이 사실상 무한히 있다. LAPA는 이 영상을 VLA 사전학습에 쓰는 방법을 제안한다. 이 글은 논문의 논지를 따라가되, 로보틱스 배경이 얕은 독자도 각 설계의 이유를 납득할 수 있도록 IDM, VQ-VAE 같은 개념을 필요한 자리에서 쌓아가며 진행한다.

---

## 1. 문제 — 비디오는 넘치는데 VLA는 왜 못 쓰는가

### 1.1 두 개의 벽

논문은 인터넷 비디오를 로봇 학습에 쓰는 데 두 장애물이 있다고 명시한다.

**⛔ 벽 1 — 행동 라벨이 없다.** 비디오에는 픽셀만 있다. BC(behavior cloning)는 (관측, 행동) 쌍을 요구하는데, 비디오에는 그 "행동" 칸이 비어 있다.

**⛔ 벽 2 — 몸과 환경이 다르다.** 사람 손과 7-DoF 그리퍼는 움직이는 방식이 다르고, 웹 영상의 장면은 로봇 작업대와 다르다. 로봇끼리도 사정은 비슷하다. 데이터셋마다 행동 표현(EE delta인지 joint인지, 좌표계, 제어 주기)이 제각각이어서, 논문은 GT 행동으로 사전학습하면 데이터셋 간 positive transfer가 줄어들 수 있다고 지적한다.

### 1.2 기존 접근은 벽을 어떻게 다뤘나

논문의 related work를 벽 1 기준으로 정리하면 다음과 같다.

| 계열 | 대표 | 비디오에서 얻는 것 | 한계 |
|---|---|---|---|
| 시각 표현 사전학습 | R3M (Ego4D) | 좋은 visual encoder | 행동 자체는 배우지 않는다 |
| 비디오 생성 + IDM | UniPi | 미래 프레임(video plan)을 만들고 IDM으로 행동 추출 | 긴 horizon에서 계획이 틀린다. IDM에 라벨이 필요하다 |
| 사람 동작 retargeting | 손 자세 추정 기반 | 손 궤적 → 로봇 행동 | 태스크 특화이거나, 같은 환경의 정렬된 사람-로봇 데이터가 필요하다 |
| IDM 의사라벨 | VPT | 라벨로 학습한 IDM이 비디오에 행동을 붙인다 | IDM 학습 자체에 라벨이 필요하고, 환경이 바뀌면 IDM이 무너진다 |
| Latent action | Genie, LAPO | 관측에서 뽑은 잠재 행동 | 게임 도메인이고 VLA가 아니다 |

### 1.3 그래서 논문이 던지는 질문

> 로봇 행동 라벨 없이 비디오만으로 VLA를 사전학습할 수 있는가? 그 사전학습이 GT 행동 사전학습을 따라잡거나, embodiment가 바뀌는 상황에서 넘어설 수 있는가? 사람 영상만으로도 되는가?

벽 1은 2~3절에서 뚫리고, 벽 2는 4절에서 회수된다. 그 과정이 이 논문의 서사다.

---

## 2. 배경 — IDM, FDM, 그리고 행동을 잠재변수로 두기

### 2.1 로봇 학습의 세 함수

| 함수 | 입력 → 출력 | 질문 | 생성모델·LLM 대응 |
|---|---|---|---|
| **IDM** (inverse dynamics model) | $(x_t,\ x_{t+1}) \to a_t$ | 이 변화를 일으킨 행동은? | 라벨러, 토크나이저 인코더 |
| **FDM** (forward dynamics model) | $(x_t,\ a_t) \to x_{t+1}$ | 이 행동을 하면 어떻게 되나? (= world model) | 조건부 생성기 |
| **정책** (BC) | $(x_t,\ \ell) \to a_t$ | 지시를 수행하려면 지금 무엇을 해야 하나? | 언어모델 |

- $x_t$ — 시점 $t$의 이미지 관측
- $a_t$ — 행동 (예: 7-DoF end-effector delta)
- $\ell$ — 언어 지시

### 2.2 VPT의 닭과 달걀

비디오를 쓰는 가장 직접적인 방법은 VPT식이다. IDM으로 비디오의 모든 프레임에 행동 라벨을 붙이고, 그 라벨로 정책을 BC 학습한다. 그런데 IDM을 학습하려면 결국 $(x_t,\ x_{t+1},\ a_t)$ 라벨이 필요하다. 라벨을 없애려고 만든 도구가 라벨을 요구하는 셈이다. 게다가 그 IDM은 라벨이 있는 소수 도메인에서 학습되므로, 모양이 다른 비디오에 적용하면 틀린 라벨을 붙인다. 이 문제는 5.2절의 실험에서 실제로 확인된다.

### 2.3 LAPA의 탈출구 — 행동을 잠재변수로

LAPA의 발상은 간단하다. $a$를 모른다면 $a$를 **잠재변수 $z$** 로 두고, IDM과 FDM을 하나의 오토인코더로 묶어 동시에 학습한다.

```
  (x_t, x_{t+H}) --[ IDM = encoder ]--> z_t --[ FDM = decoder, given x_t ]--> x_hat_{t+H}
                                         ^
                                         discrete bottleneck: s tokens, |C| codes each
```

- 인코더는 두 프레임 사이에 무슨 일이 있었는지를 $z_t$로 압축한다 (IDM 역할)
- 디코더는 $x_t$와 $z_t$만으로 $x_{t+H}$를 복원한다 (FDM 역할)
- 라벨은 쓰지 않는다. 손실은 재구성 오차뿐이다

그런데 여기에는 이상한 점이 있다. 디코더는 $x_t$를 이미 입력으로 받는다. 그렇다면 인코더는 $x_{t+H}$의 정보를 $z_t$에 통째로 담아 넘겨서 재구성 손실을 0으로 만들 수도 있지 않은가? 그 경우 $z_t$가 굳이 "행동"이 될 이유가 없다.

> ### 💡 $z$가 "행동"이 되는 이유는 정보 병목이다
>
> 핵심은 $z_t$의 용량이다. 기본 설정에서 $z_t$는 길이 $s=4$의 토큰열이고, 각 토큰은 크기 $\lvert C\rvert=8$의 코드북에서 고른다.
>
> $$\lvert\mathcal{Z}\rvert\ =\ \lvert C\rvert^{s}\ =\ 8^4\ =\ 4096\qquad\Rightarrow\qquad \log_2\lvert\mathcal{Z}\rvert\ =\ s\log_2\lvert C\rvert\ =\ 12\ \text{bits}$$
>
> - $\mathcal{Z}$ — 가능한 latent action 전체의 집합
> - $\lvert C\rvert$ — 코드북 크기(vocab), $s$ — 토큰 수
>
> 이미지 한 장을 복원하는 데 드는 정보는 12비트와 비교할 수 없이 크다. 그래서 인코더는 $x_{t+H}$를 통째로 넘길 수 없고, **재구성 손실을 가장 많이 줄이는 12비트만** 골라 담는다. 디코더가 $x_t$에서 이미 알 수 있는 정적인 장면은 담을 필요가 없다. 남는 것은 두 프레임 사이의 가장 큰 변화이고, 로봇 조작 영상에서 그 변화는 대체로 팔과 손의 움직임이다.
>
> 구조도 같은 방향을 가리킨다. 양자화되는 벡터가 두 프레임 임베딩의 **차분** $d_t = e_{t+H} - e_t$이므로(3.1절), 코드는 애초에 변화량을 보도록 귀납 편향이 걸려 있다.
>
> 다만 정확히 말하면 $z$가 담는 것은 **"행동"이 아니라 "가장 두드러진 시각적 변화"** 다. 카메라가 움직이는 영상이라면 카메라 움직임이 코드를 차지한다. 이 차이는 5.6절과 7절에서 다시 문제가 된다.

이 모델은 한마디로 **행동의 토크나이저**다. 논문도 이를 BPE에 빗댄다. EE 위치나 joint 각도 같은 사전 정의된 행동 단위 없이, 데이터에서 "원자적 동작"의 어휘를 학습한다는 뜻이다. 이미지 생성 쪽에서 보면 VQGAN 같은 토크나이저가 **이미지 한 장**을 압축하는 반면, 이 모델은 **첫 프레임이 주어졌을 때의 조건부 변화**를 압축한다는 점이 다르다.

IDM을 라벨 회귀 문제가 아니라 조건부 압축 문제로 바꾼 것, 이것이 LAPA의 핵심 트릭이고 벽 1은 여기서 뚫린다.

---

## 3. 방법 — 세 단계

```
  STAGE 1: LAQ                STAGE 2: LATENT PRETRAIN      STAGE 3: ACTION FINETUNE
  ----------------------      ------------------------      ---------------------------
  video only, no labels       video + language              small robot set w/ actions
  (x_t, x_{t+H}) -> z_t       VLM(x_t, l) -> z_t            VLM(x_t, l) -> a_t
  VQ-VAE (~300M)              7B LWM + latent head          latent head dropped
  enc = IDM, dec = FDM        BC on pseudo labels           new head: 7 dims x 256 bins
```

- **Stage 1 (LAQ)** — 라벨 없는 비디오로 latent action 토크나이저를 학습한다
- **Stage 2 (Latent Pretraining)** — 그 토크나이저로 비디오에 의사 행동을 붙이고, VLM이 이를 예측하도록 BC 학습한다
- **Stage 3 (Action Finetuning)** — 소량의 로봇 라벨로 latent를 실제 행동에 연결한다

Stage 1과 2에는 같은 사전학습 데이터셋을 쓴다. 논문은 latent pretraining을 거친 모델을 통칭 LAPA라 부른다.

### 3.1 Latent Action Quantization (LAQ)

Genie의 latent action model을 C-ViViT 구조로 재현하고 몇 가지를 고친, 약 300M 규모의 모델이다.

```
  x_t, x_{t+H} --> [patch emb -> spatial tf -> causal tf] --> e_t, e_{t+H}
                                                                   |
                                                      d_t = e_{t+H} - e_t
                                                                   |
                                                     [VQ: nearest code] --> z_t
                                                                             |
  x_t --> [patch emb, stop-grad] --> [DECODER: cross-attn + spatial tf] <----+
                                                     |
                                                     v
                                               x_hat_{t+H}    (L2 loss vs x_{t+H})
```

**⓵ 차분 벡터와 양자화**

$$d_t\ =\ e_{t+H}\ -\ e_t,\qquad z_t\ =\ \arg\min_{z_k\in C}\ \lVert d_t - z_k\rVert^2$$

- $x_t,\ x_{t+H}$ — 현재 프레임과 윈도 $H$만큼 뒤의 프레임
- $e_t,\ e_{t+H}$ — 두 프레임이 patch embedding, spatial transformer, causal transformer를 거친 연속 임베딩
- $d_t$ — 두 임베딩의 차분. 양자화 대상이다
- $C=\{z_k\}$ — 코드북, $z_t$ — $d_t$에 가장 가까운 코드
- 토큰 수 $s$는 양자화 직전 CNN의 kernel·stride·padding으로 정해진다

**⓶ NSVQ — 양자화 오차를 노이즈로 대체한다**

argmin은 미분이 불가능하다. 표준 VQ-VAE는 straight-through estimator(STE)로 이를 우회하지만, LAPA는 VQ-VAE에서 흔한 gradient collapse를 피하려고 NSVQ를 쓴다.

$$\hat d_t\ =\ d_t\ +\ \frac{\lVert d_t - z_t\rVert}{\lVert v\rVert}\,v,\qquad v\sim\mathcal N(0,\ I)$$

- $\hat d_t$ — 학습 중 디코더에 실제로 들어가는 벡터
- $v$ — 랜덤 방향 벡터. 정규화한 뒤 양자화 오차의 크기만큼 스케일한다

| | 표준 VQ-VAE (STE) | NSVQ |
|---|---|---|
| 순전파 | 양자화된 $z_t$ | $d_t$ + 양자화 오차와 **같은 크기**의 랜덤 노이즈 |
| 역전파 | $z_t$의 gradient를 $d_t$로 복사하는 근사 | $d_t$로 직접 흐르고, 코드북은 $\lVert d_t - z_t\rVert$ 항을 통해 갱신된다 |
| 코드북 활용 | commitment loss | 학습 초기 **codebook replacement**로 안 쓰이는 코드를 재배치 |

양자화가 만드는 오차의 크기는 흉내 내되, gradient 경로는 끊지 않는다는 설계다.

**⓷ 디코더와 손실**

$$\hat x_{t+H}\ =\ D\Big(\mathrm{CrossAttn}\big(\mathrm{sg}[p_t];\ \hat d_t\big)\Big),\qquad \mathcal L_{\text{LAQ}}\ =\ \lVert x_{t+H} - \hat x_{t+H}\rVert_2^2$$

- $p_t$ — $x_t$의 patch embedding, $\mathrm{sg}[\cdot]$ — stop-gradient
- $\mathrm{CrossAttn}(\cdot\,;\,\cdot)$ — query는 $\mathrm{sg}[p_t]$, key와 value는 $\hat d_t$
- $D$ — spatial transformer만으로 된 디코더. 입력이 두 프레임뿐이라 temporal 부분이 없다

**⓸ 설계 선택의 이유**

| 선택 | 이유 |
|---|---|
| Cross-attention (Genie는 additive embedding) | 경험적으로 더 의미 있는 latent action을 잡는다 (논문 보고) |
| $\mathrm{sg}[p_t]$ | representation collapse 방지. 구체적인 메커니즘은 논문에 설명되어 있지 않다 |
| 두 프레임만 입력 (Genie는 과거 여러 프레임) | 계산 비용. 과거 관측 추가는 향후 과제로 남긴다 |
| 윈도 $H$ — 로봇 0.6초, 사람 영상 2.4초 | 사람 영상에는 정적인 프레임이 많다. Bridge(5Hz)에서는 $H=3$ |

학습이 끝나면 **인코더는 latent IDM, 디코더는 latent world model**이 된다. 인코더는 Stage 2의 라벨러로, 디코더는 5.6절의 신경망 롤아웃에 쓰인다.

### 3.2 Latent Pretraining

LAQ 인코더로 사전학습 비디오의 모든 프레임에 $z_t$ 라벨을 붙이고, VLM이 이를 예측하도록 학습한다.

$$\mathcal L_{\text{latent}}(\theta)\ =\ -\,\mathbb E_{(x_t,\ \ell,\ z_t)}\Big[\sum_{k=1}^{s}\log p_\theta\big(z_t^{(k)}\mid x_t,\ \ell,\ z_t^{(<k)}\big)\Big]$$

- $z_t^{(k)}$ — latent action의 $k$번째 토큰
- $\ell$ — 비디오 클립의 언어 지시
- $\theta$ — VLM 파라미터. 백본은 LWM-Chat-1M (7B)이다

논문은 손실식을 명시하지 않는다. 위 식은 "VLM이 latent action을 예측하는 BC"라는 서술을 causal LM의 토큰 예측으로 옮긴 것이다. 구현상으로는 LM head 대신 **별도의 latent action head**(MLP 한 층, 출력 크기 $\lvert C\rvert$)를 붙이고, vision encoder는 얼리고 언어모델은 푼다. 입력은 현재 이미지 한 장과 지시문이다.

LLM 사전학습과 같은 구조다. 다만 다음에 올 "단어"가 텍스트가 아니라 **다음에 일어날 변화**를 나타내는 토큰이다. 행동 라벨이 필요 없으므로, 언어 지시가 붙은 비디오라면 무엇이든 사전학습 데이터가 될 수 있다.

> ### ⚠️ 팩트체크 — $x_{t+1}$과 $x_{t+H}$의 표기 비일관
>
> 3.1절은 인코더 입력을 $(x_t,\ x_{t+H})$로 정의하는데, 3.2절은 "$x_{t+1}$이 주어졌을 때 $x_t$에 라벨을 붙인다"고 쓴다. 부록 F가 Bridge에서 $H=3$을 기본값으로 명시하므로, 3.2절의 $x_{t+1}$은 "다음 윈도 프레임"을 느슨하게 쓴 표기로 읽어야 한다.

### 3.3 Action Finetuning

latent action은 실제 로봇이 실행할 수 없다. 그래서 소량의 라벨 궤적(delta EE)으로 fine-tune한다.

$$\mathcal L_{\text{FT}}(\theta)\ =\ -\,\mathbb E_{(x_t,\ \ell,\ a_t)}\Big[\sum_{j=1}^{7}\log p_\theta\big(b_t^{(j)}\mid x_t,\ \ell,\ b_t^{(<j)}\big)\Big],\qquad b_t^{(j)}\ =\ \mathrm{bin}_j\big(a_t^{(j)}\big)$$

- $a_t^{(j)}$ — 7-DoF 행동의 $j$번째 차원
- $\mathrm{bin}_j$ — 차원별로 각 bin에 같은 수의 데이터가 들어가도록 나누는 이산화 (OpenVLA·RT-2 방식, 256 bins)
- $b_t^{(j)}$ — 이산화된 행동 토큰

이 식도 논문에 명시되지는 않았고, OpenVLA 방식을 따른다는 서술을 옮긴 것이다.

여기서 눈여겨볼 선택이 있다. **latent action head를 버리고 새 action head를 초기화한다.** 논문은 LAPO처럼 latent head를 남기고 그 위에 "latent → 실제 행동" 디코딩 head를 추가하는 방식도 시도했지만, 재초기화가 더 좋았다고 보고한다(7B라는 모델 크기 덕분일 것이라 추정한다). vision encoder를 얼리고 언어모델 전체를 푸는 것은 사전학습과 같다.

그렇다면 질문이 생긴다. 사전학습에서 공들여 학습한 latent head를 버린다면, 사전학습이 남긴 것은 무엇인가?

> ### 💡 전이되는 것은 $z$가 아니라 "다음에 무엇이 일어나야 하는가"라는 백본의 표현이다
>
> latent pretraining의 출력은 두 부분으로 쪼갤 수 있다.
>
> $$p_\theta\big(z_t^{(k)}\mid x_t,\ \ell,\ z_t^{(<k)}\big)\ =\ \mathrm{softmax}\Big(\underbrace{W}_{\text{latent head}}\ \underbrace{h_\theta\big(x_t,\ \ell,\ z_t^{(<k)}\big)}_{\text{backbone}}\Big)$$
>
> - $h_\theta$ — 7B 백본의 hidden state. **finetune에서 유지된다**
> - $W$ — latent action head. **finetune에서 버려진다**
>
> head는 얇은 판독기일 뿐이고, 지식은 $h_\theta$에 있다. "이 장면에서 이 지시를 수행하려면 다음에 어떤 변화가 일어나야 하는가"를 가볍게 읽어낼 수 있는 표현이다. 새 head는 그 표현 위에서 **같은 의도를 이 로봇의 7-DoF 좌표로 읽는 법**만 배우면 된다. 백본도 함께 풀려 있으니 표현 자체도 조금씩 조정된다.
>
> LLM으로 치면 next-token 사전학습 뒤에 LM head를 떼고 새 head를 붙여 fine-tune하는 것과 같은 구조다. 사전학습 목적의 출력 공간은 버려지고 표현만 남는다.
>
> 이 해석은 검증 가능한 예측을 하나 만든다. 사전학습에서 온 능력은 **어느 물체로, 어느 방향으로** 같은 거친 의도에 강하고, **언제 정확히 쥐는가** 같은 미세 운동은 finetune 라벨의 양에 좌우될 것이다. 5.4절의 실환경 결과가 정확히 이 패턴을 보인다.

---

## 4. 왜 작동하는가

논문이 제시하는 설명은 세 가지다.

**⓵ Embodiment에 무관한 공유 행동 공간** — latent action은 픽셀 변화로 정의되므로, WidowX든 Franka든 사람 손이든 같은 어휘로 표현된다. 실제로 Open-X의 서로 다른 embodiment에 같은 코드를 넣으면 비슷한 움직임이 복원된다(Fig. 6). 반면 GT 행동 사전학습은 데이터셋마다 다른 행동 공간을 섞어야 하므로 positive transfer가 줄어든다.

**⓶ 행동 단위에 대한 사전 가정이 불필요하다** — EE로 할지 joint로 할지, 어떤 해상도로 할지를 정하지 않는다. 연속 관측 사이의 delta를 가장 잘 포착하도록 end-to-end로 학습될 뿐이다.

**⓷ 작은 출력 공간 → 빠른 학습** — 사전학습 출력 공간은 $8^4$이고 OpenVLA의 행동 공간은 $256^7$이다. 논문은 모든 LAPA 모델이 1 epoch 만에 최적 성능에 도달했다고 보고한다.

세 설명은 따로 떨어진 것이 아니다. 3.3절에서 본 "거친 의도는 사전학습에서, 미세 운동은 finetune에서"라는 분업으로 묶으면, LAPA가 어떤 방법인지가 한 문장으로 정리된다.

> ### 📌 LAPA의 핵심 — "무엇을 할지"와 "이 몸으로 어떻게 할지"를 다른 데이터에서 배운다
>
> | 지식 | 어디서 배우나 | 라벨 | 실험 증거 (5.4절) |
> |---|---|---|---|
> | 무엇을 할지 — 대상 선택, 언어 조건, 대략적 궤적 | 사전학습 (아무 비디오) | 불필요 | unseen instruction 48.5 vs OpenVLA 43.4, reaching 83.3% vs 66.7% |
> | 이 몸으로 어떻게 — grasp 타이밍, 접촉 | finetune (로봇 라벨) | 필요 | early grasp 실패, pick&place 45.8 vs 54.2 |
>
> 도입부의 두 벽이 여기서 회수된다.
>
> - **벽 1 (라벨 부재)** — LAQ가 라벨 없이 IDM을 만들어 우회한다. VPT의 닭과 달걀 문제가 사라진다.
> - **벽 2 (embodiment 간극)** — 사전학습은 embodiment에 무관한 latent 공간에서 하고, embodiment 특화 부분은 finetune의 새 head로 **미룬다.** 간극을 없애는 것이 아니라, 간극이 문제가 되는 부분만 라벨이 있는 곳으로 옮긴다.

---

## 5. 실험

### 5.1 설정과 비교군

| 환경 | 행동 | 사전학습 데이터 | Finetune | 측정하려는 것 |
|---|---|---|---|---|
| Language Table (sim) | 2-DoF 밀기 | sim 181k / real 442k | 1k 또는 7k | in-domain, cross-task, cross-env |
| SIMPLER (sim) | 7-DoF WidowX | Bridgev2 60k / Sthv2 | 100 | in-domain, 사람 → 로봇 |
| 실환경 | 7-DoF Franka | Bridgev2 / Open-X 970k / Sthv2 | 450 (3 태스크 × 150) | cross-embodiment, multi-embodiment |

| 비교군 | 사전학습 | 통제하는 것 |
|---|---|---|
| Scratch | 없음 (같은 LWM 백본) | 사전학습의 순효과 |
| UniPi | 비디오 diffusion + IDM | "픽셀을 예측"하는 대안 |
| VPT | 라벨로 학습한 IDM의 의사라벨로 **같은 VLM**을 사전학습 | latent action vs 의사 GT 행동 |
| ActionVLA | GT 행동으로 **같은 LWM 백본**을 사전학습 | 사실상의 상한 |
| OpenVLA | Open-X GT 행동, Prismatic 백본 | 외부 SOTA |

### 5.2 Language Table — 라벨 0.5%로 어디까지 가나

Table 1, success rate (%).

| | In-domain Seen | In-domain Unseen | Cross-task Seen | Cross-task Unseen | Cross-env Seen | Cross-env Unseen |
|---|---|---|---|---|---|---|
| Scratch | 15.6 | 15.2 | 27.2 | 22.4 | 15.6 | 15.2 |
| UniPi | 22.0 | 13.2 | 20.8 | 16.0 | 13.6 | 12.0 |
| VPT | 44.0 | 32.8 | 72.0 | **60.8** | 18.0 | 18.4 |
| **LAPA** | 62.0 | 49.6 | 73.2 | 54.8 | 33.6 | 29.6 |
| ActionVLA | 77.0 | 58.8 | 77.0 | 58.8 | 64.8 | 54.0 |

- **In-domain** — sim 181k로 사전학습, 5개 태스크 1k로 finetune
- **Cross-task** — 같은 사전학습, separate 태스크 7k로만 finetune한 뒤 5개 태스크 모두 평가
- **Cross-env** — real 442k로 사전학습, sim 1k로 finetune (real-to-sim 간극)

In-domain에서 LAPA는 라벨을 0.5%(1k/181k)만 쓰고도 Scratch를 크게 넘어 ActionVLA와의 격차를 좁힌다. 흥미로운 것은 VPT와의 관계다. cross-task unseen에서는 VPT가 LAPA를 이기는데(60.8 vs 54.8), cross-env에서는 VPT가 Scratch 수준으로 무너진다(18.0). 같은 "의사라벨 → BC" 구조인데 왜 이렇게 갈리는가?

> ### 💡 VPT와 LAPA를 가르는 것은 "IDM을 어떤 데이터로 학습했느냐"다
>
> VPT의 IDM은 **finetune용 라벨 데이터**(1k 또는 7k 궤적)로 학습된 뒤 **사전학습 비디오**에 적용된다. LAPA의 LAQ는 **사전학습 비디오 자체**로 학습된다.
>
> | | VPT | LAPA |
> |---|---|---|
> | IDM 학습 데이터 | 타깃 도메인의 라벨 궤적 | 사전학습 비디오 (라벨 없음) |
> | 라벨 공간 | 타깃 로봇의 GT 행동 | 데이터에서 학습된 latent |
> | 라벨을 붙이는 대상 | IDM에게 OOD일 수 있다 | 정의상 in-distribution |
> | 도메인을 정렬하는 시점 | 라벨링 시점 | finetune 시점 (새 head) |
>
> - **cross-task** — 사전학습과 finetune이 같은 sim 환경이고 라벨이 7k로 넉넉하다. VPT의 IDM이 정확한 GT 공간 라벨을 만들 수 있으니 VPT가 유리하다. 논문도 라벨이 많아 IDM이 정확해졌기 때문이라고 설명한다.
> - **cross-env** — sim 1k로 학습한 IDM을 real 44만 궤적에 적용하니 라벨이 망가진다. 논문은 IDM이 환경 이동에 강건하지 않다고 본다.
>
> VPT는 도메인 정렬을 라벨링 시점에 하므로, 도메인이 어긋나면 사전학습 데이터 전체가 오염된다. LAPA는 정렬을 finetune으로 미루므로 그 위험이 없다. 2.2절에서 예고한 VPT의 약점이 이것이고, 4절의 "벽 2를 finetune으로 미룬다"는 설계가 작동하는 가장 깨끗한 증거이기도 하다.

다만 같은 백본의 ActionVLA는 모든 설정에서 LAPA보다 높고, cross-env에서는 격차가 크다(64.8 vs 33.6). embodiment가 같다면 GT 라벨이 여전히 낫다.

> ### ⚠️ 팩트체크 — cross-task의 ActionVLA는 별도 실행이 아니다
>
> 부록 Table 7·8의 ActionVLA 열은 in-domain인 Table 5·6과 태스크별 수치까지 똑같다. cross-task의 "상한"은 separate 7k finetune 조건에서 다시 돌린 결과가 아니라 in-domain 결과를 재사용한 것으로 보인다. Table 1과 부록 평균도 소폭 어긋난다(ActionVLA 77.0 vs 76.8, VPT 44.0 vs 43.6).

### 5.3 SIMPLER — 픽셀을 예측하면 번역기가 병목이 된다

Table 11. Bridgev2로 사전학습한 뒤 100개 궤적으로 finetune, success rate (%).

| Scratch | UniPi | VPT | **LAPA** | ActionVLA | OpenVLA |
|---|---|---|---|---|---|
| 34.4 | 1.3 | 51.0 | **57.3** | 63.5 | 36.4 |

UniPi가 거의 0이다. 논문에 따르면 비디오 계획 자체는 꽤 정확한데, 100개 궤적으로 학습한 IDM이 7-DoF 연속 행동을 제대로 예측하지 못해 제한 스텝 안에 물체를 잡지 못한다. **픽셀을 예측하고 나중에 행동으로 번역하는 전략은 번역기(IDM)의 라벨이 부족하면 무너진다.** OpenVLA의 낮은 점수는 SIMPLER의 real-to-sim 전이 문제로 이미 알려진 현상이라고 논문은 설명한다.

한 가지 짚어둘 점이 있다. SIMPLER의 finetune 궤적 100개는 Bridgev2로 학습한 LWM 기반 VLA의 성공 롤아웃을 걸러낸 것이다. 사실상 ActionVLA 계열의 행동 분포로 보이므로, 이 편향은 LAPA에 불리한 쪽으로 작용한다.

### 5.4 실환경 Franka — 헤드라인 결과

3 태스크(knock, cover, pick&place) × 3 일반화 유형(unseen 조합, unseen 물체, unseen 지시) × 6 롤아웃으로, 모델당 54 롤아웃이다. 점수는 부분 점수 방식이다(예: pick&place는 reach 0.25, grasp 0.5, 이동 0.75, 성공 1).

Fig. 3 · Table 13–16, success rate (%).

| 모델 | 사전학습 라벨 | Knock | Cover | Pick&Place | **Avg** |
|---|---|---|---|---|---|
| Scratch | — | 13.9 | 38.7 | 11.1 | 21.2 |
| ActionVLA (Bridge) | GT | 33.3 | 42.3 | 22.2 | 32.6 |
| OpenVLA (Bridge) | GT | 25.0 | 47.8 | 19.4 | 30.8 |
| **LAPA (Bridge)** | 없음 | 25.0 | 42.4 | **43.1** | **36.8** |
| OpenVLA (Open-X) | GT | 38.9 | 38.6 | **54.2** | 43.9 |
| **LAPA (Open-X)** | 없음 | **52.8** | **51.7** | 45.8 | **50.1** |

**⓵ Bridge 사전학습 (WidowX → Franka)** — 같은 LWM 백본에서 LAPA가 ActionVLA를 이긴다(36.8 vs 32.6). 논문의 가설은 Bridge의 대부분이 pick-and-place라서 GT 행동 사전학습이 WidowX 행동 공간에 과적합되었고, 이것이 Franka로 옮길 때 방해가 되었다는 것이다. 실제로 격차는 거의 pick&place 한 태스크에서 나온다(43.1 vs 22.2). 백본과 데이터가 같고 embodiment만 바뀐 비교이므로, **"라벨을 버린 쪽이 이긴다"는 주장의 가장 깨끗한 증거**가 이것이다.

**⓶ Open-X 사전학습** — 데이터를 Bridge에서 Open-X로 늘리면 두 모델 모두 좋아지고, LAPA가 OpenVLA를 평균에서 이긴다(50.1 vs 43.9). 일반화 유형별로도 세 유형 모두 앞선다(Table 2: unseen 조합 57.8 vs 46.2, unseen 물체 43.9 vs 42.1, unseen 지시 48.5 vs 43.4). 가장 큰 차이는 언어 조건화 쪽에서 나고, unseen 물체의 차이는 1.8%p에 그친다.

그런데 pick&place에서는 진다(45.8 vs 54.2). 논문은 실패 대부분이 너무 이른 grasp이고, reaching은 오히려 LAPA가 높다고 보고한다(83.33% vs 66.67%). 3.3절에서 한 예측이 그대로 맞는다. 어느 물체로 갈지는 사전학습이 알려주지만, 한 궤적에 한두 번뿐인 grasp는 150개 라벨로 배우기에 부족하다.

> ### ⚠️ 팩트체크 — "SOTA VLA 대비 +6.22%"를 읽는 법
>
> 기여 목록의 +6.22는 Table 16의 50.09 − 43.87이다. 수치는 맞지만 네 가지를 함께 봐야 한다.
>
> - **⓵ 단위** — 상대 %가 아니라 부분 점수 기준 **%p**다. strict 성공률로는 35.19 vs 27.78이다.
> - **⓶ 백본 교란** — LAPA는 LWM-Chat-1M, OpenVLA는 Prismatic 백본이다. Open-X에서 같은 백본으로 GT 사전학습한 ActionVLA는 없다. 논문도 효율의 일부를 LWM 백본 덕으로 돌린다(5.6절).
> - **⓷ Finetune 레시피 차이** — OpenVLA는 LoRA, batch 32로 train action accuracy 95%까지 학습했고, LAPA는 LM 전체를 풀고 batch 128에 image augmentation을 썼다(부록 C). 논문은 OpenVLA에서 LoRA와 full FT가 비슷했다고 밝히지만 augmentation 차이는 남는다.
> - **⓸ 표본 크기** — 셀당 6 롤아웃이다. paired 비교(부록 D)에서 LAPA 승 31.5%, OpenVLA 승 16.7%, 무승부 51.9%다.
>
> 따라서 "OpenVLA를 이겼다"보다 **"같은 백본에서 embodiment가 바뀔 때 GT 사전학습보다 낫다"**(위의 Bridge 비교)가 이 논문에서 더 견고한 주장이다.

### 5.5 사람 영상만으로 — Something-Something V2

이 절이 논문의 원래 동기다. 사전학습에 Something-Something V2(Sthv2, 사람이 일상 물체를 다루는 약 22만 클립)만 쓰고, 윈도는 2.4초로 둔다.

| 평가 | Scratch | UniPi | VPT | **LAPA (Human)** | 참고 |
|---|---|---|---|---|---|
| SIMPLER (Table 12) | 34.4 | 0.7 | 45.8 | **52.1** | LAPA (Bridge) 57.3, 10% 데이터 50.0 |
| 실환경 평균 (Table 16) | 21.2 | — | — | **34.0** | OpenVLA (Bridge) 30.8 |

ActionVLA는 사람 영상에 로봇 행동 라벨이 없어서 애초에 학습할 수 없다.

사람 영상만으로 사전학습한 LAPA가 로봇 데이터(Bridge)로 사전학습한 OpenVLA를 실환경 평균에서 넘는다. 다만 태스크별로 보면 knock 30.6, cover 47.9, pick&place 23.6으로 편차가 크다. 논문은 이를 사전학습 데이터의 태스크 분포로 설명한다.

Table 4, 평가 태스크와 같은 태스크의 사전학습 궤적 수 (어휘 매칭 기준).

| 태스크 | Bridgev2 | Open-X | Sthv2 |
|---|---|---|---|
| Knocking | 2 | 7,969 | 6,655 |
| Covering | 898 | 5,026 | 6,824 |
| Pick & Place | 10,892 | 911,166 | 3,272 |

Bridge에는 knocking이 사실상 없고 Sthv2에는 많다. 그래서 embodiment 간극이 더 큰데도 knocking에서는 Sthv2 쪽이 낫다(30.6 vs LAPA (Bridge) 25.0). 반대로 pick&place는 Bridge에 훨씬 많고, 결과도 뒤집힌다(23.6 vs 43.1). **embodiment 일치보다 사전학습 영상에 그 스킬이 얼마나 있었나가 더 중요할 수 있다**는 관찰이다. 어휘 매칭이 거친 추정이라는 점은 논문도 인정한다.

> ### ⚠️ 팩트체크 — "internet-scale video"와 실제 실험 사이의 거리
>
> abstract와 intro는 인터넷 규모 비디오를 동기로 내세우지만, 실제로 쓴 사람 영상은 Sthv2다. 크라우드소싱으로 수집한 짧고 정제된 클립에 템플릿형 캡션이 붙은 데이터로, 편집되지 않은 웹 영상과는 거리가 있다. 사람 영상에 대한 스케일링도 10%와 100% 두 점뿐이다(논문도 향후 과제로 인정한다). 웹 규모로의 확장은 이 논문이 보인 것이 아니라 외삽이다.

### 5.6 효율, 스케일링, 그리고 latent가 실제로 담는 것

**사전학습 효율** — LAPA (Open-X)는 H100 8장으로 34시간(272 H100-hours), OpenVLA는 21,500 A100-hours가 들었다. 논문은 이 효율이 두 곳에서 온다고 본다. 하나는 LWM 백본이 비디오 next-frame 생성으로 사전학습되어 있다는 점이다. 같은 Bridge 데이터와 같은 목적에서 LWM 기반 ActionVLA는 3 epoch, Prismatic 기반 OpenVLA는 30 epoch에 최적에 도달했다. 다른 하나는 latent 출력 공간이 작다는 점이다.

> ### ⚠️ 팩트체크 — "30배 이상 효율"의 계산 근거
>
> 논문은 H100이 A100보다 2~3배 빠르다고 가정한다. 그대로 계산하면 21,500 / (272 × 3) ≈ 26배에서 21,500 / (272 × 2) ≈ 40배 사이이고, 기여 목록의 "over 30x"는 이 범위의 중간 이상을 택한 것이다. 272시간에 LAQ 학습 비용이 포함되는지는 명시되어 있지 않다. 또 $8^4$ vs $256^7$ 비교는 사전학습 출력 공간에만 해당하며, finetune 후의 LAPA도 OpenVLA와 같은 $256^7$ 공간을 출력한다.

**스케일링 (Fig. 5)** — LAQ 모델 크기, 데이터 비율, latent 토큰 길이, vocab의 네 축을 키운다. 여기서 "model scaling"은 7B VLA가 아니라 **LAQ의 크기**(30M → 300M)라는 점에 주의해야 한다. 모든 축에서 키울수록 좋아지지만, 최적의 latent 공간은 데이터의 행동 복잡도에 따라 다르다. 시각적으로 단순한 Language Table(2-DoF)에서는 토큰 길이보다 vocab을 늘리는 쪽이 훨씬 효과적이었다(Fig. 16). 윈도 $H$는 극단적으로 크지 않은 한 결과가 안정적인데, 논문은 300M 규모의 LAQ가 너무 큰 시각적 변화를 모델링하지 못하기 때문으로 본다. finetune 데이터를 줄여도 LAPA는 Scratch를 일관되게 앞선다(Fig. 15b).

**latent가 실제로 담는 것 (논문 5.2절, 부록 E)** — 2.3절의 콜아웃에서 예고한 "행동"과 "시각적 변화"의 차이가 여기서 드러난다.

| 데이터 | 관찰 |
|---|---|
| Language Table (vocab 8, 길이 1) | 8개 코드가 8개 방향(좌·전, 좌·후, 우·후, 우측 조금, 우, 후, 정지, 전)에 대응하고, GT 2D 행동 공간에서 깔끔하게 군집한다 (Fig. 12–13) |
| Open-X | 같은 코드를 넣으면 embodiment가 달라도 비슷한 움직임이 복원된다 (Fig. 6) |
| Sthv2 (egocentric) | 코드가 손 움직임뿐 아니라 **카메라 움직임**도 담는다 (Fig. 14) |

세 번째 줄이 정보 병목 해석의 직접 증거다. $z$는 행동이 아니라 가장 두드러진 시각적 변화를 담고, egocentric 영상에서는 머리가 움직이며 생기는 카메라 이동이 그 변화에 포함된다. 논문은 이를 내비게이션 등으로 확장할 수 있는 장점으로 제시하지만, 조작 정책 입장에서는 코드 용량의 일부가 행동과 무관한 신호에 쓰인다는 뜻이기도 하다.

**신경망 롤아웃 (Fig. 7)** — 사전학습만 마친 LAPA가 latent action을 내고, LAQ 디코더가 다음 프레임을 생성하는 루프를 닫는다. "냄비에서 브로콜리를 꺼내라"는 지시에 팔이 브로콜리로 다가가 집어 올리는 영상이 생성된다. 3.1절의 "인코더 = IDM, 디코더 = world model"이라는 이중 역할이 정책과 합쳐져 **순수 신경망 시뮬레이터**가 되는 셈이다. 논문은 여러 계획을 생성해 가장 좋은 것을 고르는 test-time compute 확장까지 전망하지만, 제시된 근거는 정성적 예시 하나다.

---

## 6. 위치잡기 — 이웃 연구들 속에서

LAPA는 세 계보의 교차점에 있다.

**⓵ VLA** — RT-2, OpenVLA, Octo. 이들의 행동 라벨 의존이 LAPA의 출발점이다.

**⓶ 비디오로부터의 로봇 학습** — 표현 사전학습(R3M), 비디오 생성(UniPi), 사람 동작 retargeting, IDM 의사라벨(VPT).

**⓷ Latent action** — Genie(인터랙티브 world model), ILPO·LAPO(게임 정책).

가장 가까운 사촌은 VPT와 Genie·LAPO다.

| | UniPi | VPT | Genie / LAPO | **LAPA** |
|---|---|---|---|---|
| 비디오에서 추출·예측하는 것 | 미래 **픽셀** | **의사 GT 행동** | **latent action** | **latent action 토큰** |
| IDM 학습에 라벨 | 필요 | 필요 | 불필요 | 불필요 |
| 최종 모델 | video diffusion + IDM | BC 정책 | Genie는 world model, LAPO는 게임 정책 | 7B monolithic VLA |
| 도메인 | 로봇 | 원 논문은 Minecraft (이 논문에서는 같은 VLM으로 재구현) | 2D 게임, Procgen | 실로봇 조작 |

한 문장으로 위치를 잡으면, **LAPA는 VPT의 파이프라인(비디오 라벨링 → BC 사전학습)에서 라벨이 필요한 IDM을 Genie의 비지도 latent action model로 교체하고, 정책을 VLM으로 키운 것**이다.

행동을 latent로 압축하는 다른 계열(Play-LMP, VQ-BeT, QueST)과도 구분해야 한다. 이들은 **GT 행동**을 토큰화해 멀티모달리티를 다루는 것이 목적이고, LAPA는 **관측**에서 행동을 도출한다. 둘 다 "action tokenizer"라 불릴 수 있지만 입력이 다르다.

"비디오에서 무엇을 예측할 것인가"라는 축으로 보면 트레이드오프가 선명해진다.

| 예측 대상 | 정보량 | 라벨이 필요한 시점 | 약점 (이 논문의 실험 기준) |
|---|---|---|---|
| 픽셀 (UniPi) | 가장 풍부 | 번역(IDM) 단계 | 번역기 라벨이 부족하면 붕괴 (SIMPLER 1.3) |
| 의사 GT 행동 (VPT) | 타깃 로봇 공간 | 라벨링 단계 | 도메인이 어긋나면 라벨 오염 (cross-env 18.0) |
| latent action (LAPA) | 스텝당 12 bits | finetune 단계 | 미세 운동 부족 (early grasp) |

그렇다면 latent action이라는 재료는 이후 어떻게 쓰였을까? LAPA는 latent를 이산 토큰으로 두고, finetune에서 head째 버렸다. 이산화는 LAQ 학습의 병목으로는 필요하지만, 정책이 그 이산 토큰을 그대로 예측해야 할 이유는 따로 없다.

> ### 🔗 GR00T N1 (NVIDIA, arXiv:2503.14734) — latent action을 연속 임베딩으로 가져간 후속 채택
>
> GR00T N1은 행동 라벨이 없는 사람 egocentric 영상과 생성된 neural trajectory에 LAPA의 VQ-VAE latent action 모델을 그대로 적용한다. 다만 정책에 넘기는 형태가 다르다.
>
> | | LAPA | GR00T N1 |
> |---|---|---|
> | 정책이 받는 latent | 이산 코드 ($8^4$) | 양자화 **이전**의 연속 임베딩 |
> | 정책 손실 | 토큰 cross-entropy (자기회귀) | flow matching (실제 행동과 같은 손실) |
> | 실제 행동 데이터와의 관계 | 사전학습은 latent만, finetune에서 head 교체 | 같은 사전학습에 섞되, 별도의 "LAPA" embodiment로 취급 |
>
> 이산 병목의 귀납 편향은 LAQ 학습 단계에서 이미 얻었으니, 정책에는 정보 손실이 적은 연속 표현을 넘기는 쪽으로 진화한 것으로 읽을 수 있다.

---

## 7. 한계

논문이 스스로 밝힌 것과, 읽으며 추가로 짚어둘 만한 것을 함께 정리한다.

**논문이 인정한 한계**

- **미세 운동** — grasp 같은 fine-grained 동작에서 GT 행동 사전학습보다 약하다. 논문은 latent 공간을 키우면 나아질 것으로 본다.
- **추론 지연** — 7B 자기회귀 VLA라서 실시간 추론 지연이 있다. 작은 head가 더 높은 주파수로 행동을 내는 계층 구조를 대안으로 언급한다.
- **조작 이외의 영상** — 자율주행, 내비게이션, 풍경 영상 등으로의 적용은 탐색하지 않았다.

**추가로 짚을 지점**

- **"행동" ≠ "시각적 변화"** — 2.3절과 5.6절에서 본 대로 $z$는 가장 두드러진 시각적 변화를 담는다. 카메라 ego-motion이나 다른 사람·물체의 움직임도 코드를 차지하며, 편집되지 않은 웹 영상일수록 이 오염은 심해질 것이다. "웹 규모 확장"의 실질적 장애물이다.
- **병목은 사라진 게 아니라 옮겨졌다** — latent pretraining에는 "비디오 + 언어 지시" 쌍이 필요하다. Sthv2에는 템플릿 캡션이 있지만, 웹 영상에는 행동 라벨뿐 아니라 태스크 수준의 지시도 없다. 라벨 병목이 행동에서 언어로 이동한 것이다.
- **데이터 소스별 수작업 튜닝** — 윈도 $H$를 로봇 0.6초, 사람 영상 2.4초로 따로 정했다. 이질적인 웹 영상에서는 이 손잡이를 소스마다 맞춰야 한다.
- **헤드라인 비교의 교란** — 5.4절에서 본 대로 백본과 finetune 레시피가 섞여 있다. 같은 백본에서 latent 사전학습이 GT 사전학습을 이긴 사례는 Bridge → Franka 하나이고, 그 격차도 사실상 한 태스크에서 나온다.
- **입력 정보가 적다** — LAQ는 두 프레임만 보고, 정책은 이미지 한 장과 지시문만 본다. 가려짐이나 속도처럼 시간 문맥이 필요한 상황에 대한 검증이 없다.

---

## 8. 마치며 — 이 논문이 시사하는 것

LAPA의 기여는 VQ-VAE도 VLA도 아니다. 둘 다 이미 있던 재료다. 진짜 기여는 **IDM을 라벨 없는 조건부 압축 문제로 바꿔서, 라벨 없는 비디오를 VLA 사전학습 코퍼스로 만든 것**이다.

그리고 이 논문은 LLM·생성모델 배경을 가진 사람에게 유난히 잘 읽힌다.

- **LAQ = 행동 토크나이저** — VQGAN이 이미지 한 장을 압축한다면, LAQ는 첫 프레임이 주어졌을 때의 변화를 압축한다.
- **Latent pretraining = 라벨 없는 코퍼스로 하는 next-token 사전학습** — 다만 텍스트 대신 "다음 변화" 토큰을 예측한다.
- **Action finetune = head를 교체하는 SFT** — 사전학습 출력 공간은 버리고 표현만 가져간다는 점까지 같다.
- **인코더 = IDM, 디코더 = world model** — 정책과 world model이 한 번의 비지도 학습에서 함께 나온다. world model 쪽에서 보면, 행동 조건부 비디오 생성기를 행동 라벨 없이 얻는 방법이기도 하다.

무엇보다 5.5절의 Table 4가 던지는 질문이 크다. 하류 성능이 embodiment 일치보다 **사전학습 영상의 스킬 분포**를 더 따라간다면, 로봇 파운데이션 모델의 데이터 전략은 "로봇 데이터를 얼마나 모을까"에서 **"어떤 스킬이 담긴 영상을 얼마나 모을까"** 로 바뀐다. 현장 작업 영상처럼 라벨은 없지만 스킬 밀도가 높은 영상이 쌓여 있는 도메인에서는 특히 그렇다.

---

## 부록 — 용어 정리

| 용어 | 정의 |
|---|---|
| **latent action** | 두 프레임 사이의 변화를 VQ-VAE로 이산화한 토큰열. 기본 설정은 길이 4, vocab 8 ($8^4$) |
| **LAQ** (Latent Action Quantization) | latent action을 비지도로 학습하는 VQ-VAE. 인코더는 latent IDM, 디코더는 latent world model |
| **IDM** (inverse dynamics model) | $(x_t,\ x_{t+1}) \to a_t$. 관측 변화로부터 행동을 추정 |
| **FDM** (forward dynamics model) | $(x_t,\ a_t) \to x_{t+1}$. 행동의 결과를 예측하는 world model |
| **NSVQ** | 양자화 오차를 같은 크기의 랜덤 노이즈로 대체해 gradient 경로를 유지하는 VQ 학습 기법 |
| **latent pretraining** | VLM이 $(x_t,\ \ell)$로부터 latent action을 예측하도록 하는 BC 사전학습 |
| **action finetuning** | latent head를 버리고 새 head로 실제 로봇 행동(256 bins × 7 차원)을 학습 |
| **ActionVLA** | 같은 LWM 백본을 GT 행동으로 사전학습한 비교군. 사실상의 상한 |
| **partial success** | 하위 단계(reach, grasp, 이동, 성공)에 부분 점수를 주는 실환경 평가 방식 |

**원문** — [arXiv:2410.11758](https://arxiv.org/abs/2410.11758) · **프로젝트 페이지** — latentactionpretraining.github.io
