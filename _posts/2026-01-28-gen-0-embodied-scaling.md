---
layout: paper
lang: ko
ref: gen-0-embodied-scaling
kind: tech-review
title: "GEN-0: Embodied Foundation Models That Scale with Physical Interaction (GEN-0)"
date: 2026-01-28 12:00:00 -0800
paper_date: 2025-11-04
venue: "Generalist AI Blog · 연구 블로그 (arXiv 미게재)"
tags: [Robot-Foundation-Model, Scaling-Law, Pretraining, Embodied-AI, Tech-Review]
authors: "Generalist Team"
affiliations: "Generalist AI"
summary: "대규모 물리 상호작용 데이터로 사전학습한 충분히 큰 모델에서 사전학습 데이터량과 post-training 성능 사이에 power law가 성립한다는 주장을, 무엇을 측정했고 그 측정이 무엇을 말해주는지를 중심으로 읽었습니다."
paper_url: "https://generalistai.com/blog/gen-0"
---

> **핵심 주장**: **로봇 파운데이션 모델도 LLM처럼 scaling law를 따릅니다.** 대규모 실세계 물리 상호작용 데이터로 사전학습하고 모델이 충분히 크면(블로그 기준 7B 이상), 사전학습 데이터량과 downstream post-training 성능 사이에 power law가 성립합니다. 이 곡선을 쓰면 목표 성능에 필요한 데이터량을 미리 견적할 수 있습니다.

---

## 들어가며

로봇 파운데이션 모델의 표준 레시피는 VLM 위에 행동 출력을 얹는 것입니다. 웹 규모 vision-language 사전학습에서 얻은 의미 이해를 로봇 제어로 옮겨오고, 목표 태스크는 teleoperation 시연으로 fine-tuning합니다. PaLM-E, RT-2, π₀가 모두 이 구도에 속합니다.

GEN-0는 이 레시피에 빠진 것을 지목하였습니다. LLM을 지금의 자리에 올려놓은 것은 특정 아키텍처보다 "compute와 데이터를 늘리면 성능이 예측 가능하게 오른다"는 scaling law였습니다. 다만 로봇 도메인 자체에서는 그런 관계가 확립된 적이 없고, GEN-0는 그 관계를 로봇에서 처음 보이겠다고 나섰습니다. 이 "처음"이 어디까지 맞는 말인지는 6절에서 다시 따집니다.

전제를 하나 밝혀둡니다. GEN-0는 논문이 아닌 연구 블로그이고, 아키텍처와 학습 절차가 공개되지 않았습니다. 이 때문에 이 글에서는 **무엇을 측정했고, 그 측정이 무엇을 말해주는가**를 중심으로 읽습니다. 측정을 제대로 읽으려면 LLM scaling 문헌의 도구가 필요하므로, 필요한 개념은 나올 때마다 설명합니다.

---

## 1. 문제: 로봇에는 왜 scaling law가 없었나

### 1.1 VLM 전이로 얻지 못하는 것

vision-language 사전학습을 디딤돌로 삼으면 로봇은 semantic generalization을 얻습니다. "빨간 컵을 집어라"라는 지시에서 무엇이 빨간 컵이고 집는다는 게 무엇인지 압니다. 하지만 컵을 얼마의 힘으로, 어떤 손가락 궤적으로 쥐는지, 미끄러지면 어떻게 고쳐 쥐는지 같은 **sensorimotor 지식**은 웹 이미지와 텍스트에 거의 없습니다.

GEN-0는 이 sensorimotor 지식을 scaling으로 얻으려 합니다. 블로그는 이를 위해 아키텍처, 학습 절차, 데이터 엔진이 모두 필요하다고 밝혔습니다. 이것을 장애물 관점에서 다시 정리하면 세 개의 벽이 됩니다.

### 1.2 세 개의 벽

**⛔ 벽 1: 데이터 regime이 없습니다** LLM의 scaling law는 수천억 토큰 규모에서 관측됩니다. 로봇 teleoperation 데이터는 사람이 로봇을 한 대씩 조종해야 생기므로 규모가 몇 자릿수 부족합니다. 즉, scaling 곡선을 그릴 x축 자체가 없는 셈입니다.

**⛔ 벽 2: 모델을 키우면 제어가 느려집니다** LLM은 답하기 전에 원하는 만큼 생각해도 됩니다. 반면 물리 세계는 추론이 끝날 때까지 기다려주지 않습니다. 모델이 커질수록 추론 지연이 길어지고, 지연은 그대로 제어 품질 저하로 이어집니다. scaling하려는 대상인 모델 크기가 제어를 방해하는 요인이 되는 것입니다.

**⛔ 벽 3: "예측 가능"을 무엇으로 재나** LLM에는 pretraining loss라는 싸고 매끄러운 지표가 있습니다. 반면 로봇의 진짜 목표인 성공률은 실물 rollout이 필요해서 비싸고 노이즈가 큽니다. scaling law를 세우려면 싸고 매끄러우면서도 성공률과 같은 방향으로 움직이는 proxy가 있어야 합니다.

벽 1과 벽 2는 블로그가 직접 언급한 장애물입니다. 벽 3은 블로그가 따로 이름 붙이지 않았지만, 실험 설계 전체를 이 문제에 대한 답으로 읽을 수 있습니다.

### 1.3 그래서 블로그가 던지는 질문

> 물리 상호작용 데이터로 직접 사전학습해도 LLM처럼 "데이터와 compute를 늘리면 downstream 성능이 예측 가능하게 오른다"가 성립하는가? 성립한다면 어떤 조건에서인가?

2절에서는 이 질문을 읽는 데 필요한 도구를 준비하고 벽 2가 정확히 무엇인지 살펴봅니다. 3절은 세 벽에 대한 GEN-0의 답을, 4절은 벽 1을 넘고 나서 새로 드러난 조건을 다룹니다.

---

## 2. 배경: scaling law의 문법과 벽 2의 정체

### 2.1 Kaplan식 scaling law: 모델마다 바닥이 있다

GEN-0의 실험은 모두 LLM scaling 문헌의 틀을 로봇에 옮긴 것입니다. 출발점은 Kaplan et al.(2020)의 멱법칙입니다.

$$L(N)=\left(\frac{N_c}{N}\right)^{\alpha_N},\qquad L(D)=\left(\frac{D_c}{D}\right)^{\alpha_D}$$

| 기호 | 의미 |
|---|---|
| $L$ | test loss (LLM에서는 cross-entropy) |
| $N$ | 모델 파라미터 수 |
| $D$ | 학습 데이터량 (토큰 수) |
| $N_c,\ D_c$ | 적합으로 얻는 스케일 상수 |
| $\alpha_N,\ \alpha_D$ | 멱지수 |

양변에 로그를 취하면 다음과 같습니다.

$$\log L=\alpha_D\log D_c-\alpha_D\log D$$

- log-log 그래프에서 기울기가 $-\alpha_D$인 직선이 됩니다
- "scaling law가 성립한다"는 말은 이 직선이 여러 자릿수에 걸쳐 유지된다는 뜻입니다

두 변수를 함께 넣은 식은 다음과 같습니다.

$$L(N,D)=\left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D}+\frac{D_c}{D}\right]^{\alpha_D}$$

- $(N_c/N)^{\alpha_N/\alpha_D}$ 항: 모델 크기가 부족해서 생기는 오차
- $D_c/D$ 항: 데이터가 부족해서 생기는 오차

이 식에서 중요한 것은 극한입니다. $N$을 고정하고 $D\to\infty$로 보내면 다음과 같이 됩니다.

$$L(N,D)\ \xrightarrow{\ D\to\infty\ }\ \left(\frac{N_c}{N}\right)^{\alpha_N}$$

**즉 loss는 모델 크기가 정하는 바닥에 닿습니다.** 작은 모델은 데이터를 아무리 더 줘도 이 바닥 아래로 내려가지 못합니다. 이 때문에 크기별 loss 곡선을 compute 축 위에 겹쳐 그리면 보통 다음과 같은 모양이 나옵니다.

```
 loss (log)
   ^
   | *
   |  *   +
   |   *    +     o
   |    * * * * *   +      o                <- small N: hits floor early
   |                  + + + + +    o        <- mid N: floor later
   |                                o  o    <- large N: still falling
   +------------------------------------------> compute (log)
```

- small N: 일찍 바닥에 닿아 평평해집니다
- large N: 같은 compute 구간에서 계속 내려갑니다

이 그림은 4절에서 GEN-0의 "ossification" 주장을 읽을 때 다시 씁니다.

### 2.2 Transfer scaling과 ossification: pretrain → finetune 설정

LLM 문헌에는 사전학습 자체의 scaling과 다른 종류의 scaling이 하나 더 있습니다. 사전학습의 이점이 fine-tuning 이후까지 얼마나 이어지는가를 다루는 scaling입니다. Hernandez et al.(2021)은 이 이점을 데이터량으로 환산하였습니다.

$$D_T = D_E - D_F$$

- $D_F$: 실제로 쓴 fine-tuning 데이터량
- $D_E$: 사전학습 없이(from scratch) 같은 성능을 내는 데 필요한 데이터량
- $D_T$: effective data transferred. 사전학습의 가치를 fine-tuning 데이터 몇 개 분량인지로 환산한 값

같은 논문은 **ossification**이라는 현상에도 이름을 붙였습니다. 모델이 작고 fine-tuning 데이터가 많을 때, 사전학습한 모델이 from scratch 모델보다 오히려 못해지는 현상입니다. 사전학습이 weight를 굳혀서 새 분포를 받아들이지 못하게 만든다는 의미입니다. 사전학습을 과하게 한 LLM일수록 fine-tuning 후 성능이 떨어진다는 Springer et al.(2025)의 관찰도 같은 계열에 속합니다.

여기서 두 가지를 기억해둡니다.

- **⓵** ossification은 **pretrain → finetune 설정**에서 나타나는 현상입니다 (4절에서 다시 다룹니다)
- **⓶** "사전학습이 fine-tuning 데이터를 얼마나 대체하나"는 $D_T$에 대한 질문입니다 (5.2절에서 다시 다룹니다)

### 2.3 벽 2의 정체: action chunking과 추론 지연

현재 VLA 대부분은 **action chunking**을 씁니다. 한 번 추론할 때 $H$스텝 분량의 행동을 한꺼번에 뽑고, 그 청크를 open-loop로 실행하는 방식입니다.

$$a_{t:t+H}\ \sim\ \pi_\theta(\cdot \mid o_t,\ g)$$

- $a_{t:t+H}$: 시점 $t$부터 $H$스텝 분량의 행동 청크
- $o_t$: 시점 $t$의 관측 (이미지, proprioception)
- $g$: 언어로 주어진 목표
- $\pi_\theta$: 파라미터 $\theta$를 가진 정책

문제는 추론에 걸리는 시간 $\delta$입니다. 관측 $o_t$로 추론을 시작해서 행동이 나올 무렵이면 세계는 이미 $\delta$만큼 진행되어 있고, 모델이 클수록 $\delta$도 커집니다. 이를 동기식으로 돌리면 다음과 같습니다.

```
(a) synchronous chunking
  infer  [===]           [===]           [===]
  act         >>>>>>>>>>      >>>>>>>>>>      >>>>
```

- 추론하는 동안 로봇이 멈추거나, 이전 청크를 끝까지 open-loop로 실행합니다
- 청크 경계에서 행동이 끊깁니다

이 문제에 대한 해법이 로봇 분야에 없었던 것은 아닙니다. 대표적으로 두 계열이 있습니다.

```
(b) System1-System2 (e.g. Helix)
  S2 big   [==========]  [==========]  [==========]
  S1 small  > > > > > > > > > > > > > > > > > > > >

(c) inference-time guidance (e.g. RTC)
  act    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  infer       [=== next chunk, inpainted ===]
```

그럼에도 GEN-0는 이 해법들에 기대지 않고 새로운 방식이 필요하다고 밝혔습니다. 그 이유는 두 해법과 scaling의 관계에 있습니다.

> ### 💡 기존 해법은 지연을 풀지만 scaling과 긴장 관계에 있다
>
> | | 작동 방식 | 지연을 푸는 방법 | scaling과의 긴장 |
> |---|---|---|---|
> | **(b) System1-System2** (Figure Helix) | 큰 VLM(S2)이 느리게 latent를 내고, 작은 고속 정책(S1)이 실제 행동을 냄 | 행동은 작은 S1이 빠르게 냄 | 행동을 내는 부분은 작은 채로 남음. 큰 모델을 키워도 그 이득이 반사 동작과 손놀림까지 닿지 않음 |
> | **(c) inference-time guidance** (RTC, Black et al., 2025) | 현재 청크를 실행하는 동안 다음 청크를 생성함. 지연 구간에 실행될 행동은 고정하고, 나머지는 그 뒤에 자연스럽게 이어지도록 inpainting함 | 청크 경계를 매끄럽게 이음 | 지연 $\delta$가 길어질수록 고정해야 하는 구간도 길어져 반응성이 떨어짐. 모델이 클수록 불리한 구조는 그대로임 |
>
> **두 해법이 "큰 모델은 느리다"는 전제를 받아들이고 그 결과를 관리하는 쪽이었다면, GEN-0가 원하는 것은 큰 모델 자체가 제어 루프 안에 있으면서도 지연에 발목 잡히지 않는 구조입니다.** 3.2절의 Harmonic Reasoning이 바로 이 지점을 겨냥합니다.

---

## 3. 방법: 세 벽에 대한 GEN-0의 답

GEN-0가 각 벽에 내놓은 답과 공개 수준을 먼저 정리하면 다음과 같습니다.

| 벽 | GEN-0의 답 | 공개 수준 |
|---|---|---|
| ⛔ 벽 1: 데이터 regime | 27만 시간 규모의 자체 데이터 엔진 | 규모와 인프라는 공개. 수집 장치와 action 표현은 미공개 |
| ⛔ 벽 2: 실시간성 | Harmonic Reasoning | 이름과 방향만 공개 |
| ⛔ 벽 3: 측정 | next-action validation error를 proxy로 쓰고 실물 A/B로 검증 | figure 수준. 계수는 미공개 |

### 3.1 데이터 엔진 (벽 1)

GEN-0는 자체 데이터로 x축을 만들었습니다. 블로그가 밝힌 규모는 다음과 같습니다.

| 항목 | 규모 |
|---|---|
| 누적 데이터 | 27만 시간 이상의 실세계 manipulation trajectory |
| 수집처 | 전 세계 수천 곳의 가정, 창고, 작업장 |
| 증가 속도 | 주당 1만 시간 이상, 계속 가속 중 |
| 비교 | 2025년 11월 기준 공개된 대형 로봇 데이터셋보다 몇 자릿수 큼 (Figure 5) |

데이터를 모으는 것만큼 옮기고 처리하는 것도 문제입니다. 블로그에 따르면 수집 거점의 upload 대역폭을 확보하려고 전용 인터넷 회선을 새로 깔았고, multi-cloud 계약을 맺고 custom upload 장비를 만들었으며, O(10K) core 규모로 멀티모달 데이터를 연속 처리해 수십 PB를 압축했습니다. 그 결과 학습 하루에 실세계 경험 6.85년 분량을 흡수할 수 있다고 밝혔습니다.

**데이터량뿐 아니라 처리량으로 봐도 LLM급 pipeline입니다.** 6.85년은 약 6만 시간이므로, 27만 시간 전체를 한 번 도는 데 약 4.5일이 걸립니다.

블로그는 이 데이터를 in-house robotics dataset이라 부르고, 수집 수단을 "전 세계의 hardware 네트워크와 수천 대의 데이터 수집 장치 및 로봇"이라고 썼습니다. 다만 어떤 장치로 무엇을 기록했는지, 행동을 어떤 표현으로 저장했는지는 공개되지 않았습니다.

### 3.2 Harmonic Reasoning (벽 2)

**2.3절에서 본 두 해법이 지연 문제를 구조(S1/S2)나 추론 시점의 보정(RTC)으로 푸는 방식이었다면, GEN-0는 학습 방식 자체로 풀겠다고 밝혔습니다.** 블로그의 설명을 정리하면 다음과 같습니다.

- **정의**: sensing 토큰과 acting 토큰의 비동기·연속시간(asynchronous, continuous-time) stream 사이에 조화로운 상호작용을 만드는 새로운 학습 방식입니다. 모델은 생각과 행동을 동시에 하도록 학습됩니다
- **주장**: 이 방식 덕분에 System1-System2 아키텍처나 inference-time guidance에 의존하지 않고도 매우 큰 모델 크기까지 scaling할 수 있습니다
- **사례**: camera kit 조립. 천을 상자에 넣고, 판지 트레이를 접고, 카메라를 비닐에서 꺼내 넣고, 작은 플랩까지 끼워 상자를 닫은 뒤 비닐을 버리는 장기 태스크입니다. 이 과정을 명시적인 subtask 구분 없이 단일 stream 안에서 수행합니다

2.3절의 그림에 이어 붙이면 다음과 같습니다.

```
(d) Harmonic Reasoning (claimed)
  sense  s  s s   s  s s  s   s  s
  act     a a  a   a a  a  a a   a
```

- (a)~(c)는 모두 "관측 → 추론 → 행동 청크"가 번갈아 일어나는 구조를 전제합니다
- (d)는 sensing과 acting이 고정 주기 없이 시간축 위에서 겹쳐 흐른다는 주장입니다

구현은 공개되지 않았지만, "비동기·연속시간 stream"이라는 표현에서 직접 끌어낼 수 있는 해석은 두 가지입니다.

- **⓵ event 단위 배치**: 고정된 $\Delta t$마다 (관측, 행동) 쌍을 두는 대신, timestamp가 붙은 event 단위로 sensing 토큰과 acting 토큰을 배치합니다
- **⓶ 연속 시간 인코딩**: 토큰의 순서 index 대신 연속 시간을 position encoding으로 씁니다

블로그는 이 방식을 LLM의 test-time reasoning, 즉 답하기 전에 더 오래 생각하는 방식과 대비시켰습니다. LLM 쪽에서 "생각하면서 동시에 행동하는" 구조에 대응하는 것을 찾으면 다음과 같습니다.

> ### 💡 가장 가까운 LLM 쪽 대응물은 full-duplex 음성 모델이다
>
> 일반 챗봇은 사용자가 말을 마친 뒤에 답합니다. 차례를 주고받는 구조이고, 2.3절 (a)의 동기식 청크와 같은 모양입니다. 반면 full-duplex 음성 대화 모델(예: Moshi)은 사용자의 음성 stream과 모델 자신의 음성 stream을 병렬로 모델링하며, 덕분에 상대가 말하는 도중에도 맞장구를 치거나 끼어들 수 있습니다.
>
> | 음성 대화 | 로봇 제어 |
> |---|---|
> | 듣기 stream | sensing 토큰 stream |
> | 말하기 stream | acting 토큰 stream |
> | 차례를 주고받는 챗봇 | 동기식 action chunking |
> | full-duplex 모델 | Harmonic Reasoning (주장) |
>
> 이 대응은 블로그의 문구에서 끌어낸 비유일 뿐이며, GEN-0가 실제로 이렇게 구현되었다는 근거는 아닙니다.

**벽 2에 대해서는 해법의 이름과 방향만 제시된 셈입니다.** 토큰화, 시간 인코딩, loss, 추론 주기가 모두 공개되지 않았고, (b)(c)와 비교한 실험도 없습니다.

### 3.3 측정 설계 (벽 3)

GEN-0는 scaling 곡선의 y축으로 **next-action validation prediction error**를 씁니다. held-out 데이터의 관측을 주고 다음 행동을 예측하게 한 뒤, 정답 행동과의 오차를 잽니다. Table 1에서는 이 오차를 MSE로 정의합니다.

$$\text{MSE}_{\text{val}}=\lVert a^\star-\hat a\rVert_2^2$$

- $a^\star$: held-out 데이터의 정답 행동
- $\hat a$: 모델이 예측한 행동

LLM의 validation loss처럼, 실물 rollout 없이 계산할 수 있는 매끄러운 proxy입니다. 블로그의 실험은 이 proxy를 두 겹으로 활용합니다.

```
 pretrain (size N, compute C) -----------------------> zero-shot val error  (Fig 1)
 pretrain (data subset D) ---> post-train (fixed) ---> val error            (Fig 2, 4)
                                                  \--> robot success        (Fig 3)
```

- **proxy의 scaling**: Figure 1, 2, 4는 모두 validation error를 y축으로 씁니다
- **proxy의 타당성**: Figure 3은 같은 추세가 실물 성공률에서도 나타나는지를 blind A/B 평가로 확인합니다

여기서 분명히 해둘 점이 있습니다. Figure 1의 "zero-shot"은 완전히 withheld된 장기 태스크 데이터에 대한 **offline 예측 오차**이며, 로봇이 그 태스크를 실제로 수행한 결과가 아닙니다.

---

## 4. 왜 작동하는가: 벽 1을 넘자 드러난 조건

블로그는 로봇에서 그동안 scaling이 관측되지 않은 이유를 두 가지로 설명하였습니다. 고데이터 regime이 없었고, 그 regime에서 충분히 큰 모델도 없었다는 것입니다. 벽 1을 넘어 고데이터 regime에 들어서자 두 번째 조건이 드러났습니다.

| 모델 크기 | 관측 (Figure 1) |
|---|---|
| 1B | 복잡하고 다양한 sensorimotor 데이터를 흡수하지 못함. 학습이 진행될수록 새 정보를 받아들이지 못함 |
| 6B | 사전학습의 이점이 나타나기 시작하고, 강한 multi-task 능력을 보임 |
| 7B+ | 대규모 사전학습 데이터를 내재화하고, 수천 step의 post-training만으로 downstream 태스크에 전이함 |

Figure 1의 축은 다음과 같습니다.

- **y축**: 완전히 withheld된 장기 태스크에 대한 next-action validation prediction error (낮을수록 좋음)
- **x축**: 사전학습 compute. GEN-0 7B의 값을 1.0으로 정규화했습니다
- **색**: 모델 크기

블로그는 이후 10B+까지 키웠고, 새 태스크에 적응하는 데 필요한 post-training이 점점 줄어드는 것을 관측했다고 밝혔습니다. 다만 수치는 제시하지 않았습니다.

블로그는 이 결과를 Moravec's paradox와 연결하였습니다. LLM 문헌의 ossification은 O(10M) 파라미터 규모에서 관측됐는데, 로봇에서는 O(1B)에서 나타났다는 것입니다. 사람에게 쉬운 지각과 손재주가 추상적 추론보다 오히려 더 많은 계산을 요구하듯, 물리 세계의 지능(physical commonsense)은 compute 관점에서 활성화 임계가 더 높을 수 있다는 해석입니다.

다만 2.2절에서 본 ossification을 떠올리면, GEN-0가 관측한 것이 LLM 문헌의 그 ossification과 같은 현상인지 의문이 생깁니다.

> ### ⚠️ 이름은 같지만 측정한 것이 다르다
>
> 블로그 각주 8이 스스로 밝히듯, LLM 문헌의 ossification은 **pretrain → finetune 설정**에서 나타나는 현상입니다(2.2절의 ⓵). 반면 GEN-0가 Figure 1에서 관측한 것은 **순수 사전학습 중에** zero-shot 일반화가 정체되는 "ossification 유형의 행동"입니다.
>
> 한 걸음 더 들어가면, 순수 사전학습 중 작은 모델의 곡선이 compute를 늘려도 멈추는 현상은 2.1절의 $L(N)$ 바닥과 텍스트 서술만으로는 구분되지 않습니다. 크기별 곡선을 겹쳐 그렸을 때 작은 모델이 먼저 평평해지는 것은 Kaplan compute plot에서 흔히 보는 모양이기 때문입니다. 둘을 구분하는 관건은 1B 곡선이 **정체**하는지, 아니면 **상승**하는지(학습할수록 held-out 오차가 커지는지)입니다. 캡션과 본문만으로는 이를 판별할 수 없습니다.

Moravec 해석에는 대안도 있습니다. 시간당 정보량이 텍스트보다 훨씬 많은 고주파 멀티모달 데이터라면, 더 큰 $N$이 필요한 이유가 도메인 고유의 "지능 임계"가 아니라 단순히 데이터의 엔트로피 때문일 수 있습니다. 두 해석을 가려내려면 토큰 수나 정보량으로 정규화한 비교가 필요한데, 그런 실험은 없습니다.

한편 블로그의 헤드라인은 "7B에서의 phase transition"입니다. 공개된 모델 크기만으로 전이 위치를 7B로 특정할 수 있는지는 따져볼 필요가 있습니다.

> ### ⚠️ 팩트체크: "7B phase transition"의 해상도
>
> 공개된 모델 크기는 1B, 6B, 7B 세 가지입니다. 1B와 6B 사이에 측정점이 없으므로, 전이 위치에 대해서는 "1B와 6B 사이 어딘가"라고까지만 말할 수 있습니다. 본문도 6B부터 사전학습의 이점이 나타난다고 썼습니다. 헤드라인의 "7B"는 figure의 해상도보다 더 정밀한 표현이며, 분명하게 말할 수 있는 것은 1B가 일찍 정체한다는 점뿐입니다.

---

## 5. 실험

### 5.1 사전학습 데이터량과 post-training 성능 (Figure 2, 3)

**설계**: 사전학습 데이터의 서로 다른 subset으로 학습한 checkpoint들을 준비하고, 각 checkpoint를 동일한 multi-task language-conditioned 데이터로 SFT합니다. 16개 task set을 동시에 학습하며, 다음 세 종류가 포함됩니다.

| 종류 | 예 |
|---|---|
| dexterity | Lego 조립 |
| 산업 workflow | 패스트푸드 포장 |
| generalization | "_ anything"류 태스크 |

**결과**

- **Figure 2**: 사전학습 데이터가 많을수록 post-training의 validation loss와 16개 task set 전부의 next-action prediction error가 개선됩니다
- **Figure 3**: 실물 로봇에서 blind A/B, closed-loop rollout으로 평가했습니다. task 데이터 5.6시간(전체의 1%)만으로 post-training해도 사전학습 데이터가 많을수록 평균 성공률이 오릅니다. 최고 성능은 전체 사전학습 데이터와 550시간 이상의 task 데이터를 함께 쓸 때 나오며, 일부 경우에는 최대 99%에 이릅니다
- **데이터 분리**: 사전학습 데이터와 post-training 데이터는 서로 다른 사람이 전혀 다른 환경에서 수집했으므로 겹치지 않습니다

**Figure 3은 벽 3을 넘는 마지막 고리입니다.** offline proxy에서 본 추세가 실물 성공률에서도 이어진다는 증거이기 때문입니다. 다만 이것으로 GEN-0의 실제 성공률 수준까지 알 수 있는지, 특히 "최대 99%"를 대표값으로 읽어도 되는지는 별개의 문제입니다.

> ### ⚠️ 팩트체크: 99%는 peak이다
>
> 본문의 99%는 특정 경우의 peak이며, 평균 성공률은 제시되지 않았습니다. Figure 3이 보여주는 것은 사전학습 데이터가 늘수록 성공률이 오른다는 추세이며, GEN-0의 전형적인 성공률을 나타내지는 않습니다.

post-training 데이터(5.6시간, 550시간 이상)가 어떤 방식으로 수집되었는지는 GEN-0 글에 나오지 않습니다.

### 5.2 Power law (Figure 4)

downstream 태스크의 데이터와 fine-tuning budget을 고정하고 사전학습 데이터 크기만 바꾸면, downstream 오차는 다음 멱법칙으로 예측됩니다.

$$L(D)=\left(\frac{D_c}{D}\right)^{\alpha_D}$$

- $L$: post-training 후 downstream 태스크의 next-action validation error
- $D$: 사전학습 데이터 크기 (action trajectory 수)
- $D_c$: 적합으로 얻는 스케일 상수
- $\alpha_D$: 멱지수. log-log 그래프에서 기울기가 $-\alpha_D$인 직선이 됩니다

블로그가 공개한 예시는 Clothes Handling(옷 분류, 뒤섞인 옷 정리, 단추 채우기, 옷걸이에 걸기)이며, 사전학습 데이터가 10억 trajectory일 때의 성능을 예측합니다.

형태는 2.1절의 Kaplan $L(D)$와 똑같습니다. 다만 형태가 같다고 같은 것을 재는 것은 아닙니다.

> ### 📌 Figure 4는 pretraining scaling이 아니라 transfer scaling이다
>
> | | Kaplan $L(D)$ | GEN-0 Figure 4 |
> |---|---|---|
> | $L$ | 사전학습 test loss | **post-training 후** downstream error |
> | $D$ | 사전학습 토큰 수 | 사전학습 trajectory 수 |
> | 고정하는 것 | 모델 크기가 충분히 큼 | downstream 데이터와 fine-tuning budget |
> | 성격 | pretraining scaling | **transfer scaling** |
>
> 블로그가 내세우는 용도는 두 가지입니다. 하나는 "사전학습 데이터를 늘리면 post-training 데이터를 얼마나 아낄 수 있는가"인데, 이것은 정확히 2.2절의 effective data transferred $D_T$에 대한 질문입니다(2.2절의 ⓶). 다른 하나는 "목표 오차에 도달하려면 사전학습 데이터가 얼마나 필요한가"입니다. 블로그는 여기에 모델 크기 scaling law를 결합하면 태스크별로 사전학습 compute와 데이터의 최적 배분을 예측할 수 있다고 주장하였습니다. Chinchilla류의 분석이지만 블로그가 Chinchilla를 직접 언급하지는 않았습니다.

블로그는 이 곡선을 파트너 태스크 논의와 데이터 견적에 쓴다고 명시하였습니다. 다만 견적 도구로 쓰기에는 공개된 근거가 부족합니다.

> ### ⚠️ 외삽의 근거가 공개되지 않았다
>
> **⓵ 계수 미공개**: $D_c$, $\alpha_D$, fit 오차, 데이터 점 개수가 공개되지 않았습니다. 공개된 fit은 Clothes Handling 하나뿐이고, 측정한 모든 태스크에 적용된다는 것은 서술로만 제시됩니다.
>
> **⓶ 바닥항이 없습니다**: 순수 멱법칙이므로 $D\to\infty$이면 $L\to 0$입니다. 그런데 같은 상황에서도 사람마다 다르게 행동하는 시연 데이터라면, next-action 오차에는 줄일 수 없는 바닥이 있는 것이 자연스럽습니다. Chinchilla(Hoffmann et al., 2022)가 상수항을 두는 이유도 여기에 있습니다.
>
> $$L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$$
>
> - $E$: 줄일 수 없는 손실 (irreducible loss)
> - $A,\ B,\ \alpha,\ \beta$: 적합 상수
>
> 10억 trajectory까지의 외삽 결과는 이 바닥항을 넣느냐 마느냐에 크게 좌우됩니다.
>
> **⓷ 단위 환산 불가**: $D$의 단위는 trajectory 수인데, 27만 "시간"과 환산할 수 있는 평균 trajectory 길이가 공개되지 않았습니다. 따라서 10억 trajectory가 몇 시간에 해당하는지 알 수 없습니다.
>
> **⓸ 성공률 fit 없음**: power law는 validation error에 대한 것입니다. 성공률에 대해서는 Figure 3의 추세만 있습니다.

### 5.3 데이터 혼합의 과학 (Table 1)

**설계**: 8개 사전학습 데이터셋으로 각각 학습한 모델을 10개의 장기 task set에 fine-tune해 비교합니다. task set은 dexterity, applications(실세계 응용), generalization 세 그룹으로 나뉩니다. 데이터셋은 외부 데이터 수집 파트너(data foundry) A, B, C와 수집 방식(Class)의 조합으로 구분됩니다.

| Class | 수집 방식 |
|---|---|
| Class 1 | 특정 태스크 데이터 |
| Class 2 | Class 1과 Class 3의 중간 |
| Class 3 | do-anything형 데이터 |

지표는 두 가지입니다. 하나는 3.3절의 $\text{MSE}_{\text{val}}$이고, 다른 하나는 reverse KL입니다. LLM 배경에서 보면 예측 오차가 이미 있는데 reverse KL을 따로 두는 이유가 자연스럽게 궁금해집니다.

> ### 💡 SFT는 forward KL을 줄이고, reverse KL은 정책 자신의 샘플을 본다
>
> $p$를 데이터(정답) 분포, $q$를 정책 분포라고 하면, KL은 방향에 따라 성격이 달라집니다.
>
> $$D_{KL}(p\,\|\,q)=\mathbb{E}_{a\sim p}\!\left[\log\frac{p(a)}{q(a)}\right],\qquad D_{KL}(q\,\|\,p)=\mathbb{E}_{a\sim q}\!\left[\log\frac{q(a)}{p(a)}\right]$$
>
> - 왼쪽은 forward KL입니다. 데이터에서 뽑은 샘플 위에서 평가합니다
> - 오른쪽은 reverse KL입니다. 정책이 뽑은 샘플 위에서 평가합니다
>
> | | forward | reverse |
> |---|---|---|
> | 기대값을 취하는 샘플 | 데이터 $a\sim p$ | 정책 $a\sim q$ |
> | 큰 벌점을 받는 경우 | $p$에 있는 행동에 $q$가 확률을 거의 주지 않을 때 | $q$가 내놓은 샘플이 $p$ 아래에서 확률이 낮을 때 |
> | 성질 | mode-covering | mode-seeking |
> | LLM 쪽 대응 | MLE / SFT | RLHF의 KL 페널티, on-policy distillation |
>
> 블로그가 인용한 f-divergence 관점의 모방학습 문헌에서도 BC(=SFT)는 forward KL 최소화에 해당합니다. SFT 손실은 데이터 샘플에서만 계산되므로, 정책이 **실제로 뱉는 샘플의 질**을 직접 보지 못합니다. reverse KL은 그 빈자리를 채우는 지표입니다.

블로그는 reverse KL을 Monte-Carlo로 추정하였습니다. 정책 샘플 $M$개 각각에 단위분산 Gaussian을 씌운 혼합분포 $q$와, 정답 행동에 단위분산 Gaussian을 둔 $p$를 씁니다.

$$q(a)=\frac{1}{M}\sum_{j=1}^{M}\mathcal{N}(a;\ \hat a_j,\ I),\qquad p(a)=\mathcal{N}(a;\ a^\star,\ I)$$

$$\widehat{D}_{KL}(q\,\|\,p)\approx\frac{1}{M}\sum_{m=1}^{M}\Big[\log q(\hat a_m)-\log p(\hat a_m)\Big]$$

- $\hat a_m$: 정책이 뽑은 $m$번째 행동 샘플
- $a^\star$: 데이터의 정답 행동
- $M$: 정책 샘플 수 (미공개)
- $I$: 단위 공분산 행렬

블로그의 판정 규칙은 다음과 같습니다. 예측 오차와 reverse KL이 모두 낮은 모델은 SFT post-training에 유리하고, 예측 오차는 높지만 reverse KL이 낮은 모델은 분포적으로 multimodal이어서 RL post-training에 유리합니다. 이 규칙이 왜 이런 모양인지는 추정기가 실제로 무엇을 재는지를 풀어 보면 드러납니다.

> ### 💡 reverse KL 추정기는 "½·MSE + 샘플 뭉침 항"이다
>
> 두 Gaussian 밀도의 로그를 풀어 쓰면 다음과 같습니다.
>
> $$\log p(\hat a_m)=-\tfrac{1}{2}\lVert\hat a_m-a^\star\rVert^2-\tfrac{d}{2}\log 2\pi,\qquad \log q(\hat a_m)=\log\Big[\tfrac{1}{M}\textstyle\sum_{j}e^{-\frac{1}{2}\lVert\hat a_m-\hat a_j\rVert^2}\Big]-\tfrac{d}{2}\log 2\pi$$
>
> - $d$: 행동 벡터의 차원. 두 식의 $-\tfrac{d}{2}\log 2\pi$는 빼면 상쇄됩니다
>
> 이를 추정기에 대입하면 정확히 다음 식이 됩니다.
>
> $$\widehat{D}_{KL}=\underbrace{\frac{1}{2M}\sum_{m=1}^{M}\lVert\hat a_m-a^\star\rVert^2}_{\text{(i)}}\;+\;\underbrace{\frac{1}{M}\sum_{m=1}^{M}\log\Big[\frac{1}{M}\sum_{j=1}^{M}e^{-\frac{1}{2}\lVert\hat a_m-\hat a_j\rVert^2}\Big]}_{\text{(ii)}\ \in\ [-\log M,\ 0]}$$
>
> - **(i)**: 샘플이 정답에서 얼마나 떨어져 있는가. 샘플 MSE의 절반입니다
> - **(ii)**: 샘플끼리 얼마나 뭉쳐 있는가. 모든 샘플이 한 점에 모이면 0이고, 서로 멀리 흩어질수록 $-\log M$에 가까워집니다
>
> **reverse KL이 낮다는 것은 정확하거나(i가 작다) 샘플이 퍼져 있거나(ii가 작다) 둘 중 하나입니다.** MSE와 짝지어 보면 두 경우가 구분됩니다.
>
> | MSE | Rev KL | 분해식으로 본 해석 | 블로그의 판정 |
> |---|---|---|---|
> | 낮음 | 낮음 | 정확하고 일관된 정책 | SFT post-training에 유리 |
> | 높음 | 낮음 | 평균적으로는 빗나가지만 샘플이 넓게 퍼져 있음 → multimodal | RL post-training에 유리 |
>
> RL 쪽이 유리한 이유도 여기서 드러납니다. RL post-training은 정책 샘플 중 좋은 것을 강화하는 과정이라, 샘플이 다양할수록 탐색 폭이 넓어집니다. LLM에서 base 모델의 샘플 다양성(pass@k)이 RL로 끌어올릴 수 있는 여지를 정한다는 직관과 같습니다.
>
> 다만 두 가지 단서가 붙습니다.
>
> - **⓵ 교과서적 mode-seeking과는 다릅니다**: $p$가 샘플마다 봉우리가 하나인 Gaussian이므로 이 추정기가 mode-seeking을 직접 재지는 않습니다. 실제로 재는 것은 "정답과의 거리"와 "샘플 분산"의 조합입니다
> - **⓶ 절대값은 역산할 수 없습니다**: MSE를 샘플 기준으로 계산했는지 평균 기준으로 계산했는지, 그리고 $M$ 값이 공개되지 않았습니다

이제 표를 읽을 수 있습니다. 값은 ×10⁻³ 단위이고, 판독 열은 위 분해식에 따른 해석입니다.

| Dataset | Pred Err (Dex / App / Gen) | Rev KL (Dex / App / Gen) | 판독 |
|---|---|---|---|
| A · Class 1 | 3.077 / 3.342 / 3.090 | 2.006 / 2.589 / 1.981 | KL이 세 그룹 모두에서 최고 |
| A · Class 2 | 3.062 / 3.333 / 3.065 | 1.887 / 2.446 / 1.939 | 중간 |
| A · Class 3 | 3.057 / 3.313 / 3.059 | 1.983 / 2.461 / 1.902 | 중간 |
| A · Class 2+3 | 3.160 / 3.419 / 3.157 | 1.841 / 2.286 / 1.855 | MSE 높음, KL 낮음 → RL형 |
| B · Class 1 | 3.027 / 3.304 / 3.046 | 1.893 / 2.461 / 1.923 | MSE 낮음, KL 중간 |
| B · Class 2 Objs | 3.144 / 3.411 / 3.160 | 1.847 / 2.332 / 1.867 | MSE 높음, KL 낮음 → RL형 |
| B · Class 2 Skills | 3.020 / 3.292 / 3.053 | 1.826 / 2.423 / 1.903 | MSE 최저권, Dex KL 최저 → SFT형 |
| C · Class 3 | 3.062 / 3.321 / 3.079 | 1.921 / 2.369 / 1.910 | 중간 |

표에서 읽을 만한 신호는 두 가지입니다.

- **혼합 효과가 가장 일관된 신호입니다**: Partner A의 Class 2+3은 세 그룹 **모두**에서 Class 2나 Class 3 단독보다 예측 오차가 높고 reverse KL이 낮습니다. 분해식으로 읽으면, 두 수집 방식을 섞자 정책 샘플이 더 넓게 퍼졌다는 뜻입니다
- **같은 파트너, 같은 Class라도 성격이 갈립니다**: Partner B의 Class 2를 물체 다양화(Objs)와 스킬 다양화(Skills)로 나누자, Objs는 RL형 사분면에, Skills는 SFT형 사분면에 놓입니다. Class 라벨보다 "무엇을 다양화했는가"가 모델의 성격을 더 크게 가릅니다

블로그의 결론은 세 가지입니다.

- **⓵** 데이터의 양보다 품질과 다양성이 중요합니다
- **⓶** 데이터 혼합을 설계하면 성격이 서로 다른 사전학습 모델을 얻을 수 있습니다
- **⓷** 여러 수집 전략을 대규모로 운영하면서 어떤 데이터가 사전학습을 가장 개선하는지 계속 A/B 테스트하고, 그 결과를 바탕으로 파트너에게 무엇을 어떻게 수집할지 피드백합니다

다만 이 표에 나타난 차이를 얼마나 믿을 수 있는지는 따로 확인해야 합니다.

> ### ⚠️ 팩트체크: 차이가 작고 통계 정보가 없다
>
> 그룹 안에서 최대값과 최소값의 차이는 예측 오차가 3.7~4.6%, reverse KL이 6.8~13.3%입니다. 분산이나 신뢰구간이 없고, 데이터셋마다 크기를 맞췄는지도 공개되지 않아서 순위를 얼마나 믿을 수 있는지 판단할 수 없습니다. SFT와 RL 중 어느 쪽에 유리하다는 판정도 서술로만 제시되며, 실제로 SFT나 RL post-training을 해본 결과는 표에 없습니다.
>
> 사소하지만 각주에도 오기가 있습니다. Minka의 divergence 리포트는 1988년으로 표기되어 있지만 실제로는 2005년이고, 각주 12와 13이 같은 링크를 가리키며, Kaplan et al.은 2021년으로 표기되어 있지만 arXiv 공개는 2020년입니다.

---

## 6. 위치잡기: 이웃 연구들 속에서

블로그는 related work를 각주 정도로만 다뤘습니다. 인용된 연구와, 인용되지 않았지만 가까운 위치에 있는 연구를 함께 놓아야 GEN-0의 위치가 보입니다.

| 계보 | 대표 연구 | GEN-0와의 관계 |
|---|---|---|
| VLM 전이형 RFM | PaLM-E, RT-2 | 대립각. GEN-0가 "빠진 것이 있다"고 지목한 레시피 |
| LLM scaling law | Kaplan, Hernandez, Springer | 분석 틀을 그대로 가져옴 |
| 실시간 실행 | Figure Helix (S1/S2), RTC | Harmonic Reasoning이 차별화하려는 대상 |
| 계층형 장기 태스크 | Hi Robot (Shi et al., 2025) | 고수준 VLM이 언어 subtask를 명시적으로 내려주는 설계. GEN-0는 반대로 명시적 subtask가 없는 단일 stream을 내세움 |

들어가며에서 GEN-0가 로봇 scaling law를 "처음" 보이겠다고 나섰다고 썼습니다. 이 "처음"은 앞선 연구와 비교해 범위를 좁혀 읽어야 합니다.

> ### 🔗 Lin et al. (2024): 먼저 나온 로봇 데이터 scaling 연구
>
> *Data Scaling Laws in Imitation Learning for Robotic Manipulation* (ICLR 2025 oral)은 GEN-0보다 약 1년 앞서 로봇 모방학습의 데이터 scaling을 체계적으로 조사하였습니다. 4만 개가 넘는 시연을 모으고 1만 5천 회가 넘는 실물 rollout으로 평가한 결과, 정책의 일반화 성능이 학습 환경 수와 물체 수에 대해 대략 멱법칙을 따른다고 보고하였습니다.
>
> | | Lin et al. (2024) | GEN-0 (2025) |
> |---|---|---|
> | 대상 | single-task 모방학습 정책 | 대규모 사전학습 파운데이션 모델 |
> | x축 | 학습 환경 수, 물체 수 | 사전학습 데이터량(trajectory), 모델 크기, compute |
> | y축 | 새 환경과 새 물체에서의 일반화 성능 (실물 rollout) | post-training 후 validation error (+ 실물 성공률 추세) |
> | 질문 | 데이터를 어떻게 다양화해야 일반화되는가 | 사전학습이 downstream으로 얼마나 전이되는가 |
>
> GEN-0 블로그는 Lin et al.을 인용하지 않았습니다. 따라서 GEN-0의 "처음"은 **대규모 사전학습 → post-training 전이의 scaling을 실증한 첫 사례**로 읽는 것이 정확하다고 판단됩니다.

---

## 7. 한계

블로그가 스스로 인정한 한계와, 읽으면서 추가로 짚어둘 만한 지점을 함께 정리합니다.

**블로그가 인정한 한계**

- 사실상 없습니다. 각주 8에서 용어 정의가 다르다고 밝힌 것과, 더 자세한 내용은 향후 글에서 다루겠다는 예고 정도입니다.

**추가로 짚을 지점**

- **구현 비공개**: 블로그 스스로 scaling의 요건으로 아키텍처, 학습 절차, 데이터 엔진을 꼽았지만, 공개된 것은 데이터 엔진의 규모뿐입니다. Harmonic Reasoning은 이름과 방향만 있습니다(3.2절).
- **데이터 구성 비공개**: 사전학습 데이터의 수집 장치, action 표현, embodiment 정렬 방식이 공개되지 않았습니다(3.1절).
- **성공률 근거가 얇습니다**: power law는 validation error 기준이고, 성공률에 대해서는 추세와 peak 99%만 있습니다(5.1절, 5.2절).
- **Cross-embodiment**: 6DoF, 7DoF, 16+DoF semi-humanoid 로봇에서 테스트했다는 언급뿐이고 수치가 없습니다. 여러 RFM이 공통으로 내세우는 속성이라 GEN-0만의 차별점으로 보기는 어렵다고 판단됩니다.
- **Table 1의 통계**: 차이가 작은데 신뢰구간이 없습니다(5.3절).

---

## 8. 마치며: 이 글이 시사하는 것

**GEN-0의 기여는 로봇 도메인에서 scaling 곡선을 그릴 수 있는 데이터 regime을 만들고, 그 위에서 사전학습의 이점이 downstream으로 예측 가능하게 전이된다는 것을 보였다는 점입니다.** 특정 아키텍처가 기여의 중심은 아닙니다.

LLM이나 diffusion 배경을 가진 사람에게 이 글은 유난히 익숙하게 읽힙니다.

- **분석 도구는 모두 LLM에서 왔습니다**: Kaplan의 멱법칙, Hernandez의 transfer scaling, forward/reverse KL이 그렇습니다. 새로운 것은 그 도구를 쓸 수 있게 해준 x축입니다.
- **Table 1은 data mixture 연구의 로봇판**: 다만 mixture의 재료가 "어느 파트너가 어떤 방식으로 모았는가"라는 운영 단위이며, 데이터 수집 운영 자체가 실험 변수가 됩니다.
- **Harmonic Reasoning은 full-duplex의 물리판**: 차례를 주고받는 방식에서 벗어나 동시에 흐르는 stream으로 가려는 방향입니다. 구현이 공개되면 가장 먼저 확인할 대목입니다.

마지막으로, 이 power law를 **데이터 견적 도구로 제시받는 입장**이라면 확인할 것이 세 가지 있습니다.

| 확인 항목 | 이유 |
|---|---|
| ⓵ 해당 태스크 계열의 $\alpha_D$, $D_c$와 fit 오차 | 공개된 fit은 Clothes Handling 하나뿐임 |
| ⓶ 바닥항 $E$를 넣은 fit과의 비교 | 순수 멱법칙은 외삽할 때 낙관적인 결과를 낼 수 있음 |
| ⓷ 목표 validation error와 성공률의 대응 관계 | power law는 오차에 대한 것이며, 성공률에 대한 것은 아님 |

---

## 부록: 용어 정리

| 용어 | 정의 |
|---|---|
| **scaling law** | 성능(loss)이 모델 크기, 데이터량, compute의 멱함수로 예측되는 관계. log-log 그래프에서 직선이 됨 |
| **transfer scaling** | 사전학습량과 fine-tuning 후 downstream 성능 사이의 scaling. GEN-0 Figure 4가 이 형태임 |
| **effective data transferred** | $D_T=D_E-D_F$. 사전학습의 가치를 fine-tuning 데이터량으로 환산한 값 |
| **ossification** | LLM 문헌에서는 작은 모델에서 사전학습이 오히려 fine-tuning을 방해하는 현상. GEN-0는 순수 사전학습 중 zero-shot 일반화가 정체되는 현상에 이 이름을 빌려 씀 |
| **action chunking** | 한 번의 추론으로 여러 스텝의 행동을 뽑아 open-loop로 실행하는 방식 |
| **Harmonic Reasoning** | sensing과 acting 토큰의 비동기·연속시간 stream을 동시에 다루도록 학습하는 GEN-0의 방식. 구현은 미공개 |
| **forward / reverse KL** | $D_{KL}(p\,\Vert\,q)$ / $D_{KL}(q\,\Vert\,p)$. 앞의 것은 mode-covering(SFT에 해당), 뒤의 것은 mode-seeking |
| **data foundry** | 외부 데이터 수집 파트너 |
| **Class 1 / 2 / 3** | 각각 특정 태스크 / 중간 / do-anything형 수집 방식 |

**원문**: [GEN-0 (Generalist AI Blog, 2025-11-04)](https://generalistai.com/blog/gen-0) · **관련**: [Lin et al., arXiv:2410.18647](https://arxiv.org/abs/2410.18647)
