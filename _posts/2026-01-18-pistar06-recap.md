---
layout: paper
lang: ko
ref: pistar06-recap
kind: paper-review
title: "π*₀.₆: a VLA That Learns From Experience (RECAP)"
date: 2026-01-18 12:00:00 -0800
paper_date: 2025-11-18
venue: "arXiv preprint · arXiv:2511.14759"
tags: [VLA, Reinforcement-Learning, Offline-RL, Advantage-Conditioning, Robot-Foundation-Model, Paper-Review]
authors: "Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, Danny Driess, Michael Equi, Adnan Esmail, Yunhao Fang, Chelsea Finn, Catherine Glossop, Thomas Godden, Ivan Goryachev, Lachy Groom, Hunter Hancock, Karol Hausman, Gashon Hussein, Brian Ichter, Szymon Jakubczak, Rowan Jen, Tim Jones, Ben Katz, Liyiming Ke, Chandra Kuchi, Marinda Lamb, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Yao Lu, Vishnu Mano, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Charvi Sharma, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, Will Stoeckle, Alex Swerdlow, James Tanner, Marcel Torne, Quan Vuong, Anna Walling, Haohuan Wang, Blake Williams, Sukwon Yoo, Lili Yu, Ury Zhilinsky, Zhiyuan Zhou"
affiliations: "Physical Intelligence"
summary: "RECAP은 가치함수로 각 행동이 데이터 평균보다 나았는지 판정하고, 그 판정을 텍스트 조건 하나로 넣는 조건부 SFT(CFG식 정책 개선)입니다. 사람 시연·자율 롤아웃·교정을 한 레시피로 흡수해, 배포 경험만으로 flow VLA의 처리량을 2배 이상 끌어올립니다."
paper_url: "https://arxiv.org/abs/2511.14759"
---

> **핵심 주장** — **거대한 flow-matching VLA를 RL로 개선하는 데 정책경사는 필요 없습니다.** 가치함수로 데이터 속 각 행동이 "평소보다 나았는지"를 판정하고, 그 판정을 텍스트 조건 하나로 넣어 조건부 SFT를 하면 됩니다. 이 레시피 하나가 사람 시연, 자율 롤아웃, 사람 교정을 모두 흡수하고, 배포 경험만으로 가장 어려운 태스크의 처리량을 2배 이상 끌어올립니다.

---

## 들어가며

VLA의 표준 레시피는 웹·로봇 혼합 데이터로 대규모 사전학습을 한 뒤, 목표 태스크의 teleoperation 시연으로 SFT하는 것입니다. 이 레시피로 만든 정책은 시연을 흉내 낼 뿐이라서, 원리상 시연보다 나아질 수 없습니다.

반면 사람은 처음엔 서툴러도 반복해서 시도하고 실수에서 배우며, 결국 가르쳐준 사람보다 능숙해집니다. Physical Intelligence는 π*₀.₆ 논문에서 VLA에게 이 "연습"을 시키는 방법으로 **RECAP**(RL with Experience and Corrections via Advantage-conditioned Policies)을 제시하였습니다. 로봇을 실제로 배포해 자율 시행과 사람 교정을 모으고, 가치함수로 각 행동의 좋고 나쁨을 매긴 뒤, 그 판정을 조건으로 VLA를 다시 학습하는 방식입니다. 그 결과로 나온 모델은 에스프레소를 13시간 연속으로 만들고, 처음 보는 가정에서 2시간 넘게 멈춤 없이 빨래를 개고, 실제 공장에서 포장 박스를 조립했습니다.

이 논문의 핵심 트릭은 diffusion을 다뤄본 사람에게 이미 익숙한 **classifier-free guidance**입니다. 이 글에서는 논문의 논지를 따라가면서, 각 설계 결정의 이유를 이해하는 데 필요한 RL 개념을 해당 지점마다 정리했습니다.

---

## 1. 문제 — 모방학습의 천장과 세 개의 벽

### 1.1 시연이 줄 수 없는 두 가지

**모방학습 정책은 오차 누적(compounding error)을 겪고, 잘해야 시연 데이터만큼만 잘합니다.** 시연으로는 얻을 수 없는 것이 두 가지 있습니다.

**⓵ 자기 실수의 교정** — 배포된 정책이 실제로 빠지는 실패 상태는 시연에 없습니다. 시연자는 자기가 조종할 때 나오는 궤적만 보여줄 수 있습니다.

**⓶ 시연자를 넘는 속도와 강건성** — teleoperation보다 빠르고 매끄러운 동작은 시연을 아무리 모아도 나오지 않습니다. 모방학습의 목표가 시연 분포 그 자체이기 때문입니다.

개념적으로 그리면 다음과 같습니다.

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

- 모방학습(SFT)은 데이터를 늘려도 시연 품질 아래에서 포화합니다
- 천장을 뚫으려면 정책이 **자기 경험**에서 배워야 하며, 이것이 RL입니다

다만 논문의 서론은 이 원리를 범용·확장 가능한 로봇 학습 시스템으로 구현하는 데 세 가지 난제가 있다고 명시하였습니다. 이 글에서는 이 셋을 벽으로 명명하고 끝까지 추적합니다.

### 1.2 세 개의 벽

**⛔ 벽 1 — 거대하고 표현력 높은 VLA에 RL 걸기**

**flow VLA 전체를 end-to-end로 RL한 사례는 없습니다.** π₀.₆의 행동은 860M 파라미터의 flow-matching action expert가 만듭니다. 대부분의 RL 알고리즘은 정책의 로그우도 $\log\pi_\theta(a\mid o)$나 미분 가능한 샘플링을 요구하는데, flow 모델은 이것을 싸게 내주지 않습니다. 논문의 related work는 기존 시도를 두 부류로 정리하였습니다. PPO를 VLA에 직접 적용한 계열은 실세계로 효율적으로 확장하기 어렵고, VLA 위에 RL을 얹는 계열(잔차 정책, action head 미세조정, 행동 선택·정제, diffusion 노이즈 공간 정책)은 대체로 이산 행동이나 단순 가우시안 분포를 씁니다.

**⛔ 벽 2 — 출처가 제각각인 데이터**

실제로 쌓이는 데이터는 다음과 같이 섞입니다.

```
  data source          behavior policy            label
  -------------------  -------------------------  --------------
  demos (pretrain)     many human teleoperators   success / fail
  demos (task l)       human teleoperator         success / fail
  autonomous, iter k   pi_l^(k-1)                 success / fail
  corrections          pi_l^(k-1) + human         success / fail
```

**데이터를 만든 정책(behavior policy)이 전부 다릅니다.** PPO나 REINFORCE 같은 on-policy 방법은 정책이 바뀔 때마다 과거 데이터를 버려야 합니다. 실물 로봇에서 에피소드 하나하나는 사람의 시간과 장비 가동을 먹는 비싼 자원이라, 버리는 것은 사치입니다.

**⛔ 벽 3 — 실세계의 보상**

시뮬레이터처럼 성공을 자동 판정해 주는 함수가 없습니다. 사람이 에피소드 결과를 판정해야 하고, 그 판정은 모호하거나 확률적일 수 있습니다. 게다가 현장에서 의미 있는 목표는 성공률만이 아닙니다. **얼마나 빨리** 해내는지, 즉 시간당 처리량(throughput)이 함께 올라가야 합니다.

### 1.3 그래서 논문이 던지는 질문

> 사람의 보상 피드백과 개입을 포함하는 **하나의 범용 RL 레시피**로, 배포 경험만 가지고 거대한 flow VLA의 강건성과 처리량을 **동시에** 끌어올릴 수 있는가? 그리고 그 보상 신호를 사후학습뿐 아니라 **사전학습 단계부터** 넣을 수 있는가?

세 벽이 무엇이고 RECAP이 각각을 어떻게 넘는지가 이 논문의 서사입니다.

---

## 2. 배경 — 세 벽을 이해하기 위한 최소한의 RL

RL에 익숙하다면 3절로 건너뛰어도 됩니다.

### 2.1 가치함수와 n-step advantage

이 논문은 감가율을 쓰지 않습니다($\gamma=1$, 유한 horizon). 관측 $o_t$는 Markov 상태로 간주합니다(로봇 RL의 흔한 단순화라고 논문이 각주로 밝혔습니다).

$$V^{\pi}(o_t)=\mathbb{E}\Big[\sum_{t'=t}^{T}r_{t'}\Big],\qquad A^{\pi}(o_t,a_t)=\mathbb{E}_{\rho_\pi}\Big[\sum_{t'=t}^{t+N-1}r_{t'}+V^{\pi}(o_{t+N})\Big]-V^{\pi}(o_t)$$

- $o_t$: 관측. 카메라 이미지와 로봇 관절 상태
- $r_t$: 스텝 $t$의 보상, $T$: 에피소드 마지막 스텝
- $V^\pi(o_t)$: $o_t$에서 시작해 정책 $\pi$를 따를 때의 기대 누적 보상
- $A^\pi(o_t,a_t)$: 행동 $a_t$가 "평균적인 $\pi$"보다 얼마나 나았는지를 나타냅니다. 처음 $N$스텝은 실제 보상을 더하고, 그 이후는 $V$로 대신합니다(n-step 추정)
- $\rho_\pi$: 정책과 환경 동역학이 만드는 궤적 분포

### 2.2 Policy extraction — 가치함수를 정책으로 바꾸는 세 갈래

**가치함수를 얻었다고 정책이 저절로 좋아지지는 않습니다.** "이 행동이 평균보다 낫다"는 판정을 정책의 파라미터 변화로 옮기는 단계가 필요하고, 이를 **policy extraction**이라 부릅니다. 방법은 크게 세 갈래입니다.

| 갈래 | 대표 | 정책에 요구하는 것 | 이 설정에서의 문제 |
|---|---|---|---|
| 정책경사 | PPO, REINFORCE, SAC(재매개변수화) | 로그우도 또는 미분 가능한 샘플링 | flow 모델은 로그우도가 비쌈 (벽 1). on-policy면 데이터를 버림 (벽 2) |
| 가중 회귀 | AWR, CRR, IQL식 추출 | advantage 가중치를 곱한 지도학습 | advantage 낮은 데이터를 사실상 버림 → 여과된 모방에 가까움 |
| 조건화 | UDRL, RCP, Decision Transformer, CFGRL | 조건부 지도학습 | RECAP이 택한 길 |

첫 갈래가 벽 1에 부딪히는 이유는 수식으로 보면 분명합니다. flow ODE $\dot z_s=v_\theta(z_s,s\mid o)$로 노이즈 $z_0$에서 행동 $a=z_1$을 만드는 모델의 정확한 로그우도는 순간 변수변환 공식으로 주어집니다.

$$\log\pi_\theta(a\mid o)=\log p_0(z_0)-\int_0^1\nabla_z\cdot v_\theta(z_s,s\mid o)\,ds$$

- $p_0$: 노이즈 prior $\mathcal N(0,I)$
- $v_\theta$: 학습된 속도장, $z_s$: 시간 $s$에서의 중간 상태
- $\nabla_z\cdot v_\theta$: 속도장의 발산(Jacobian의 trace)

샘플 하나의 로그우도를 얻으려면 ODE를 풀면서 Jacobian trace를 시간에 걸쳐 적분해야 합니다. 대형 VLA에서 PPO가 요구하는 확률비 $\pi_\theta/\pi_{\text{ref}}$를 매 샘플 정확히 계산하는 것은 사실상 불가능합니다. 이 때문에 flow 모델에 PPO를 쓰려는 연구들은 ELBO류 근사(DPPO, FPO)에 기대며, 이 논문의 PPO baseline도 그 방식입니다(5절에서 다시 다룹니다).

### 2.3 정규화 RL — "기준 정책보다 조금 낫게"의 닫힌 해

같은 데이터로 여러 번 gradient step을 밟으려면, 데이터를 만든 정책 $\pi_{\text{ref}}$ 근처에 머물도록 정규화하는 것이 표준입니다.

$$\mathcal J(\pi,\pi_{\text{ref}})=\mathbb E_{\tau\sim\rho_\pi}\Big[\sum_t r_t\Big]-\beta\,\mathbb E_{o}\Big[D\big(\pi(\cdot\mid o)\,\Vert\,\pi_{\text{ref}}(\cdot\mid o)\big)\Big]$$

- $\pi_{\text{ref}}$: 기준 정책. 보통 학습 데이터를 수집한 behavior policy
- $D$: 두 분포 사이의 발산, $\beta$: 정규화 강도(라그랑주 승수)

$D$가 KL이면 잘 알려진 닫힌 해가 나오며, AWR의 이론적 근거가 이것입니다.

$$\hat\pi(a\mid o)\ \propto\ \pi_{\text{ref}}(a\mid o)\,\exp\big(A^{\pi_{\text{ref}}}(o,a)/\beta\big)$$

RECAP이 기반으로 삼는 것은 이것과 가깝지만 덜 알려진 자매 결과입니다.

$$\hat\pi(a\mid o)\ \propto\ \pi_{\text{ref}}(a\mid o)\ p\big(I\mid A^{\pi_{\text{ref}}}(o,a)\big)^{\beta},\qquad p(I\mid A)=\frac{g\big(A^{\pi_{\text{ref}}}(o,a)\big)}{\int g\big(A^{\pi_{\text{ref}}}(o,a')\big)\,da'}$$

- $I$: "이 행동은 $\pi_{\text{ref}}$보다 개선이다"라는 사건
- $p(I\mid A)$: 행동 $a$가 개선일 확률
- $g$: 임의의 단조증가 함수
- $\beta$: 재가중의 날카로움

이렇게 정의한 $\hat\pi$는 다음을 보장합니다.

$$\mathcal J(\hat\pi)\ \ge\ \mathcal J(\pi_{\text{ref}})$$

직관은 단순합니다. $\pi_{\text{ref}}$의 행동 분포를 "개선일 확률"로 재가중하면 반드시 나아집니다. 다만 `exp(A/β)` 형태로도 충분해 보이는데 굳이 이 형태를 쓰는 이유는, 3.3절에서 Bayes 규칙 한 줄로 드러납니다.

---

## 3. 방법 — RECAP

### 3.0 전체 구조

RECAP은 세 서브루틴의 반복입니다.

- **⓵ 데이터 수집** — VLA를 태스크에 돌리고, 에피소드마다 성공/실패 라벨을 답니다. 필요하면 사람이 개입해 교정합니다
- **⓶ 가치함수 학습** — 지금까지 모은 모든 데이터로, 실패를 감지하고 완료까지 남은 시간을 판단하는 다중 태스크 가치함수를 학습합니다
- **⓷ advantage 조건부 학습** — 가치함수에서 얻은 advantage로 최적성 지표를 만들어 VLA 입력에 넣고 학습합니다

단계마다 바뀌는 것은 각 서브루틴에 넣는 데이터뿐입니다.

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

- **사전학습**은 수만 시간 분량의 다중 로봇·다중 태스크 시연에 대해 ⓶⓷만 수행합니다. 즉, 보상 신호가 사전학습부터 들어갑니다
- **사후학습**은 매 반복마다 직전 반복의 모델 대신 **사전학습 체크포인트에서** 다시 미세조정합니다. 여러 반복에 걸친 drift를 피하기 위해서입니다
- 논문에 따르면 반복 1회만으로도 크게 개선되는 경우가 많았습니다

이제 보상부터 시작해 각 부품을 차례로 살펴보겠습니다.

### 3.1 보상 — 성공 라벨 하나로 속도까지 담기 (벽 3)

사람이 할 일은 에피소드마다 성공/실패 라벨 하나를 다는 것뿐입니다. 논문은 이 라벨에서 보상을 다음과 같이 만듭니다.

$$r_t=-1\ \ (t<T),\qquad r_T=0\ \ \text{(success)},\qquad r_T=-C_{\text{fail}}\ \ \text{(failure)}$$

- $T$: 에피소드 마지막 스텝
- $C_{\text{fail}}$: 실패 에피소드의 가치를 충분히 낮추는 큰 상수

이 보상 아래에서 스텝 $t$ 이후의 return은 다음과 같습니다.

$$R_t=\sum_{t'=t}^{T}r_{t'}=-(T-t)\ \ \text{(success)},\qquad R_t=-(T-t)-C_{\text{fail}}\ \ \text{(failure)}$$

- $R_t$: 스텝 $t$부터 에피소드 끝까지의 실제 누적 보상

**성공 에피소드에서 return은 남은 스텝 수의 음수입니다.** 따라서 가치함수는 "성공까지 몇 스텝 남았나"를 예측하게 됩니다. 태스크마다 전형적인 길이가 크게 다르므로, 태스크별 최대 에피소드 길이로 나눠 $(-1,0)$ 구간으로 정규화합니다.

여기서 헷갈리는 점이 하나 있습니다. 사람이 주는 정보는 에피소드당 1비트로, LLM RL의 outcome reward처럼 희소합니다. 그런데 매 스텝 $-1$이 붙어 있으니, 이것이 희소 보상인지 조밀 보상인지가 분명하지 않습니다.

> ### 💡 정보원은 희소하고 신호는 조밀하다 — 시간 페널티가 처리량을 만든다
>
> **라벨링 비용은 희소하고, 학습 신호는 조밀합니다.** 사람은 에피소드당 성공 여부 하나만 줍니다. 반면 스텝당 $-1$은 사람이 줄 필요 없는 공짜 정보라서, 라벨 비용을 늘리지 않고 시간 축의 구배를 만듭니다.
>
> LLM 쪽으로 옮기면 "정답 여부 outcome reward + 길이 페널티"와 같은 모양입니다. GRPO의 순수 outcome reward에는 "더 빨리 풀어라"라는 압력이 없지만, 이 보상은 성공 여부와 속도를 하나의 스칼라로 묶습니다.
>
> | | 순수 outcome reward | RECAP의 보상 |
> |---|---|---|
> | 사람이 주는 정보 | 에피소드당 성공 여부 | 에피소드당 성공 여부 (동일) |
> | 가치함수가 배우는 것 | 성공 확률 | 성공까지 남은 시간 (실패는 큰 음수) |
> | 속도에 대한 압력 | 없음 | 있음 |
>
> 이 설계 때문에 RECAP의 개선은 성공률보다 **처리량**에서 더 크게 나타나며, 5절에서 이를 확인합니다.

### 3.2 분포형 가치함수 — 데이터 혼합 정책의 Monte Carlo critic (벽 2·3)

가치함수는 return을 스칼라로 회귀하지 않고, $B=201$개 bin 위의 분포로 분류합니다.

$$\min_\phi\ \mathbb E_{\tau\in\mathcal D}\Big[\sum_{o_t\in\tau}H\big(R^B_t(\tau),\ p_\phi(V\mid o_t,\ell)\big)\Big]$$

$$V^{\pi_{\text{ref}}}(o_t,\ell)=\sum_{b}p_\phi(V=b\mid o_t,\ell)\,v(b)$$

- $R^B_t(\tau)$: 궤적 $\tau$의 스텝 $t$ 이후 실제 return을 bin으로 이산화한 값
- $p_\phi(V\mid o_t,\ell)$: 관측과 언어 명령이 주어졌을 때 가치의 bin 분포
- $\ell$: 언어 입력. 태스크 프롬프트와 수행 방식을 조절하는 메타데이터
- $H$: cross-entropy, $v(b)$: bin $b$가 대표하는 가치값
- $\mathcal D$: 지금까지 모은 데이터 전체

가치함수는 VLA와 같은 아키텍처를 쓰되, Gemma 3에서 초기화한 670M 규모의 작은 VLM 백본을 씁니다. 과적합을 막기 위해 소량의 멀티모달 웹 데이터와 함께 공동학습합니다. 백본이 작기 때문에 VLA 학습 중에 on-the-fly로 advantage를 계산해도 추가 비용이 거의 없습니다.

**이 가치함수는 실수 탐지기이자 진척 속도계로 작동합니다.** 논문의 가치함수 시각화(Fig. 4)를 보면, 로봇 팔이 개어둔 셔츠를 구기는 순간 가치가 떨어지고 복구하는 구간에서 다시 오릅니다. 실패 에피소드에서는 물체를 넘어뜨리는 순간 가치가 급락합니다.

분포형으로 만든 이유를 논문은 Bellemare et al.(2017) 인용 외에 따로 설명하지 않았습니다. 태스크마다 스케일이 크게 다른 다중 태스크 회귀에서 cross-entropy 분류가 MSE보다 안정적이고, VLM의 토큰 분류 헤드와 자연스럽게 맞물린다는 점이 실용적인 이유로 보입니다(필자 평가).

다만 이 식에는 이상한 점이 있습니다. 논문 스스로 이 추정기를 "on-policy"라고 부르며, 고전적인 off-policy Q 추정기보다 덜 최적이지만 단순하고 신뢰할 만하다고 밝혔습니다. 반면 1.2절의 표에서 봤듯이 데이터는 사람 시연, 여러 세대의 정책, 사람 교정이 뒤섞인 전형적인 off-policy 데이터입니다. on-policy 추정기로 off-policy 데이터를 다룬다는 것은 언뜻 모순으로 보입니다.

> ### 💡 V가 추정하는 것은 "현재 정책"이 아니라 "데이터 혼합 정책"의 가치다
>
> **구분해야 할 것은 V가 어떤 정책의 가치를 추정하느냐입니다.** 데이터셋 $\mathcal D$의 실제 return에 Monte Carlo 회귀를 하면, 그 결과는 $\mathcal D$를 만든 정책들의 혼합 $\pi_{\text{ref}}$의 가치 $V^{\pi_{\text{ref}}}$가 됩니다. 여기서 "on-policy"는 **"데이터를 만든 정책 자신을 평가한다"**는 뜻이며, "새 데이터만 쓴다"는 뜻으로 쓰인 것이 아닙니다.
>
> 그러면 advantage $A^{\pi_{\text{ref}}}$의 의미도 정해집니다. **"이 데이터 속 평균적인 행동보다 나았나."** 2.3절의 보장 $\mathcal J(\hat\pi)\ge\mathcal J(\pi_{\text{ref}})$ 역시 정확히 이 혼합 정책 대비 **한 스텝의 개선**입니다.
>
> | | off-policy Q-learning | RECAP의 MC $V^{\pi_{\text{ref}}}$ |
> |---|---|---|
> | 추정 대상 | 개선된 정책의 Q (부트스트랩) | 데이터 혼합 정책의 V |
> | 한 번의 개선 폭 | 여러 스텝의 개선이 가능 | 혼합 정책 대비 한 스텝 |
> | 주된 위험 | OOD 행동 과대추정, 발산 | 거의 없음 (실제 return 회귀) |
>
> 한 번에 멀리 가지 못하는 대신, 반복이 그 폭을 메웁니다. $k$번째 정책의 롤아웃이 $\mathcal D$에 들어가면 $\pi_{\text{ref}}$ 자체가 한 단계 올라가고, 그 위에서 다시 한 스텝 개선합니다. 근사적인 policy iteration으로 읽는 것이 정확하다고 판단됩니다.

advantage는 단계별로 다르게 계산합니다.

**사후학습 — 50스텝 lookahead.** 같은 궤적에서 50스텝 뒤의 관측을 씁니다. 50스텝 안에 에피소드가 끝나지 않는다면(정규화 전 단위로) 다음과 같이 정리됩니다.

$$A(o_t,a_t)=\sum_{t'=t}^{t+49}r_{t'}+V(o_{t+50})-V(o_t)=V(o_{t+50})-\big(V(o_t)+50\big)$$

- $V(o_t)+50$: 데이터의 평균적 정책이라면 50스텝 뒤에 도달했을 가치
- 즉 advantage는 **"50스텝 동안 데이터 평균보다 더 전진했나"**를 잽니다. 속도가 advantage에 직접 들어가는 경로가 여기입니다

**사전학습 — 에피소드 끝까지($N=T$).** return에서 $V(o_t)$를 빼는 고분산 추정이지만, 가치함수 호출 한 번으로 on-the-fly 계산이 가능합니다. 논문은 대규모 다중 태스크 데이터에서 경험적으로 잘 작동했다고 보고하였습니다.

### 3.3 정책 추출 — advantage를 "조건"으로 넣는다 (벽 1·2)

2.3절에서 미뤄둔 질문에 답할 차례입니다. 개선 확률 $p(I\mid o,a)$에 Bayes 규칙을 적용합니다.

$$p(I\mid o,a)=\frac{\pi_{\text{ref}}(a\mid I,o)\,p(I\mid o)}{\pi_{\text{ref}}(a\mid o)}\quad\Longrightarrow\quad p(I\mid o,a)\ \propto_a\ \frac{\pi_{\text{ref}}(a\mid I,o)}{\pi_{\text{ref}}(a\mid o)}$$

- $\pi_{\text{ref}}(a\mid I,o)$: 데이터 중 "개선이었던 행동들"만의 조건부 분포
- $\pi_{\text{ref}}(a\mid o)$: 데이터 전체의 행동 분포
- $p(I\mid o)$: $a$와 무관하므로 정규화 상수에 흡수됩니다

이것을 2.3절의 식에 대입하고 언어 조건을 붙이면 논문의 Eq. 2가 됩니다.

$$\hat\pi(a\mid o,\ell)\ \propto\ \pi_{\text{ref}}(a\mid o,\ell)\left(\frac{\pi_{\text{ref}}(a\mid I,o,\ell)}{\pi_{\text{ref}}(a\mid o,\ell)}\right)^{\beta}$$

$\beta=1$이면 식이 극적으로 단순해집니다.

$$\beta=1\quad\Longrightarrow\quad\hat\pi(a\mid o,\ell)=\pi_{\text{ref}}(a\mid I,o,\ell)$$

**개선 확률을 명시적으로 계산할 필요가 사라지며, 이것이 `exp(A/β)` 대신 이 형태를 쓰는 이유입니다.** 조건부 분포 $\pi_{\text{ref}}(a\mid I,o,\ell)$와 무조건부 분포 $\pi_{\text{ref}}(a\mid o,\ell)$ 두 개를 모방학습으로 배우기만 하면, 개선된 정책이 그 안에 들어 있습니다.

개선 지표는 태스크별 임계값으로 이진화합니다.

$$I_t=\mathbf 1\big(A^{\pi_{\text{ref}}}(o_t,a_t,\ell)>\epsilon_\ell\big)$$

- $\epsilon_\ell$: 태스크 $\ell$의 개선 임계값

조건부와 무조건부를 함께 학습하고, 둘의 비율을 거듭제곱해 분포를 날카롭게 만듭니다. diffusion을 다뤄봤다면 이 구조가 낯설지 않을 것입니다.

> ### 💡 이것은 classifier-free guidance 그 자체다
>
> 부록에 따르면 $\beta>1$로 추론할 때 flow 샘플링이 따르는 score는 다음과 같습니다.
>
> $$\nabla_a\log\pi_\theta(a\mid o)+\beta\big(\nabla_a\log\pi_\theta(a\mid I,o)-\nabla_a\log\pi_\theta(a\mid o)\big)$$
>
> 이미지 생성의 CFG와 문자 그대로 같은 식입니다.
>
> $$\nabla_x\log p(x)+w\big(\nabla_x\log p(x\mid c)-\nabla_x\log p(x)\big)$$
>
> 여기서 $x$는 생성할 이미지, $c$는 조건(텍스트 프롬프트), $w$는 guidance scale입니다.
>
> | CFG (이미지 생성) | RECAP |
> |---|---|
> | 조건 $c$ (텍스트 프롬프트) | 개선 지표 $I$ |
> | 암묵적 classifier $p(c\mid x)$ | 개선 확률 $p(I\mid o,a)$ |
> | guidance scale $w$ | $\beta$ |
> | 학습 시 조건 dropout | 학습 시 $I$를 30% 확률로 생략 |
> | $w$ 과대 → 과포화, 아티팩트 | $\beta$ 과대 → 행동이 support 경계로 밀려 과격한 동작 |
>
> 이 대응이 CFGRL(Frans et al., 2025)의 핵심 명제, 즉 **diffusion guidance는 제어 가능한 정책 개선 연산자**라는 주장입니다. 프롬프트 충실도를 올리는 손잡이가 곧 행동 최적성을 올리는 손잡이가 되며, RECAP은 이 원리를 대형 VLA로 옮겼습니다.

다만 RECAP은 CFGRL과 한 가지를 다르게 합니다. **CFGRL이 $\epsilon=0$(advantage가 양수면 개선)으로 고정하고 추론 시 $\beta$를 조절했다면, RECAP은 학습 시 임계값 $\epsilon_\ell$로 날카로움을 조절하고 기본 추론은 $\beta=1$로 합니다.** 이유는 두 가지입니다.

- **⓵ 과격한 동작** — 높은 $\beta$는 행동 분포를 학습된 support의 모서리로 밉니다
- **⓶ 자기회귀 부분 미적용** — CFG는 flow 부분에만 작동하고, 모델의 자기회귀 출력(서브태스크 텍스트, 이산 행동 토큰)에는 영향을 주지 못합니다

필요할 때만 $\beta\in[1.5,\ 2.5]$ 정도의 온건한 guidance를 곁들입니다.

> ### ⚠️ 팩트체크 — 임계값 $\epsilon_\ell$의 기술이 본문과 부록에서 어긋난다
>
> 본문(V-D)은 $\epsilon_\ell$을 "태스크별로 가치함수가 예측한 **값**의 30% 백분위수"로 정한다고 썼습니다. 반면 부록(F)은 "시연 데이터의 약 30%가 **양의 advantage**를 갖도록" 정한다고 썼습니다. 30번째 백분위수를 임계값으로 두면 약 70%가 양성이 되므로 방향이 반대이고, 대상도 값과 advantage로 다릅니다.
>
> **부록이 더 구체적이고 방법의 취지("상위 일부만 positive")와도 맞으므로, 상위 약 30%가 positive라고 읽는 것이 타당합니다.** 부록에 따르면 사후학습에서는 약 40%가 양성이 되도록 잡고, 시연 기반 정책이 성공률은 높지만 느린 T셔츠·반바지 태스크에서는 약 10%만 양성이 되도록 올렸습니다. 즉 $\epsilon_\ell$은 사실상 태스크별로 조정하는 **분위수 하이퍼파라미터**입니다.

학습 손실은 두 항의 음의 로그우도입니다(Eq. 3).

$$\min_\theta\ \mathbb E_{\mathcal D_{\pi_{\text{ref}}}}\Big[-\log\pi_\theta(a_t\mid o_t,\ell)-\alpha\log\pi_\theta(a_t\mid I_t,o_t,\ell)\Big]$$

- 첫 항: 무조건부 모델. 데이터 혼합 정책 $\pi_{\text{ref}}$를 그대로 모방
- 둘째 항: 조건부 모델. $I_t$와 함께 모방
- $\alpha$: 두 항의 균형. 실제로는 $\alpha$를 튜닝하지 않고, $I_t$를 30% 확률로 입력에서 빼는 dropout으로 대체합니다. 한 모델이 두 분포를 모두 표현하게 됩니다
- $\mathcal D_{\pi_{\text{ref}}}$: 지금까지 모은 **모든** 데이터

**사람 교정 구간은 advantage와 무관하게 $I_t=\text{True}$로 강제합니다.** 전문가의 교정 행동은 항상 좋은 행동이라는 가정입니다.

다만 이 손실은 실패 궤적, 느린 시연, 교정 전의 실수까지 전부 로그우도로 학습하므로, 나쁜 행동까지 모방하게 되지 않느냐는 의문이 생깁니다.

> ### 💡 실패 데이터는 버려지지 않고 "대조군"이 된다
>
> 나쁜 행동은 $I=\text{negative}$ 라벨과 함께 학습됩니다. 모델은 나쁜 행동이 어떻게 생겼는지를 알게 되지만, 추론 때는 $I=\text{positive}$로만 샘플링하므로 그쪽 분포에서 행동을 뽑지 않습니다.
>
> | | 여과 BC | AWR | Advantage conditioning |
> |---|---|---|---|
> | 학습 손실 | 성공 궤적만 NLL | advantage 지수 가중 NLL | 전체 NLL + 입력에 $I$ |
> | 나쁜 데이터 | 버림 | 가중치 거의 0 | negative 라벨로 학습 |
> | 좋고 나쁨의 단위 | 궤적 | 상태-행동 | 상태-행동 |
> | 추론 | 그대로 샘플 | 그대로 샘플 | $I=\text{positive}$ (필요시 CFG) |
>
> 정밀하게 말하면, $\beta=1$일 때 RECAP이 목표로 하는 분포 $\pi_{\text{ref}}(a\mid I,o)$ 자체는 "양성 행동만의 분포"입니다. **목표 분포만 보면 AWR·여과 BC와 같은 가족이며, 차이는 학습 신호에 있습니다.** 음성 데이터도 공유 백본과 무조건부·음성 분기를 지도하므로 버려지지 않고, 조건부와 무조건부의 대비가 $\beta>1$ guidance의 재료가 됩니다. 논문이 AWR을 "데이터를 버리는 여과 모방"이라고 비판한 지점도 정확히 이 학습 신호의 차이입니다.

### 3.4 π₀.₆에 끼워 넣기 — flow 로그우도 없이 (벽 1)

RECAP의 기반 모델 π₀.₆은 π₀.₅를 개선한 VLA입니다. π₀.₅ 대비 다른 로봇 플랫폼의 사전학습 데이터를 보강하고, 기반 VLM을 Gemma 3 4B로, action expert를 860M 파라미터로 키웠습니다. 구조의 핵심은 세 가지입니다.

- **Knowledge Insulation (KI)** — 연속 행동(flow)과 이산 토큰(FAST로 토큰화한 행동 포함)을 함께 end-to-end로 학습하되, flow action expert의 gradient가 백본으로 흘러가지 않도록 stop-gradient를 겁니다
- **서브태스크 예측** — 모델이 다음 서브태스크를 텍스트 $\hat\ell$로 먼저 생성하고("커피잔 집기" 등), 행동은 그 뒤에 생성되어 $\hat\ell$에 조건부가 됩니다. 추론 시 서브태스크 예측은 행동 생성보다 낮은 빈도로 돕니다
- **출력** — 50Hz 관절 각도와 그리퍼 명령으로 이루어진 action chunk $a_{t:t+H}$

$\hat\ell$을 먼저 예측하므로 전체 로그우도는 세 항으로 분해됩니다.

$$\log\pi_\theta(a_{t:t+H},a^{\ell}_{t:t+H},\hat\ell\mid o_t,\ell)=\log\pi_\theta(\hat\ell\mid o_t,\ell)+\log\pi_\theta(a^{\ell}_{t:t+H}\mid o_t,\ell,\hat\ell)+\log\pi_\theta(a_{t:t+H}\mid o_t,\ell,\hat\ell)$$

- $\hat\ell$: 예측한 서브태스크 텍스트 (자기회귀)
- $a^{\ell}_{t:t+H}$: FAST 이산 행동 토큰 (자기회귀, KI의 보조 목적)
- $a_{t:t+H}$: 연속 action chunk (flow matching). action expert는 이산 토큰을 입력으로 받지 않으므로 둘은 독립적으로 예측됩니다

**π*₀.₆가 여기에 추가한 것은 텍스트 입력 하나입니다.** $I_t$가 참이면 `Advantage: positive`, 거짓이면 `Advantage: negative`를 넣습니다.

```
 [imgs][q][task + metadata][subtask l_hat][Advantage: +/-][FAST] --> action expert --> a_{t:t+H}
                                          ^
                                          I_t: after l_hat, before actions
```

- 지표가 서브태스크 $\hat\ell$ **뒤**, 행동 **앞**에 놓이므로 행동의 로그우도만 영향을 받습니다
- "무엇을 할지"(서브태스크)는 그대로 두고, "어떻게 할지"(행동)만 최적성 조건에 걸립니다

연속 행동의 정확한 로그우도는 여전히 계산할 수 없으므로, flow matching 손실을 로그우도의 하한으로 대신합니다(Eq. 4).

$$\log\pi_\theta(a_{t:t+H},a^{\ell}_{t:t+H}\mid I_t,o_t,\ell,\hat\ell)\ \ge\ \mathbb E_{\eta,\omega}\Big[\log p_\theta(a^{\ell}_{t:t+H}\mid I_t,o_t,\ell,\hat\ell)-\alpha_\eta\big\lVert\omega-a_{t:t+H}-f_\theta(a^{\eta,\omega}_{t:t+H},I_t,o_t,\ell,\hat\ell)\big\rVert^2\Big]$$

$$a^{\eta,\omega}_{t:t+H}=\eta\,a_{t:t+H}+(1-\eta)\,\omega,\qquad\omega\sim\mathcal N(0,I)$$

- $\eta\in[0,1]$: flow matching 시간, $\omega$: 가우시안 노이즈
- $a^{\eta,\omega}$: 노이즈를 섞은 행동
- $f_\theta$: action expert가 출력하는 속도
- $\alpha_\eta$: 노이즈 수준별 손실 가중치
- 논문은 flow matching과 diffusion의 대응, 그리고 diffusion 손실이 ELBO로 해석된다는 결과를 근거로 이 합을 전체 행동 로그우도의 대략적 하한으로 봅니다

결국 flow의 로그우도는 끝까지 계산하지 못합니다. 그럼에도 벽 1을 넘을 수 있는 이유는 정책 추출의 형태에 있습니다.

> ### 💡 RL 목적이 조건부 SFT 손실이 되는 순간 벽 1이 사라진다
>
> **RECAP의 정책 추출은 "조건부 최대우도"입니다.** 연속 행동에 대해서는 **평소 사전학습에 쓰던 flow matching 손실을 그대로** 쓰면 됩니다. 임의의 행동에서 로그우도를 평가할 일도, 확률비를 계산할 일도, trust region을 걸 일도 없습니다.
>
> | | PPO (논문의 baseline) | RECAP |
> |---|---|---|
> | 정책에 요구 | 현재/기준 정책의 확률비 | 조건부 생성 손실 |
> | flow 처리 | 1-step diffusion ELBO로 로그우도 근사 (FPO식) | 표준 flow matching 손실 |
> | 안정화 | SPO식 trust region 필요 | 불필요 |
> | 데이터 | 본질적으로 on-policy | 누적 데이터 전부 |
>
> 정규화는 3.3절의 $\hat\pi\propto\pi_{\text{ref}}\cdot(\text{ratio})^{\beta}$ 안에 들어 있습니다. 이 식은 데이터 분포의 **재가중**이라, $\hat\pi$의 support는 $\pi_{\text{ref}}$ 밖으로 나가지 못합니다. 즉 behavior regularization이 손실의 페널티 항으로 들어가는 대신 **목표 분포의 구조 안에** 내장되어 있습니다.
>
> 이 설계 덕분에 사전학습과 사후학습이 **완전히 같은 손실과 인프라**를 씁니다. "모든 단계에 보상을 넣는다"는 주장이 가능한 이유가 여기에 있습니다.

### 3.5 데이터 수집 — 사람 교정과 자율 경험의 분업

사후학습의 흐름은 다음과 같습니다.

- **⓵ 초기 SFT** — 태스크 시연으로 미세조정하되 $I_t=\text{True}$로 고정합니다. 논문은 이것이 약간 더 좋은 결과를 냈다고 밝혔습니다. 이 단계의 결과가 반복 0회차 정책 $\pi^0_\ell$입니다
- **⓶ 수집** — 일부 에피소드는 완전 자율로 돌리고, 일부는 원격조종자가 지켜보다가 필요할 때 개입합니다(**human-gated DAgger**). 교정 여부와 무관하게 에피소드 전체(자율 구간 + 교정 구간)를 데이터에 넣습니다
- **⓷ 재학습** — 누적 데이터 전체로 가치함수를 먼저 재학습하고, 갱신된 $I_t$로 정책을 재학습합니다

사람이 실시간으로 교정해 준다면 DAgger만으로 충분하고, 가치함수와 RL은 필요 없어 보일 수도 있습니다.

> ### 💡 교정은 큰 실수와 탐색을, RL은 미세한 품질을 맡는다
>
> **교정은 큰 실수를 고치고 탐색의 벽을 넘는 역할을 하며, DAgger 이론이 가정하는 "최적 지도"와는 다릅니다.** 논문은 교정만으로는 모든 문제를 고칠 수 없다고 밝혔습니다. 자율 실행 중의 개입은 흐름을 끊는 사건이고, 숙련된 조작자도 교정의 품질을 일관되게 유지하거나 전체 속도 같은 미묘한 측면을 개선할 수는 없습니다.
>
> | | 사람 교정 (human-gated DAgger) | 자율 경험 + 가치함수 |
> |---|---|---|
> | 잘 고치는 것 | 치명적 실수, 탐색이 막힌 상황 | 속도, 동작의 매끄러움, 미세 품질 |
> | 한계 | 흐름 단절, 품질 비일관, 미세 개선 불가 | 정책이 한 번도 안 해본 해법은 찾지 못함 |
> | RECAP에서의 처리 | $I_t=\text{True}$ 강제 | 가치함수로 advantage를 매겨 조건화 |
>
> 이 분업을 가장 잘 보여주는 것이 T셔츠·반바지 태스크입니다. 정책 속도가 전문가 데이터 수집자에 가까워지자 교정을 제공하는 것 자체가 어려워졌고, 이 태스크는 결국 교정 없이 자율 데이터만으로 개선했습니다. **정책이 사람 수준에 도달하면 사람 교정의 가치가 떨어지고, 그 너머는 RL만 갈 수 있다고 판단됩니다.**

---

## 4. 왜 작동하는가

1절에서 명명한 세 벽을 회수하면 다음과 같습니다.

| 벽 | RECAP의 장치 | 왜 뚫리는가 |
|---|---|---|
| **벽 1** 거대 flow VLA | advantage를 텍스트 조건으로 넣어 조건부 SFT | 로그우도·확률비·trust region이 필요 없음. 사전학습과 같은 손실과 인프라를 그대로 사용 (3.3절, 3.4절) |
| **벽 2** 이질적 데이터 | 혼합 정책의 MC 가치함수 + 전량 조건부 학습 + 교정은 양성 강제 | "누가 만든 데이터인가"를 묻지 않고 "그 데이터의 평균보다 나았나"만 물음. 실패 데이터도 대조군으로 사용 (3.2절, 3.3절) |
| **벽 3** 실세계 보상 | 성공 라벨 + 스텝당 $-1$ → time-to-go 가치함수, 다중 태스크 사전학습 | 라벨 비용은 에피소드당 1비트, 학습 신호는 스텝마다. 가치함수가 실수와 진척 속도를 포착함 (3.1절, 3.2절) |

여기에 세 가지가 더 받쳐줍니다.

**⓵ 이론적 뒷받침** — 한 번의 추출은 데이터 혼합 정책 대비 개선이 보장되고(2.3절), 반복하면 그 혼합 정책 자체가 올라갑니다(3.2절). 알고리즘 전체가 근사적 policy iteration의 구조를 갖습니다.

**⓶ RL 사전학습** — 수만 시간의 사람 시연도 품질이 제각각입니다. 사전학습 단계에서 시연마다 advantage를 매기면, 모델은 수많은 태스크에 걸쳐 "좋은 실행과 나쁜 실행의 차이"를 이미 배운 상태로 사후학습에 들어갑니다. 논문은 offline RL로 사전학습한 π*₀.₆에 고품질 SFT를 결합한 모델이 offline RL 사전학습 없는 표준 SFT보다 낫고, 이후 로봇 데이터 RL의 좋은 출발점이 된다고 보고하였습니다.

**⓷ 처리량 개선의 원천** — 시연 SFT는 시연 속도를 넘을 수 없습니다. 반면 50스텝 advantage는 "데이터 평균보다 빨리 전진한 구간"을 양성으로 고르므로, 반복할수록 평균 속도가 올라갑니다. T셔츠 태스크에서 양성 비율을 10%까지 좁힌 것은 이 효과를 의도적으로 키운 선택입니다.

이를 종합하면 RECAP에서 새로운 것이 무엇인지가 드러납니다.

> ### 📌 RECAP의 기여는 부품이 아니라 조합과 스케일이다
>
> **RECAP에서 새로운 것은 조합과, 그 조합이 작동함을 보인 스케일입니다.** 논문 스스로 인정하듯 RECAP의 개별 부품(MC 가치함수, advantage 조건화, CFG식 개선, human-gated DAgger)은 모두 선행 연구에 있습니다.
>
> - 사전학습부터 실세계 반복까지 **하나의 손실**로 관통합니다
> - flow 기반 대형 VLA **전체를 end-to-end로** RL합니다
> - 사람 시연, 자율 롤아웃, 사람 교정이라는 **이질적 데이터를 한 레시피**로 흡수합니다
> - 그 결과, 범용 RL 레시피가 실세계 배포 경험만으로 VLA의 강건성과 **처리량**을 크게 올릴 수 있음을 처음으로 보였습니다

---

## 5. 실험

### 5.1 설정과 태스크

반복 개선 실험은 정적 양팔 시스템에서 수행되었습니다. 평행 그리퍼가 달린 6-DoF 팔 두 개를 50Hz 관절 위치로 제어하고, 관측은 관절·그리퍼 상태와 카메라 3대(양팔 사이 베이스 카메라 1대, 손목 카메라 2대)의 이미지입니다. 사전학습 데이터는 여러 종류의 로봇에서 수집되었습니다.

| 태스크 | 내용 | 성공 기준 | 제한 |
|---|---|---|---|
| Laundry (T셔츠·반바지) | 바구니에서 꺼내 펴고 접어 우상단에 쌓기 | 한 벌을 접어 적재 | 200 s |
| Laundry (diverse) | 11종 의류로 학습. 지표는 가장 어려운 버튼셔츠로만 측정 | 접어 스택에 올리기 | 500 s |
| Laundry (실패 제거) | 고정 초기 배치의 주황색 T셔츠 한 벌 | 칼라가 위를 향하게 정확히 접기 (엄격) | 200 s |
| Cafe (더블 에스프레소) | 포터필터 → 그라인딩 → 탬핑 → 장착 → 컵 → 추출 → 서빙 | 치명적 실수 없이 전 단계 완료 | 200 s |
| Box assembly | 평판 골판지 → 조립 → 라벨 부착 → 크레이트 적재 (실제 공장) | 평판에서 적재까지 | 600 s |

지표는 두 가지입니다.

- **처리량(throughput)** — 시간당 성공 횟수. 성공률과 속도를 하나의 실용적 수치로 묶습니다
- **성공률** — 평가자가 여러 품질 항목을 판정하고 이를 합산해 만든 성공 라벨의 비율

### 5.2 비교 모델

이름이 비슷해 헷갈리기 쉬우므로 먼저 정리합니다.

| 이름 | 사전학습 | 태스크 적응 | 역할 |
|---|---|---|---|
| π₀.₅ | 지도학습 | — | 이전 세대, RL 없음 |
| π₀.₆ | 지도학습 | — | advantage 입력 없음 |
| π*₀.₆ (RL pretrained) | offline RL | — | 사전학습 단계의 RECAP |
| π*₀.₆ offline RL + SFT | offline RL | 태스크 시연 SFT ($I$=True 고정) | 반복 0회차, 즉 데이터 수집 정책 |
| **π*₀.₆ (ours)** | offline RL | + 자율 경험·교정으로 RECAP 반복 | 최종 모델 |
| AWR | 지도학습 (π₀.₆) | 같은 로봇 데이터, advantage 가중 회귀 | 추출법 비교 |
| PPO | 지도학습 (π₀.₆) | 같은 로봇 데이터, DPPO/FPO식 + SPO 제약 | 추출법 비교 (2.2절에서 예고) |

### 5.3 결과

정확한 막대값은 그림(Fig. 7–12)에만 있고 본문 텍스트에는 수치가 거의 나오지 않습니다. 아래는 본문이 문장으로 보고한 결과입니다.

| 질문 | 설정 | 보고된 결과 |
|---|---|---|
| **Q1** 전체 효과 | 5개 모델 × 전 태스크 (Fig. 7, 8) | diverse laundry와 espresso에서 처리량 2배 이상, 실패율 약 절반. diverse laundry를 제외한 모든 태스크에서 성공률 90% 이상 |
| **Q2** 반복 효과 | T셔츠: 자율 데이터만, 반복당 300 궤적 (로봇 4대) / Box: 반복당 자율 600 + 개입 360 (Fig. 9, 10) | T셔츠: 2회 반복으로 처리량 +50%, 1회차에 성공률 90% 돌파 / Box: 1회차에 처리량이 떨어졌다가 2회차에 2배, 접기·라벨 성공률 약 90% |
| **Q3** 추출법 비교 | T셔츠, 동일 데이터 (Fig. 11) | AWR·PPO 모두 offline RL + SFT 모델을 거의 넘지 못함. PPO는 trust region을 0.01로 작게 잡아 안정화했지만 성능이 낮고, AWR은 성공률은 괜찮으나 느려서 처리량이 낮음 |
| **Q4** 실패 제거 | 엄격 기준 (칼라 위), 반복당 600 궤적 × 2회 (Fig. 12) | 성공률 97%, 빠른 속도 |

**3.1절에서 예고한 대로, 개선은 성공률보다 처리량에서 두드러집니다.** 증거는 세 곳에서 나옵니다.

- 쉬운 T셔츠 태스크는 SFT 단계에서 이미 성공률이 최대치에 가깝지만, 최종 모델에서 처리량이 여전히 크게 오릅니다
- T셔츠 반복 실험에서 1회차가 성공률을 90% 위로 올린 뒤, 2회차는 주로 처리량을 올립니다
- AWR은 성공률은 괜찮지만 처리량이 낮습니다. 속도 개선을 끌어내는 데서 추출법의 차이가 드러납니다

Q1의 "2배"가 **어떤 기준선 대비**인지도 짚어둘 필요가 있습니다. 본문은 이 향상이 offline RL + SFT 모델에서 최종 모델로 가는 차이, 즉 **로봇에서 수집한 경험 데이터를 추가한 효과**라고 명시하였습니다. π₀.₆ 대비가 아니므로 사전학습 개선 효과와 섞이지 않은 순수한 "경험으로부터의 RL" 효과를 잰 것이며, 오히려 보수적인 비교라고 할 수 있겠습니다.

Box assembly의 실패는 대부분 제한 시간 초과이며, 특히 마지막 크레이트 적재 단계에서 실패가 남습니다.

정량 평가 밖에서도 실사용 수준의 시연을 보여주었습니다. 에스프레소를 13시간 연속으로 만들었고, 새 가정에서 처음 보는 빨래를 2시간 넘게 중단 없이 접었으며, 실제 공장에서 포장용 박스를 조립했습니다.

### 5.4 데이터 규모

부록 F가 밝힌 데이터 규모를 보면, 대부분 **1~2회 반복, 태스크당 수백 에피소드** 수준입니다.

| 태스크 | 반복 | 자율 에피소드 | 교정 에피소드 |
|---|---|---|---|
| T셔츠·반바지 | 2 | 반복당 300 | 0 |
| Diverse laundry | 명시 없음 | 450 | 287 |
| 실패 제거 | 2 | 약 1000 | 280 + 378 |
| Box assembly | 2 | 반복당 600 | 반복당 360 |
| Cafe | 1 | 414 | 429 |

이 표를 본문의 주장과 대조하면 한 군데가 어긋납니다.

> ### ⚠️ 팩트체크 — "개입 데이터 없이" 실패를 제거했다는 주장
>
> Q4의 결론에서 본문은 개입 데이터나 추가 시연 없이 순수하게 RL만으로 특정 실패 모드를 제거했다고 밝혔습니다. 반면 부록 F는 실패 제거 실험에서 자율 데이터와 교정 데이터를 **모두 수집했다**고 쓰고, 위 표처럼 교정 에피소드 수(280 + 378)까지 적었습니다. 교정 데이터를 수집만 하고 학습에 쓰지 않았을 가능성도 있지만, 원문은 이를 명시하지 않았습니다. 따라서 "순수 RL로 실패 제거"는 확인이 필요한 주장으로 판단됩니다.
>
> 같은 부록에서 Box의 반복당 600개를 본문은 "자율 시행", 부록은 "시연"으로 다르게 기술하였습니다(경미).

Q3 비교에는 설정상 짚을 점이 하나 더 있습니다.

> ### ⚠️ 팩트체크 — 추출법 비교(Q3)의 출발점이 다르다
>
> **Fig. 11의 격차에는 추출법의 차이와 사전학습의 차이가 섞여 있습니다.** AWR과 PPO가 **π₀.₆(지도학습 사전학습)**에서 출발했다면, RECAP 최종 모델은 **π*₀.₆(RL 사전학습)**에서 출발합니다. 비교도 가장 쉬운 T셔츠 태스크 하나에서만 이루어졌습니다. 논문은 "baseline이 RECAP이 모은 더 좋은 데이터를 쓰므로 오히려 유리하다"고 공정하게 밝혔지만, 이 교란 요인은 언급하지 않았습니다.

---

## 6. 위치잡기 — 이웃 연구들 속에서

RECAP은 여러 계보의 교차점에 있습니다.

| 계보 | 대표 연구 | RECAP과의 관계 |
|---|---|---|
| 개입형 모방학습 | DAgger, HG-DAgger, BC-Z, RaC | human-gated DAgger를 채택하되 자율 경험·RL과 결합 |
| 실세계 로봇 RL | QT-Opt, SERL, MT-Opt, RoboCat | 대형 VLA와 장기·정밀 조작으로 확장 |
| VLA + PPO 계열 | VLA-RL, SimpleVLA-RL, πRL | 실세계로 효율적으로 확장하기 어렵다고 비판 |
| VLA 위에 얹는 RL | 잔차 정책, ConRFT, PA-RL, V-GPS, DSRL, RLDG | 이산·단순 가우시안 행동. RECAP은 flow VLA 전체를 end-to-end로 학습 |
| 가치함수 기반 end-to-end VLA RL | CO-RFT, GRAPE, VLAC, Self-Improving EFM | 이산 행동 모델, on-policy 정책경사, 비교적 단순한 태스크 |
| 보상·가치 조건부 정책 | UDRL, RCP, Decision Transformer, RvS, CFGRL | 방법론의 직계 조상 |

가치함수 기반 end-to-end 계열을 조금 더 풀면 다음과 같습니다. CO-RFT는 grasping 시연에 Cal-QL을 적용하되 온라인 개선 단계가 없습니다. GRAPE는 VLA 롤아웃에 대한 사람 선호로 DPO를 겁니다. VLAC와 Self-Improving Embodied Foundation Models는 RECAP처럼 **완료까지 남은 시간** 형태의 가치함수를 쓰지만, 그 가치로 각각 PPO와 REINFORCE를 겁니다. 같은 time-to-go 가치함수에서 출발해 **정책경사로 가느냐, 조건화로 가느냐**가 갈림길입니다.

가장 가까운 사촌은 방법론의 직접적 뿌리인 CFGRL입니다. RECAP이 CFGRL에서 무엇을 가져오고 무엇을 바꿨는지는 다음과 같습니다.

> ### 🔗 CFGRL (Frans et al., 2025) — 이론은 그대로, 날카로움의 손잡이는 학습 시점으로
>
> RECAP의 개선 확률 정리(2.3절)와 CFG 대응(3.3절)은 CFGRL에서 왔습니다. 차이는 그 원리를 무엇에, 어떻게 적용했느냐입니다.
>
> | | CFGRL | RECAP |
> |---|---|---|
> | 개선 조건 | $\epsilon=0$ (advantage가 양수면 개선) | 태스크별 분위수 임계값 $\epsilon_\ell$ |
> | 날카로움 조절 | 추론 시 $\beta$ (guidance) | 학습 시 $\epsilon_\ell$, 추론은 $\beta=1$ 기본 |
> | 대상 모델 | diffusion 정책 | 자기회귀 + flow 혼합 대형 VLA |
> | 적용 범위 | 정책 개선 연산자로서의 guidance | 사전학습부터 실세계 반복까지, 사람 교정 통합 |
>
> RECAP이 손잡이를 추론 시점에서 학습 시점으로 옮긴 이유는 3.3절의 두 가지(과격한 동작, 자기회귀 부분 미적용)입니다. 대형 VLA가 자기회귀 출력과 flow 출력을 함께 갖는다는 구조적 사실이 설계를 바꾼 것으로 볼 수 있겠습니다.

참고로 이 블로그에서 다룬 [PLD](/notes/pld-self-improving-vla/)는 이 논문의 related work에서 "VLA 위에 잔차 정책을 RL로 학습하고 되증류하는" 계열로 인용되었습니다.

---

## 7. 한계

논문이 스스로 밝힌 것과, 읽으며 추가로 짚어둘 만한 것을 함께 정리합니다.

**논문이 인정한 한계**

- **완전 자율이 아닙니다** — 보상 라벨, 개입, 에피소드 리셋을 모두 사람이 담당합니다. 논문은 고수준 정책으로 리셋 절차를 추론하는 등, VLA가 자동화의 새 길을 열 수 있다고 전망하였습니다
- **탐색이 순진합니다** — 정책의 확률성과 사람 개입에 기대는 greedy 탐색입니다. 초기 모방 정책이 이미 합리적일 때는 괜찮지만 개선의 여지가 큽니다
- **반복 offline입니다** — 데이터를 배치로 모으고 재학습하는 반복이며, 정책과 가치함수가 실시간으로 갱신되는 concurrent online RL은 아닙니다. 논문은 이 선택이 편의를 위한 것이라고 밝혔습니다
- **MC 가치 추정** — off-policy Q 추정기보다 덜 최적이며, 확장은 향후 과제로 남겼습니다

**추가로 짚을 지점**

- **교정 = 항상 양성이라는 가정** — 논문 스스로 교정 품질이 일관되지 않다고 말하면서, 교정 행동은 무조건 $I_t=\text{True}$로 강제합니다. 둘 사이에 긴장이 있습니다
- **임계값의 태스크별 조정** — $\epsilon_\ell$을 30%, 40%, 10%로 태스크와 단계마다 다르게 잡습니다. "범용 레시피"라는 주장을 다소 약화시키는 부분으로 판단됩니다
- **평가 대상은 태스크별 specialist입니다** — 방법 요약은 최종 generalist를 처음부터 학습한다고 썼지만, 실험 결과는 태스크별로 미세조정한 모델입니다(문맥상 추론). 반복 실험은 단일 플랫폼이고, diverse laundry는 버튼셔츠 한 종으로만 측정했습니다
- **보상이 사람 판정입니다** — 여러 품질 항목을 합산한 평가자 라벨이 보상의 원천입니다. 서론이 "모호한 보상"을 난제로 꼽았지만, 라벨 노이즈가 결과에 미치는 영향은 측정하지 않았습니다
- **수치 보고 방식** — 핵심 결과가 막대그래프로만 제시되어, 인용할 때는 본문이 문장으로 확정한 수치(2배, 절반, 90%, 97% 등)에 기대야 합니다

---

## 8. 마치며 — 이 논문이 시사하는 것

**RECAP의 기여는 RL을 조건부 SFT로 환원하는 한 가지 형식을 찾아, 사전학습부터 실세계 배포까지 같은 손실로 관통시켰다는 점입니다.** 새로운 RL 알고리즘을 제시한 것은 아니지만, 이 형식 덕분에 RL을 위해 따로 짜야 할 기계장치(확률비, trust region, 전용 critic 업데이트 루프)가 거의 사라지고, 이미 돌아가는 대규모 SFT 파이프라인이 그대로 RL 파이프라인이 됩니다.

또한 이 논문은 LLM·diffusion 배경을 가진 사람에게 유난히 잘 읽힙니다.

- **CFG = 정책 개선 연산자.** diffusion에서 guidance scale을 올려 프롬프트 충실도를 높이는 것과, 여기서 $\beta$를 올려 행동의 최적성을 높이는 것은 같은 연산입니다. 과포화 아티팩트와 과격한 동작이라는 부작용까지 대응합니다.
- **텍스트 토큰 하나로 RL을.** advantage를 `Advantage: positive` 토큰으로 넣는 방식은 LLM의 reward-token 조건화(Quark, SteerLM, Decision Transformer 계열)와 같은 설계 철학입니다. 알고리즘보다 **인프라 재사용**이 더 큰 이점이라고 판단됩니다.
- **품질 인지 사전학습.** 수만 시간의 시연에 advantage를 매겨 사전학습하는 것은 LLM 사전학습 데이터에 품질 태그를 붙이는 발상과 닮았습니다. "모든 시연이 같은 품질은 아니다"라는 사실을 모델에게 명시적으로 알려주는 셈입니다.
- **처리량을 1급 지표로.** time-to-go 가치함수는 "성공했나"와 "얼마나 빨리"를 한 스칼라로 묶습니다. 실제 배포에서 의미 있는 것은 시간당 처리량이므로, 보상 설계와 평가 지표가 배포 목표에 정렬되어 있다는 점이 이 논문의 실용적 강점이라고 할 수 있겠습니다.
- **가치함수 = 런타임 진척 모니터.** 실수하는 순간 값이 떨어지는 가치함수는, agentic 시스템에서 실패를 감지하고 개입을 요청하는 트리거로도 쓸 수 있습니다.

정리하면, 사람이 연습으로 숙련에 이르듯 VLA도 배포 경험으로 시연자의 천장을 넘을 수 있으며, RECAP은 그 연습을 가장 단순한 형태, 즉 좋았던 행동에 "positive"라고 적어주는 방식으로 구현했습니다.

---

## 부록 — 용어 정리

| 용어 | 정의 |
|---|---|
| **RECAP** | RL with Experience and Corrections via Advantage-conditioned Policies. 데이터 수집 → 가치함수 학습 → advantage 조건부 학습의 반복 |
| **policy extraction** | 학습된 가치함수로부터 개선된 정책을 얻는 단계. 정책경사, 가중 회귀, 조건화의 세 갈래가 있음 |
| **advantage conditioning** | advantage를 이진 지표로 바꿔 정책 입력에 넣고, 전체 데이터로 조건부 지도학습하는 추출법 |
| **$\pi_{\text{ref}}$** | 데이터를 만든 정책들의 혼합. 사람 시연자와 과거 정책들이 섞여 있음 |
| **개선 지표 $I_t$** | advantage가 태스크별 임계값 $\epsilon_\ell$을 넘으면 참. 텍스트로 입력됨 |
| **$\epsilon_\ell$** | 개선 임계값. 실제로는 양성 비율(30%, 40%, 10% 등)을 정하는 분위수 |
| **$\beta$** | 추론 시 guidance 강도. 1이면 조건부 샘플링, 1보다 크면 CFG |
| **time-to-go 가치함수** | 성공까지 남은 스텝 수의 음수를 예측하는 가치함수. 실패는 큰 음수 |
| **human-gated DAgger** | 사람이 위험을 판단해 자율 실행에 개입하는 교정 방식 |
| **Knowledge Insulation (KI)** | 이산 토큰과 flow 행동을 함께 학습하되, action expert의 gradient가 백본으로 흐르지 않게 막는 학습법 |
| **throughput** | 시간당 성공 횟수. 성공률과 속도를 함께 반영하는 지표 |

**원문** — [arXiv:2511.14759](https://arxiv.org/abs/2511.14759) · **프로젝트 페이지** — [pi.website/blog/pistar06](https://pi.website/blog/pistar06)
