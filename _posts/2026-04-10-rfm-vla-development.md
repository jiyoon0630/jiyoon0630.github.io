---
layout: paper
lang: ko
ref: rfm-vla-development
kind: tech-review
title: "Robot Foundation Model 기술: VLA의 발전과 학습 방법론"
date: 2026-04-10 12:00:00 -0700
tags: [Robot-Foundation-Model, VLA, Diffusion-Policy, Imitation-Learning, Reinforcement-Learning, Tech-Review]
summary: "VLA가 Regression에서 출발해 Diffusion과 웹 규모 VLM이라는 두 갈래를 거쳐 하나로 합류하기까지의 흐름을 정리하였습니다. 이어서 VLA를 학습시키는 데이터와 Pre-training·Post-training 방법론을 다룹니다."
---

> **핵심 정리** — **현재 VLA의 최전선은 VLM의 시각·언어 지식을 유지하면서 연속 Action을 생성하는 구조입니다.** Regression으로 Action을 직접 예측하던 출발점(Era 1)의 두 한계에서, Diffusion으로 Multimodality를 확보하는 갈래(Era 2A)와 웹 규모 VLM의 지식을 활용하는 갈래(Era 2B)가 나뉘었고, 두 갈래는 Era 3에서 합류했습니다. 이 글에서는 그 발전 과정과 함께, VLA를 학습시키는 데이터와 Pre-training·Post-training 방법론을 대표 연구를 통해 정리했습니다.

---

## Section 1. VLA(Vision Language Action) 발전

### 1-1. [Era 1] Naive Regression으로 Action 생성

> ### 💡 한 줄 요약
>
> **"이미지를 보고 관절을 얼마나 움직일지 직접 예측하라"는 가장 단순한 접근. 최초의 출발점이었지만, 정밀한 동작과 대규모 학습에 한계가 있었습니다.**

로봇 조작(Robot Manipulation)은 전통적으로 “perception → state estimation → planning → control”이라는 파이프라인으로 이루어졌으며, 각 모듈을 사람이 별도로 설계하고 튜닝해야 했기 때문에 새로운 task마다 엔지니어링 비용이 크고, 모듈 간 오차가 누적되는 문제가 있었습니다. 딥러닝이 떠오르던 시기, 일부 Computer Science 과학자/공학자들은 이 지점에서 근본적인 질문을 던지기 시작했습니다.

> *카메라 이미지와 로봇의 센서 값을 받아서 로봇 관절 Action 값을 직접 출력하는 하나의 신경망을 end-to-end로 학습시키면 안 될까?*

[![Era 1 — Naive Regression으로 Action 생성](/assets/img/notes/rfm/era1.jpg)](/assets/img/notes/rfm/era1.jpg)
*[Era 1] Naive Regression으로 Action 생성 (클릭하면 원본 크기로 열립니다)*

#### 1-1-1. End-to-End Visuomotor (JMLR 2016)

2016년, 당시 UC Berkeley에서 Pieter Abbeel 교수, Trevor Darrell 교수의 연구실(BAIR)에서 Postdoc으로 일하던 Sergey Levine은 박사 과정생 Chelsea Finn과 함께 perception부터 control까지 end-to-end로 학습하는 알고리즘을 제안했습니다.

**카메라로 본 장면과 로봇 팔의 현재 자세 정보를 합쳐서, 다음에 각 관절을 얼마나 움직여야 하는지를 예측하는 최초의 이미지 기반 Policy 모델(CNN 기반)**을 만들었습니다.

> ### 💡 Policy 모델
>
> 로봇이 현재 보고 느끼는 것(카메라 이미지, 관절 센서)을 입력받아, 다음에 취할 행동(관절 움직임)을 출력하는 의사결정 모델

로봇의 Action은 이산적인 분류(class) 값이 아니라 연속적인 실수 값이기 때문에, 자연스럽게 Regression 방식으로 학습하게 됩니다.

즉, 사람이 시연한 정답 행동(Ground Truth)과 모델이 예측한 행동 사이의 오차를 MSE(Mean Squared Error) Loss로 줄여나가는 지도학습 방식입니다. 쉽게 말해, "정답 관절 토크가 5인데 모델이 3을 예측했으면 그 차이 2를 줄이도록 반복 학습"하는 것입니다.

단일 태스크, 단일 카메라 이미지만 입력받는다는 한계가 있었지만, 사람이 설계한 perception-planning-control 파이프라인 없이 이미지에서 로봇 행동을 직접 매핑할 수 있음을 최초로 실증했다는 점에서, 이후 모든 **Robot Foundation Model 연구의 출발점이 된 논문**입니다.

> ### 💡 Physical AI 스타트업 생태계의 핵심 인물: Sergey Levine, Chelsea Finn, Pieter Abbeel
>
> **Pieter Abbeel** 교수는 2017년 자신의 제자들과 함께 로봇 AI 스타트업 **Covariant**를 창업했습니다. 물류 창고에서 로봇 팔이 다양한 물체를 집고 분류할 수 있는 범용 AI(Covariant Brain)를 개발했으며, 2024년 Amazon이 Covariant의 기술을 라이선스하고 창업자들을 영입하면서 사실상 인수되었습니다.
>
> **Sergey Levine**(당시 Postdoc → 현 UC Berkeley 교수)과 **Chelsea Finn**(당시 박사 과정생 → 현 Stanford 교수)은 2024년 함께 **Physical Intelligence(π)**를 공동 창업했습니다. 로봇을 위한 범용 Foundation Model을 만드는 것을 목표로, 2024년 Jeff Bezos가 리드한 <span class="tex2jax_ignore">&#36;400M</span> Series A 투자를 유치하며 <span class="tex2jax_ignore">&#36;2B</span> 기업가치를 인정받았습니다. 이 회사에서 나온 $\pi_0, \pi_{0.5}$는 현재 Robot Foundation Model 연구의 최전선에 있습니다.
>
> 2016년 하나의 논문에서 함께 연구했던 사제 관계가, 오늘날 Physical AI 산업을 이끄는 창업자들로 이어진 셈입니다.

#### 1-1-2. BC-Z (PMLR 2022, Google)

Google은 Sergey Levine의 연구를 Multi-task로 확장하여, 이미지와 로봇 센서 값 외에 자연어 또는 사람 비디오를 추가 입력으로 받아 100개 이상의 Task를 하나의 모델에 학습했습니다. 그 결과, 학습에 포함되지 않았던 24개의 새로운 Task에 대해서도 평균 44%의 성공률로 일반화(zero-shot)를 달성했습니다.

"어떤 작업을 할지"를 알려주는 자연어 명령이나 사람 시범 영상을, 각각의 변환기를 통해 AI가 이해할 수 있는 하나의 압축된 숫자 묶음(Embedding)으로 요약한 뒤, 이 요약 정보를 이미지 처리 신경망(CNN)의 중간 단계에 끼워 넣어 주입합니다.

구체적으로는 단순히 정보를 이어 붙이는 것(concat)이 아니라, 작업 종류에 따라 모델이 이미지를 해석하는 방식 자체를 바꿔주는 기법(FiLM, Feature-wise Linear Modulation)을 적용했습니다. Task에 따라 모델이 이미지를 보는 방식 자체를 바꾸도록 학습하는 것입니다.

> ### 💡 FiLM(Feature-wise Linear Modulation)
>
> 입력된 Embedding 벡터를 이용해 모델 중간 feature를 변환하는 기법입니다.
>
> 구체적으로는 FiLM에서는 학습 가능한 파라미터 스케일 값 $\gamma$와 시프트 값 $\beta$를 사용합니다. 모델 중간 feature에 대해 $\gamma \times \text{feature} + \beta$로 변환하여 feature를 얼마나 참조할지를 결정하게 됩니다.
>
> 단순히 정보를 이어 붙이는 concat과 달리, 네트워크가 입력을 처리하는 내부 표현 자체를 조건에 따라 변조할 수 있어 더 효과적인 conditioning이 가능합니다.

**Task Conditioning 개념은 하나의 모델이 입력되는 자연어나 비디오에 따라 서로 다른 Task를 수행할 수 있게 해준다는 점에서, 이후 범용 Robot Foundation Model로 나아가는 핵심 설계 원칙**이 됩니다.

#### 1-1-3. Action Chunking Transformer (arXiv 2023)

앞선 연구들은 현재 시점의 카메라 이미지와 로봇 센서값을 관측하여 Action 하나만을 예측했기 때문에, 매 timestep마다 신경망 추론이 필요하여 추론 지연(latency)이 실시간 제어의 병목이 되고, prediction 오차가 매 스텝 누적(compounding error)되어 궤적이 점점 벗어나는 문제가 있었습니다.

Stanford의 Chelsea Finn 교수 연구실 박사 과정생이었던 Tony Z. Zhao는 이 문제를 해결하기 위해, 현재 시점에서 미래 $k$ 스텝의 Action을 한 번에 묶어서 예측하는 Action Chunking 개념을 제안했습니다.

한 번의 예측으로 여러 스텝을 커버하므로 모델이 의사결정을 내리는 횟수 자체가 줄어들어 오차 누적이 완화되고, 겹치는 Action Chunk를 앙상블(temporal ensemble)하면 시간적으로 부드러운(smooth) 궤적이 만들어집니다.

이 **Action Chunking은 이후 Diffusion Policy, $\pi_0$ 등 거의 모든 Robot Foundation Model 연구에서 표준적으로 채택되며, RFM 연구의 기본 단위**가 됩니다.

> ### 💡 Tony Zhao와 Sunday Robotics
>
> **Tony Zhao**는 ACT 연구를 통해 "충분한 데이터만 있으면 저비용 하드웨어로도 정교한 조작이 가능하다"는 확신을 얻었고, 로봇 연구가 실제 제품으로 전환될 수 있는 시점이 왔다고 판단하여 2024년 Stanford 박사 과정을 중퇴했습니다. 뒤에서 소개할 Diffusion Policy 논문의 저자 **Cheng Chi**와 함께 **Sunday Robotics**를 공동 창업했습니다.
>
> Sunday Robotics의 가정용 로봇 Memo는 500가구 이상의 실제 가정에서 Skill Capture Glove라는 웨어러블로 1,000만 건의 가사 데이터를 수집하여 학습하며, 2026년 Series B에서 <span class="tex2jax_ignore">&#36;165M</span>을 유치하며 기업가치 <span class="tex2jax_ignore">&#36;1.15B</span>를 인정받았습니다.

#### 1-1-4. [Era 1] 한계점

앞서 소개드린 세 논문은 모두 Regression 기반으로 로봇 Action을 예측합니다. 이 접근법에는 구조적으로 두 가지 한계가 있었습니다.

1. **Multimodal Action을 반영하지 못함**
    - 로봇이 Task를 수행하려고 할 때, 왼쪽으로 돌아가도 되고, 오른쪽으로 돌아가도 되는 **다양한 올바른 궤적이 존재하는 상황을 Multimodality라고 부릅니다.** Regression은 다양한 궤적을 올바르게 모델링하는 것이 아니라, 평균 오차를 줄이는 방식으로 학습되기 때문에 여러 궤적의 평균을 출력(mode averaging)하는 경우가 발생합니다.
    - ACT는 이를 해결하고자 했지만, 부분적으로 완화됐을 뿐 복잡한 multimodality 해결은 실패했습니다.
2. **Foundation Model을 만들기 위한 데이터 부족**
    - Robot Foundation Model을 학습하려면 로봇 Teleoperation으로 수집한 대규모 시연 데이터(관측-Action 쌍)가 필요하지만, Teleoperation 데이터를 수만~수십만 episode 이상으로 확장하는 것은 시간과 비용 면에서 현실적으로 어렵습니다.
    - 한편 웹에는 수십억 건의 비디오·이미지·텍스트 데이터가 존재하지만, 이 데이터에는 로봇 Action 값이 포함되어 있지 않아 Regression 학습의 Ground Truth로 활용할 수 없습니다. 세 논문의 모델 구조는 이러한 웹 데이터를 활용할 수 있는 파이프라인을 갖추고 있지 않습니다.

> ### 💡 Multimodality와 Mode averaging 문제
>
> 로봇이 테이블 위의 컵을 집으려 할 때, 왼쪽으로 돌아가도 되고 오른쪽으로 돌아가도 됩니다. 이처럼 하나의 상황에서 올바른 경로가 여러 개 존재하는 것을 Multimodality(다중 모드)라고 합니다.
>
> 문제는 Regression(회귀) 방식의 학습에서 발생합니다.
>
> Regression은 "정답과의 오차를 줄여라"는 방식으로 학습하기 때문에, 왼쪽 경로와 오른쪽 경로가 모두 정답인 데이터를 보면 두 경로의 평균, 즉 가운데로 직진하는 경로를 출력합니다. 가운데에는 장애물이 있으므로 충돌합니다.
>
> 이것이 **Mode Averaging(모드 평균화)** 문제입니다. 여러 정답 중 하나를 골라야 하는 상황에서, 어느 쪽도 고르지 못하고 어정쩡한 중간값을 내는 것입니다.
>
> 뒤에서 다룰 Diffusion Policy는 이 문제를 해결하기 위해 등장한 핵심 기법입니다.
>
> ![Mode averaging 문제](/assets/img/notes/rfm/mode-averaging.svg)
> *두 경로 모두 올바르지만, Regression은 그 평균을 출력해 장애물에 부딪힙니다*

이 두 가지 한계가 각각 “[Era 2A] Multimodality 확보”, “[Era 2B] Web-scale로 학습된 VLM(Vision Language Model)의 구조 및 지식 활용”로 분기하는 동기가 됩니다.

---

### 1-2. [Era 2A] Multimodality 확보

> ### 💡 한 줄 요약
>
> **같은 상황에서 올바른 경로가 여러 개일 때 어정쩡한 중간값을 내는 문제를, 이미지 생성 AI 기술(Diffusion)을 로봇에 도입하여 해결했습니다. 현재 거의 모든 로봇 AI의 동작 생성 표준이 된 핵심 전환점입니다.**

Era 1의 Regression 방식은 같은 관측에서 여러 올바른 궤적이 존재할 때 그 평균을 출력하는 mode averaging 문제를 안고 있었습니다. Multimodality(로봇 Task 수행 시 다양한 올바른 궤적이 존재함) 확보를 위한 시도는 크게 두 갈래로 나뉩니다.

[![Era 2A — Multimodality 확보](/assets/img/notes/rfm/era2a.jpg)](/assets/img/notes/rfm/era2a.jpg)
*[Era 2A] Multimodality 확보 (클릭하면 원본 크기로 열립니다)*

#### 1-2-1. [Era 2A①] 로봇 Action을 코드북으로 정의해서 명시적으로 Mode 분리

##### 1-2-1-1. Behavior Transformer (NeurIPS 2022)

NYU의 Lerrel Pinto 교수 연구실의 Nur Muhammad Shafiullah는 Multimodality 문제를 "이산 mode 선택 + 연속 보정"이라는 2단계 구조로 해결하고자 했습니다.

먼저 학습 데이터의 Action 궤적을 클러스터링(K-means)하여 $K$개의 대표 Action(코드북)을 사전에 구축합니다.

추론 시에는 (1) Transformer가 K개 중 하나의 mode를 이산적으로 선택하고, (2) 선택된 mode의 중심으로부터의 잔차(offset)를 연속 값으로 예측하여 최종 Action을 생성합니다.

여기서 주목할 부분은 $K$개 중 하나의 mode를 이산적으로 선택한다는 점입니다. Regression은 모든 mode의 평균을 출력하지만, 이산 분류는 하나의 mode를 명시적으로 선택합니다. 왼쪽 궤적과 오른쪽 궤적이 각각 별도의 bin으로 존재하므로, 어정쩡한 중간값이 나오지 않습니다.

이 접근은 Multimodality를 명시적으로 해결했지만, 코드북을 데이터에 맞춰 사전에 구축해야 한다는 한계가 있었습니다. $K$ 값 선택이 어렵고, 데이터가 바뀌면 코드북을 다시 만들어야 하며, 고차원 Action space(행동 공간)에서는 클러스터링 품질이 떨어집니다.

> ### 💡 Action Space (행동 공간)
>
> 로봇이 취할 수 있는 모든 행동의 범위를 수학적으로 정의한 것입니다.
>
> 예를 들어 7축 로봇 팔이라면, 7개 관절 각각의 회전량(또는 토크)을 하나의 숫자로 표현하므로 Action space는 7차원 연속 공간이 됩니다.
>
> 게임 캐릭터의 Action space가 {상, 하, 좌, 우}처럼 이산적이고 유한한 것과 달리, 로봇의 Action space는 각 관절이 연속적인 실수 값을 가지므로 차원이 높고 연속적입니다.
>
> 휴머노이드처럼 관절 수가 많아지면 30차원 이상으로 확장되며, 이 고차원 연속 공간에서 정확한 Action을 생성하는 것이 Robot Foundation Model 연구의 핵심 난제입니다.

이후, 같은 연구실에서 클러스터링 방식(K-means)을 학습 가능한 모델(VQ-VAE)로 대체하여 코드북을 end-to-end로 학습하는 VQ-BeT(ICML 2024)를 발표하며 이 한계를 개선했으나, 이산 코드북 방식은 본질적으로 Action space(행동 공간)를 유한한 코드 수로 나누기 때문에 연속적이고 정밀한 궤적 표현에 한계가 있기 때문에 후속 연구에서 주류로 자리잡지는 못했습니다.

> ### 💡 Lerrel Pinto와 ARI
>
> **Lerrel Pinto**는 UC Berkeley에서 Pieter Abbeel과 함께 Postdoc로 일한 뒤, 현재는 NYU 조교수로 재직하며 로봇 학습 연구실(GRAIL)을 이끌고 있습니다. 연구실에서 진행한 BeT, VQ-BeT, Robot Utility Models 등의 연구를 기반으로, **Assured Robot Intelligence(ARI)**를 창업했습니다.
>
> ARI는 노동력 부족이 심각한 산업/엔터프라이즈 현장에 범용 휴머노이드를 배치하는 것을 목표로 하며, 현재 스텔스 모드로 운영하고 있습니다.

#### 1-2-2. [Era 2A②] 이미지 생성 분야 Diffusion을 도입해서 Multimodality 분포 모델링

##### 1-2-2-1. Diffusion Policy (RSS 2023)

Columbia University의 Shuran Song 교수 연구실 박사 과정생이었던 Cheng Chi는 완전히 다른 접근을 취했습니다. 이미지 생성 분야에서 multimodal distribution 모델링에 성공한 Diffusion 기법(DDPM)을 로봇 Action 생성에 도입한 것입니다.

핵심 아이디어는 로봇 Action에 노이즈를 점진적으로 추가하는 forward process와, 노이즈로부터 원본 Action을 복원하는 reverse(denoising) process를 학습하는 것입니다. 추론 시에는 가우시안 분포에서 추출된 순수 랜덤 노이즈 $a_T\sim \mathcal{N}(0, I)$에서 출발하여 T번의 denoising step을 거쳐 최종 Action chunk를 생성합니다.

> ### 💡 Diffusion(확산 모델)
>
> 선명한 사진 위에 TV 노이즈(지직거림)를 조금씩 덮어서 결국 아무것도 안 보이는 모래폭풍 화면으로 만드는 과정(Forward process)과, 반대로 모래폭풍 화면에서 노이즈를 한 겹씩 벗겨내어 원래 사진을 복원하는 과정(Reverse process)을 AI에게 반복 학습시키는 생성 모델 기법입니다.
>
> 학습이 끝나면, AI는 아무런 의미 없는 모래폭풍에서 출발해서도 그럴듯한 새로운 사진을 만들어낼 수 있게 됩니다.
>
> 2020년 DDPM(Denoising Diffusion Probabilistic Models)으로 정립되었으며, Stable Diffusion, DALL-E 등 이미지 생성 서비스의 핵심 기술입니다.
>
> Diffusion Policy는 이 원리를 그대로 적용하되, 사진 대신 "로봇의 향후 동작 시퀀스(Action chunk)"를 생성합니다. **즉, 랜덤 노이즈에서 출발하여 노이즈를 단계적으로 제거하면, 현재 장면에 맞는 정교한 로봇 동작이 만들어지는 것**입니다.

현재 이미지와 센서 관측 값은 denoising 과정에 FiLM 방식(설명: → 1-1-2)으로 네트워크에 주입되어, 모델이 “어떤 상황에서, 노이즈 제거의 어느 단계에 있는지”를 동시에 반영하며 Action을 생성하게 됩니다.

> ### 💡 왜 Diffusion은 mode averaging이 없을까?
>
> Regression은 "정답 하나를 내놓아라"는 구조이기 때문에, 여러 정답이 있으면 그 평균으로 수렴합니다.
>
> 반면 Diffusion은 매번 랜덤한 노이즈에서 출발합니다. 출발점이 다르면 도착지도 달라집니다. 왼쪽에 가까운 노이즈에서 출발하면 왼쪽 경로로, 오른쪽에 가까운 노이즈에서 출발하면 오른쪽 경로로 자연스럽게 끌려갑니다.
>
> 즉, Diffusion은 여러 정답의 평균을 내는 것이 아니라, 매번 여러 정답 중 하나를 선택하는 구조이므로 mode averaging이 발생하지 않습니다.

ACT에서는 Action Chunking을 위해 Conditional VAE(CVAE) encoder로 입력값을 압축하고 Transformer decoder로 미래 스텝까지를 autoregressive하게 풀어내는 복잡한 설계가 필요했습니다.

반면 Diffusion은 "노이즈 → 원본 복원"이라는 프로세스 자체가 출력의 형태에 무관하기 때문에 단일 Action $a_t$을 생성하든, Action Chunk $[a_t, ..., a_{t+k}]$를 생성하든 추가적인 아키텍처 변경 없이 출력의 shape만 바꾸면 자연스럽게 Action Chunking이 되는 구조입니다.

Diffusion Policy의 한계는 추론 비용입니다. 하나의 Action chunk를 생성하기 위해 T번의 denoising step이 필요하므로, 실시간 제어에서 latency가 발생합니다. 이 문제는 이후 Flow Matching, Consistency Model 등 빠른 샘플링 기법으로 개선되어 갑니다.

**Diffusion Policy는 Multimodality 해결, Action chunking 통합, 안정적 학습이라는 세 가지를 동시에 달성하며 현재 Robot Foundation Model의 Action 생성 표준**으로 자리 잡았습니다. 이후 $\pi_0$, GR00T N1, Helix 등 주요 RFM 업체 모델들이 모두 Diffusion 기반(또는 그 변형인 Flow Matching) Action 생성을 채택하고 있습니다.

> ### 💡 Cheng Chi
>
> **Cheng Chi**는 앞서 소개한 ACT의 저자인 **Tony Zhao**와 함께 **Sunday Robotics**를 공동 창업한 인물입니다. 현재 Sunday Robotics에서 CTO를 맡고 있습니다.

##### 1-2-2-2. RDT-1B (ICLR 2025)

Diffusion Policy가 단일 로봇, 단일 테스크에서 Multimodality 해결을 실증했다면, RDT-1B는 이를 대규모 cross embodiment 로봇(서로 다른 체형의 로봇들을 하나의 모델로 통합 학습하는 것)을 위한 Foundation Model로 확장한 연구입니다.

Tsinghua University의 Huazhe Xu가 이끌었으며, 핵심 변경은 Diffusion Policy의 CNN을 확장성이 좋은 Diffusion Transformer(DiT)로 교체한 것입니다.

> ### 💡 DiT(Diffusion Transformer)
>
> 이미지 생성 분야에서 제안된 구조로, Stable Diffusion 3, SORA 등 최신 이미지/영상 생성 모델의 backbone입니다.
>
> LLM/VLM에서 사용하는 Autoregressive Transformer가 토큰을 한 개씩 순차 생성하는 반면, DiT는 노이즈가 섞인 전체 데이터를 입력받아 한 번에 병렬로 denoising합니다.
>
> 이러한 특성 덕분에 연속적이고 시간적으로 상관된 로봇 Action chunk를 생성하는 데 구조적으로 적합하여, CogACT, GR00T N1 등 최신 Robot Foundation Model의 Action Expert로 채택되고 있습니다.

서로 다른 로봇은 관절 수와 Action 차원이 다르기 때문에, 하나의 모델로 학습하려면 이질적인 Action space(행동 공간)를 통합해야 합니다.

RDT-1B는 모든 로봇의 Action을 하나의 통일된 공간으로 정렬한 뒤, 사용하지 않는 차원은 마스킹 처리합니다. 이렇게 하면 서로 다른 로봇의 데이터가 하나의 Diffusion Transformer에서 함께 학습될 수 있습니다.

> ### 💡 Cross-embodiment의 핵심 메커니즘 “Loss Masking”
>
> 서로 다른 로봇은 관절 수가 다릅니다. 이들을 하나의 모델로 함께 학습시키려면, 먼저 가장 관절이 많은 로봇에 맞춰 통일된 답안지를 만듭니다.
>
> ```
> 통일된 Action 공간 (최대 차원 = 14라고 가정)
> __ = 빈 차원 (해당 로봇에 존재하지 않는 관절)
>
> 7축 로봇 팔 (Franka):   [a1, a2, a3, a4, a5, a6, a7, __, __, __, __, __, __, __]
> 12축 양팔 로봇 (ALOHA): [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, __, __]
> 14축 휴머노이드:         [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14]
> ```
>
> 관절이 7개뿐인 Franka 로봇은 나머지 7칸이 비게 됩니다.
>
> 이때 비어있는 칸을 "채점하지 않음" 처리하는 것이 Loss Masking입니다.
>
> 시험에 비유하면, 14문제짜리 시험지를 모든 학생에게 동일하게 나눠주되, 7축 로봇은 1~7번만 채점하고 8~14번은 "해당 없음"으로 건너뛰는 것입니다.
>
> 다만 시험을 푸는 과정에서는 전체 14문제의 맥락을 함께 읽게 되므로, Franka에서 학습한 "물체를 집는 팔 움직임 패턴"이 다른 로봇의 팔 관절에도 간접적으로 전이될 수 있습니다.

#### 1-2-3. [Era 2A] 한계점

Era 2A, 특히 Diffusion을 기반으로 하는 연구들은 mode averaging을 구조적으로 해결했으며, Cross-embodiment로의 확장 가능성을 실증했지만, 여전히 로봇 Teleoperation 데이터에 의존하고 있어 데이터 확장성 문제는 미해결로 남아 있었습니다.

이 한계를 해결하기 위해, 웹 규모 데이터로 사전 학습된 VLM의 구조와 지식을 직접 활용하는 접근이 등장합니다.

---

### 1-3. [Era 2B] Web-scale로 학습된 VLM(Vision Language Model)의 구조 및 지식 활용

> ### 💡 한 줄 요약
>
> **ChatGPT 같은 대규모 AI가 인터넷에서 배운 시각·언어 지식을 로봇에 물려주는 접근. "피곤한 사람에게 좋은 음료를 집어줘"라고 하면 에너지 드링크를 고를 수 있게 되었지만, 동작의 정밀도를 희생하는 한계가 있었습니다.**

Era 2A가 "Action을 어떻게 잘 생성할 것인가"에 집중했다면, Era 2B는 완전히 다른 질문에서 출발합니다.

> *웹 규모 데이터로 사전 학습된 VLM(Vision Language Model)의 언어 이해, 시각 추론, 상식 지식을 로봇 제어에 어떻게 전이할 것인가?*

VLM은 본질적으로 Autoregressive/Next-token prediction 방식으로 이산 토큰을 순차 생성하는 모델입니다. 따라서 연속적인 로봇 Action을 VLM 파이프라인에 태우려면, Action을 이산 토큰으로 변환하는 것이 가장 자연스러운 접근이 됩니다.

> ### 💡 Autoregressive/Next-token Prediction
>
> LLM/VLM이 텍스트를 생성하는 핵심 동작 방식입니다. LLM/VLM은 입력을 한꺼번에 처리해서 결과를 한 번에 내놓는 것이 아니라, 출력을 토큰 단위로 하나씩 순차 생성합니다.
>
> 구체적으로, 입력된 텍스트는 어휘 사전을 통해 토큰 시퀀스로, 이미지는 Vision Encoder를 통해 시각 토큰 시퀀스로 변환됩니다. 그러면 모델은 이 토큰 시퀀스를 보고 "다음에 올 토큰이 무엇인지"를 예측합니다.
>
> 예를 들어 ["오늘", "날씨", "가"]라는 토큰이 입력되면 다음 토큰 "좋다"를 예측하고, 이를 다시 입력에 붙여 ["오늘", "날씨", "가", "좋다"] → 다음 토큰 예측 → ... 을 반복합니다.
>
> 이처럼 자기 자신의 이전 출력을 입력으로 되먹이며 순차 생성하는 방식을 Autoregressive(자기회귀)라고 부르며, ChatGPT가 글자를 한 글자씩 타이핑하듯 생성하는 것이 바로 이 과정입니다.

[![Era 2B — 웹 규모 VLM의 구조와 지식 활용](/assets/img/notes/rfm/era2b.jpg)](/assets/img/notes/rfm/era2b.jpg)
*[Era 2B] Web-scale로 학습된 VLM의 구조 및 지식 활용 (클릭하면 원본 크기로 열립니다)*

#### 1-3-1. Robotics Transformers 2 (PMLR 2023, Google DeepMind)

Google DeepMind는 사전 학습된 VLM(PaLI-X, PaLM-E)을 로봇 제어에 직접 활용하는 핵심 아이디어를 제안했습니다. 로봇 Action을 텍스트 토큰으로 표현하여 VLM의 언어 출력과 동일한 형식으로 학습시키는 것입니다.

구체적으로, RT-2는 7-DoF 로봇 Action(end-effector 위치/회전 변화 + gripper 압력)의 각 차원을 0~255 사이의 정수로 양자화한 뒤, VLM 어휘 사전에서 가장 사용 빈도가 낮은 256개의 토큰을 Action 전용으로 재활용하여 매핑합니다.

예를 들어 하나의 Action이  $[1, 128, 91, 241, 5, 101, 127, 217]$이라는 토큰 시퀀스가 됩니다. VLM 입장에서는 텍스트를 생성하는 것과 로봇 Action을 생성하는 것이 동일한 next-token prediction이 되므로, 모델 구조를 전혀 변경하지 않고, 로봇 데이터를 웹에서 수집된 이미지-텍스트 데이터와 함께 학습(co-fine-tuning)할 수 있습니다.

> ### 💡 VLM 단어사전(Vocabulary)
>
> LLM/VLM은 텍스트를 처리할 때 문자 그대로가 아니라, 미리 정의된 어휘 사전에 등록된 토큰(token) 단위로 변환합니다.
>
> 예를 들어 "로봇이 컵을 집는다"라는 문장은 ["로봇", "이", "컵", "을", "집", "는다"]와 같은 토큰 시퀀스가 되고, 각 토큰은 어휘 사전에서 고유한 정수 번호(예: 4521, 12, 8837, ...)를 가집니다. VLM은 이 정수 시퀀스를 입력받아 다음 토큰의 정수 번호를 예측하는 방식(next-token prediction)으로 작동합니다.
>
> 어휘 사전의 크기는 모델마다 다르지만 보통 수만 개의 토큰으로 구성됩니다.

> ### 💡 Co-fine-tuning
>
> 구글의 RT-2는 로봇 데이터만으로 fine-tuning하면 VLM의 원래 지식이 catastrophic forgetting으로 사라지는 문제를 발견했습니다.
>
> 이를 방지하기 위해 로봇 데이터와 웹스케일 언어-이미지 데이터를 함께 학습(co-fine-tuning)하여 원래의 시각-언어 능력을 유지하면서 로봇 제어 능력을 추가했습니다.

**RT-2는 이를 통해 VLA(Vision-Language-Action) 모델이라는 새로운 패러다임을 확립했습니다.** 웹 데이터에서 학습한 시각-언어 지식이 로봇 제어로 전이되어, 학습에 없던 물체를 인식하거나("피곤한 사람에게 좋은 음료를 집어줘" → 에너지 드링크 선택), chain-of-thought 추론으로 다단계 작업을 수행하는 reasoning capability를 보여주었습니다.

> ### 💡 왜 Action을 이산 토큰으로 바꿔야 하는가?
>
> VLM 파이프라인은 softmax → cross-entropy loss → sampling이라는 이산 선택 구조를 전제합니다.
>
> 연속 값을 그대로 넣으려면 모델 구조 자체를 바꿔야 하지만, Action을 이산 토큰으로 변환하면 VLM의 기존 구조를 그대로 활용할 수 있습니다.
>
> 즉, 이산 토큰화는 VLM 활용을 위한 최소 변경 해법이었습니다.

> ### 💡 Reasoning Capability
>
> RT-2는 로봇 데이터에서 직접 학습하지 않은 추론 능력을 VLM으로부터 물려받았습니다.
>
> 예를 들어, "피곤한 사람에게 좋은 음료를 집어줘"라는 명령을 받으면, 로봇 학습 데이터에는 "피곤함 → 에너지 드링크"라는 연결이 없음에도 VLM이 웹에서 학습한 상식을 활용하여 에너지 드링크를 선택합니다.
>
> 또한 "망치 대용으로 쓸 수 있는 물체를 집어줘"라는 명령에 돌을 선택하는 등, 물체의 물리적 특성에 대한 상식 추론도 가능합니다.
>
> 이는 로봇 데이터만으로는 절대 학습할 수 없는 능력으로, VLM의 웹 규모 지식이 로봇 제어로 전이된 핵심 증거입니다.

#### 1-3-2. OpenVLA (arXiv 2024)

RT-2의 패러다임은 강력했지만, 모델과 데이터가 비공개였기 때문에 연구 커뮤니티가 재현하거나 확장하기 어려웠습니다.

Stanford Univ Chelsea Finn 교수, Percy Liang 교수의 지도를 받는 박사 과정생 Moo Jin Kim과 Physical Intelligence 핵심 Engineer인 Karl Pertsch는 이를 오픈소스로 재현하여 연구 커뮤니티의 베이스라인이자 Physical Intelligence $\pi_0$의 기술적 시초 역할을 했습니다.

OpenVLA는 Vision encoder(DINOv2, SigLIP)를 활용하여 공간적 특징과 의미적 이해를 동시에 강화하고, Llama-2 7B를 언어 backbone으로 사용하여 이산화된 Action 토큰을 next-token prediction 방식으로 생성합니다.

Open X-Embodiment 데이터셋(21개 기관, 22종 로봇, 100만개 이상 에피소드)으로 학습되어 로봇 cross-embodiment 일반화를 지원합니다.

#### 1-3-3. [Era 2B] 한계점

VLM의 지식을 로봇에 전이하는 데 성공했지만, VLM 파이프라인의 이산 토큰 구조 자체가 한계를 만들었습니다.

1. **양자화 손실:** 연속적인 로봇 Action을 256-bin으로 이산화하면 정밀도가 떨어집니다. FAST가 이를 완화했지만, 본질적으로 이산 토큰은 연속 값을 완벽히 표현할 수 없습니다.
2. **Multimodality 한계의 회귀:** Next-token prediction 방식의 action 생성은 각 차원을 순차적으로 예측하므로, Era 2A에서 Diffusion이 해결한 Multimodality 문제가 다시 제한적으로 나타납니다.
3. **추론 속도:** VLM의 Next-token prediction 생성 자체가 느리고, Action chunking을 위해 더 많은 토큰을 생성해야 하면 latency가 누적됩니다.

---

### 1-4. [Era 3] VLM 지식 활용 + Continuous Action 생성을 통한 정밀도, Multimodality 확보

> ### 💡 한 줄 요약
>
> **Era 2A의 정밀한 동작 생성(Diffusion)과 Era 2B의 언어·시각 지식(VLM)을 합친 것이 현재 최전선입니다. Physical Intelligence $\pi_0$, NVIDIA GR00T N1, Figure AI Helix 등 주요 업체들이 모두 이 구조를 채택하고 있습니다.**

Era 2A는 Diffusion으로 연속 Action의 multimodality를 해결했지만 VLM 지식을 활용하지 못했고, Era 2B는 VLM의 지식을 활용했지만 Action을 이산 토큰으로 양자화하면서 정밀도와 multimodality를 희생했습니다.

Era 3는 이 두 갈래가 합류하여, VLM의 시각-언어 추론 능력을 유지하면서 연속 Action을 생성하는 방법을 탐색합니다.

합류의 본질은 단순합니다. VLM 파이프라인의 출력단에서 "언어 Head + 분류 예측(cross entropy loss)"를 "Action Expert + 연속값 예측(flow/diffusion loss)"로 교체하는 것입니다. 이를 통해 양자화 손실이 제거되고, Action chunking과 multimodality가 동시에 달성됩니다.

[![Era 3 — VLM 지식 + Continuous Action](/assets/img/notes/rfm/era3.jpg)](/assets/img/notes/rfm/era3.jpg)
*[Era 3] VLM 지식 활용 + Continuous Action 생성 (클릭하면 원본 크기로 열립니다)*

#### 1-4-1. [Era 3①] Diffusion 기반 Action Expert 트랜드

##### 1-4-1-1. CogACT (arXiv 2024, Microsoft Research Asia)

CogACT는 합류의 가장 직관적인 형태를 제안했습니다. VLM을 학습하지 않고(freeze), 그 출력 feature를 별도의 DiT(Diffusion Transformer) Action Expert에 conditioning으로 전달하여 연속 Action chunk를 생성합니다.

VLM은 이미지와 언어를 이해하는 "인지(cognition)" 역할만 담당하고, Action Expert가 "행동(action)" 생성을 전담하는 구조입니다. 이 분리를 통해 VLM의 사전 학습된 지식이 보존되며, Action Expert만 스케일업하면 성능이 향상되는 양상도 확인되었습니다.

구체적으로, VLM이 추출한 인지 feature는 AdaLN(Adaptive Layer Normalization)을 통해 DiT Action Expert의 각 레이어에 주입되어, "어떤 장면에서 어떤 언어 명령이 주어졌는지"에 따라 Action 생성 과정 자체가 변조됩니다.

다만 VLM이 freeze되어 있기 때문에, VLM이 로봇 Action에 중요한 정보를 선별적으로 추출하는 형태로 학습이 이루어지지 않는다는 한계가 있습니다.

> ### 💡 AdaLN(Adaptive Layer Normalization)
>
> DiT에서 조건 정보를 주입하는 방식입니다. BC-Z에서 소개한 FiLM(설명: → 1-1-2)이 네트워크의 중간 feature를 $\gamma \times \text{feature} + \beta$로 변환했던 것과 원리가 같지만, 적용 위치가 다릅니다.
>
> FiLM은 CNN에 적용되는 반면, AdaLN은 Transformer 내부에 적용됩니다.
>
> Transformer는 각 레이어에서 값의 분포를 안정시키기 위해 정규화(Normalization) 과정을 거치는데, AdaLN은 이 정규화 직후에 입력된 조건 정보에 따라 스케일 값 $\gamma$와 시프트 값 $\beta$를 적용합니다.
>
> 즉, "어떤 조건이냐에 따라 Transformer 각 레이어가 정보를 처리하는 방식 자체가 달라지는" 구조입니다.
>
> DiT 원논문(CVPR 2023)에서 제안되었으며, CogACT, GR00T N1 등 Diffusion Transformer 기반 Action Expert에서 표준적으로 사용됩니다.

##### 1-4-1-2. $\pi_0$ (arXiv 2024, Physical Intelligence)

CogACT가 "VLM 비학습 + 별도 Action Expert"라는 합류의 기본 형태를 제시했다면, Physical Intelligence의 $\pi_0$는 이 구조를 한 단계 발전시킨 모델입니다.

CogACT는 VLM을 학습하지 않고, 출력 feature만 별도의 DiT에 전달하는 분리 구조였지만, $\pi_0$는 VLM과 Action Expert를 하나의 Transformer 안에 통합(Transfusion 방식)하고, Diffusion 대신 더 효율적인 Flow Matching을 채택했습니다.

> ### 💡 Transfusion 방식
>
> 일반적인 AI 모델은 한 가지 형태의 데이터만 다룹니다. 텍스트 모델은 텍스트만, 이미지 생성 모델은 이미지만 학습하는 식입니다.
>
> Transfusion은 하나의 모델이 서로 다른 형태의 데이터를 각각에 맞는 학습 방식으로 동시에 배우는 구조입니다.
>
> 원래 이미지 생성 분야에서 제안된 개념으로, 하나의 모델이 텍스트는 "다음 단어 맞추기" 방식으로, 이미지는 "노이즈 제거" 방식으로 동시에 학습했습니다.
>
> $\pi_0$는 이 아이디어를 로봇에 적용하여, 언어는 "다음 단어 맞추기" 방식(cross-entropy loss)으로, 로봇 동작은 "노이즈에서 동작을 복원하는" 방식(flow matching loss)으로 학습합니다.
>
> 즉, 하나의 모델이 "언어를 이해하는 능력"과 "연속적인 로봇 동작을 생성하는 능력"을 동시에 갖게 됩니다.

구체적으로, $\pi_0$는 Google이 개발한 VLM인 PaliGemma를 backbone으로 사용합니다. 다만, 로봇 제어를 위해 로봇 상태(관절 각도)와 Action에 해당하는 토큰을 입력 시퀀스에 추가했습니다.

이 토큰들은 VLM이 사전 학습 때 본 적 없는 새로운 모달리티이므로, 기존 이미지/언어 토큰과는 별도로 처리되도록 하나의 Transformer 안에 Self-attention 연산을 공유하는 두 세트의 weights를 두었습니다.

- **Weights 1 (VLM expert):** 이미지·언어 토큰 처리
- **Weights 2** (**Action expert)**: 로봇 상태·Action 토큰 처리

> ### 💡 Self-Attention
>
> Transformer의 핵심 연산으로, 시퀀스 내의 모든 토큰이 서로를 참조하여 "어떤 토큰이 나에게 중요한지"를 가중치로 계산하는 메커니즘입니다.
>
> 예를 들어, "로봇이 빨간 컵을 집는다"라는 시퀀스에서 "집는다" 토큰은 self-attention을 통해 "빨간"과 "컵"에 높은 가중치를 부여하고, 상대적으로 덜 중요한 "로봇이"에는 낮은 가중치를 부여합니다.
>
> 이 과정을 거치면 각 토큰의 표현에 문맥 정보가 반영됩니다.
>
> $\pi_0$에서는 Action 토큰이 self-attention을 통해 이미지/언어 토큰을 참조하여, "지금 장면에서 어떤 시각 정보가 이 Action에 중요한지"를 학습하게 됩니다.

입력되는 모든 토큰(이미지, 언어, 로봇 상태, Action)은 하나의 시퀀스로 통합(concat)되어 같은 attention 연산을 거칩니다.

다만 attention mask로 참조 방향이 제한됩니다. Attention 이후, 각 토큰은 자신이 속한 expert로 routing되어 처리됩니다.

```
입력 시퀀스 (하나로 concat):
[img1, img2, ..., lang1, lang2, ... | state | action1, action2, ..., action50]
 ←───── VLM expert가 처리 ────────-─→  ←────-──── Action expert가 처리 ────────→

각 Transformer 레이어의 처리 과정:
  (1) Self-Attention: 전체 토큰이 하나의 attention 연산 참여 (mask로 방향 제한)
  (2) MLP 분기:  img/lang 토큰 → VLM expert
                state/action 토큰 → Action expert
  → 이 과정이 전체 레이어에 걸쳐 반복
```

CogACT가 VLM 출력을 하나의 벡터로 압축하여 AdaLN(설명:→ 1-4-1-1)으로 전달하는 것과 달리, $\pi_0$의 Action expert는 attention 블록을 공유하기 때문에 VLM의 모든 토큰을 직접 참조하므로 더 세밀한 시각-Action 정렬이 가능합니다.

> ### 💡 Blockwise Causal Attention Mask ($\pi_0$ 정보 참조 규칙)
>
> π₀는 이미지, 언어, 로봇 상태, Action 토큰을 하나의 시퀀스로 합쳐서 처리하는데, 이때 "누가 누구의 정보를 참고할 수 있는가"에 대한 규칙을 설정합니다.
>
> 비유하면, 선배 연구원(이미지/언어)이 작성한 분석 보고서를 신입사원(Action)은 자유롭게 참고할 수 있지만, 반대로 신입사원의 메모가 선배의 보고서 내용을 바꿔서는 안 되는 것과 같습니다.
>
> ```
> Block 1: [이미지, 언어]    Block 2: [로봇 상태]    Block 3: [Action]
>
> 누가 누구를 참조(attend)할 수 있는가:
>
>                     Block 1    Block 2    Block 3
>                    (img/lang)  (state)   (action)
> Block 1 (img/lang):   ✅         ❌         ❌
> Block 2 (state):      ✅         ✅         ❌
> Block 3 (action):     ✅         ✅         ✅
> ```
>
> 이 설계의 핵심은 이미지/언어 토큰이 Action 토큰의 영향을 받지 않는다는 점입니다.
>
> 덕분에 VLM이 웹에서 학습한 시각-언어 지식(예: "유리컵은 깨지기 쉽다")이 로봇 데이터 학습 과정에서 오염되지 않고 그대로 보존됩니다.
>
> 반대로 Action 토큰은 모든 정보를 참고할 수 있어, VLM이 이해한 장면 정보를 세밀하게 활용하여 정교한 동작을 생성합니다.

> ### 💡 Diffusion vs Flow Matching
>
> 둘 다 랜덤 노이즈에서 출발하여 원하는 결과물(로봇 동작)을 만들어내는 생성 모델이지만, 목적지까지 가는 경로가 다릅니다.
>
> Diffusion은 구불구불한 산길로 내려오는 것과 같습니다. 수학적으로 복잡한 경로(DDPM의 forward/reverse process)를 따르기 때문에, 목적지에 도달하려면 많은 단계(denoising step)를 거쳐야 합니다.
>
> Flow Matching은 곧장 직선 도로로 내려오는 것과 같습니다. 노이즈와 원본 사이를 직선으로 잇는 단순한 보간을 학습하기 때문에, 더 적은 단계로 같은 품질의 결과물에 도달할 수 있습니다.
>
> 목적지(생성 품질)는 동일하지만 Flow Matching이 더 빠르게 도착하므로, 실시간 로봇 제어처럼 속도가 중요한 상황에 유리합니다. $\pi_0$가 Diffusion 대신 Flow Matching을 채택한 이유입니다.

$\pi_0$는 8종의 로봇 embodiment에서 cross-embodiment 일반화를 달성하고, 세탁물 접기와 같은 고난도 dexterous manipulation 태스크를 수행했습니다.

다만 전체 모델을 동기적으로 실행하므로 추론 속도가 ~10Hz에 그쳐, 빠른 반응이 필요한 휴머노이드 전신 제어에는 한계가 있었습니다.

#### 1-4-2. [Era 3②] System 2(인지/계획)-System 1(제어) 2 Stage 트랜드

CogACT, $\pi_0$는 VLM과 Action Expert를 하나의 파이프라인에서 동기적으로 실행합니다. 하지만 VLM의 추론(이미지 인코딩 + 언어 이해)은 무겁고 느린 반면, 로봇의 관절 제어는 가볍고 빨라야 합니다.

이 속도 불일치를 해결하기 위해 인지와 제어를 비동기적으로 분리하는 아키텍처가 등장합니다.

> ### 💡 System 1 / System 2
>
> 노벨 경제학상 수상자 Daniel Kahneman이 저서 Thinking, Fast and Slow(2011)에서 제안한 인간 인지의 이중 프로세스 이론입니다.
>
> System 2는 느리지만 의식적이고 논리적인 사고(예: 수학 문제 풀기), System 1은 빠르고 직관적이며 자동화된 반응(예: 자전거 타기)을 담당합니다.
>
> AI 분야에서는 최근 LLM의 추론 능력을 강화하기 위해 이 개념이 활발히 차용되고 있습니다. OpenAI의 o1/o3 모델이 답을 바로 출력하지 않고 chain-of-thought로 "생각하는 시간"을 갖는 것이 System 2적 접근이고, 일반적인 LLM의 즉각적인 next-token prediction이 System 1에 해당합니다.
>
> 로봇 분야에서는 이를 더 직관적으로 적용하여, 느리지만 풍부한 장면 이해(VLM = System 2)와 빠른 관절 제어(Action Expert = System 1)를 물리적으로 분리하고 비동기 병렬로 실행하는 아키텍처가 등장합니다.

##### 1-4-2-1. GR00T N1 (arXiv 2025, NVIDIA)

$\pi_0$가 하나의 Transformer 안에서 VLM과 Action Expert를 통합한 것과 달리, NVIDIA의 GR00T N1은 앞서 소개한 Daniel Kahneman의 System 2/System 1 이론을 명시적인 아키텍처 설계 원칙으로 채택하여, 이 둘을 물리적으로 완전히 분리된 두 네트워크로 구성합니다.

VLM은 웹 규모 데이터로 학습된 풍부한 시각-언어 추론 능력을 갖추고 있지만 느립니다. Era 3의 핵심인 "VLM 지식 활용"과 "고주파 연속 Action 생성"을 동시에 달성하려면, VLM의 느린 인지를 빠른 제어와 분리하는 것이 자연스러운 귀결이며, GR00T N1은 이 구조를 최초로 실현했습니다.

- **System 2(인지/계획,~1Hz):** VLM이 이미지와 언어를 처리하여 "지금 장면에서 무엇을 해야 하는지"를 이해하고, 그 결과를 Task embedding으로 압축
- **System 1 (제어, ~30Hz):** 별도의 네트워크인 DiT 기반 Action Expert가 Task embedding + 로봇 관측값 입력 받아 "관절을 실제로 어떻게 움직일지"에 해당하는 연속 Action chunk를 flow matching으로 생성 (이미지, 언어 불필요)

GR00T N1은 System 2와 System 1을 분리된 네트워크 구조로 디자인하기 위해, VLM의 정보를 Action Expert에 어떻게 전달할 것인지와 각 네트워크를 어떻게 실행할 것인지 두 가지 측면에서의 trade-off가 발생했습니다.

- 정보 전달 방식
    - $\pi_0$의 Action expert는 attention 블록 공유를 통해 VLM의 모든 토큰(예: 1200개의 시각 토큰)을 직접 참조합니다. "빨간 컵 옆의 파란 컵을 집어"라는 명령이 주어지면, Action expert가 "빨간 컵"에 해당하는 시각 토큰과 "파란 컵"에 해당하는 시각 토큰을 개별적으로 attend하여 정확한 위치를 파악할 수 있습니다.
    - 반면 GR00T N1의 System 1은 VLM이 장면 전체를 하나의 벡터로 압축한 Task embedding만 입력으로 받습니다. 1200개 토큰에 담긴 공간적 디테일이 하나의 벡터로 압축되는 과정에서 "어느 컵이 정확히 어디에 있는지"와 같은 세밀한 정보가 손실될 수 있습니다.
    - 즉, π₀는 토큰 수준의 세밀한 시각-Action 정렬이 가능하고, GR00T N1은 장면의 전역적인 의미만 전달됩니다.
- 실행 방식

    ```
    시간축 비동기 동작:
    System 2: [계획1.............] [계획2.............] [계획3...........]
    System 1: [a][a][a][a][a][a][a][a][a][a][a][a][a][a][a][a][a][a]...
               System 2가 계획을 업데이트하는 동안, System 1은 이전 계획으로 계속 Action 생성
    ```

    - $\pi_0$는 하나의 Transformer 안에 VLM과 Action Expert가 통합되어 있으므로, Action을 생성할 때마다 이미지 인코딩을 포함한 전체 forward pass를 실행해야 합니다. 이미지 인코딩은 연산량이 크기 때문에 ~10Hz가 한계입니다.
    - GR00T N1은 두 네트워크가 물리적으로 분리되어 있으므로, System 2(VLM)가 ~1초마다 task embedding을 업데이트하는 동안 System 1(DiT)은 이전에 받아둔 임베딩과 로봇 관측값만으로 독립적으로 Action을 생성합니다.
    - System 1은 이미지를 처리할 필요가 없어 연산이 가볍고, ~30Hz 이상의 고주파 제어가 가능합니다.

이 물리적 분리 구조는 모듈 교체와 분산 배포에도 유리합니다. VLM만 더 큰 모델로 교체하거나, VLM은 클라우드에서, DiT는 로봇 온보드 GPU에서 실행하는 식의 유연한 구성이 가능합니다.

이와 같은 이유로 GR00T N1은 휴머노이드 실시간 제어 연구의 시작점이라는 평가를 받고 있습니다.

##### 1-4-2-2. $\pi_{0.5}$ (arXiv 2025, Physical Intelligence)

Physical Intelligence는 $\pi_0$를 확장하여, VLM이 실제로 자연어 서브태스크 계획을 생성하는 plan-and-execute 구조를 제안했습니다.

$\pi_0$에서 VLM expert의 출력은 Action expert가 attention으로 참조하는 데만 사용되었지만(텍스트 생성 X), $\pi_{0.5}$에서는 VLM backbone이 원래의 역할인 텍스트 생성도 함께 수행합니다. 즉, VLM의 원래 구조에 있던 언어 Head를 살려 텍스트를 출력하도록 하였습니다.

같은 VLM backbone이 한쪽으로는 Action expert에 시각-언어 정보를 제공하면서, 다른 한쪽으로는 "1. 접시를 집는다 → 2. 싱크대에 놓는다 → 3. 행주로 닦는다"와 같은 자연어 subtask를 생성합니다. Action expert는 이 subtask를 추가 조건으로 받아 각 단계의 Action chunk를 순차 실행합니다.

이 구조의 장점은 계획이 사람에게 설명가능한 형태로 보이게 되고, 또 교정 가능하다는 점입니다. 로봇이 실수하면 "아니, 그거 말고 왼쪽 거"라고 구두 피드백으로 계획을 수정할 수 있습니다.

또한 VLM 학습에 사용하던 Web-scale 데이터(이미지+텍스트)와 로봇 데이터를 co-fine-tuning으로 함께 학습하여, 학습 환경 밖의 실제 가정집에서 동작하는 최초의 open-world VLA를 달성했습니다.

다만 $\pi_0$의 동기적 실행 구조를 그대로 유지하므로 추론 속도가 ~10Hz에 머물러, 휴머노이드 적용에는 한계가 있습니다.

> ### 💡 GR00T N1 vs $\pi_{0.5}$ — 계획 표현의 차이
>
> GR00T N1의 System 2 출력은 벡터(임베딩)이므로 사람이 내용을 확인하거나 수정할 수 없습니다. 반면 $\pi_{0.5}$의 계획은 자연어 텍스트이므로 사람이 읽고 교정할 수 있습니다. 투명성과 사람-로봇 협업 측면에서 중요한 설계 차이입니다.

##### 1-4-2-3. Helix 02 (2026, Figure AI)

GR00T N1은 System 2(인지/계획)와 System 1(제어)의 2계층 구조로 ~30Hz 제어를 달성했지만, 휴머노이드에는 근본적인 문제가 하나 더 있습니다.

팔 로봇(Franka 등)은 고정된 베이스 위에 있어 밸런스가 불필요하지만, 두 발로 서 있는 휴머노이드는 항상 넘어질 위험이 있습니다.

물건을 들면 무게중심이 변하고, 팔을 뻗으면 몸이 기울어지므로 밀리초 단위의 즉각적인 밸런스 보정이 필요합니다. 30Hz(~33ms 간격)로는 이 반사적 보정이 충분히 빠르지 않습니다.

Figure AI는 이 문제를 해결하기 위해 인간 신경계의 “대뇌 피질(인지) → 운동 피질(제어) → 척수 반사(밸런스)”를 모사한 3계층 비동기 아키텍처를 제안했습니다.

- **System 2 (인지/계획, ~7-10Hz):** VLM이 카메라 이미지와 자연어 명령을 처리하여 "지금 장면에서 무엇을 해야 하는지"를 이해하고, Task embedding으로 압축합니다. (GR00T N1의 System 2와 동일 역할)
- **System 1 (전신 제어,~200Hz):** System 2 출력인 Task embedding과 로봇 관측값, 손목 카메라, 촉각 센서를 입력받아 전신 관절 목표를 생성합니다. (구체적인 모델 구조 미공개)
- **System 0 (반사 제어,~1kHz):** 인간 모션 데이터 1,000시간으로 학습된 신경망이 관절 센서, IMU, 접촉 센서만을 입력받아 밀리초 단위의 밸런스·접촉·전신 협응을 수행합니다. System 1이 생성한 관절 목표에 대해 "이대로 움직이면 넘어지니까 발목 토크를 이만큼 보정해"라는 미세 조정을 1kHz로 적용합니다.

> ### 💡 왜 System 0이 이미지를 보지 않는가?
>
> System 2 → System 1 → System 0으로 갈수록 입력 차원이 급격히 줄어듭니다. (이미지 (150K) → 태스크 임베딩 (4K) → 관절 목표 (30) → 토크 보정 (30))
>
> 특히, System 0은 관절 각도, IMU, 접촉 센서 등 저차원 신호만 처리하므로 1kHz라는 극도로 빠른 주기가 가능합니다.
>
> 이미지 처리를 건너뛰는 것이 속도의 핵심이라는 점은 GR00T N1의 System 1과 같은 설계 철학입니다.

---

## Section 2. VLA 학습 방법론 (학습 알고리즘)

### 2-1. 학습 데이터

> ### 💡 한 줄 요약
>
> **로봇 학습 데이터는 "로봇이 등장하는가?" × "행동 라벨이 있는가?"의 2×2로 분류됩니다. 가장 직접적인 데이터(로봇 조종 데이터)는 비싸고, 가장 풍부한 데이터(인터넷 영상)는 행동 라벨이 없는 것이 핵심 딜레마입니다.**

Robot Foundation Model을 학습시키려면 결국 데이터가 필요합니다. 로봇 학습 데이터는 학습 성격에 따라 학습에 활용하는 방식이 완전히 달라지기 때문에, 데이터의 지형부터 정리할 필요가 있습니다.

로봇 학습 데이터를 분류하는 가장 직관적인 기준은 두 가지입니다. **로봇이 등장하는 데이터인가?** 그리고 **행동(action) 라벨이 있는가?**

이 두 축으로 2×2 사분면을 그리면, 현재 RFM 생태계에서 사용되는 거의 모든 데이터를 분류할 수 있습니다.

![RFM 학습 데이터 사분면](/assets/img/notes/rfm/data-quadrant.svg)
*RFM 학습 데이터의 네 사분면 — 로봇 유무 × Action label 유무*

#### 2-1-1. [Q1] 로봇 Teleoperation 데이터 — 가장 직접적이지만 가장 비싼 데이터

사람이 조이스틱, VR 컨트롤러 방식으로 로봇을 원격 조종하면서 수집한 데이터입니다.

매 시점(timestep)마다 로봇의 상태(카메라 영상, 관절 각도)와 사람이 내린 행동 명령(그리퍼 이동량, 관절 토크 등)이 쌍으로 기록됩니다.

VLA 학습에 가장 직접적으로 쓸 수 있는 "gold standard" 데이터이지만, 수집 비용이 매우 높습니다. 로봇 하드웨어가 필요하고, 숙련된 오퍼레이터가 실시간으로 조종해야 하며, 한 에피소드를 수집하는 데 수 분이 걸립니다.

현재 공개된 대표적인 대규모 데이터셋으로는 Google의 RT-X 프로젝트가 만든 Open X-Embodiment(OXE) 데이터셋이 있습니다. 21개 기관, 22종 로봇에서 수집한 약 100만 개 이상의 에피소드를 합친 것으로, "데이터를 합치면 로봇 성능이 올라간다"는 가설을 실증한 데이터셋입니다.

비공개 데이터까지 포함하면, Physical Intelligence(π0 학습용), Google DeepMind(RT-2, Gemini Robotics 학습용), NVIDIA(GR00T 학습용) 등 주요 회사들이 자체 로봇 플릿이나 파트너십을 통해 대규모 Teleoperation 데이터를 수집하고 있습니다.

> ### 💡 로봇 플릿(Robot Fleet)
>
> 로봇 플릿은 동일하거나 유사한 로봇을 여러 대 운용하는 것을 말합니다.
>
> Google DeepMind가 RT-2 학습 데이터를 수집할 때 사무실 곳곳에 로봇 팔 수십 대를 배치해두고 동시에 Teleoperation 데이터를 수집한 것이 대표적인 예입니다.

#### 2-1-2. [Q3] 웹스케일 데이터 — 로봇도 없고 action도 없지만, 규모가 압도적

인터넷에서 수집한 이미지-텍스트 쌍, 비디오, 텍스트 데이터입니다.

로봇과 직접적인 관련은 없지만, 규모가 Q1과는 비교가 되지 않습니다. Q1이 수십만~수백만 에피소드 단위라면, Q3는 수십억 개의 이미지-텍스트 쌍, 수천만 시간의 비디오 단위입니다.

이 데이터는 로봇이 포함되지 않기 때문에 VLA에 직접 로봇 action을 가르치는 데는 쓸 수 없지만, VLM(Vision-Language Model) 사전학습 과정에 "세상이 어떻게 생겼는지, 언어가 시각과 어떻게 연결되는지"를 학습하는 데 사용됩니다.

현재 대부분의 VLA는 웹스케일 데이터로 pretrain된 VLM을 backbone으로 가져와서 그 위에 action head를 얹는 구조입니다.

RT-2는 PaLI/PaLM을, OpenVLA는 Llama-2를, π0는 PaliGemma를 backbone으로 사용합니다. 이 backbone들이 웹스케일 데이터에서 학습한 vision-language 지식이 로봇의 시각적 이해와 언어 지시 이해 능력으로 전이(transfer)되는 것입니다.

#### 2-1-3. [Q2] 파생 데이터 — 원래는 action이 없었지만, 추출해서 만든 데이터

Q2분면은 인간이 작업하는 영상(요리, 청소, 조립 등)에서 AI 모델을 활용해 로봇 action label을 추출하여 생성한 파생 데이터입니다. 여기서 핵심적인 역할을 하는 것이 로봇 action label 추출을 위한 AI 모델, Inverse Dynamics Model(IDM)입니다.

> ### 💡 Inverse Dynamics Model(IDM)과 LAPA 미리보기
>
> IDM은 연속된 두 프레임을 보고 "이 변화를 만든 action은 무엇인가?"를 추론하는 모델입니다.
>
> 전통 로보틱스에서는 로봇의 질량, 관절 길이 등 물리 파라미터를 기반으로 역동역학 방정식을 풀어서 action을 계산했습니다. 반면 학습 기반 IDM은 물리 방정식 없이, 연속된 두 프레임의 시각적 변화만 보고 "어떤 행동이 이 변화를 만들었는가"를 데이터로부터 학습합니다.
>
> 로봇 스펙을 몰라도 되고, 사람 손 영상에도 적용할 수 있다는 점에서 확장성이 훨씬 큽니다. 다만 추출되는 action은 정확한 물리량이 아니라 추상적인 표현(latent action)이므로, 실제 로봇 제어에 쓰려면 이를 타겟 로봇 H/W 특성에 맞는 Action을 생성할 수 있도록 학습하는 단계가 추가로 필요합니다.
>
> **LAPA(Latent Action Pretraining, ICLR 2025)**는 이 아이디어를 파운데이션 모델 스케일로 확장한 연구입니다.
>
> "action이 뭔지 명시적으로 정의하지 않더라도, 프레임 간 변화 자체를 추상적인 표현(latent action)으로 인코딩하겠다"는 발상으로, 인터넷의 방대한 비디오를 pre-training 데이터로 활용할 수 있는 길을 열었습니다.

Q2의 가치는 데이터 규모에 있습니다. Q1(Teleoperation)은 수집 비용 때문에 규모 확장에 한계가 있지만, 인간 작업 영상은 인터넷에 사실상 무한히 존재합니다. IDM를 통해 이 영상들에서 action 정보를 추출할 수 있다면, pre-training 데이터의 규모를 비약적으로 늘릴 수 있습니다.

#### 2-1-4. [Q4] 로봇 영상 (action label 없음) — 아직 활용이 제한적이지만 가치가 커지는 영역

로봇이 촬영된 영상이지만 Teleoperation 로그가 아니라서 action label이 없는 데이터입니다.

로봇의 자율 탐색(autonomous exploration) 영상, 로봇 시연 영상(YouTube 등), 또는 로봇이 배포되어 작업하는 과정에서 action 로그 없이 영상만 기록된 경우가 해당됩니다.

현재 순수하게 Q4만으로 구성된 대규모 공개 데이터셋은 많지 않습니다. 대부분의 로봇 데이터셋은 텔레오퍼레이션으로 수집되어 action label이 함께 있기 때문입니다(Q1).

하지만 Q4의 활용 가능성은 점점 커지고 있습니다. Q2와 마찬가지로 LAPA/IDM을 적용하면 Q1으로 변환하여 RFM 사전학습에 이용할 수 있고, action label 없이도 다음 프레임 예측 방식으로 world model 사전학습에 활용할 수 있습니다.

---

### 2-2. Pre-training (사전학습): 범용 or 특정 산업환경 중심 로봇 지능의 기반 만들기

> ### 💡 한 줄 요약
>
> **인터넷 지식을 물려받은 AI에 로봇 동작을 가르치는 단계. 로봇 조종 데이터로 직접 가르치는 것이 기본이지만, 최근에는 인터넷 영상에서 동작 정보를 AI가 스스로 추론하여 학습하는 접근이 급부상하고 있습니다.**

Pre-training 단계의 목적은 두 가지입니다.

1. **웹 규모로 학습된 VLM(Vision-Language Model)의 지식을 로봇에 전이(transfer)하는 것**
    - "빨간 컵을 집어라"라는 지시를 이해하려면, 모델이 "빨간"이 무슨 색인지, "컵"이 어떻게 생겼는지, "집다"가 어떤 행위인지를 이미 알고 있어야 합니다.
    - 이 지식은 인터넷의 수십억 개 이미지-텍스트 쌍(Q3 데이터)에서 학습됩니다.
2. **다양한 로봇과 환경에 걸쳐 범용적인 action space(행동 공간)를 구축하는 것**
    - 특정 로봇 한 대에서만 동작하는 모델이 아니라, 여러 종류의 로봇(다관절 팔, 휴머노이드, 모바일 매니퓰레이터 등)에서 공통적으로 작동할 수 있는 행동 표현을 만드는 것이 목표입니다.
    - 이를 위해 대규모 텔레오퍼레이션 데이터(Q1)와, 인간 비디오에서 추출한 파생 데이터(Q2)가 활용됩니다.

> ### 💡 Action Space (행동 공간)
>
> 1-2-1-1에서 Action space를 "로봇이 취할 수 있는 모든 행동의 범위"로 정의했다면, 학습 관점에서 Action space는 "모델이 출력해야 하는 정답의 형태"를 결정합니다.
>
> 예를 들어 7축 로봇 팔의 Action space는 7차원 연속 공간이고, 양팔 로봇 ALOHA는 14차원 연속 공간입니다.
>
> 서로 다른 로봇의 데이터를 하나의 모델로 학습하려면, 이 이질적인 Action space를 통합해야 합니다.
>
> RDT-1B(설명: → 1-2-2-2. RDT-1B (ICLR 2025))가 모든 로봇의 Action을 최대 차원으로 정렬하고 빈 차원을 마스킹한 것이 대표적인 해법입니다.
>
> **Pre-training에서 범용 Action space를 구축하는 것이 중요한 이유는, 이후 post-training 단계에서 타겟 로봇에 적응(adaptation)할 때의 출발점이 되기 때문입니다.**
>
> Pretrain 단계에서 다양한 로봇의 Action space를 충분히 학습해 둘수록, 새로운 로봇에 소량의 데이터만으로도 빠르게 적응할 수 있는 기반이 만들어집니다.

현재 대부분의 VLA는 이미 pretrain된 VLM을 backbone으로 가져온 뒤, 그 위에 action head를 붙여서 로봇 데이터로 학습하는 방식으로 두가지 목적을 달성하고 있습니다.

> ### 💡 VLA 사전학습 파이프라인
>
> **Step 1: 사전학습된 VLM 준비**
>
> 첫 번째 단계는 VLM pretrain인데, 이 단계는 VLA 연구자가 직접 수행하는 것이 아니라 이미 학습이 완료된 VLM을 가져오는 것입니다.
>
> PaliGemma, Llama-2, PaLI 같은 모델들이 Q3 데이터(웹스케일 이미지-텍스트 쌍)로 수십억 개의 데이터를 학습하여 vision-language 지식을 이미 갖추고 있고, VLA 연구자는 이 모델을 backbone으로 가져와서 출발합니다.
>
> **Step 2: VLA에 로봇 데이터로 action 학습**
>
> 두 번째 단계가 본격적인 VLA pretrain입니다. 가져온 VLM backbone 위에 action head를 추가하고, Q1 데이터(Teleoperation)로 action space(행동 공간)를 학습시킵니다.
>
> 선택적으로 Q2 데이터(파생 데이터)를 함께 사용하여 데이터 규모를 확장하기도 합니다.
>
> 이 과정에서 VLM backbone은 완전히 고정(freeze)하거나 매우 낮은 learning rate로 fine-tune하여 기존 지식이 훼손되지 않도록 합니다.
>
> 이 두 단계를 거치면 "세상을 보고, 언어를 이해하고, 행동을 출력하는" 범용 VLA가 만들어집니다.

위에서 설명한 두 가지 목적을 달성하기 위한 핵심 질문은 결국 "로봇을 가르칠 데이터를 어디서, 어떻게 확보하는가?"입니다.

로봇에게 동작을 가르치는 가장 직관적인 방법은, 사람이 직접 로봇을 조종한 시연 데이터를 모아서 "이 상황에선 이렇게 움직여라"고 따라하게 하는 것입니다(**Behavior Cloning**).

여기에 웹 데이터를 함께 섞어 학습시키면 VLM이 원래 갖고 있던 시각-언어 지식이 훼손되지 않도록 보존할 수 있습니다(**Co-fine-tuning**). 이 두 기법은 현재 VLA 사전학습의 표준 접근법입니다.

그런데 근본적인 병목이 있습니다. 사람이 로봇을 직접 조종해서 데이터를 모으는 것은 비용이 매우 높아, 아무리 노력해도 수십만 에피소드가 한계입니다. 반면 인터넷에는 요리, 청소, 조립 등 사람이 손으로 작업하는 영상이 수억 건 이상 존재합니다. 문제는 이 영상에 로봇 관절 값(action label)이 없다는 것입니다.

최근 급부상하고 있는 **Learning from Observation(LfO)** 접근은 이 한계를 정면으로 돌파합니다.

핵심 아이디어는 간단합니다. 요리 영상에서 "도마 위에 양파가 있는 장면" 다음에 "양파가 잘려 있는 장면"이 나오면, AI가 두 장면을 비교하여 "그 사이에 칼로 양파를 썰었을 것"이라는 동작 정보를 스스로 추론하는 것입니다. 이렇게 추출된 동작 표현(Latent Action)으로 사전학습하면, 로봇 시연 데이터 없이도 대규모 학습이 가능해집니다.

대표 연구인 LAPA는 인간 영상만으로 pretrain한 모델이 로봇 데이터로 pretrain한 모델을 성공률에서 앞지르는 결과를 보여주며, 이 접근의 가능성을 실증했습니다.

아래에서는 이 세 가지 접근 — **Behavior Cloning, Co-fine-tuning/Knowledge Insulating, Learning from Observation** — 을 순서대로 살펴봅니다.

#### 2-2-1. Imitation Learning & Behavior Cloning

##### 2-2-1-1. Imitation Learning

전통적인 로보틱스에서 로봇을 "가르치는" 방법은 크게 두 갈래로 나뉘어 왔습니다. 하나는 엔지니어가 직접 코드로 로봇의 동작을 프로그래밍하는 방식이고, 다른 하나는 사람이 시범을 보여주면 로봇이 그걸 보고 배우는 방식입니다.

후자를 로보틱스 분야에서는 **Learning from Demonstration (LfD)**이라고 불렀으며, 1980~90년대 산업용 로봇 연구에서 매우 활발하게 다뤄졌습니다.

> ### 💡 LfD (Learning from Demonstration) 파이프라인
>
> 당시 LfD의 전형적인 파이프라인은 이렇습니다.
>
> 사람이 로봇 팔을 직접 잡고 원하는 동작을 시연합니다(kinesthetic teaching). 이때 로봇의 관절 각도, 속도, 힘 등이 시계열 데이터로 기록됩니다. 그 다음 이 궤적(trajectory)을 수학적으로 모델링해서 재현 가능한 형태로 만듭니다.
>
> 이 단계에서 쓰인 대표적인 도구가 Dynamic Movement Primitives (DMP)나 GMM/GMR (Gaussian Mixture Model/Regression) 같은 전통적 궤적 표현 방법들입니다.

이 LfD 전통이 ML/AI 커뮤니티와 만나면서 좀 더 일반화된 용어로 정리된 것이 **Imitation Learning (IL)**입니다.

##### 2-2-1-2. Behavior Cloning

**Behavior Cloning (BC)**이라는 용어는 1989년 카네기멜론의 자율주행 연구(ALVINN)에서 처음 확립되었습니다.

카메라 영상을 입력으로 받아 스티어링 각도를 신경망으로 직접 예측하는, 지금 기준으로 보면 매우 단순한 시도였지만, "사람의 행동 데이터를 그대로 복제(clone)한다"는 핵심 아이디어가 여기서 시작되었습니다.

BC의 학습 구조는 일반적인 지도학습(supervised learning)과 동일합니다. 전문가(사람)가 로봇을 조작한 데이터, 즉 "이 상황(state)에서 이 행동(action)을 했다"는 쌍을 대량으로 수집하고, 모델이 상황(state)를 입력받아 action을 출력하도록 loss를 최소화합니다.

**현재 주요 VLA 모델들의 사전학습(pre-training)도 본질적으로 이 BC입니다.** 다만 규모가 달라졌을 뿐입니다. OpenVLA는 97만 개의 로봇 에피소드를 학습했고, π₀는 수천~수만 시간 규모의 정밀한 로봇 제어(dexterous manipulation) 데모 데이터를 기반으로 BC를 수행했습니다.

BC가 이렇게 오랫동안, 그리고 지금까지도 기본 학습 방법으로 쓰이는 이유는 명확합니다. 구현이 단순하고, 강화학습의 복잡한 알고리즘을 설계할 필요가 없으며, 고품질 데모 데이터만 있으면 바로 학습이 가능합니다.

하지만 BC에는 세 가지 근본적인 한계가 있습니다.

1. **분포 이동(distributional shift) 문제**

    ```
    [분포 이동에 의한 에러 누적]

    학습 시: 전문가 trajectory     ●──●──●──●──●  (전문가가 방문한 state들)
    배포 시: VLA 출력 trajectory.  ●──●──◎──◎──◎  (◎ = 학습 데이터에 없는 state)
                                      ↑
                                  첫 실수 발생 → 이후 state가 전부 미지의 영역
                                  → 에러가 누적되며 발산
    ```

    - 운전교습소에서만 연습한 초보 운전자가 처음 시내에 나가면 당황하는 것과 같습니다. 교습소에서 안 겪어본 상황이 오면 대처를 못 하는 것처럼, BC로 학습한 모델도 학습 데이터에 없던 상황을 만나면 올바른 행동을 출력하지 못합니다.
    - BC는 학습 시 전문가가 방문한 state에서만 데이터를 봅니다. 그런데 배포 시 VLA가 아주 약간이라도 실수를 하면, 전문가가 한 번도 방문하지 않은 state로 진입하게 됩니다.
    - 그 state에서의 올바른 action은 학습한 적이 없으니, 또 실수를 하고, 그 실수가 다음 state를 더 이상한 곳으로 밀어내고, 이런 식으로 에러가 시간에 따라 누적(compounding)됩니다.
    - 이걸 수식으로 보면, T 스텝짜리 task에서 BC의 에러는 제곱배로 증가합니다. 즉, 작업이 길어질수록 실패 확률이 제곱으로 커집니다.
    - 짧은 pick-and-place 같은 작업에서는 괜찮지만, "부엌을 정리해줘" 같은 수십 분짜리 장기 작업(long-horizon task)에서는 BC만으로는 매우 취약해집니다.
    - 이 문제는 뒤에서 설명할 **post-training 단계의 RL이나 DAgger 계열 기법으로 보완**하는 것이 현재의 표준 접근입니다.
2. **VLM 지식 보존 문제**
    - 현재 VLA는 웹 규모로 사전학습된 VLM(PaliGemma, Llama-2 등)을 backbone으로 가져온 뒤, 그 위에 action head를 붙여서 로봇 데이터로 학습하는 구조입니다.
    - 그런데 BC로 로봇 데이터만 학습시키면, VLM이 원래 가지고 있던 vision-language 지식이 훼손될 수 있습니다. 이를 catastrophic forgetting 현상이라고 부릅니다.
    - 예를 들어 "유리컵은 깨지기 쉬우니까 조심해서 집어야 한다"는 상식을 VLM이 알고 있었는데, 로봇 데이터만으로 fine-tuning하면 이런 지식이 사라질 수 있습니다.
    - 이 문제는 뒤에서 설명할 **pre-training 단계의 Co-fine-tuning 기법으로 보완**하는 것이 현재의 표준 접근입니다.
3. **데이터 규모의 한계**
    - Q1 데이터(Teleoperation)는 수집 비용이 높아 규모 확장에 근본적인 한계가 있습니다. 인터넷에는 인간 작업 영상이 사실상 무한히 존재하지만, action label이 없어 BC에 직접 사용할 수 없습니다.
    - 이 문제는 뒤에서 설명할 **pre-training 단계의 Learning from Observation 기법으로 보완**하는 연구들이 학계를 넘어 산업계에도 자리 잡고 있습니다.

#### 2-2-2. VLM 지식 보존 전략: Co-fine-tuning & Knowledge Insulating

앞서 BC의 한계점 2에서 다룬 catastrophic forgetting 문제를 다시 짧게 짚으면, VLM의 지식을 보존하자니 action 학습이 부족하고, action을 잘 학습시키자니 VLM 지식이 훼손되는 딜레마가 있었습니다.

이 딜레마를 해결하는 방식은 크게 두 갈래로 나뉩니다.

하나는 웹 데이터와 로봇 데이터를 동일한 학습 루프 안에서 동시에 학습시키는 **co-fine-tuning**이고, 다른 하나는 VLM backbone을 구조적으로 보존하면서 action 학습만 별도로 수행하는 **지식 절연(knowledge insulating)** 접근입니다.

##### 2-2-2-1. Co-fine-tuning: RT-2 (Google DeepMind)

Co-fine-tuning을 가장 정면으로 구현한 모델은 구글의 RT-2(설명: → 1-3-1. Robotics Transformers 2 (PMLR 2023, Google DeepMind))입니다.

RT-2는 VLM pretrain 데이터(웹 이미지-텍스트)와 로봇 텔레오퍼레이션 데이터를 약 1:1 비율로 섞어서 동시에 학습했습니다. 이것이 가능했던 것은 RT-2의 아키텍처 설계 덕분입니다.

RT-2는 로봇 action을 이산 토큰으로 변환하여 VLM의 단어 사전에 편입시켰기 때문에, 웹 데이터와 로봇 데이터가 구조적으로 같은 형식의 학습 데이터가 됩니다.

> ### 💡 RT-2에서 웹 데이터와 로봇 데이터가 같은 분류 방식으로 학습되는 원리
>
> VLM은 원래 "이미지를 보고 텍스트를 생성하는" 모델입니다. 학습 시의 loss는 next-token prediction, 즉 "다음 토큰이 무엇인가?"를 맞추는 cross-entropy loss입니다.
>
> RT-2는 로봇 action(예: 관절 각도 7개)을 256개 구간으로 양자화하여 텍스트 토큰과 동일한 형태로 만들었습니다.
>
> 그 결과 웹 데이터에서는 "이미지 → 텍스트 토큰 예측"이 되고, 로봇 데이터에서는 "이미지 + 언어 지시 → 텍스트 토큰 + action 토큰 예측"이 됩니다.
>
> 두 경우 모두 동일한 next-token prediction loss(cross-entropy)를 사용하므로, 하나의 학습 루프 안에서 배치(batch) 단위로 웹 데이터와 로봇 데이터를 자연스럽게 섞을 수 있습니다.
>
> 별도의 loss 설계나 multi-task 학습 프레임워크가 필요하지 않았던 것입니다.

이 co-training 덕분에 RT-2는 웹에서 학습한 시각적 추론 능력을 로봇 제어에 직접 전이할 수 있었습니다.

학습 데이터에 없던 물체(예: "코카콜라 캔을 집어라")를 인식하고 조작하거나, "쓰레기인 것을 골라서 버려라" 같은 의미적 추론이 필요한 지시를 수행할 수 있었던 것이 대표적인 사례입니다.

##### 2-2-2-2. Knowledge Insulating

모든 VLA가 RT-2처럼 co-training을 하는 것은 아닙니다.

Action을 이산 토큰이 아닌 연속 벡터로 다루는 모델(Diffusion, Flow Matching 기반)은 웹 데이터의 텍스트 loss와 로봇 데이터의 action loss가 형태가 다르기 때문에, RT-2와 같은 방식의 co-fine-tuning이 구조적으로 쉽지 않습니다.

$\pi_0$(설명: → 1-4-1-2. $\pi_0$ (arXiv 2024, Physical Intelligence))는 가장 구조적인 접근을 취했습니다. PaliGemma VLM을 backbone으로 사용하되, action 생성을 별도의 flow matching expert로 완전히 분리했습니다.

$\pi_0$의 후속 knowledge insulating(arXiv 2025) 연구에서는 전체 모듈 구조 중 VLM 부분은 학습하지 않거나(freeze) 매우 낮은 learning rate로 학습하여 원래의 지식이 변하지 않도록 하고, action expert만 로봇 데이터로 집중 학습합니다.

최근에는 이 knowledge insulating 문제의 해법으로, VLM(고수준 추론)과 action expert(실시간 모터 제어)를 아키텍처 수준에서 분리하는 dual-system 구조(NVIDIA GR00T N1, Physical Intelligence $\pi_{0.5}$, Figure AI Helix 02)가 표준으로 자리잡고 있습니다.

이 분리 덕분에 action expert를 로봇 데이터로 집중 학습할 때 VLM backbone에 영향을 주지 않도록 제어할 수 있어, 지식 보존과 action 학습을 구조적으로 양립시킬 수 있게 됩니다.

#### 2-2-3. Learning from Observation (LfO)

Co-training이 "VLM 지식을 보존하면서 로봇 action을 학습하는" 전략이었다면, LfO는 BC의 세 번째 한계인 데이터 규모 문제를 정면으로 다룹니다.

Q1 데이터(Teleoperation)는 수집 비용이 높아 규모 확장에 한계가 있는 반면, 인터넷에는 인간이 작업하는 영상이 사실상 무한히 존재합니다. 문제는 이 영상에 action label이 없다는 것입니다.

LfO는 "action label이 없는 영상에서도 로봇이 학습할 수 있는 정보를 추출하겠다"는 접근입니다.

LfO의 핵심 도구는 앞에서 잠깐 소개한 **Inverse Dynamics Model(IDM)**입니다.

전통 로보틱스에서는 로봇의 질량, 관절 길이 등 물리 파라미터를 기반으로 역동역학 방정식을 풀어서 action을 계산했습니다. 반면 학습 기반 IDM은 물리 방정식 없이, 연속된 두 프레임의 시각적 변화만 보고 "어떤 행동이 이 변화를 만들었는가"를 데이터로부터 학습합니다.

로봇 스펙을 몰라도 되고, 사람 손 영상에도 적용할 수 있다는 점에서 확장성이 훨씬 큽니다.

다만 추출되는 action은 정확한 물리량(관절 토크, end-effector 좌표 등)이 아니라 추상적인 표현(latent action)이므로, 실제 로봇 제어에 쓰려면 이를 타겟 로봇 H/W 특성에 맞는 Action을 생성할 수 있도록 학습하는 단계가 추가로 필요합니다.

이 IDM 아이디어를 파운데이션 모델 스케일로 확장한 것이 LAPA입니다.

##### 2-2-3-1. Latent Action Pretraining from Videos (ICLR 2025)

LAPA는 KAIST 서민준 교수(Config Intelligence CEO)와 당시 학생이던 예성현, 장요엘(현 NVIDIA GEAR Lab)을 중심으로, Microsoft Research, NVIDIA와의 공동 연구로 진행되었습니다.

action label 없는 인터넷 비디오에서 추상적인 표현(latent action)을 추출하여 VLA를 사전학습하는 방법론을 최초로 제안한 논문으로, LfO 기반 대규모 pretrain의 시초라고 할 수 있습니다.

> ### 💡 LAPA 저자들의 이후 행보
>
> **서민준** 교수는 현재 KAIST 김재철AI대학원 부교수로 재직하면서, Physical AI 스타트업 **Config Intelligence**를 공동 창업하여 대표를 맡고 있습니다.
>
> Config Intelligence는 LAPA에서 제시한 "인간 행동 데이터에서 로봇 학습 데이터를 만든다"는 아이디어를 산업화한 회사로, 로봇 학습용 인간 행동 데이터를 대규모로 생산·공급하면서 자체 로봇 파운데이션 모델을 개발하고 있습니다.
>
> LAPA의 주저자인 **예성현**과 **장요엘**은 이후 NVIDIA GEAR Lab(Jim Fan, Yuke Zhu가 이끄는 Project GR00T 팀)에 합류하여 world model 연구를 주도하고 있습니다.
>
> 장요엘은 GEAR Lab에서 world model 팀을 리드하며, DreamGen(2025.05) → DreamDojo(2026.02) → DreamZero(2026.02)로 이어지는 연구 시리즈를 예성현과 함께 이끌었습니다.
>
> 특히 DreamZero는 비디오와 action을 동시에 모델링하는 World Action Model(WAM)이라는 새로운 개념을 제시했으며, 젠슨 황이 GTC 2026에서 **NVIDIA GR00T N2**로 명명하며 차세대 로봇 파운데이션 모델로 발표했습니다.
>
> 즉 LAPA에서 시작된 latent action 방법론이, NVIDIA에서 world model과 RFM을 통합하는 World Action Model로 발전한 것입니다.

Human Ego-Centric 비디오 데이터는 사람으로부터 취득되기 때문에 로봇에 동기화된 Action Label을 갖추기 어렵습니다. LAPA는 이 문제를 정면으로 다룹니다.

> *로봇 action이 뭔지 명시적으로 정의하지 않더라도, 일반적인 비디오로부터 연속된 프레임 사이의 변화 자체를 하나의 action으로 볼 수 있지 않은가?*

핵심 아이디어를 비유하면 이렇습니다. 요리 영상에서 "도마 위에 양파가 있는 장면" 다음에 "양파가 잘려 있는 장면"이 나오면, 그 사이에 "칼로 양파를 썰었다"는 동작이 있었을 것입니다.

LAPA는 AI가 영상의 앞 장면과 뒷 장면을 비교하여 "그 사이에 어떤 동작이 있었을 것이다"를 스스로 추론하게 합니다.

이렇게 추론된 동작 표현을 Latent Action(잠재 동작)이라고 부릅니다. 로봇의 관절 각도나 그리퍼 좌표 같은 정확한 물리량은 아니지만, "이 방향으로 이 정도 크기의 변화가 있었다"는 핵심 정보를 담고 있습니다.

LAPA가 중요한 이유는 데이터에 있습니다. Latent Action은 로봇 없이 촬영한 일반 영상에서도 추출할 수 있기 때문에, 인터넷에 사실상 무한히 존재하는 비디오(Q3)를 사전학습 데이터로 활용할 수 있게 됩니다.

비싼 로봇 Teleoperation 데이터 없이도 대규모 학습이 가능해지는 것입니다.

특히 논문에서 제시한 인상적인 결과는, 인간 비디오만으로 pretrain한 LAPA가 로봇 데이터(Bridge V2)로 사전학습한 OpenVLA를 평균 성공률에서 능가했다는 것입니다.

이는 로봇-인간 간 신체 구조 차이가 큼에도 불구하고, 인간 영상에서 학습한 조작 지식이 로봇으로 효과적으로 전이될 수 있음을 보여주며, Human Ego-centric 비디오 데이터의 대규모 활용 가능성을 시사합니다.

##### 2-2-3-2. DreamDojo (arXiv 2026, NVIDIA)

DreamDojo는 LAPA와 같은 "action label 없는 영상에서 latent action을 추출하여 pretrain"이라는 맥락의 연구이지만, world model 쪽에서 접근합니다.

NVIDIA의 연구팀이 발표했으며, 44,000시간의 Human Ego-centric 비디오를 사전학습에 활용했습니다.

> ### 💡 DreamDojo의 위치
>
> DreamDojo는 앞서 소개한 LAPA의 주저자 **예성현, 장요엘**이 NVIDIA GEAR Lab에 합류한 뒤 주도한 연구로, 44,000시간의 인간 1인칭 비디오를 world model 사전학습에 활용했습니다.
>
> LAPA가 latent action으로 VLA를 만들었다면, DreamDojo는 같은 방법론으로 world model을 만든 것으로, 이후 DreamZero(GR00T N2)에서 RFM와 world model이 통합되는 흐름의 중간 단계에 해당합니다.

LAPA와의 핵심 차이는 두 가지입니다.

첫째, LAPA에서는 프레임 간 변화를 코드북(codebook) 형태의 이산적인 로봇 action(추상적인 latent action)으로 표현했는데, DreamDojo는 이를 연속적인(continuous) 로봇 action으로 확장했습니다.

이산 코드는 "256개 중 하나를 고르는" 방식이라 표현력에 한계가 있지만, 연속 로봇 action은 실수 벡터로 표현되기 때문에 세밀한 조작의 미세한 차이까지 담을 수 있습니다.

둘째, LAPA가 VLA구축을 목적으로 추상적인 latent action의 가능성을 보여주었다면, DreamDojo는 같은 추상적인 latent action 방법론을 world model에 적용하여 "이 action을 취하면 세상이 어떻게 변할까?"를 시뮬레이션하는 모델을 만들었습니다.

즉 latent action이 VLA 뿐 아니라 world model에서도 유효하다는 것을 검증한 셈이며, 이것이 이후 DreamZero에서 비디오와 action을 동시에 모델링하는 World Action Model(WAM)로 통합되는 초석이 됩니다.

---

### 2-3. Post-training (사후학습): 적용 타겟 환경에 적응하고, 스스로 개선하기(Self-improving)

> ### 💡 한 줄 요약
>
> **범용으로 학습된 모델을 실제 타겟 로봇에 맞추고, 실전 경험을 통해 스스로 개선하는 단계. 사람의 교정과 강화학습을 결합하여, 배포 후에도 성능이 계속 좋아지는 자기개선 로봇이 현실화되고 있습니다.**

Pre-training이 "세상을 보고, 언어를 이해하고, 행동을 출력하는, 범용적인 로봇 지능의 기반을 만드는" 단계였다면, post-training은 그 기반을 특정 로봇과 태스크에 맞게 적응(adaptation)시키고, 배포 후 실제 경험을 통해 성능을 개선하는 단계입니다.

예를 들어 pretrain에서 Franka 로봇 팔 데이터로 학습했는데, 실제 배포 대상은 ALOHA 양팔 로봇일 수 있습니다. 관절 구조, 센서 구성, action을 표현하는 방식이 다를 수 있기 때문에 그대로는 동작하지 않습니다.

또한 Behavior Cloning에서 실제 환경에서 VLA 모델의 action으로 만들게 된 state가 학습 데이터와 다를 때 발생하는 분포 이동 문제(설명: → 2-2-1-2)도 post-training 단계에서 해결해야 합니다.

Post-training의 목적은 크게 두 가지로 정리됩니다.

1. **타겟 로봇의 로봇 action 스펙에 맞게 adaptation:** Pretrain에서 구축한 범용 지식(vision-language 이해, 조작 패턴 등)은 유지하면서, 출력층만 타겟 로봇에 맞게 조정하는 것이 핵심
2. **배포 후 자기개선(self-improvement)**: 로봇이 실제 환경에서 작업하면서 쌓는 경험 데이터(성공, 실패, 사람의 개입 등)를 활용하여 VLA 모델을 지속적으로 개선하는 것

post-training에서 사용되는 데이터는 주로 Q1(타겟 로봇의 소규모 Teleoperation 데이터)과 배포 후 수집되는 경험 데이터입니다.

Pre-training이 수십만~수백만 에피소드를 사용했다면, post-training은 수십~수천 에피소드 규모로 훨씬 작습니다.

#### 2-3-1. Behavior Cloning (BC)

가장 기본적인 post-training 방법은 대규모 데이터로 사전학습된 VLA를 타겟 로봇의 소량 데이터로 Behavior Cloning 방식으로 학습하는 것입니다.

기술적으로는 pre-training과 동일한 BC를 사용하되, 데이터와 목적이 다릅니다. Pre-training이 "여러 로봇의 데이터를 합쳐서 범용 action space(행동 공간)을 구축"하는 것이었다면, Post-training은 "특정 로봇의 데이터로 해당 로봇의 action space(행동 공간)에 정밀하게 맞추는" 것입니다.

실무적으로 가장 중요한 설계 결정은 "모델의 어느 부분을 학습시킬 것인가"입니다.

전체 모델을 Post-training하면 타겟 로봇에 잘 맞지만, 소량 데이터로 전체를 학습시키면 pre-training에서 쌓은 범용 지식이 훼손될 수 있습니다(catastrophic forgetting).

반면 action head만 fine-tuning하면 지식 보존은 되지만 adaptation이 불충분할 수 있습니다.

OpenVLA(설명: → 1-3-2. OpenVLA (arXiv 2024))는 이 문제를 LoRA(Low-Rank Adaptation)로 해결했습니다. 모델 전체의 가중치를 바꾸는 대신, 각 layer에 작은 저랭크(low-rank) 행렬을 추가하여 이것만 학습시킵니다.

원래 가중치는 고정되므로 pretrain 지식이 보존되면서도, 저랭크 행렬이 타겟 로봇에 맞는 미세 조정을 담당합니다.

#### 2-3-2. DAgger (Dataset Aggregation) (AISTATS 2011)

DAgger는 당시 Carnegie Mellon University의 Machine Learning Department에 있던 Stéphane Ross가 주저자로, 그의 지도교수인 J. Andrew Bagnell과 함께 발표한 알고리즘입니다.

핵심은 단순합니다. Behavior Cloning의 분포 이동 문제를 "학습 루프 안에서 전문가를 계속 개입시키는 것"으로 해결하자는 것입니다.

> ### 💡 J. Andrew Bagnell과 Aurora Innovation
>
> Carnegie Mellon University **J. Andrew Bagnell** 교수는 CMU Robotics Institute에서 로봇 학습과 구조화된 예측을 오랫동안 연구해 온 인물로, 이후 Google 자율주행 프로젝트(현 Waymo)의 전 리더 Chris Urmson과 함께 자율주행 스타트업 **Aurora Innovation**을 공동 창업했습니다.
>
> Aurora는 2021년 SPAC을 통해 나스닥에 상장되었고, 대형 트럭 자율주행(L4)에 집중하고 있습니다.

BC의 문제를 다시 짚으면, 학습 데이터는 전문가가 방문한 state에서만 수집되는데, 배포 시 VLA는 전문가와 다른 state를 방문하게 됩니다. DAgger는 이 gap을 "그러면 VLA 모델이 실제로 방문하는 state에서도 전문가 라벨을 받자"라는 직관으로 해결합니다.

"dataset을 aggregation(누적)한다”를 목표로 매 라운드마다 VLA 모델이 실제로 만드는 state 분포에서 전문가 라벨을 추가로 수집하고, 이전 데이터와 합쳐서 재학습합니다. 이렇게 하면 VLA 모델이 방문하는 state와 학습 데이터의 state 분포 사이의 gap이 점점 줄어듭니다.

하지만 실용적 한계가 큽니다. 매 라운드마다 로봇을 실제로 실행시켜야 하고, 그때마다 전문가(사람)가 옆에서 라벨을 달아줘야 합니다. 로봇 수백 대를 배포하는 상황에서 매번 사람이 개입하는 것은 현실적으로 불가능합니다.

후속 연구인 HG-DAgger(Human-Gated DAgger) (ICRA 2019)는 전문가가 모든 state에 라벨을 다는 대신, VLA 모델이 실행되는 것을 모니터링하다가 위험하거나 잘못된 행동을 할 때에만 조종권을 가져와서(takeover) 직접 시범을 보이는 방식을 제안합니다.

"항상 가르치는 선생님"에서 "지켜보다가 필요할 때만 개입하는 선생님"으로 전환한 셈입니다. 개입이 발생한 구간의 (state, expert_action) 쌍만 수집하므로 라벨링 비용이 크게 줄어듭니다.

#### 2-3-3. Advantage 기반 RL (RECAP): Physical Intelligence $\pi_{0.6}^*$

##### 2-3-3-1. $\pi_{0.6}^*$ (arXiv 2025, Physical Intelligence)

Physical Intelligence가 $\pi_0$, $\pi_{0.5}$에 이어 발표한 모델로, VLA 분야에서 "배포된 모델이 실제 경험을 통해 자기개선할 수 있는가?"를 대규모로 실증한 첫 번째 사례입니다.

$\pi_{0.6}^*$의 핵심 기법인 RECAP (RL with Experience and Corrections via Advantage-conditioned Policies)은 DAgger 계열의 현대적 확장이면서, 동시에 RL의 프레임워크를 취합니다.

작동 방식은 다음과 같습니다.

1. **Coaching with correction(교정을 통한 지도)**
    - 로봇이 자율 실행 중 실수를 하면, 사람이 Teleoperation으로 개입하여 교정해 주는 것입니다. 앞에서 다룬 DAgger(설명: → 2-3-2. DAgger (Dataset Aggregation) (AISTATS 2011))와 동일한 구조입니다.
    - 로봇이 실제로 만드는 state에서 전문가 시범을 받기 때문에, BC의 근본적 한계였던 분포 이동 문제를 직접 해결합니다.
    - 다만 사람의 교정만으로는 한계가 있습니다. 명확한 실수는 교정할 수 있지만, 속도를 높이거나 미세한 동작의 디테일을 개선하는 것까지 사람이 일일이 가르치기는 어렵습니다.
2. **Practicing with reinforcement**(강화 방식으로 연습)
    - 사람이 개입하지 않는 구간에서, 에피소드 단위로 "이 시도는 성공이었나 실패였나"를 판단하고, 성공한 행동 패턴은 강화하고 실패한 행동 패턴은 억제합니다.
    - 이 과정에서 각 trajectory에 advantage(이점) 값을 부여하고, 이 advantage를 conditioning token으로 넣어 학습합니다.
    - 추론(inference) 시에는 항상 "높은 advantage"를 조건으로 넣어서 성공 패턴의 행동을 생성하도록 유도합니다.

> ### 💡 Practicing with reinforcement의 핵심: Value function과 Advantage
>
> 로봇이 테이블 위의 물건을 정리하는 과정에서 수십 개의 동작을 수행합니다.
>
> 최종적으로 실패했을 때, "도대체 어느 동작이 문제였는가?"를 찾아내는 것이 핵심입니다. 이것을 RL에서는 **Credit Assignment(기여도 할당)** 문제라고 부릅니다. 이를 해결하기 위해 두 가지 도구를 사용합니다.
>
> - **Value Function(가치 함수)**은 "지금 상황이 얼마나 유리한가"를 예측하는 모델입니다. 바둑에 비유하면, 현재 판세에서 승률이 몇 %인지를 실시간으로 계산하는 AI 해설과 같습니다.
> - **Advantage(이점)**는 특정 동작 전후로 이 승률이 얼마나 변했는지를 나타냅니다. 예를 들어 로봇이 컵을 잡다가 미끄러뜨리는 동작 이후 성공 확률이 80%에서 30%로 떨어졌다면, 해당 동작에 낮은 Advantage 점수(-50)가 부여됩니다. 반대로 컵을 안정적으로 집어 올린 동작은 높은 점수를 받습니다.
>
> 학습 시에는 각 동작에 이 점수를 태그처럼 붙여서 "이 상황에서 높은 점수를 받은 동작은 이것이었다"를 학습합니다.
>
> 추론(실제 운용) 시에는 항상 "최고점" 모드로 실행하여, 성공 패턴의 동작만 재현하도록 유도합니다.

RECAP으로 학습한 $\pi_{0.6}^*$은 배포 환경에서 세탁물 접기(50종의 새로운 의류, 새로운 가정 환경), 박스 조립 및 라벨링(초콜릿 포장용 59개 박스, 실제 공장), 에스프레소 만들기(오전 5시 30분부터 오후 11시 30분까지 연속 운용)를 수행했습니다.

가장 어려운 태스크에서는 처리량(throughput)과 성공률이 모두 2배 이상 개선되었고, 박스가 겹쳐 올라왔을 때 여분을 내려놓고 다시 시작하거나, 접다가 실패한 옷을 펴고 다시 접는 행동 등 BC만으로는 불가능했던 에러 회복이 자연스럽게 나타났습니다.

#### 2-3-4. Reward Model 기반 RL: DYNA-1

Physical Intelligence $\pi_{0.6}^*$의 특징 알고리즘인 RECAP이 성공/실패를 기준으로 advantage 점수를 산정해 학습에 반영하는 방식으로 자기개선을 구현했다면, 또 다른 RL의 접근법인 Reward model(보상 모델)을 학습하여 Post-training 학습 루프 안에 배치하는 방식도 있습니다.

##### 2-3-4-1. DYNA-1 (2025, Dyna Robotics)

> ### 💡 Dyna Robotics
>
> Dyna Robotics는 Caper AI를 <span class="tex2jax_ignore">&#36;350M</span>에 매각한 창업자 Lindon Gao(CEO)와 전 Google DeepMind 연구원 Jason Ma가 공동 창업한 Physical AI 스타트업입니다.
>
> "Dyna"라는 사명은 Richard Sutton이 1991년에 제안한 Dyna 아키텍처에서 가져온 것으로, Sutton의 Dyna는 "실제 경험으로 environment model을 학습하고, 그 model 안에서 가상 경험(imagination)을 생성하여 VLA 모델을 추가 학습하는" model-based RL의 원형입니다.
>
> 2025년 3월 CRV, First Round Capital 공동 리드로 <span class="tex2jax_ignore">&#36;23.5M</span> Seed 라운드를 마감한 뒤, 같은 해 9월 RoboStrategy, CRV, First Round Capital 리드로 <span class="tex2jax_ignore">&#36;120M</span> Series A를 클로징했으며, Salesforce Ventures, NVentures(NVIDIA), Amazon Industrial Innovation Fund, Samsung Next 등이 참여했습니다.

DYNA-1은 closed model이기 때문에 아키텍처와 학습 방법론의 전체 디테일은 공개되어 있지 않습니다.

다만 공동 창업자 Jason Ma의 연구와 공개된 테크 블로그를 종합하면, 자체 개발한 foundation reward model을 post-training 학습 루프의 중심에 상시 배치하여 reward model의 피드백으로 VLA 모델을 반복 개선하는 구조, 즉 RM-in-the-loop training이 핵심 접근인 것으로 추정됩니다.

> ### 💡 Foundation reward model이란?
>
> 로봇 RL에서 reward function은 "로봇이 지금 잘 하고 있는가?"를 판단하는 함수입니다.
>
> 전통적으로 이 reward function은 태스크마다 사람이 수작업으로 설계해야 했습니다. "냅킨의 네 꼭짓점이 일정 거리 이내로 모이면 +1" 같은 규칙을 엔지니어가 직접 코딩하는 것입니다.
>
> 태스크가 바뀌면 처음부터 다시 설계해야 하고, 복잡한 태스크일수록 규칙을 정의하기가 매우 어렵습니다.
>
> LLM 분야에서 "foundation model"이 하나의 모델로 다양한 언어 태스크를 수행하듯, foundation reward model은 하나의 모델로 다양한 로봇 태스크의 progress를 범용적으로 평가할 수 있는 reward model을 뜻합니다.

Foundation reward model은 인터넷 상의 대규모 인간 비디오로부터 학습한 VLM(Vision Language Model)을 활용하여, 로봇의 현재 상태가 task 완료 상태(goal)에 얼마나 가까운지를 reward로 출력하는 방식으로 구현됩니다.

Goal까지의 시간적 거리(temporal distance)가 가까울수록 높은 reward를 부여하므로, reward가 높으면 task가 원활히 진행되고 있다는 뜻이고, 낮으면 그렇지 않다고 볼 수 있습니다.

사전학습된 VLM은 "물체를 집는 동작", "냅킨을 접는 과정" 같은 작업의 시각적 진행 단계에 대한 사전 지식(world knowledge)을 보유하고 있기 때문에 사람 작업 영상이 로봇 task progress 평가하는데 전이될 수 있게 되는 것이 특징입니다.

DYNA-1은 Foundation reward model을 post-training 학습 루프에 배치하여 VLA 모델이 reward를 높이는 쪽으로 학습이 되게 하며, 실패 케이스를 자동으로 reward model 개선에 반영(online adaptation)하여 더 나은 학습을 할 수 있게 하는 closed-loop을 구성하는 것이 핵심입니다.

> ### 💡 Physical Intelligence와 Dyna Robotics의 전략적 차이
>
> $\pi_{0.6}^*$와 DYNA-1은 "배포 후 경험으로 자기개선"이라는 같은 철학을 공유하지만, 지향하는 방향이 근본적으로 다릅니다.
>
> Physical Intelligence는 다양한 embodiment — 모바일 매니퓰레이터, 양팔 로봇, 고정형 로봇 등 — 에서 범용적으로 작동하는 generalist를 지향합니다. 하나의 모델이 세탁물 접기, 박스 조립, 에스프레소 만들기, 침대 정리, 식기 정리 등 다양한 태스크를 수행하며, embodiment가 바뀌어도 같은 모델이 작동합니다.
>
> 반면 Dyna Robotics는 저비용 정지형 양팔 로봇이라는 하드웨어를 고정하고, 태스크별 상용 성능(99%+ 성공률, 24시간 연속 운용, 상용 품질 기준 충족)을 극대화하는 데 집중합니다.
>
> 전자는 "하나의 모델, 다양한 로봇과 태스크"이고, 후자는 "하나의 로봇, 태스크별 최고 성능"입니다.
>
> 다만 현재까지 DYNA-1의 공개된 성과는 주로 냅킨 접기, 빨래 접기, 컵 채우기 같은 정지형 양팔의 경량 dexterity 작업에 집중되어 있습니다. 높은 payload가 요구되거나 태스크 복잡성이 높은 산업 현장 — 제조 조립, 중량물 처리, 복잡한 공구 사용 등 — 에서의 검증은 블로그 상에서 확인이 어렵습니다.
>
> Dyna가 "general-purpose robot"과 "physical AGI"를 궁극적 목표로 내세우고 있지만, 현재의 비즈니스 모델은 use-case를 고정하고 해당 태스크의 성능을 RL로 극대화하는 구조에 가깝습니다.
>
> 이 접근이 다양한 산업 환경과 더 복잡한 태스크로 얼마나 빠르게 확장될 수 있는지가 향후 관건이 될 것입니다.
