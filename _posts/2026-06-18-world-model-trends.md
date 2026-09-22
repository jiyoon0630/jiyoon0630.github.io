---
layout: paper
lang: ko
ref: world-model-trends
kind: tech-review
title: "World Model 기술 현황: 데이터·학습·평가로 번지는 세 가지 트렌드"
date: 2026-06-18 12:00:00 -0700
tags: [World-Model, Robot-Foundation-Model, VLA, Video-Generation, Evaluation, Tech-Review]
summary: "World Model이 어디에서 왔고 어디로 가는지 — 학습 데이터 생성, VLA 학습과의 결합, 그리고 평가 병목이라는 세 갈래로 로봇 개발 파이프라인 전체에 번지고 있습니다."
---

> **핵심 정리** — World Model은 물리 법칙을 사람이 명시적으로 설계하는 대신, 데이터로부터 암묵적으로 학습하는 접근입니다. 최근의 흐름은 하나로 모입니다. **World Model이 로봇 개발 파이프라인의 데이터·학습·평가 전 단계로 침투하고 있다는 것**입니다. 이 글에서는 World Model이 무엇이고 어디에서 왔는지, 그리고 그 세 가지 트렌드를 대표 연구를 통해 정리했습니다.

---

## Introduction: World Model의 부상과 RFM에서 World Action Model로의 흐름

이 글은 앞서 정리했던 **RFM(Robot Foundation Model)의 발전 흐름**에 이어, 최근 Physical AI 분야에서 빠르게 부상하고 있는 **World Model의 발전 흐름과 트렌드**를 정리한 것입니다.

RFM을 정리하면서는 카메라 이미지(Vision)와 자연어 명령(Language)을 입력받아 로봇의 동작(Action)을 직접 출력하는 VLA(Vision Language Action) 모델이 어떻게 발전해왔는지를 살펴보았습니다. 

그리고 그 마지막 흐름으로, VLA가 단순히 action을 출력하는 것을 넘어 물리적 환경 자체를 생성·예측할 수 있는 World Model과 통합되는 방향, 즉 World Action Model로 나아가고 있음을 짧게 언급한 바 있습니다. 

바로 그 지점에서 출발하여, World Model이 무엇이고 어디에서 왔으며 어디로 향하고 있는지를 본격적으로 다뤄보겠습니다.

**World Model이라는 용어는 최근 1~2년 사이 NVIDIA, Google DeepMind, Physical Intelligence 등 빅테크/주요 스타트업들이 일제히 사용하기 시작하면서 로보틱스 분야의 핵심 키워드로 자리잡았습니다.** 

**NVIDIA는 CES 2025에서 Cosmos를 World Foundation Model로 발표하며 Physical AI의 핵심 인프라로 내세웠고, Google DeepMind는 Genie 시리즈로 영상만으로 상호작용 가능한 가상 환경을 생성해냈으며, Physical Intelligence를 비롯한 RFM 기업들도 RFM/VLA 모델에 World Model을 결합하기 시작했습니다.** 

즉 World Model은 일부 연구실의 실험이 아니라, 산업 전반이 동시에 베팅하고 있는 흐름이 되었습니다. 다만 이 용어가 빠르게 확산된 만큼, 그 정의가 회사마다, 적용 분야마다 조금씩 다르게 쓰이고 있어 혼란이 큰 것도 사실입니다. 

어떤 곳에서는 게임처럼 조작 가능한 가상 세계를 생성하는 기술을 World Model이라 부르고, 어떤 곳에서는 로봇 학습용 합성 데이터를 만드는 시뮬레이터를 World Model이라 부르며, 또 어떤 곳에서는 정책과 결합되어 미래를 예측하는 모델을 World Model이라 부릅니다. 

적용되는 application이 다를 뿐 그 뿌리는 하나로 이어져 있지만, 표면적으로는 전혀 다른 기술처럼 보이기 때문에 많은 사람들이 가장 혼란스러워하는 지점이기도 합니다.

따라서 본격적인 발전사를 다루기에 앞서, World Model이 정확히 무엇이며 우리에게 가장 익숙한 Simulation Tool과 어떻게 다른지부터 명확히 정리하고자 합니다. 

덧붙여, 앞서 RFM을 살펴보며 정리했듯 RFM 개발의 핵심은 결국 "데이터를 어떻게 확보하고, 어떻게 학습시키며, 어떻게 평가할 것인가"에 있습니다. 

**World Model은 바로 이 데이터·학습·평가의 세 단계 모두에 걸쳐 활용되고 있는 기술**이라는 점에서, 그 흐름을 이해해 두는 것은 Physical AI 분야의 큰 그림을 파악하는 데 도움이 될 것으로 생각됩니다.

---

## Section 1. Background: World Model이란 무엇인가

### 1-1. World Model vs 물리 엔진 기반 시뮬레이터(Physics-based Simulator)

World Model을 가장 정확하게 이해하는 방법은, 가장 유사하면서도 본질적으로 다른 기술인 물리 엔진 기반 시뮬레이터와 직접 비교하는 것입니다.

- **물리 엔진 기반 시뮬레이터(Physics-based Simulator)**: 물리 엔진 기반 시뮬레이터(MuJoCo, Isaac Sim 등)는 가상 환경을 명시적(explicit)으로 구성합니다.
    - 시뮬레이션 환경 안에는 **사람이 미리 정의한 에셋(asset)**들이 존재합니다. 테이블, 특정 로봇, 조작 대상 물체, 방의 구조 등이 모두 3D 모델로 사전에 설계되어 있습니다.
    - 그리고 이 에셋들이 **서로 어떻게 상호작용하는지를 계산하는 물리 엔진(physics engine)**이 필요합니다. 로봇이 팔을 움직였을 때 주변 물체가 어떻게 밀리고 부딪히는지를, 질량·마찰·접촉 같은 물리 법칙을 명시적으로 계산하여 환경에 반영하는 것입니다.
- **World Model 기반 시뮬레이터(World Model-based Simulator)**: World Model은 물리를 암묵적(implicit)으로 다룹니다. 물리 법칙을 사람이 디테일하게 설계하는 것이 아니라, **데이터로부터 학습(data-driven)**하도록 하는 접근입니다.
    - AI 모델이 대량의 영상에서 프레임 간의 변화를 보고 **"이 상황에서 이 행동을 하면 다음 장면은 이렇게 변한다"를 직접 배우는 것**입니다.
    - 물체가 떨어지고, 액체가 흐르고, 문이 열리는 등의 dynamics가 인터넷 영상 속에 이미 암묵적으로 담겨 있으며, 충분히 큰 생성 모델은 이를 명시적인 물리 supervision 없이도 학습할 수 있다는 것이 핵심 전제입니다.

![World Model vs 물리 엔진 시뮬레이터](/assets/img/notes/world-model/world-model-vs-simulator.jpg)
*World Model과 물리 엔진 기반 시뮬레이터의 대비 (출처: [arXiv:2507.00917](https://arxiv.org/abs/2507.00917))*

이 차이가 중요한 이유는 **“sim2real gap”** 때문입니다. 

**물리 엔진 기반 시뮬레이터는 현실의 물리를 근사하기 때문에, 시뮬레이션에서 학습한 정책이 실제 로봇에 배포되었을 때 시뮬레이션과 현실의 불일치로 인해 성능이 저하되는 문제가 구조적으로 발생합니다.** 

이는 시뮬레이션 기반 로봇 학습의 가장 지배적인 실패 모드로 알려져 있습니다. 케이블 정리, 변형되는 물체 조작, 다양한 표면에서의 마찰 같은 dynamics는 물리 엔진으로 정확히 모사하기가 매우 까다롭습니다.

World Model은 실제 데이터로 학습되기 때문에 현실의 dynamics와 본질적으로 더 가깝다는 점에서 **sim2real gap을 줄일 수 있는 대안**으로 주목받고 있습니다. 

**정리하면, 물리 엔진 기반 시뮬레이터가 "사람이 세계를 명시적으로 설계하는" 방식이라면, World Model은 "AI가 세계를 데이터로부터 암묵적으로 학습하는" 방식이라고 할 수 있겠습니다.**

### 1-2. Preliminary: 알아두면 좋은 배경지식

본격적인 발전사를 다루기 전에, 반복적으로 등장하는 배경지식과 핵심 용어를 미리 정리하겠습니다.

#### 1-2-1. NVIDIA 시뮬레이션 스택의 세 층위 (Omniverse · Isaac Sim · PhysX)

로보틱스 시뮬레이션을 논할 때 Omniverse, Isaac Sim, PhysX가 자주 함께 언급되지만, 이 셋은 같은 층위의 경쟁 도구가 아니라 서로 다른 추상화 레벨에 위치한 기술들입니다. 

현업에서 자주 섞어 쓰는 용어들이므로, NVIDIA 생태계 기준으로 그 관계를 정리할 필요가 있습니다.

![NVIDIA 시뮬레이션 스택](/assets/img/notes/world-model/nvidia-sim-stack.jpg)
*NVIDIA 시뮬레이션 스택 (출처: [NVIDIA Developer Blog](https://developer.nvidia.com/blog/design-your-robot-on-hardware-in-the-loop-with-nvidia-jetson/))*

1. **Omniverse — 기반 플랫폼(개발 OS)**
    - **digital twin과 physical AI 시뮬레이션 애플리케이션을 개발하기 위한 라이브러리·마이크로서비스 모음**입니다.
    - USD(Universal Scene Description) 기반의 씬 구성과 RTX 기반 photorealistic 렌더링을 제공하는, 가장 아래를 받치는 기반 플랫폼입니다.
    
    > ### 💡 Digital Twin (NVIDIA 생태계 정의 기준)
    >
    > Digital Twin은 현업에서 많이 쓰는 단어이지만, NVIDIA 생태계에서는 두 가지 의미로 쓰이므로 구분이 필요합니다.
    >
    > - **본래 의미:** 실제 환경(공장·창고 등)을 OpenUSD 기반으로 Omniverse 안에 구축한 물리 정확 가상 복제본입니다. 즉 사람이 **명시적으로 자산과 물리를 정의한 explicit 복제본**이며, 1-1에서 말한 '시뮬레이터' 계열에 속합니다.
    > - **확장된 용법:** NVIDIA의 Cosmos 논문은 World Foundation Model을 '세계의 digital twin', VLA model을 '자기 자신의 digital twin'이라고 명명합니다. 이때의 digital twin은 명시적 복제본이 아니라 **implicit하게 학습된 World Model 자체**를 가리킵니다.
    >
    > 같은 'digital twin'이 명시적인 시뮬레이터와 World Model 양쪽을 가리키므로 문맥에 따른 구분이 필요합니다.
    
2. **Isaac Sim — 시뮬레이션 애플리케이션**
    - **Omniverse 위에 구축된 로보틱스 시뮬레이션 플랫폼**입니다.
    - 고정밀 물리(PhysX 엔진)와 photorealistic 렌더링을 제공하며, 합성 데이터 생성(synthetic data generation)과 테스트·검증에 초점을 둡니다.
    - digital twin 안에서 로봇을 학습·테스트하는 응용 계층에 해당합니다.
3. **PhysX / Mujoco / Newton — 물리 엔진**
    - **가장 밑단에서 실제 물리(강체·연성체의 dynamics)를 계산하는 엔진**입니다.
    - MuJoCo는 가볍고 접촉 물리(contact physics)가 정확하여 학계의 표준으로 자리잡았으며, Isaac Sim 안의 엔진으로 통합될 수도 있고 단독으로 사용될 수도 있습니다.

세 층위의 관계를 정리하면 다음과 같습니다.

```
NVIDIA 시뮬레이션 스택 (상위 → 하위)

▸ [기반 플랫폼] Omniverse
    digital twin 구축 · USD 씬 · RTX 렌더링

  ▸ [시뮬레이션] Isaac Sim
      Omniverse 위 로보틱스 시뮬레이션 앱
      합성데이터 생성 · 학습 · 테스트

    ▸ [물리 엔진] PhysX / MuJoCo / Newton ...
        강체 · 연성체 dynamics 계산 (교체 가능)

※ MuJoCo는 Isaac Sim의 백엔드로 들어갈 수도 있고, 단독으로도 쓰이는 독립 물리 엔진 (가볍고 접촉 물리 정확)
```

여기서 짚어둘 점은, 이 세 층위는 **물리 엔진 기반 시뮬레이터(Physics-based Simulator)**에 속한다는 것입니다.

자산과 물리를 사람이 명시적으로 정의한다는 점에서, 셋 모두 World Model(implicit)과 대비되는 시뮬레이터 진영에 위치한다고 할 수 있습니다.

#### 1-2-2. 핵심 용어 사전

본문에서 반복 등장하는 World Model 관련 핵심 용어를 미리 정의합니다. 

- **Neural Trajectory**
    - **World Model이 생성한 합성 로봇 비디오(synthetic robot video) 데이터**를 뜻합니다.
    - "neural"은 신경망이 생성했다는 의미이고, "trajectory"는 로봇의 작업 궤적 영상을 의미합니다.
    - 실제 teleoperation 없이 World Model로 만들어낸 학습용 데이터입니다.
- **Rollout**
    - **VLA/World Model이 작업을 처음부터 끝까지 한 번 수행하면서 생긴 (관측, action)의 시퀀스 전체, 즉 한 번의 시도(에피소드) 전체 기록**을 뜻합니다.
    - 실제 로봇으로 수행한 것은 real-world rollout, World Model이 만든 상상 공간(imagination) 안에서 수행한 것은 imagined(synthetic) rollout으로 구분됩니다.
- **World Foundation Model (WFM)**
    - **특정 도메인에 국한되지 않는 범용 World Model을 뜻합니다.**
    - LLM의 foundation model과 같은 맥락으로, 하나의 범용 모델을 학습해 두고 downstream 용도(특정 로봇·환경)에 맞게 fine-tune하는 것을 전제로 합니다. NVIDIA Cosmos가 대표적입니다.
- **World Action Model (WAM)**
    - **미래의 world state(비디오)와 action을 함께(jointly) 예측하는 모델**을 뜻합니다. (= VLA + World Model)
    - 기존 VLA가 action만 출력하는 것과 달리, 비디오 생성과 action 생성을 하나의 모델에서 동시에 다룬다는 점이 핵심입니다.
    - World Model과 VLA의 통합을 상징하는 키워드입니다. NVIDIA 최신 모델인 DreamZero(26.02)가 대표적입니다.

### 1-3. 세 가지 World Model 트렌드

World Model의 발전사를 살펴보면, 결국 하나의 큰 흐름으로 수렴합니다. 바로 **World Model이 로봇 개발 파이프라인의 전 단계로 침투하고 있다는 것**입니다. 

로봇 Action을 생성하는 RFM 모델(= 로봇 Policy 모델)을 개발하는 과정은 크게 **(1) 데이터 확보, (2) 학습, (3) 평가**의 세 단계로 나눌 수 있습니다. 

![로봇 Policy 개발 파이프라인](/assets/img/notes/world-model/policy-pipeline.jpg)
*로봇 Policy 개발 파이프라인의 세 단계와, World Model이 각 단계로 번져가는 방향*

1. **[트렌드1. 데이터] World Model로 학습 데이터를 만든다 (Section 3)**
    - World Model이 합성 비디오·rollout을 생성하여 VLA 학습 데이터를 공급하는 흐름입니다.
2. **[트렌드2. 학습] World Generation이 VLA 구조에 결합된다 (Section 4)**
    - World Model의 비디오 생성 능력이 VLA 자체의 학습·추론 구조에 직접 결합되는 흐름이며, World Action Model이 그 정점에 있습니다.
3. **[트렌드3. 평가] World Model이 평가 병목을 푼다 (Section 5)**
    - 아직 초기 단계이지만, 로봇 정책 평가의 고질적 병목을 World Model로 자동화하려는 흐름입니다.

이 세 가지는 각각 독립적인 것이 아니라, World Model이라는 하나의 기술이 로봇 개발의 모든 단계로 번져 나가는 과정으로 이해하는 것이 적절합니다. 

이어지는 Section 2에서는 이러한 트렌드를 가능하게 한 기술적 토대를, Section 3~5에서는 세 가지 트렌드를 각각의 대표 연구를 통해 살펴보겠습니다.

> ### 💡 로봇 Policy 모델
>
> 로봇이 현재 보고 느끼는 것(카메라 이미지, 관절 센서)을 입력받아, 다음에 취할 행동(관절 움직임)을 출력하는 의사결정 모델로 강화학습에서 유래되었습니다.

특히 World Model을 로봇 개발 파이프라인 안에 넣으려는 움직임은 빅테크 Robotics Lab(NVIDIA GEAR Lab, Meta, Google DeepMind)부터 Robot Learning 분야의 주요 스타트업(Physical Intelligence, Generalist AI), 그리고 신생 스타트업까지 폭넓게 관찰됩니다. 

---

## Section 2. World Model을 떠받친 기술 축

World Model이 최근 급부상한 것은 어느 한 연구의 돌파구 때문이 아니라, 서로 다른 분야에서 발전해 온 세 갈래의 기술이 한 곳으로 수렴했기 때문입니다. 

이 장에서는 World Model의 발전을 떠받친 세 가지 기술 축을 정리합니다. 

하나는 **World Model이라는 개념 자체의 계보(RL → 게임 → 로봇)**이고, 다른 두 축은 이 계보에 지속적으로 기술적 근간이 된 **Video Generation 기술**과 **Inverse Dynamics Model(IDM) 기술**입니다.

### 2-1. World Model의 발전: “강화학습(RL) → 게임 → 로봇”

World Model이라는 용어의 기원은 의외로 로보틱스가 아니라 강화학습(RL)입니다.

2018년 강화학습 연구자 David Ha와 Jürgen Schmidhuber가 발표한 "World Models" 논문이 이 용어를 처음으로 정립했습니다. 

핵심 아이디어는, **에이전트가 실제 환경과 직접 상호작용하는 대신 환경의 dynamics를 학습한 내부 모델("world model")을 만들고, 그 안에서 마치 꿈을 꾸듯(in imagination) 정책을 학습한 뒤 실제 환경으로 전이한다**는 것이었습니다. 

흥미로운 점은, 이 논문이 그 아이디어를 처음 실증한 무대가 자동차 레이싱과 2D 슈팅 같은 게임 환경이었다는 것입니다. 즉 World Model은 출발부터 게임과 깊이 얽혀 있었습니다.

> ### 💡 강화학습(RL) 기본 용어
>
> World Model이 RL에서 출발했기에, 이 글에는 RL 용어가 종종 등장합니다.
>
> - **에이전트(agent):** 스스로 판단하여 행동하는 주체입니다. 로봇이 대표적인 에이전트입니다.
> - **환경(environment):** 에이전트가 행동을 가하는 대상 세계입니다. 로봇이 놓인 작업 공간, 게임 화면 등이 이에 해당합니다.
> - **정책(policy):** 에이전트가 "현재 상황에서 어떤 행동을 할지"를 결정하는 의사결정 모델입니다. 로봇의 action을 출력하는 VLA 모델이 곧 정책(policy)에 해당합니다.

이 RL 뿌리에서 두 갈래의 응용이 갈라져 나옵니다.

#### 2-1-1. 게임 갈래 — Google DeepMind Genie
Google DeepMind가 2024년 발표한 Genie는 라벨이 없는 인터넷 게임 영상만으로 학습하여, 단 한 장의 이미지 프롬프트로부터 조작 가능한(action-controllable) 가상 환경을 생성해냈습니다. 

이후 Genie 2는 이를 3D로 확장하여, 키보드·마우스 입력에 반응하는 플레이 가능한 환경을 만들어냈습니다.

Genie의 핵심은 **latent action**입니다. **영상에 action 라벨이 전혀 없음에도, 프레임 간 변화로부터 "어떤 조작이 이 변화를 만들었는가"를 스스로 추론하여 조작 인터페이스를 학습합니다.** 

이 latent action 개념은 뒤에서 다룰 로봇 갈래(LAPA, DreamGen)와 정확히 같은 메커니즘이며, 게임과 로봇이라는 전혀 다른 두 응용을 잇는 연결고리입니다.

> ### 💡 Latent Action
>
> 일반적인 인터넷 영상에는 "이 장면에서 어떤 버튼을 눌렀는지, 로봇 관절을 얼마나 움직였는지" 같은 action 라벨이 붙어 있지 않습니다.
>
> **Latent Action은 이런 라벨 없이도, 연속된 두 프레임의 변화만 보고 "그 사이에 어떤 행동이 있었을 것"이라는 점을 추상적인(latent, 잠재적인) 형태로 스스로 정의해낸 것입니다.**
>
> 실제 버튼·관절 값을 모르더라도, 변화를 일으킨 행동을 모델 내부의 표현으로 인코딩하는 것이라 이해하면 됩니다.
>
> 이 개념 덕분에 action 라벨이 없는 방대한 영상을 학습에 활용할 수 있게 되며, 그 구체적인 작동 방식은 이후 LAPA 논문과 함께 자세히 다루겠습니다.

#### 2-1-2. 로봇 갈래 — NVIDIA Cosmos
로봇 분야에서 World Model이 부상한 동기는 명확합니다. **기존 물리 엔진 기반 시뮬레이터의 sim2real gap이 너무 크기 때문에, 현실 세계를 데이터로 학습하여 모사해보자**는 것입니다.

NVIDIA는 CES 2025에서 Cosmos를 World Foundation Model(WFM)로 발표하며 이 흐름을 주도하고 있습니다. Cosmos는 Physical AI를 위한 범용 World Model로, 도메인별 fine-tune을 전제로 한 foundation model입니다.

> ### 💡 Cosmos 라인업 (Predict / Transfer / Reason)
>
> Cosmos는 2.5 세대까지 단일 모델이 아니라 역할이 다른 모델군으로 구성됩니다. World Model 주제와 직접 관련된 것은 Predict이지만, 나머지도 함께 알아두면 좋습니다.
>
> - **Cosmos Transfer:** 시뮬레이션 영상이나 구조적 입력(depth·segmentation 등)을 photorealistic한 영상으로 변환하는 모델입니다. Sim2Real을 위한 도구입니다.
> - **Cosmos Predict:** 텍스트·이미지·비디오로부터 미래의 world state(비디오)를 생성하는 핵심 World Model입니다.
> - **Cosmos Reason:** 영상과 물리적 상황을 이해·추론하는 Physical AI용 Vision-Language Model입니다.
>
> 이 세 가지는 처음에는 별도 모델이었으나, 최신 Cosmos 3에서 하나의 모델로 통합되었습니다.

정리하면, **World Model은 강화학습에서 시작하여 게임(Genie)과 로봇(Cosmos)이라는 두 응용으로 갈라졌지만, 적용되는 application이 다를 뿐 그 뿌리는 "데이터로부터 세계의 dynamics를 학습한다"는 하나의 개념으로 이어져 있다**고 할 수 있겠습니다.

### 2-2. Video Generation의 발전과 World Model로의 흡수

World Model의 발전을 이야기할 때 빼놓을 수 없는 것이 Video Generation 기술입니다. 

**World Model은 결국 "다음 장면이 어떻게 펼쳐질지"를 영상으로 생성하는 기술이므로, Video Generation의 발전이 곧 World Model의 성능으로 직결되기 때문입니다.** 

실제로 최근의 World Model은 당대 최신 Video Generation 기술을 backbone으로 적극 흡수하는 양상을 보입니다.

Diffusion 모델이 발전하면서 이미지 생성을 넘어 비디오 생성으로 확장되었고, 비디오의 품질은 비약적으로 향상되었습니다. 이 과정에서 비디오 생성 아키텍처는 크게 두 가지 구조로 나뉩니다.

#### 2-2-1. DiT-only 구조
**Diffusion Transformer(DiT)를 기반으로, 비디오의 모든 프레임을 한 번에 병렬로 생성하는 방식**입니다. 

모든 프레임이 서로를 동시에 참조하는 full-sequence attention을 사용하기 때문에, 시간적 일관성과 모션 품질이 뛰어납니다.

> ### 💡 Diffusion Transformer(DiT)와 full-sequence attention
>
> - **Diffusion Transformer(DiT):** 비디오·이미지 생성에 쓰이는 diffusion 모델의 한 구조로, 기존에 쓰이던 U-Net 대신 Transformer를 backbone으로 채택한 것입니다. Transformer는 LLM에서 입증된 것처럼 데이터·모델 크기를 키울수록 성능이 안정적으로 향상되는(scalable) 특성이 있어, 대규모 비디오 생성 모델의 표준 구조로 자리잡았습니다. Open Sora 이후 사실상 모든 고품질 비디오 생성 모델이 이 DiT 계열입니다.
> - **Attention:** Transformer의 핵심 연산으로, 출력의 각 요소가 입력의 어느 부분을 얼마나 "참조"할지를 계산하는 메커니즘입니다. 비디오에서는 "지금 생성하는 프레임이 다른 어떤 프레임들을 참조할 것인가"를 결정합니다.
> - **full-sequence attention (= bidirectional/양방향):** 영상의 모든 프레임이 과거·미래를 가리지 않고 서로를 동시에 참조하는 방식입니다. 1번 프레임이 10번 프레임을 참조하고, 10번이 다시 1번을 참조하는 식으로 전 구간이 양방향(bidirectional)으로 얽혀 있습니다. 덕분에 영상 전체의 일관성과 매끄러운 모션을 얻지만, 미래 프레임이 과거에 영향을 주는 구조이기 때문에 "생성 도중 개입"이 원천적으로 불가능합니다.

**OpenAI Sora, Google Veo, Alibaba Wan** 등 우리가 잘 아는 대표적인 고품질 비디오 생성 모델들이 이 구조에 해당합니다.

다만 한계가 있습니다. **모든 프레임이 미래까지 동시에 참조하기 때문에, 일단 생성이 시작되면 전체 시퀀스가 미리 결정되어 버립니다.** 

**즉 중간에 사용자가 개입하여 다음 전개를 바꿀 수 없고, 연산량이 시퀀스 길이의 제곱으로 증가합니다.** 고정 길이의 "완성된 영상"을 만드는 데는 적합하지만, 실시간으로 상호작용하는 데는 부적합합니다.

#### 2-2-2. AR+DiT 구조
**비디오를 chunk 단위로 순차 생성하면서, 각 chunk는 이전(과거) 프레임만 참조하도록 causal attention을 적용하는 방식**입니다. 

KV-cache를 활용하여 효율적으로 긴 영상을 이어 붙일 수 있습니다.

> ### 💡 Autoregressive(AR), causal attention, KV-cache
>
> 세 용어는 모두 **"과거만 보고 순차적으로 생성한다"**는 하나의 원리에서 나온, 서로 맞물린 개념입니다.
>
> - **Autoregressive(AR, 자기회귀):** 앞에서 생성한 결과를 다시 입력으로 받아 다음을 이어서 생성하는 방식입니다. LLM이 단어를 하나씩 순차적으로 뱉어내는 것과 같은 원리로, 비디오에서는 앞 구간(chunk)을 만든 뒤 그것을 보고 다음 구간을 생성합니다.
> - **causal attention (= 단방향):** 앞서 본 full-sequence(bidirectional) attention과 반대로, 각 프레임이 과거 프레임만 참조하고 미래는 보지 못하도록 제한한 방식입니다. "원인이 결과보다 먼저 온다"는 인과(causal) 순서를 강제하는 것으로, 현실의 시간 흐름과 일치하기 때문에 매 순간 들어오는 입력(action)에 따라 다음 전개가 달라질 수 있습니다.
> - **KV-cache:** AR 생성은 매번 과거 전체를 다시 참조해야 하므로, 그대로 두면 길이가 길어질수록 연산이 폭증합니다. KV-cache는 이미 계산해 둔 과거 프레임의 참조 정보(Key·Value)를 저장해 두고 재사용하는 기법으로, 긴 영상을 효율적으로 이어 붙일 수 있게 해줍니다. LLM이 긴 문장을 빠르게 이어 쓰는 데 쓰는 것과 같은 기법입니다.

이 구조의 대표적 전환점이 Diffusion Forcing(NeurIPS 2024)으로, autoregressive 생성(next-token prediction)과 full-sequence diffusion의 장점을 결합했습니다. 

이후 Self-Forcing, Rolling Forcing 등으로 발전하며 긴 영상의 안정적 생성을 가능하게 했습니다.

> ### 💡 왜 World Model에는 AR+DiT 구조가 중요한가
>
> 게임이든 로봇이든, World Model이 "상호작용 가능한 세계"가 되려면 결정적인 조건이 하나 있습니다. **미래가 과거에 영향을 주어서는 안 된다는 것, 즉 시간의 인과성(causality)입니다.**
>
> 사용자나 로봇이 매 순간 action을 입력하면 그에 따라 다음 장면이 달라져야 하는데, 모든 프레임을 동시에 참조하는 DiT-only(bidirectional) 구조는 이미 전체 영상이 결정되어 있어 이런 개입이 불가능합니다.
>
> **따라서 과거만 참조하는 AR+DiT 구조가 World Model에 더 적합하며, "Video Generation 모델을 World Model로 만드는" 핵심 작업은 bidirectional 구조를 causal 구조로 전환하는 것이라고 할 수 있습니다.**

#### 2-2-3. Video World Model로의 수렴
OpenAI Sora가 "video generation models as world simulators(세계 시뮬레이터로서의 비디오 생성 모델)"라는 관점을 제시한 이후, 비디오 생성 기술을 World Model의 backbone으로 삼는 흐름이 본격화되었습니다. 

Google Genie 2/3, Cosmos 등이 모두 이 흐름 위에 있으며, 이렇게 비디오 생성 능력을 갖춘 World Model을 **Video World Model**이라 부릅니다.

핵심은, 고품질 비디오를 생성하는 능력(DiT 계열)과 실시간 상호작용 능력(AR/causal 계열)을 결합하는 것이 곧 좋은 Video World Model을 만드는 길이라는 점입니다. 

뒤에서 다룰 NVIDIA DreamZero가 사전학습된 video diffusion 모델(Alibaba의 Wan)을 적극 활용하는 것도 바로 이 맥락입니다. 

### 2-3. Inverse Dynamics Model(IDM)의 발전과 Latent Action

World Model이 비디오(미래 장면)를 생성할 수 있게 되었다 하더라도, 로봇을 실제로 움직이려면 그 비디오에 대응하는 action이 필요합니다. 

**비디오와 action을 잇는 다리 역할을 하는 것이 바로 Inverse Dynamics Model(IDM)**이며, 이 IDM의 발전이 World Model의 세 번째 기술 축입니다.

> ### 💡 Inverse Dynamics Model(IDM)
>
> IDM은 연속된 두 프레임을 보고 "이 변화를 만든 action은 무엇인가"를 역으로 추론하는 모델입니다.
>
> **전통 로보틱스에서는 로봇의 질량·관절 길이 같은 물리 파라미터로 역동역학 방정식을 풀어 action을 계산했지만, 학습 기반 IDM은 물리 방정식 없이 두 프레임의 시각적 변화만으로 action을 데이터로부터 학습합니다.**
>
> 로봇 스펙을 몰라도 되고 사람 손 영상에도 적용할 수 있어 확장성이 큽니다.

이 IDM 아이디어를 foundation model 스케일로 확장한 대표 연구가 **LAPA(Latent Action Pretraining from Videos, ICLR 2025)**입니다.

**LAPA는 KAIST 서민준 교수(Config Intelligence CEO)와 예성현·장요엘(전 NVIDIA GEAR Lab)을 중심으로 진행된 연구로, action 라벨이 없는 인터넷 비디오에서 latent action을 추출하여 VLA를 사전학습하는 방법론을 최초로 제시했습니다.**

> ### 💡 Latent Action의 핵심 발상
>
> 기존 IDM이 "두 프레임 사이의 action을 추론하는 도구"였다면, LAPA는 이를 뒤집어 "프레임 간 변화 자체를 latent action이라는 새로운 공간으로 정의하겠다"고 선언했습니다.
>
> **즉 로봇 action을 명시적으로 정의하지 않더라도, 일반적인 비디오의 프레임 변화를 하나의 추상적 action으로 인코딩하는 것입니다.**
>
> 이 덕분에 action 라벨이 전혀 없는 인터넷 비디오를 사실상 무제한으로 사전학습에 활용할 수 있게 됩니다.

여기서 주목할 점은, 이 Latent Action이 앞서 2-1에서 본 게임 갈래(Genie)의 latent action과 본질적으로 같은 메커니즘이라는 것입니다. 

**Genie가 게임 영상에서 조작 인터페이스를 latent action으로 학습했다면, LAPA는 로봇·인간 작업 영상에서 로봇 action을 latent action으로 학습한 것**입니다. 

게임과 로봇이라는 두 갈래가 결국 같은 기술 뿌리를 공유하고 있음을 다시 한번 확인할 수 있습니다.

그리고 이 Latent Action / IDM 계보는 앞으로 3장, 4장에서 설명할 트렌드로 직접 이어집니다.

- **비디오 World Model이 미래 장면을 생성하면, IDM 또는 latent action으로 그로부터 action을 복원하여 학습 데이터를 만들 수 있습니다.**
- **나아가 비디오 생성과 action 추론을 하나의 모델에서 동시에 수행하면, World Model과 VLA가 통합된 World Action Model이 됩니다.**

즉 "비디오를 생성하는 능력(2-2)"과 "비디오에서 action을 추출하는 능력(2-3)"이 결합되면서, World Model이 단순한 영상 생성기를 넘어 로봇 개발 파이프라인의 데이터·학습·평가 전 단계로 침투할 수 있는 기술적 토대가 마련되었다고 볼 수 있습니다.

이어지는 Section 3부터는 이 토대 위에서 전개되는 세 가지 트렌드를 차례로 살펴보겠습니다.

---

## Section 3. [트렌드 1. 데이터] World Model로 학습 데이터를 만든다

World Model이 로봇 개발에 침투하는 첫 번째 방식은 가장 직관적입니다. **World Model이 만들어낸 영상을 학습 데이터로 활용하는 것**입니다. 

**RFM 기업들이 공통적으로 겪는 가장 큰 병목이 "실제 로봇으로 데이터를 취득하는 비용"인 만큼, 데이터를 생성으로 대체할 수 있다면 그 가치는 매우 큽니다.**

이 장에서 다루는 연구들의 공통점은 **World Model과 VLA가 서로 분리되어 있고, World Model의 산출물(비디오)을 일단 '데이터'로 만들어 둔 뒤 그 데이터로 VLA를 별도 학습(offline)한다**는 점입니다. 다음 Section 4에서 다룰 "World Generation이 VLA 구조에 직접 결합되는" 방식과는 구분됩니다.

이 분류 기준 안에서, World Model이 어떻게 데이터를 만들어내는지에 따라 세 가지 방식으로 나누어 살펴보겠습니다. 

단순한 외형 변환부터 시작하여, 비디오를 생성한 뒤 action을 복원하는 방식, 그리고 기존 VLA를 상상 공간 안에 넣고 굴리는 방식 순으로 World Model의 개입 정도가 점점 깊어집니다.

### 3-1. Simulation 데이터의 시각적 변환(photoreal): NVIDIA Cosmos Transfer

![NVIDIA Cosmos Transfer](/assets/img/notes/world-model/cosmos-transfer.jpg)
*시뮬레이션 데이터의 시각적 변환 (출처: [arXiv:2503.14492](https://arxiv.org/abs/2503.14492))*

가장 단순한 방식은, **기존 시뮬레이터가 만든 영상의 "겉모습"만 현실적으로 바꿔주는 것**입니다.

물리 엔진 기반 시뮬레이터(Isaac Sim 등)는 물리 계산은 정확하지만, 렌더링된 영상이 실제 카메라 영상과는 차이가 큽니다. 이 시각적 차이가 sim2real gap의 큰 부분을 차지합니다. 

**Cosmos Transfer는 이 지점을 공략합니다. 시뮬레이션 영상이나 구조적 입력(depth·segmentation map 등)을 입력 받아, 동일한 장면을 photorealistic한 영상으로 변환해줍니다.**

여기서 중요한 점은, 이 방식에서는 action을 World Model이 만들지 않는다는 것입니다. **action은 이미 시뮬레이터가 가지고 있던 원본 그대로이고, World Model(Cosmos Transfer)은 영상의 외형만 현실적으로 바꿔주는 역할에 한정**됩니다. 

**즉 "시뮬레이터가 만든 (action, 영상) 쌍에서, 영상의 시각적 사실성만 끌어올려 학습 데이터의 품질을 높이는" 가장 보수적인 형태의 데이터 생성이라고 할 수 있습니다.**

### 3-2. 비디오 생성 후 IDM으로 action 복원: NVIDIA DreamGen (CoRL 2025)

![DreamGen 데이터 생성 흐름](/assets/img/notes/world-model/dreamgen-flow.jpg)
*DreamGen — 비디오를 생성한 뒤 IDM으로 action을 복원하는 데이터 생성 흐름*

다음 방식은 한 걸음 더 나아갑니다. **World Model이 영상 자체를 새로 생성하고, 그 영상으로부터 action을 복원하여 학습 데이터를 만드는 것**입니다. 

여기서 생성된 비디오와 action을 잇는 것이 바로 2-3에서 설명한 IDM입니다. 대표 연구는 **NVIDIA GEAR Lab의 DreamGen(CoRL 2025)**입니다.

핵심 흐름은 다음과 같습니다.

1. video world model(NVIDIA Cosmos Predict 사용)을 타겟 로봇에 맞게 소량의 데이터로 학습시킵니다.
2. 초기 프레임과 언어 지시만 주어 대량의 합성 로봇 비디오, 즉 neural trajectory를 생성합니다. (→ 1-2-2 핵심 용어 사전)
3. 생성된 비디오에는 action 라벨이 없으므로, IDM 또는 latent action model로 pseudo-action을 복원합니다.
4. 이렇게 만들어진 (영상, pseudo-action) pair 데이터로 VLA를 학습합니다.

여기서 주목할 점은, **비디오를 생성하는 World Model과 action을 복원하는 IDM이 분리된 외부 모듈이며, 그 산출물이 어디까지나 별도 VLA를 학습시키기 위한 '데이터'라는 것입니다.** 

학습이 끝난 뒤 로봇을 실제로 움직이는 것은 학습된 VLA이고, World Model은 데이터를 만들어 건네준 뒤 배포 단계에서는 빠집니다.

> ### 💡 DreamGen이 보여준 것
>
> **DreamGen의 의미는 "teleoperation 없이도 새로운 동작·환경에 대한 일반화 데이터를 만들 수 있다"는 것을 실증**했다는 데 있습니다.
>
> 단 하나의 작업(pick-and-place)에 대한 실제 데이터만으로 video world model을 적응시킨 뒤, **사람이 한 번도 시연하지 않은 새로운 동작들의 합성 데이터를 생성하여 로봇이 그 동작들을 수행하게 만들었습니다.**
>
> 데이터 취득 비용이라는 RFM의 가장 큰 병목을 World Model로 우회할 수 있음을 보여준 사례입니다.

### 3-3. Policy-in-the-loop: 기존 VLA를 상상 공간에서 굴려 데이터를 만든다

세 번째 방식은 World Model의 개입이 가장 깊습니다. 

**이미 학습된 기존 VLA를 World Model이 만든 상상 공간(imagination) 안에 집어넣고, 마치 실제처럼 작업을 수행하게 한 뒤, 그 결과(rollout)를 데이터로 활용하는 것**입니다. (→ 1-2-2 핵심 용어 사전) 이를 **policy-in-the-loop**라고 부릅니다.

> ### 💡 policy-in-the-loop와 closed-loop control
>
> 실제 로봇은 **closed-loop control 방식**으로 동작합니다. **관측하고(보고) → action을 내고(움직이고) → 다시 관측하고 → 다음 action을 내는 과정을 매 순간 반복**하는 것입니다. (open-loop은 모든 움직임을 한번에 관측한 뒤 action 수행)
>
> **policy-in-the-loop는 바로 이 closed-loop control을 World Model이 만든 상상 공간 안에서 재현**하는 것입니다.
>
> **VLA가 action을 내면 World Model이 그 결과로 바뀐 미래 관측을 생성하고, 그 관측을 다시 VLA에 돌려주어 다음 action을 내게 합니다.** 단지 "다시 보는" 관측이 실제 카메라 영상이 아니라 World Model이 생성한 영상일 뿐입니다.
>
> 앞서 3-2(DreamGen)가 World Model이 혼자 비디오를 생성하는 방식이었다면, 여기서는 **World Model이 상상 공간을 생성한 뒤, 기존 VLA가 action 생성에 직접 참여**한다는 점이 다릅니다.

이 방식의 대표 연구가 **Chelsea Finn 교수(Stanford, Physical Intelligence)**의 Ctrl-World와 그 후속인 VLAW입니다.

#### 3-3-1. Ctrl-World (ICLR 2026)

![Ctrl-World](/assets/img/notes/world-model/ctrl-world.jpg)
*Ctrl-World — world model을 구축하고, 얼린 뒤, 상상 공간에서 정책을 개선하는 3단계*

Ctrl-World는 일반 비디오로 사전학습된 video diffusion 모델(Stable Video Diffusion)을 backbone으로 삼아, 이를 **실제 로봇 데이터로 학습시켜 VLA와 상호작용하기 위한 controllable world model을 먼저 구축**합니다. 

구체적으로는 DROID 데이터셋 전량(약 95k trajectory, 564개 scene — 논문이 흔히 인용하는 76k 성공 궤적에 약 19k 실패 궤적을 더한 것으로, **성공·실패를 모두 포함**)으로 학습하여, 주어진 action에 따라 미래 관측을 multi-view로 예측하고 20초 이상 일관된 영상을 생성할 수 있는 world model을 만듭니다.

이 만들어둔 World Model 위에서, 작동 흐름은 다음과 같습니다.

1. 기존 VLA(예: π0.5-DROID)가 현재 관측을 보고 action을 출력합니다.
2. Ctrl-World가 그 action을 입력받아, 해당 action을 실행했을 때의 미래 관측을 생성합니다.
3. 생성된 관측을 다시 VLA에 돌려주어 다음 action을 받습니다. 
4. 이 과정을 반복하여 상상 공간 안에서 하나의 작업 전체(rollout)를 완성합니다. (→ 1-2-2 핵심 용어 사전)
5. 완성된 rollout 중 성공한 것만 골라, 그 (관측, action) 데이터로 VLA를 추가 학습(SFT)합니다.

> ### 💡 SFT(Supervised Fine-Tuning)와 closed-loop improvement
>
> SFT는 **"정답 데이터로 모델을 추가 학습"**하는 AI의 가장 기본적인 학습 방식입니다.
>
> Ctrl-World는 상상 공간에서 성공한 rollout을 일종의 정답 데이터로 삼아 VLA를 SFT 방식으로 학습합니다.
>
> **즉 실행(상상 공간 rollout) → 성공 사례 선별 → 그 데이터로 재학습 → 더 나은 VLA로 다시 실행이라는 순환**이 만들어지는데, 이렇게 결과를 다시 학습에 반영하는 순환을 **closed-loop improvement**라고 합니다.
>
> Ctrl-World는 이 방식으로 기존 VLA의 성공률을 44.7%p 끌어올렸습니다.
>
> 여기서 핵심은, Ctrl-World 자체는 action을 만들지 않는다는 것입니다. **action은 어디까지나 기존 VLA가 생성하고, World Model은 그 action의 결과를 영상으로 예측하여 "성공 rollout이라는 데이터"를 합성해주는 역할**을 합니다.
>
> 학습이 끝나면 실제 로봇을 움직이는 것은 개선된 VLA이고, World Model은 배포 단계에서 빠집니다.
>
> 따라서 이 연구의 본질은 평가 도구가 아니라, fidelity 높은 합성 rollout으로 VLA를 학습·개선하는 데이터 생성에 있다고 할 수 있습니다.

#### 3-3-2. VLAW (arXiv 2602.12063)
![VLAW](/assets/img/notes/world-model/vlaw-flow.jpg)
*VLAW — VLA와 World Model을 번갈아 개선하는 iterative co-improvement 루프*

VLAW는 Ctrl-World의 후속 연구(Chelsea Finn)로, VLA와 World Model을 번갈아 함께 개선하는 iterative co-improvement를 제안합니다. motivation은 바로 앞 Ctrl-World의 한계에서 나옵니다. 

**Ctrl-World처럼 시연(demonstration) 데이터로 한 번 학습해 고정한 World Model은, 정상적인 작업 영상은 잘 만들어내지만 충돌이 잦거나 변형되는 물체를 다루는 contact-rich 상황, 그리고 무엇보다 실패가 일어나는 순간의 dynamics를 충분히 담지 못합니다.** 

학습 데이터 자체에 그런 상황의 coverage가 부족하기 때문입니다. 그 결과 고정된 World Model이 만드는 합성 rollout은 정작 정책을 더 끌어올리는 데 필요한 fidelity가 부족하다는 한계가 드러납니다. (→ 1-2-2 핵심 용어 사전)

VLAW는 이 한계를, **World Model을 한 번 만들고 고정하는 대신 실제 rollout으로 계속 개선한다**는 발상으로 돌파합니다. 

**즉 Ctrl-World가 "고정된 World Model로 VLA만 개선"했다면, VLAW는 그 World Model 자체도 실제 데이터로 보강해 가며 VLA와 함께 끌어올립니다.**

작동 흐름은 다음과 같습니다.

1. 기존 VLA를 실제 로봇에서 소량 굴려 real-world rollout을 수집합니다. 이때 성공뿐 아니라 **실패 케이스**도 함께 모읍니다. 
2. 이 real-world rollout 데이터로 World Model을 fine-tune하여, **정책 실행 중 마주치는 복잡한 dynamics와 실패 양상까지 재현하도록 World Model의 fidelity를 끌어올립니다.**
3. fidelity가 높아진 World Model로 상상 공간에서 대규모 합성 rollout을 생성하고, 그 성공·실패는 **vision-language reward model**로 자동 라벨링합니다. 
4. 이렇게 만들어진 대규모 합성 데이터로 VLA를 학습하고 성능을 개선합니다.
5. 개선된 VLA로 다시 1번의 real-world rollout을 수집하여, World Model → VLA 개선을 반복합니다.

이 선순환을 통해 VLAW는 base policy 대비 절대 성공률을 39.2% 끌어올렸으며, 그중 합성 rollout 학습이 기여한 몫은 약 11.6%p로 보고하고 있습니다.

> ### 💡 Ctrl-World와 VLAW의 차이
>
> 두 연구 모두 "상상 공간 rollout으로 VLA를 개선한다"는 큰 틀은 같지만, **World Model을 손대느냐**가 갈립니다.
>
> - **Ctrl-World:** World Model을 실 데이터로 한 번 학습해 **고정**한 뒤, 그 안에서 VLA만 개선합니다. (한 방향)
> - **VLAW:** 실제 rollout(실패 포함)으로 **World Model 자체도 계속 개선**하고, 좋아진 World Model이 더 나은 합성 데이터를 만들어 VLA를 개선하며, 좋아진 VLA가 다시 더 풍부한 rollout을 만들어 World Model을 개선합니다. (양방향 순환)

> ### 💡 학습용 World Model을 그대로 평가에 쓸 수 있을까
>
> Ctrl-World와 VLAW의 rollout은 "성공/실패를 보여준다"는 점에서 평가에도 쓸 수 있을 것처럼 보입니다. 그러나 학습에 사용한 World Model로 같은 VLA를 평가하면 검증의 독립성이 깨지는 문제가 있습니다.
>
> VLAW가 fidelity를 끌어올리기 위해 실패 케이스까지 학습시킨다는 점은, **"World Model이 못 만드는 실패 모드는 평가에서 놓친다"는 한계**와 맞닿아 있습니다.

정리하면, 3장에서 살펴본 세 가지 방식은 모두 **"World Model이 만든 산출물을 데이터로 활용하여 VLA를 학습한다"**는 공통점을 가지되, World Model의 개입 정도가 단순 외형 변환(Cosmos Transfer) → 비디오 생성 후 action 복원(DreamGen) → 기존 VLA를 상상 공간에서 굴리기(Ctrl-World·VLAW) 순으로 점점 깊어진다고 정리할 수 있겠습니다. 

---

## Section 4. [트렌드 2. 학습] World Generation이 VLA 학습에 결합된다

앞선 3장에서 살펴본 세 가지 방식은 모두 World Model이 데이터를 만들어 건네준 뒤, 배포 단계에서는 빠진다는 공통점을 가지고 있었습니다. 

**실제 로봇을 움직이는 것은 어디까지나 학습된 VLA였고, World Model은 학습용 데이터를 합성해주는 외부 도구에 머물렀습니다.**

이 장에서 다루는 트렌드는 정확히 그 반대 지점에 있습니다. **World Model의 "다음 장면을 생성하는" 능력(World Generation)이 VLA 자체에 직접 결합되어, 학습은 물론 배포·추론 시점에도 함께 동작(online)하는 흐름**입니다.

3장과 다르게 4장의 기술들은 “action은 여전히 VLA가 생성하되, World Generation이 그 VLA에 결합되어 추론 파이프라인의 상시 구성요소”가 된다는 점에서 차이가 있습니다. 

즉 **"배포 시 World Model이 빠지느냐(3장) vs 함께 도느냐(4장)"**가 두 장을 가르는 핵심 대비축이라고 할 수 있겠습니다.

결합 방식에 따라 크게 두 갈래로 나누어 살펴보겠습니다.

1. **분리(decoupled)**: World Model을 VLA와 별도 모듈로 둔 채, 그 산출물을 conditioning으로 연결하는 방식 (4-1)
2. **통합(coupled)**: 비디오 생성과 action 생성을 하나의 모델 안에 합치는 방식 (4-2)

### 4-1. 분리(Decoupled) — World Model과 VLA를 분리한 채 conditioning으로 연결: π0.7 (Physical Intelligence, 2026)

![π0.7 분리 구조](/assets/img/notes/world-model/pi07-decoupled.jpg)
*π0.7 — World Model을 별도 모듈로 두고 그 산출물로 VLA를 conditioning하는 분리 구조*

가장 결합이 느슨한 형태는, **World Model을 VLA와 분리된 별도 모듈로 두되 그 산출물을 추론 시점에 VLA의 conditioning으로 주입하는 것**입니다. **Physical Intelligence의 π0.7(arXiv 2604.15483)**이 대표적입니다.

π0.7의 본체는 5B 규모의 VLA로, BAGEL 이미지 생성 모델에 기반한 별도의 경량 World Model을 둡니다.

작동 흐름은 다음과 같습니다.

1. VLA가 현재 상황을 보고, 다음에 수행할 subtask를 언어 지시로 출력합니다.
2. BAGEL 기반 World Model이 그 지시를 받아, 가까운 미래에 장면이 어떻게 되어야 하는지를 subgoal 이미지로 생성합니다.
3. VLA의 action expert가 현재 관측과 그 subgoal 이미지를 함께 conditioning으로 받아 action을 출력합니다.

그 결과, **subgoal 이미지를 주면 복잡한 task에서 성능이 개선**되며, 일부 task에서는 subgoal 없이는 실패한다고 보고하고 있습니다.

또한 **subgoal을 주면 학습이 유의미하게 빨라지는데**, **action 예측이 "현재 프레임 → 원하는 미래 프레임" 사이의 IDM 문제에 가까워지기 때문**이라고 설명됩니다. 

이런 구성을 통해 π0.7은 generalist이면서도, 일부 task에서는 specialist 모델 수준의 수행을 달성한다고 보고됩니다.

여기서 핵심은, 이 **World Model이 배포 시점에도 매 순간 subgoal을 생성하며 VLA와 함께 동작**한다는 점입니다. **모듈은 분리(decoupled)되어 있지만 추론 시 상시 결합된다는(online) 점**이 3장과 갈리는 지점이라고 판단됩니다.

> ### 💡 Ctrl-World(3-3)와 π0.7 — 같은 '별도 World Model 모듈', 무엇이 다른가
>
> **두 모델 모두 VLA와 분리된 별도 World Model을 둔다**는 점은 같습니다. **차이는 World Model이 배포 시점**에 하는 역할에 있습니다.
>
> - **Ctrl-World**: **World Model은 학습 단계에서 상상 rollout(데이터)을 만들어주고, 배포 시점에는 빠집니다.** 실제 로봇을 움직이는 것은 학습된 VLA뿐입니다.
> - **π0.7**: **World Model이 배포 시점에도 매 순간 subgoal 이미지를 생성하며, 그 이미지가 action expert의 입력으로 계속 들어갑니다.** World Model을 떼어내면 동작 자체가 달라집니다.
>
> 즉 "World Model이 데이터 생성기로서 학습 때만 쓰이느냐(3장)" vs "WM이 추론 파이프라인의 상시 구성요소냐(4장)"가 두 장을 가르는 기준입니다.

### 4-2. 통합(Coupled) — World Generation을 VLA 한 모델에 통합

분리(decoupled) 방식에서는 World Model을 외부 모듈로 두었다면, 통합(coupled) 방식에는 **비디오 생성과 action 생성을 하나의 모델 안에서 함께 학습·추론**합니다. 아래에서는 결합도가 깊어지는 순서로 두 모델을 살펴보겠습니다.

#### 4-2-1. NVIDIA DreamZero (World Action Model/GR00T N2, 2026): joint denoising

![DreamZero joint denoising](/assets/img/notes/world-model/dreamzero-joint-denoising.jpg)
*DreamZero — 비디오와 action을 한 모델에서 함께 denoising하는 통합 구조*

첫번째로 소개드릴 모델은 **NVIDIA DreamZero(arXiv 2026)**입니다. **World Action Model(WAM)이라는 명명을 전면에 내세운 대표 모델로, NVIDIA GEAR Lab(예성현, Joel 등)이 주도**하였습니다. 

논문 제목('World Action Models are Zero-shot Policies')에서 드러나듯, **task별 학습 없이 처음 보는 task·환경·로봇까지 수행하는 zero-shot 일반화**를 핵심 성과로 내세웁니다.

DreamZero는 사전학습된 video diffusion backbone(Alibaba Wan 2.1-I2V-14B)을 그대로 출발점 삼아, **미래 비디오와 action을 하나의 Diffusion Transformer 모델 안에서 함께 denoising(joint denoising)**합니다.

**핵심은 별도의 IDM 모듈이 없다**는 점입니다. 

3-2의 DreamGen이 "비디오 생성 모델 + 분리된 외부 IDM"의 2단 구조였다면, **DreamZero는 action을 비디오와 같은 denoising 과정 안의 또 하나의 modality로 다루어, 비디오 생성과 action 복원을 분리 없이 end-to-end로 수행**합니다. 

작동 흐름은 다음과 같습니다.

1. 사전학습된 video diffusion 모델(Wan 2.1-I2V-14B)을 로봇 데이터로 fine-tune하여, 하나의 Diffusion Transformer 모델이 미래 비디오와 action을 함께 생성하도록 학습합니다.
2. 추론 시, 현재 관측(비디오)·언어 지시·로봇 상태를 입력으로 받습니다.
3. DiT가 autoregressive + flow matching 방식으로, 다음 chunk의 미래 비디오 토큰과 action 토큰을 하나의 denoising 과정에서 동시에(jointly) 생성합니다. 
4. 생성된 action chunk를 실제 로봇에서 실행하고, 그 결과로 입력 받은 비디오 관측을 활용하여 생성된 프레임을 실제 프레임으로 교체하여 오차 누적을 막습니다.
5. 위 과정을 반복하며 closed-loop로 로봇 task를 수행합니다.

결과 측면에서 가장 두드러지는 것은 **zero-shot 일반화**입니다. 

기존 VLA가 의미(semantic) 일반화에는 강하지만 새로운 환경의 처음 보는 동작에는 약했던 반면, **DreamZero는 학습에 없던 task·환경에서 SOTA VLA 모델 대비 약 2배의 일반화 성능**을 보고합니다. 

예컨대 학습에 전혀 없던 task(신발끈 풀기·다림질·악수 등)에서 평균 39.5%를 수행하여, 같은 조건의 VLA(처음부터 학습 시 1% 미만, 대규모 사전학습 VLA도 16.3%)를 크게 앞섰습니다. 

더욱이 이를 **cross-embodiment 대규모 robot 학습 없이 DROID 데이터셋만으로** 달성했고, 실세계(Real 환경) 분산 평가 RoboArena에서도 1750점으로 Physical Intelligence π0.5(1622점)를 앞섰다는 점이 WAM의 잠재력을 보여주는 신호로 평가됩니다.

나아가 "비디오 생성 품질이 곧 policy 성능의 main lever"라는 것이 이 계열의 핵심 주장이며, **비디오를 잘 만드는 능력이 좋은 정책으로 직결된다는 관점을 명시적으로 내세웁니다.** NVIDIA는 GR00T N2를 이 DreamZero WAM 기반으로 공개할 예정이라고 밝혔습니다. (현재 모델 파일 미공개)

> ### 💡 zero-shot 일반화
>
> **zero-shot은 학습이나 시연 때 한 번도 본 적 없는 task를, 그 task용 추가 학습·데이터 없이 곧바로 수행하는 것을 뜻합니다.**
>
> DreamZero가 이를 해내는 이유는 출발점인 video foundation model이 웹 영상에서 "언어 지시 → 장면이 어떻게 변하는가"라는 지식을 이미 폭넓게 학습해 두었기 때문입니다.
>
> 덕분에 좁은 robot 데이터(DROID)만 보고도 처음 보는 동작·물체·환경으로 전이됩니다.
>
> 기존 VLA가 새 물체의 이름을 이해하는 의미(semantic) 일반화에는 강하지만, 새로운 환경의 처음 보는 물리적 동작에는 약했던 것과 대비되는 지점입니다.

> ### 💡 Joint denoising
>
> diffusion 모델은 noise를 점진적으로 걷어내며(denoising) 결과를 만듭니다. DreamZero는 비디오 토큰과 action 토큰을 같은 모델 안에서 동시에 denoising하여, "다음 장면"과 "그 장면을 만드는 action"을 한 번의 생성 과정에서 함께 뽑아냅니다.

> ### 💡 DROID와 RoboArena
>
> **DROID는 학습 데이터셋이자 표준 로봇 플랫폼**이고, **RoboArena는 그 DROID 플랫폼 위에서 도는 실세계(Real 환경) 평가 벤치마크**입니다. 데이터·평가 양쪽에서 반복 등장하므로 한 쌍으로 묶어 둡니다.
>
> - **DROID (Distributed Robot Interaction Dataset)**: 13개 기관이 동일한 하드웨어(Franka Panda 7-DoF 팔 + Robotiq gripper)로 수집한 대규모 실세계 조작 데이터셋입니다. 약 76k teleoperation trajectory, 564개 scene, 86개 task(~350시간 분량)로 구성되며(실패 궤적까지 포함한 공개 전량은 약 95.6k), multi-view RGB-D 영상과 자연어 지시를 포함합니다. 이 분야의 사실상 표준 데이터셋·플랫폼이라, 여러 모델이 같은 조건의 "DROID checkpoint"(π0.5-DROID, GR00T N1.6-DROID, DreamZero-DROID 등)로 서로 비교됩니다.
> - **RoboArena**: DROID 로봇 플랫폼 위에서 도는 분산형(distributed) 실세계 평가 벤치마크입니다(Chelsea Finn·Sergey Levine 등 주도). 고정된 task나 중앙집중식 대회로 평가하는 대신, **여러 기관의 evaluator가 task·환경을 자유롭게 고르되 두 정책을 double-blind pairwise로 비교**하고, 이 선호 데이터를 모아 **Elo-style 랭킹**을 산출합니다. LLM을 사람 선호로 줄세우는 LMArena의 로봇판이라 이해하면 됩니다.

#### 4-2-2. NVIDIA Cosmos 3 (GTC Taipei 2026.6): Omnimodal 단일 모델

![NVIDIA Cosmos 3](/assets/img/notes/world-model/cosmos3-omnimodal.jpg)
*Cosmos 3의 omnimodal 단일 모델 구성 (출처: [NVIDIA Cosmos Lab](https://research.nvidia.com/labs/cosmos-lab/cosmos3/))*

두번째로 소개드릴 모델은 **NVIDIA Cosmos 3(arXiv 2026)**입니다.

2-1에서 본 Cosmos의 Predict·Transfer·Reason이 별도 모델군이었다면, **Cosmos 3는 이들을 단일 모델로 통합하고 여기에 action(Policy 모델)까지 더해, language·image·video·audio·action을 하나의 모델에서 처리·생성합니다.**

구조는 **Mixture-of-Transformers(MoT)**의 two-tower입니다. **단, 두 '타워'는 별도의 두 모델이 아니라, 한 모델이 입력되는 토큰 종류에 따라 Reasoning, Generation 두 역할을 함께 수행합니다.**

기술적으로는 하나의 모델 안에서 각 layer마다 reasoning용 파라미터(Reasoner, Autoregressive)와 generation용 파라미터(Generator, diffusion)를 나눠 갖고 shared self-attention으로 연결된 형태입니다. 

작동 흐름은 다음과 같습니다.

1. language·image·video·audio·action 입력을 각 modality별 encoder로 토큰화하여 하나의 시퀀스로 받습니다.
2. 이 시퀀스가 하나의 MoT 모델을 한 번에 통과하며, 각 토큰은 종류에 따라 reasoning용(AR) 또는 generation용(diffusion) 파라미터로 처리되고 shared self-attention에서 연결됩니다.
3. 이때 attention mask가 비대칭이라 generation 토큰만 reasoning 맥락을 참조하도록(reasoner→generator 단방향) 조건화되며, 한 번의 forward pass 안에서 이해와 생성이 함께 이뤄집니다. policy 모드에서는 미래 비디오와 action을 함께(jointly) 예측합니다.
4. 어떤 입출력을 묶느냐에 따라 같은 모델이 모드를 전환합니다 — Reason(이해)·Predict(미래 비디오 생성)·Transfer(photoreal 변환)·Policy(action+video 동시 생성 = WAM 모드).

사양은 두 변형이 공개되었습니다. 

Nano는 총 16B(타워당 8B)로 workstation급, Super는 총 64B(타워당 32B)로 데이터센터급입니다(2B급 Edge 단계도 출시 예정).

Cosmos 3의 의의는, **데이터 생성(Transfer·Predict)·학습(Policy)·평가/시뮬레이션(Reason·world sim)이 한 모델로 수렴**하는 상징이라는 점입니다. 

World Model이 로봇 개발 파이프라인의 전 단계에 침투한다는 트렌드가, 단일 모델 안에서 구현된 형태로 볼 수 있겠습니다.

> ### 💡 MoT(Mixture-of-Transformers)와 two-tower
>
> MoT는 modality·역할마다 전용 weight를 두되, attention을 공유해 서로 정보를 주고받게 하는 구조입니다.
>
> 흔히 혼동되는 MoE와는 다릅니다 — **MoE가 같은 종류의 전문가 중 일부를 라우팅으로 고르는 방식이라면, MoT는 역할별로 전용 파라미터를 둡니다.**
>
> 핵심은, Cosmos 3의 reasoner와 generator가 별도의 두 모델이 아니라 하나의 모델 안에 있는 두 파라미터 집합이라는 점입니다.
>
> 각 transformer layer가 reasoning용(AR)·generation용(diffusion) 두 weight set를 갖고, 두 토큰 흐름은 shared self-attention에서만 만납니다.
>
> attention mask가 비대칭이라(AR은 causal·자기 완결, diffusion은 full·AR 맥락 참조) 정보는 reasoner→generator 단방향으로 흐르지만, 이는 모두 한 번의 forward pass 안에서 일어납니다.
>
> 따라서 **느린 VLM(System 2)이 빠른 action head(System 1)에 결과를 넘기는 dual-system과는 다른, 이해와 생성이 한 모델로 통합된 omnimodal 구조입니다.**

> ### 💡 비대칭 attention mask (causal vs full)
>
> attention mask는 각 토큰이 시퀀스 안에서 "**어떤 토큰을 참조(attend)할 수 있는지**"를 정하는 규칙입니다(2-2에서 비디오 프레임에 적용했던 causal/full attention과 같은 개념).
>
> Cosmos 3는 reasoning 토큰과 generation 토큰을 한 시퀀스로 합쳐 처리하되, 이 참조 규칙을 비대칭으로 설계합니다.
>
> 비유하면, 먼저 상황을 분석하는 분석가(reasoning)의 분석을 실행팀(generation)은 자유롭게 참고하지만, 실행팀의 작업물이 분석가의 분석을 바꾸지는 못하는 것과 같습니다.
>
> ```
> Block 1 = Reasoning 토큰 (AR)
> Block 2 = Generation 토큰 (diffusion)
>
> 누가 누구를 참조(attend)할 수 있는가:
>
>                        Block 1      Block 2
>                      (reasoning)  (generation)
> Block 1 (reasoning):    ✅           ❌
> Block 2 (generation):   ✅           ✅
>
> ※ 블록 내부: Block 1은 causal(과거만 참조), Block 2는 full(서로 다 참조)
> ```
>
> 이 설계의 핵심은 reasoning 토큰이 generation 쪽 noisy 토큰의 영향을 받지 않고 자기 완결적(self-contained)으로 유지된다는 점입니다.
>
> 덕분에 정보는 reasoner→generator 한 방향으로만 흐릅니다. 그러면서도 이것은 두 모델을 따로 돌려 결과를 넘기는 게 아니라, 같은 모델·같은 forward pass 안에서 mask 규칙만으로 구현됩니다.

> ### 💡 Dual-system (System 2 → System 1)
>
> dual-system은 로보틱스에서 흔히 쓰이는 구조로, 인간 인지의 "느린 사고 / 빠른 사고"(System 2 / System 1)에서 이름을 따왔습니다.
>
> - **System 2**: 느리지만 깊이 생각하는 부분. 보통 VLM이 장면·언어를 이해하고 다음 행동을 계획하며, 낮은 주기로 동작합니다(예: NVIDIA GR00T N1의 system 2는 ~10Hz).
> - **System 1**: 빠르게 반응하는 부분. action head가 실제 관절 움직임을 생성하며, 높은 주기로 동작합니다(예: GR00T N1의 system 1은 ~120Hz).
>
> 핵심은 이 둘이 **별도의 네트워크**라는 점입니다. System 2가 이해·계획 결과를 System 1에 넘기면, System 1이 그것을 받아 빠르게 실행하는 모듈 분리형 구조입니다. **Figure AI의 Helix와 NVIDIA GR00T**가 대표적입니다.
>
> Cosmos 3는 이와 다릅니다. reasoning과 generation이 별도 네트워크로 나뉘어 결과를 주고받는 게 아니라, 하나의 모델·하나의 forward pass 안에 통합되어 있습니다.

정리하면, 4장에서 살펴본 기술들은 **World Generation이 VLA에 결합되는 정도가 점점 깊어지는 스펙트럼**으로 이해할 수 있겠습니다. 

이어지는 Section 5에서는 World Model이 마지막 단계인 '평가'에 어떻게 침투하는지를 살펴보겠습니다.

---

## Section 5. [트렌드 3. 평가] World Model이 평가 병목을 푼다

3장과 4장이 World Model을 데이터와 학습에 활용하는 흐름이었다면, 마지막 트렌드는 '평가'입니다. 

다만 이 트렌드는 앞의 둘에 비해 아직 early-stage이며, 가능성과 함께 풀리지 않은 근본 과제가 분명히 남아 있습니다.

### 5-1. 사람이 평가하던 시대 — 평가 병목의 실체

**로봇 정책 평가의 gold standard는 실제 로봇으로 작업을 끝까지 굴린 rollout을 사람이 직접 보고 성공·실패를 채점하는 것입니다.** 문제는 이것이 지독하게 비싸다는 점입니다. 

한 사례로, OpenVLA 한 모델을 베이스라인과 비교 평가하는 데에만 4개 로봇 셋업·3개 기관에 걸쳐 2,500회 이상의 rollout과 100시간 이상의 사람 노동(장면 리셋·정책 실행·성공 채점)이 들었다고 보고됩니다.

이 병목을 체계화·자동화하려는 시도가 이미 있습니다. 

앞서 설명한 RoboArena(4-2-2)는 여러 기관의 evaluator가 DROID 플랫폼 위에서 정책들을 double-blind pairwise로 비교해 Elo-style 랭킹을 만드는 분산 실제 평가이고, Sergey Levine(UC Berkeley, Physical Intelligence)의 AutoEval(arXiv 2025)은 사람 개입을 최소화해 정책 실행·성공 채점·장면 리셋을 자동화하여 24시간 real-robot 평가를 돌립니다.

그럼에도 **real-world 평가는 본질적으로 하드웨어·실험실에 묶이고 느립니다.** AutoEval은 특정 셋업·단일 embodiment에 한정되며, generalist 정책일수록 필요한 평가 환경이 다양해져 병목은 오히려 심해진다고 볼 수 있겠습니다.

### 5-2. 강화학습(RL)을 통한 우회 방법 — rollout 비디오 없이 평가하기

평가가 비싸다는 문제에 RL 진영은 **"rollout 영상을 일일이 보지 않고도 정책을 평가·개선하는" 우회로**로 대응해 왔습니다. 대표적인 세 갈래는 다음과 같습니다.

1. **개입(intervention) 기반 — DAgger / HG-DAgger**: **사람이 정책(=VLA)을 지켜보다 실패하려는 순간 개입해 교정하는 방식**으로, 사람이 얼마나 자주 개입해야 했는가(개입 빈도) 자체가 정책 품질의 신호가 됩니다.
2. **advantage 기반 — π*0.6 (Physical Intelligence, RECAP)**: value function이 **각 action이 작업 성공에 얼마나 기여하는지(advantage)를 추정하여, 좋은 action에는 "positive", 나쁜 action에는 "negative"라는 점수**를 매깁니다. rollout 영상을 보는 대신 학습된 value로 action을 평가하고, 추론 시에는 "positive" action만 내도록 조건화하여 학습 데이터보다 나은 정책을 얻습니다.
3. **foundation reward model 기반 — Dyna**: 태스크마다 사람이 reward 규칙을 손으로 짜는 대신, **하나의 VLM 기반 reward model이 다양한 로봇 태스크의 진행도(goal까지의 temporal distance)를 범용적으로 점수화**하는 접근입니다. Dyna Robotics의 DYNA-1은 이런 foundation reward model을 post-training 루프의 중심에 상시 배치(RM-in-the-loop)하여 VLA가 reward를 높이는 쪽으로 반복 개선되도록 하고, 실패 케이스를 reward model 개선에 자동 반영(online adaptation)하는 closed-loop을 구성한 것으로 추정됩니다(DYNA-1은 closed model이라 디테일은 비공개)

이 우회로들은 빠르고 자동화하기 좋습니다. 다만 공통의 한계가 있습니다. 

**정책(=VLA)의 거동을 value·advantage·reward 같은 scalar나 성공/실패 label로 압축하기 때문에, 실제 rollout 영상을 보는 것만큼 직관적이고 정확한 평가는 되기 어렵다**는 점입니다. 

reward model은 틀리거나 reward hacking될 수 있고, advantage는 어디까지나 추정이며, 개입 빈도는 거칠어서 "어디서·왜 실패했는가"를 보여주지 못합니다.

**결국 가장 신뢰할 수 있는 평가 신호는 여전히 'rollout 영상' 자체라고 판단됩니다.**

> ### 💡 advantage conditioning (π*0.6의 평가 신호)
>
> Physical Intelligence π*0.6의 advantage conditioning은 **"rollout 영상 없이 action을 점수화"**하는 대표적 방법입니다.
>
> 먼저 value function이 현재 상태가 작업 성공에 얼마나 가까운지를 추정하고, 어떤 action을 했을 때 그 값이 오르면 advantage가 양수(좋은 action), 내리면 음수(나쁜 action)입니다.
>
> RECAP은 이 advantage를 복잡한 숫자 대신 "positive/negative"라는 텍스트로 정책에 조건으로 붙여 학습시키고, 배포 때는 "positive"만 요청해 학습 데이터보다 나은 행동을 끌어냅니다.
>
> flow matching 기반 VLA는 action 확률(log-prob)을 내주지 않아 PPO 같은 표준 RL을 그대로 쓰기 어려운데, 이 우회가 그 문제를 푼 것입니다.

### 5-3. World Model 기반 평가의 근본 과제와 떠오르는 방향

그래서 **'rollout 영상'을 활용하되 실제 하드웨어 없이 대규모로 만들자는 것이 World Model 기반 평가**입니다. 

**학습된 World Model 안에서 정책(=VLA)을 굴려 rollout 영상을 생성하고 그 영상을 채점하면, 확장 가능하고(scalable) 재현 가능한(reproducible) 평가가 됩니다.** 

다만 이 방식이 신뢰할 만한 평가가 되려면 풀어야 할 근본 과제가 남아 있습니다.

#### 5-3-1. World Model을 평가자로: WorldEval

대표 연구는 **WorldEval(arXiv 2025)**입니다. 

World Model 기반 평가의 핵심 난제는 정책(=VLA)의 action을 충실히 반영하는 영상을 생성하는 것입니다. **정책(=VLA)이 낸 action을 World Model에 그대로 넣으면 의외로 action을 따라가는 영상이 잘 만들어지지 않는데, 로봇 action이 의미적으로 모호하기 때문입니다.** 

WorldEval은 정책(=VLA) 출력을 latent action으로 바꿔 영상을 만드는 **Policy2Vec**으로 이 난제를 해결합니다. **정책의 출력을 latent action으로 바꾸고, video 생성 모델이 그 latent action을 따라 로봇 영상을 만들도록 하는 방식**입니다.

이후 생성된 rollout을 video-capable VLM(예: Gemini)이 보고 성공·실패를 자동 판정합니다(auto-labeling). 사람 채점자를 VLM으로 대체해 평가를 대규모·24시간으로 자동화하는 셈입니다.

작동 흐름은 다음과 같습니다.

1. 평가할 정책(=VLA)이 현재 관측을 보고 action을 출력합니다.
2. World Model이 그 action(latent action)을 따라 미래 관측 영상을 생성합니다.
3. 생성된 관측을 다시 정책(=VLA)에 돌려주어, 상상 공간 안에서 작업 전체(rollout)를 완성합니다.
4. 완성된 rollout 영상을 video-capable VLM(예: Gemini)이 보고 성공·실패를 자동으로 판정합니다.

이렇게 산출된 점수는 사람 개입 없이 정책(=VLA)들과 실험 중인 모델들을 줄세울 수 있으며, 위험한 동작을 사전에 걸러내는 safety detector로도 쓰입니다. **WorldEval은 이 점수가 실제 real-world 성능과 강한 상관을 보인다고 보고합니다.**

**미국 스타트업 Runway의 GWM-Robotics**(General World Model GWM-1의 로봇 분기)도 같은 흐름으로, 하드웨어 없이 8개 정책을 평가해 시뮬레이션 점수와 실제 점수가 95% 유사성을 보였다고 보고했습니다.

#### 5-3-2. 근본 과제 두 가지: 학습/평가 독립성과 fidelity

World Model 기반 평가에는 두 가지 근본 과제가 있습니다.

- **학습 독립성(train/eval leakage)**: 3-3에서 설명했듯, **학습에 쓴 World Model을 그 World Model로 학습된 같은 정책(=VLA)을 평가하는 데 그대로 쓰면 검증의 독립성이 깨집니다.**
    - 정책(=VLA)이 World Model의 편향에 overfit되어 있으면 상상 공간 안에서는 성공처럼 보여도 실제로는 실패할 수 있고, 무엇보다 World Model이 못 만드는 실패 모드는 평가에서 아예 보이지 않습니다.
    - 따라서 엄밀한 평가는 정책 학습과 독립된 World Model, 또는 real-world 검증을 요구한다고 판단됩니다.
- **fidelity(실패 재현):** 평가용 World Model이 성공 장면만 그럴듯하게 만들고 **실패를 재현하지 못하면, 실패할 정책도 성공한 것처럼 보이게 만들어 평가가 거짓이 됩니다.**
    - 이를 정면으로 다룬 dWorldEval(arXiv 2026)은 동일 action에서 실패를 self-correct하거나 hallucination하지 않고 충실히 재현하도록 사람이 수집한 실패 데이터를 학습에 넣고, 매 step마다 progress score(0~1)를 함께 생성해 "작업이 어디까지 진행되다 멈췄는가"를 자동 채점합니다.
    
    > ### 💡 평가용 World Model이 빠지기 쉬운 함정 — 실패를 못 만든다
    >
    > 좋은 정책을 가려내려면, **World Model은 실패하는 정책(=VLA)을 실패하는 영상**으로 보여줘야 합니다. 그런데 영상 품질을 위해 학습된 World Model은 정반대로 행동하는 경향이 있습니다.
    >
    > - **self-correction**: 정책이 grasp에 실패했는데도 "자연스러운 영상"을 만들려다 성공한 grasp로 고쳐 버립니다.
    > - **hallucination**: 장면에 없는 물체를 만들어내, 실제로는 불가능한 성공을 그려냅니다.
    > - **action 불일치**: 입력 action과 어긋난 영상을 만들어, 정책의 실제 행동을 반영하지 못합니다.
    >
    > 세 경우 모두 실패할 정책을 성공으로 오판하게 만듭니다.
    

이 두 과제를 풀어가는 과정에서, **단순 성공/실패를 넘어 rollout의 3D trace(궤적)를 추출해 "어디서 어떻게 틀렸는지"까지 설명하려는 explainable 평가 방향도 early-stage로 함께 떠오르고 있습니다.**

정리하면, 평가 트렌드는 아직 early-stage이지만 로봇 개발 파이프라인의 병목이 되고 있다는 점에서 앞으로 집중될 분야임은 분명합니다. 

사람이 직접 채점하던 시대(5-1)에서, rollout 영상 없이 scalar로 우회하던 RL의 방식(5-2)을 거쳐, World Model로 rollout 영상을 하드웨어 없이 대규모로 만들어 평가하는 방향(5-3)으로 나아가되, train/eval leakage라는 근본 과제와 auto-labeling·fidelity·explainability라는 보완책이 함께 떠오르고 있습니다.