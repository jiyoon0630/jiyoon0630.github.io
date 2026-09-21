---
layout: paper
lang: ko
ref: actuate-2026-rfm-data-trends
kind: tech-review
title: "Actuate 2026 참관기: RFM과 데이터 트렌드"
date: 2026-08-24 12:00:00 -0700
venue: "Actuate 2026 · San Francisco"
tags: [Robot-Foundation-Model, Data, Scaling-Law, World-Model, Conference, Tech-Review]
summary: "Actuate 2026 참관 기록 — 데이터 수집·저장 포맷의 표준화, 대규모 Pre-training + 소량 Post-training이라는 공통 레시피, 그리고 VLA 바깥으로의 아키텍처 확장."
---

> **핵심 정리** — 2026년 8월 샌프란시스코에서 열린 로보틱스 개발자 컨퍼런스 Actuate 2026의 참관 기록입니다. 주요 RFM 기업들이 **데이터 수집 방식·저장 포맷의 표준화**, **"대규모 Pre-training + 소량 Post-training"이라는 공통 학습 레시피**, 그리고 **VLA 바깥으로의 아키텍처 확장**이라는 세 방향으로 수렴하고 있다는 점을 확인할 수 있었습니다.

---

## Introduction: Actuate 2026 Conference 소개

Actuate 2026은 Robotics 데이터 시각화·관리 Platform 기업 Foxglove가 주관하여 2026년 8월 18~19일 양일간 샌프란시스코에서 열린 "로봇을 만드는 사람들을 위한 개발자 컨퍼런스"입니다.

학술 연구 발표나 제품 홍보보다는 실제로 로봇을 현장에 배포하는 엔지니어와 기술 리더가 기술적 진전을 공유하는 자리입니다.

NVIDIA를 비롯해 Avala, Encord, Scale, Lightwheel 등 데이터 관련 기업과 Foundry Robotics, Flexion, Genesis 등 로봇 기업이 스폰서로 참여하였으며, 1,000명 이상이 참석하는 로봇 생태계의 주요 네트워킹 행사로 자리잡았습니다.

특히 Physical Intelligence, Generalist AI, Genesis AI, Dyna Robotics 등 주요 RFM 기업들이 speaker로 참여하여 이번 행사에서 처음 공개하는 기술 내용까지 소개하였으며, 로보틱스 분야의 현안인 "Data 전쟁(Human Data vs UMI vs Teleop)", "RFM Architecture(VLA vs WAM)" 등이 패널 토론 주제로 다루어졌습니다.

> ### 💡 참석 주요 RFM 기업
>
> | **기업명** | **창업 시기 / 본사** | **창업자** | **펀드 규모** | **회사 소개** |
> | --- | --- | --- | --- | --- |
> | **Physical Intelligence** | 2024 / 실리콘밸리 San Francisco | • Karol Hausman(CEO, USC 박사, 前 Google DeepMind)<br>• Sergey Levine(Chief Scientist, UC Berkeley 교수, Stanford 박사)<br>• Chelsea Finn(Research Lead, Stanford 교수, UC Berkeley 박사과정 중 Pieter Abbeel·Sergey Levine 교수 지도)<br>• 외 Brian Ichter, Lachy Groom 등 7인 | 누적 약 $1.07B 펀드 조달<br>• Seed $70M('24.3)<br>• Series A $400M('24.11, Valuation $2.4B)<br>• Series B $600M('25.11, Valuation $5.6B)<br>• 신규 라운드 협의 중(보도 기준 $11B+, 미확정) | 어떤 로봇(cross-embodiment)에 적용할 수 있는 범용 foundation model(π 시리즈) 개발을 목표로, UC Berkeley·Stanford 연구진을 기반으로 Scaling, 강화학습, World Model과의 결합 등 최신 RFM 연구 트렌드 선도하는 기업 |
> | **Generalist AI** | 2024 / 실리콘밸리 San Mateo | • Pete Florence(CEO, MIT 박사과정 중 Russ Tedrake 교수 지도, 前 Google DeepMind에서 PaLM-E·RT-2 주도)<br>• Andy Zeng(Chief Scientist, Princeton 박사, 前 DeepMind)<br>• Andrew Barry(CTO, Boston Dynamics에서 Atlas·Spot·Stretch 개발) | 누적 약 $500M 펀드 조달<br>• Series A $140M(Valuation $440M)<br>• Series B $400M(’26.6, Valuation $2B)<br>• 신규 라운드 협의 중 (보도 기준 $3B, 미확정) | 하드웨어에 종속되지 않는 범용 foundation model 개발을 목표로, GEN-0/1에서 로보틱스 최초의 Scaling Law를 입증하고 GEN-1.5의 One-shot 학습까지 사전학습 스케일 확대만으로 도달한 기업 |
> | **Dyna Robotics** | 2024 / 실리콘밸리 Redwood City | • Lindon Gao(CEO, Caper AI를 $350M에 매각한 연쇄 창업자)<br>• York Yang(CTO, Caper AI를 $350M에 매각한 연쇄 창업자)<br>• Jason Ma(Chief Scientist, UPenn 박사, 前 DeepMind) | 누적 약 $143.5M 펀드 조달<br>• Seed $23.5M('25.3)<br>• Series A $120M('25.9, Valuation $600M+, LG Technology Ventures 참여) | 상업 현장에 배포 가능한 범용 manipulation 모델 개발을 목표로, 인간 비디오 100만 시간 규모의 Scaling Law와 Zero-shot 전이를 입증하고 이를 뒷받침하는 대규모 학습 인프라까지 함께 구축한 기업 |
> | **Sunday** | 2024 / 실리콘밸리 Mountain View | • Tony Zhao(CEO, Stanford 박사과정 중 Chelsea Finn 지도, ALOHA·ACT 연구자)<br>• Cheng Chi(CTO, Stanford·Columbia 박사과정 중 Shuran Song 지도, Diffusion Policy·UMI 연구자) | 누적 약 $200M 펀드 조달<br>• Series A $35M('25.11, Valuation 비공개)<br>• Series B $165M('26.3, Valuation $1.15B) | 완전 자율 가정용 로봇 개발을 목표로, Diffusion Policy·ACT·ALOHA·UMI 등 현재 RFM의 기반이 된 주요 연구를 직접 수행한 연구진이 데이터 수집부터 모델·하드웨어까지 풀스택으로 개발하는 기업 |
> | **Genesis AI** | 2024 / 실리콘밸리 Palo Alto·프랑스 Paris | • Zhou Xian(CEO, CMU 박사)<br>• Théophile Gervet(CMU 박사, 前 Mistral AI) | 누적 약 $105M 펀드 조달<br>• Seed $105M('25.7)<br>• 신규 라운드 협의 중(보도 기준 $3B, 미확정) | 범용 로봇의 대규모 보급을 목표로, 자체 개발 시뮬레이션 엔진을 강화학습 환경으로 활용하고 인간 손과 동일한 형태의 로봇 핸드를 직접 설계하여 데이터·모델·하드웨어를 풀스택으로 통합한 기업 |
> | **1X Technologies** | 2014 / 실리콘밸리 Palo Alto | • Bernt Øivind Børnich(CEO) | 누적 약 $126M 펀드 조달<br>• Series A2 $23.5M('23.3, Valuation 비공개)<br>• Series B $100M('24.1, Valuation 비공개)<br>• 신규 라운드 협의 중(보도 기준 $10B+, 미확정) | 가정용 휴머노이드 NEO의 대량 보급을 목표로, 로봇 하드웨어와 World Model을 자체 개발하는 동시에 데이터 수집 리그·시뮬레이션 도구·제어 API를 외부 개발자에게 개방하여 플랫폼으로 확장하는 기업 |
> | **Eka Robotics** | 2025 / 메사추세츠 Cambridge | • Pulkit Agrawal(CEO, MIT 교수, UC Berkeley 박사)<br>• Tuomas Haarnoja(前 Google DeepMind) | 펀드 규모 비공개<br>• '26.4 스텔스 해제 | 범용성과 속도를 동시에 갖춘 로봇 개발을 목표로, 힘(force)을 물리적 상호작용의 언어로 보는 VFA 모델을 시뮬레이션 강화학습으로 학습시켜 Sim-to-Real Gap을 극복한 기업 |
> | **Foundry Robotics** | 2025 / 실리콘밸리 San Francisco | • Adarsh Kulkarni(CEO) | 누적 약 $24.5M 펀드 조달<br>• Seed $5.5M('26.1)<br>• Seed $19M('26.4) | 제조업의 AI 기반 재설계를 목표로, 조립 공정을 검증 가능한 프리미티브 단위로 분해해 학습시키고 이를 소프트웨어로 재구성 가능한 모듈형 로봇 셀 형태로 판매하는 기업 |
> | **Veeda AI** | 2026 / 캐나다 Toronto·실리콘밸리 Mountain View | • Sanja Fidler(CEO, University of Toronto 교수, University of Ljubljana 박사, 前 NVIDIA VP of AI Research)<br>• Huan Ling(Chief Scientist, 前 NVIDIA)<br>• Zan Gojcic(CTO, 前 NVIDIA) | 누적 약 $90M+ 펀드 조달<br>• Seed $90M+('26.8) | Physical AI의 인터랙티브 학습 확장을 목표로, 생성형 World Model로 구현한 시뮬레이션 환경에서 정책의 성공 여부를 평가하고 개선하는 반복 루프(interactive loop) 구축을 지향하는 기업 |

## Key Takeaways

**1. 데이터 — 수집 방식은 갈리지만, Scaling을 위한 표준화는 빠르게 진행되고 있습니다.**
로봇 없이 데이터를 모으기 위해 UMI Gripper, Glove, Human Egocentric 방식이 병존하며, 각 사는 자체 로봇 End-effector와 동일한 형태로 수집 기구를 제작해 변환 부담을 줄입니다. 즉 데이터 전략과 하드웨어 전략은 분리된 의사결정이 아니며, 수집 인터페이스(ALOHA·UMI·GELLO)와 저장 포맷(MCAP)은 이미 사실상의 표준이 형성되었습니다.

**2. 학습 — "대규모 Pre-training + 소량 Post-training"이 공통 레시피이나, 성능은 강화학습의 몫입니다.**
Pre-training의 Scaling Law 발견으로, Generalist AI는 1분 분량 데이터와 1회 학습으로, Dyna Robotics는 13분 학습으로 새 작업을 수행하며 Post-training 규모가 세 자릿수 이상 줄었습니다. 다만 Scaling은 Generalization을 만들 뿐 배포 수준의 Performance(신뢰성 + 속도)는 만들지 못하며, 이를 메우는 공통 해법으로 강화학습이 채택되고 있습니다.

**3. 아키텍처 — VLA 바깥으로 확장이 시작되었습니다.**
Physical Intelligence의 π0.7은 VLA 앞에 World Model을 두어 다음 목표 상태를 이미지/비디오 형태로 생성해 정책의 프롬프트로 주입하는 조건화 구조를 취하고 있으며, Veeda AI는 VLA 뒤에 World Model 기반 Simulator를 두어 정책의 성공 여부를 평가하고 개선하는 폐쇄형 루프를, Google DeepMind는 VLA 앞에 Agentic Layer를 두어 고수준 추론과 다중 로봇 협업, Safety를 담당하게 하는 구조를 제시하였습니다.

공통적으로 강조된 것은 **이제 실험실이 아니라 현장에서 검증할 단계라는 점이며, 파트너십을 통한 실증이 다음 경쟁 지점이 될 것으로 판단됩니다.**

---

## 1. 데이터 Scaling을 위한 표준화 경쟁 본격화

**RFM의 성능은 결국 얼마나 좋은 데이터를 얼마나 많이 모으느냐**에 달려 있습니다.

컨퍼런스에서 주요 RFM(Robot Foundation Model) 기업들이 발표한 그들의 기술에 따르면, **(1) 데이터를 모으는 방식이 점차 확장되고 있으며, 수집 인터페이스(하드웨어)·저장 포맷·라벨링 방식이 표준화되고 있다는 점, (2) 범용 RFM 학습 레시피가 어느 정도 확립되고 있다는 점**을 확인할 수 있었습니다.

여기서 명확히 인지 할 수 있는 점은, 최근 AI 산업의 주도권은 사전학습 모델을 보유한 기업에 집중되고 있습니다. 이들이 모델의 가중치만 쥐고 있어서가 아니라, **모델 성능을 좌우하는 데이터의 조건 자체를 규정하기 때문입니다.**
어떤 데이터를 얼마나 모아야 하는지, 그것을 어떤 구조와 표준으로 정형화할지, 그리고 어떤 조합과 비율로 학습에 투입할지, 이 판단은 모두 학습 결과를 관측할 수 있는 위치에서만 내릴 수 있습니다. **데이터 사양은 모델과 독립적으로 존재하는 것이 아니라, 학습을 돌려보고 성능 변화를 확인하는 과정에서 역산되어 도출됩니다.**

따라서 **모델을 학습시켜 본 경험 없이 좋은 데이터를 만들겠다는 것은 성립하지 않습니다.** 무엇이 좋은 데이터인지를 판정할 기준을 갖지 못한 채 수집만 수행하는 것이기 때문입니다.

### 1-1. 데이터 Scaling: Robot Teleoperation → UMI Gripper 기반 Human Video → Human Egocentric Video

로봇 학습 데이터는 오랫동안 로봇 Teleoperation, 즉 사람이 실제 로봇을 원격 조작하며 기록하는 방식에 의존해 왔습니다.

이 방식의 제약은 명확합니다. Sunday 공동창업자 Cheng Chi는 로봇 한 대당 사람 한 명이 필요하므로 작업 종류를 10배 늘리면 인력과 비용도 10배 증가하며, 논문이나 프로젝트마다 로봇과 카메라를 새로 설치하고 그 배치를 유지해야 하는 운영 부담도 크다고 설명했습니다.

**현재 로봇 업계에는 다양하고 많은 데이터를 모으기 위한 네 가지 수집 방식이 활용되고 있습니다.**

| **데이터명** | **수집 방식** | **기록되는 데이터** | **Action의 정체** | **대표 기업 (Pre-training 기준)** |
| --- | --- | --- | --- | --- |
| **로봇 Teleoperation** | 사람이 컨트롤러로 실제 로봇을 조작 | • 영상: 헤드, 손목<br>• 관측값: 관절 Action 값 | 측정된 진짜 Action(예측값 X) | Physical Intelligence |
| **UMI Gripper 방식** | 로봇 그리퍼와 유사하게 생긴 기구를 파지 | • 영상: 헤드, 손목<br>• 관측값: IMU 가속도 값, 그리퍼 개폐 폭 | 예측된 Action (영상·IMU로 그리퍼 pose 복원 → delta pose를 action으로 재구성 = Retargeting) | Sunday, Generalist AI |
| **Glove 방식** | 센서 내장 글로브 착용 | • 영상: 헤드, 손목<br>• 관측값: 손가락 관절각, IMU 가속도 값, 손바닥·손가락 촉각 | 측정 + 예측된 Action<br>• 손가락 관절은 센서를 통해 측정<br>• 손목 Action은 pose 복원 → action으로 재구성(= Retargeting) | Sunday, Genesis AI |
| **Human Egocentric 방식** | 1인칭 영상 | • 영상: 헤드(경우에 따라 손목도) | 예측된 Action (손 pose 추정 → action으로 재구성 = Retargeting) | Dyna, Genesis AI, Physical Intelligence |

위에서 아래로 갈수록 수집 규모는 커지지만 직접 측정되는 정보는 줄어듭니다.

Genesis AI의 공동창업자 Theophile Gervet는 수집 규모, Action 정확도 두 축을 강조하며, 자체 개발한 Glove 및 5 Finger Hand의 강점을 어필했습니다.

Teleoperation이 Action 정확도가 최상이나 규모 측면에서는 최소이며, Human Video가 그 반대에 위치한다고 할 수 있겠습니다.

> ### 💡 Action Retargeting
>
> **로봇 Teleoperation을 제외한 세 방식은 사람의 동작(UMI Gripper 파지, Glove 착용, 사람 손)을 로봇 Embodiment로 옮기는 변환 과정**이 필요하며, 이를 **Action Retargeting**이라 합니다. 다음 요소들이 포함됩니다.
>
> 1. **스케일 정규화:** 수집자별 팔 길이·키 차이 보정
> 2. **관절 매핑:** 사람 관절과 로봇 관절의 대응 관계 설정
> 3. **IK(Inverse Kinematics):** 손끝을 목표 위치에 놓기 위해 각 관절을 몇 도씩 움직여야 하는지 계산하는 기술로, 관절이 꺾이는 한계, 로봇 팔끼리 부딪히는 상황, 팔이 닿지 않는 위치 등을 함께 고려
>
> 실무적으로는 하나의 기술로 함께 풀리는 경우가 많으며, 각 기업은 이 변환 부담을 하드웨어로 줄이거나 소프트웨어로 감당하는 선택을 하고 있습니다.
>
> 1. **하드웨어로 줄이는 방식**
>     1. Sunday 공동창업자 Cheng Chi가 박사 과정 시절 고안한 UMI Gripper는 핸드헬드 기구를 로봇 그리퍼와 동일한 형상으로 제작하여, 스케일 정규화(1) 및 관절 매핑(2) 과정을 생략하도록 합니다. 이는 기록 대상이 사람 신체가 아니라 기구이기 때문에 가능하다고 할 수 있습니다. (IK는 여전히 필요)
>     2. Genesis AI는 반면 로봇의 손을 인간 손과 동일한 형태 및 접촉 물리학을 갖도록 설계하고, 여기에 대응하는 Glove로 데이터를 수집하여 스케일 정규화(1) 및 관절 매핑(2)의 부담을 줄입니다. 사람의 손가락 관절이 로봇의 손가락 관절에 거의 그대로 대응되기 때문입니다. (IK는 여전히 필요)
> 2. **소프트웨어로 감당하는 방식**
>     1. Sunday는 현재 UMI가 아닌 Glove 방식으로 전환하였으며, Memo 로봇의 손과 동일한 기하 구조·센서 구성으로 Skill Capture Glove를 제작하였습니다. 다만 손은 정렬되어도 수집자마다 키와 팔 길이가 달라 신체 전체는 정렬되지 않으므로, Skill Transform이라는 변환 파이프라인을 통해 해결합니다.

> ### 💡 5 Finger Hand와 Glove 방식
>
> 로봇의 End-effector는 2 Finger Gripper에서 점차 5 Finger Hand로 확장되고 있습니다.
>
> **쓰레기 봉투를 묶거나 전구를 돌려 끼우는 동작, 병뚜껑을 비틀어 여는 정밀하고 섬세한 동작은 두 손가락으로는 한계가 있으며, Google DeepMind는 22 자유도를 가진 Sharpa Hands로 이를 가능하게 했다고 공개했습니다.** 인간 수준의 손재주에는 아직 거리가 있으나 진전 속도가 빠르다는 업계의 평가를 받고 있습니다.
>
> 특히, 손가락 수가 늘어나면, 접촉점에서 촉각과 힘 정보를 얻을 수 있다는 장점도 있습니다. 사람이 Manipulation 동작을 할 때를 생각해보면, 유리잔이나 라즈베리는 섬세하게 다루고, 무거운 박스는 힘을 주어 지지합니다. **Glove 방식은 손바닥·손가락 촉각 센서를 추가해 접촉력을 직접 측정할 수 있다는 장점이 있습니다. 대표적인 기업으로는 Sunday, Genesis AI가 있으며, 그들은 Contact-rich 정보를 활용해 Manipulation 성능을 개선하였다고 밝혔습니다.**
>
> 5 Finger Hand를 개발하는 기업은 데이터 수집 방식으로 Glove를 채택하게 됩니다. 앞서 설명드린 Sunday, Genesis AI 모두 Glove 방식으로 손가락 관절각(Action)과 손바닥·손가락 힘 데이터를 함께 기록합니다.
>
> 다만 모든 Task에 대해 5 Finger Hand가 유일한 답이라고 할 수는 없습니다. Foundry Robotics는 나사 조임 작업에서 2 Finger Gripper로 드라이버를 집는 대신 End-effector 자체를 드라이버로 교체하여, 모델이 삽입 동작에만 집중하도록 하드웨어 단에서 해결합니다. 이는 Task design 단계에서 복잡성을 줄여 과도한 설계와 과소 설계 사이에서 모델 성능을 극대화하는 방안으로 볼 수 있습니다.

### 1-2. 데이터 표준화 시도

#### 1-2-1. 수집 인터페이스 — UMI, ALOHA, GELLO

데이터 수집 기구를 오픈소스로 공개하여 다른 연구실과 기업이 그대로 복제해 사용하도록 만든 사례가 사실상의 표준으로 자리잡고 있습니다. 다음 세 가지가 대표적입니다.

![ALOHA](/assets/img/notes/actuate-2026/aloha.jpg)
*ALOHA — 저비용 양팔 Teleoperation 하드웨어 (Tony Zhao et al., 2023)*

**ALOHA(A Low-cost Open-source Hardware system, 2023)**는 **로봇 Teleoperation을 위한 저비용 하드웨어**입니다.

사람이 컨트롤러를 통해 손으로 직접 움직이면 로봇 팔이 동일하게 따라 움직이며, 그 과정에서 명령값과 실제값이 함께 기록됩니다.

Sunday 공동창업자 Tony Zhao가 Stanford 박사과정 중 Chelsea Finn 교수 지도 아래 발표한 연구로, 고가의 산업용 로봇 없이도 양팔 조작 데이터를 수집할 수 있게 하였습니다.

Open Source로 공개되어 있어 많은 로봇 하드웨어 업체가 이를 상용화하고 있으며, 대표적으로 **Trossen Robotics는 ALOHA 구성을 조립 완료 상태로 판매하여 Physical Intelligence 등 많은 RFM 회사에서 채택하고 있는 방식이기도 합니다.**

![GELLO](/assets/img/notes/actuate-2026/gello.jpg)
*GELLO — 보유한 로봇 팔에 맞춰 제작하는 저비용 컨트롤러 (Philipp Wu · Yide Shentu et al., 2024)*

**GELLO(General, Low-Cost, and Intuitive Teleoperation Framework, 2024)**는 ALOHA가 **특정 하드웨어에 종속된다는 한계를 개선한 방식**입니다.

ALOHA는 Controller와 작업용 로봇 팔을 한 세트로 묶어 규정하기 때문에 지정된 기종의 로봇 팔을 새로 구비해야 하지만, GELLO는 이미 보유한 **로봇 팔에 맞는 컨트롤러만 3D 프린팅과 저가 모터를 통해 별도로 만들면 되므로 어느 로봇에든 적용할 수 있습니다.**

XDOF 공동창업자 Philipp Wu와 Yide Shentu가 UC Berkeley 재학 중 발표한 연구이며, 설계와 소프트웨어가 오픈소스로 공개되어 있어 로봇 기종별 컨트롤러를 직접 만들어 쓸 수 있습니다.

**XDOF는 다양한 로봇 팔에 대해 Teleoperation 체계를 갖추는 방법을 표준화한 연구를 기반으로 대규모 범용 데이터를 확보하려는 전략을 취하고 있습니다.**

![UMI](/assets/img/notes/actuate-2026/umi.jpg)
*UMI — 로봇 없이 데이터를 수집하는 핸드헬드 그리퍼 (Cheng Chi et al., 2024)*

**UMI(Universal Manipulation Interface, 2024)**는 로봇 자체를 없앤 접근입니다. Sunday 공동창업자 Cheng Chi가 Stanford·Columbia 박사과정 중 Shuran Song 교수 지도 아래 발표한 연구입니다.

연구의 주된 목표는 2가지로, **로봇 설치를 위한 표준 센서 구성을 확립하는 것과 고가의 로봇 없이도 로봇 데이터를 수집 가능하게 하는 것**이었습니다.

저비용 3D 프린팅 그리퍼를 활용하여 Bay Area 일대에서 1,400개의 데이터를 수집하여 학습된 모델이 Stanford 캠퍼스 어디서든 작동하는 Generalization 성능을 선보였습니다.

UMI는 로봇 팔의 기종과 무관하게 End-effector로 2 Finger Gripper를 사용하는 경우라면 그대로 적용 가능하다는 점에서 확장성이 높습니다.
**Generalist AI를 비롯해 데이터 Scaling을 우선하는 다수 기업이 UMI 계열 방식을 채택하고 있습니다.**

#### 1-2-2. 라벨링 — Subtask, Subgoal Image, Metadata, 활용

**수집된 데이터에 어떤 정보를 덧붙일 것인가**도 표준이 형성되는 중입니다. Physical Intelligence는 이 영역에서 가장 구체적인 방식을 π0.7를 통해 공개하였습니다.

π0.7 모델은 네 가지 요소로 구성된 상세 멀티모달 prompt를 입력 받습니다.

1. **Task Instruction:** 수행할 작업의 자연어 설명입니다.
2. **Subtask Instruction:** 세부 단계 단위의 지시로 Task Instruction만으로는 표현하기 어려운 세부 단계를 모델에 전달합니다. 예를 들어 에스프레소를 만들라는 지시 하나로는 포터필터 삽입, 추출 대기, 액체 운반 같은 개별 동작을 구분할 수 없으나, 단계별 지시를 함께 주면 모델이 현재 어느 국면에 있는지 인식하고 그에 맞는 행동을 생성할 수 있습니다.
3. **Subgoal Images:** 1~2초 후 도달해야 할 미래 상태를 이미지로 제시하는 방식입니다. 언어로 정확히 서술하기 어려운 물리적 동작을 목표 장면으로 직접 지정할 수 있다는 점이 핵심입니다. World Model과 결합되는 지점입니다.
4. **Metadata: 데이터 품질/속도에 대한 관점을 바꾼 요소입니다. 일반적으로 저품질 데이터를 학습 데이터로 활용하면 성능이 정체하거나 하락하지만, 품질이 낮은 데이터임을 미리 라벨링해두고, 모델에 함께 입력한다면 성능 개선에 도움이 된다고 발표했습니다.**

#### 1-2-3. 저장 포맷 — Foxglove MCAP

수집한 데이터를 어떤 형식으로 저장할 것인가에서는 **이번 컨퍼런스 주최사인 Foxglove가 개발하여 Open Source로 공개한 포맷인 MCAP이 사실상 표준으로 자리잡았습니다.**

**MCAP은 로보틱스 시계열 로그를 담기 위한 컨테이너 포맷으로, 여러 센서에서 서로 다른 주기로 들어오는 영상·관절값·촉각 데이터를 하나의 파일에 시간축으로 정렬해 저장합니다.**

특정 시점을 바로 찾아 읽는 랜덤 액세스와, 데이터를 묶는 단위(chunking)를 용도에 맞게 조정할 수 있다는 점이 특징입니다.

실제 활용 케이스는 다음과 같습니다.

1. **Dyna Robotics:** 기존 H5와 프레임 단위 JPEG 조합에서 MCAP으로 전면 전환하였습니다. 자율주행 분야에서 널리 사용되며 랜덤 액세스와 유연한 chunking을 지원한다는 점이 선택 이유였습니다. 다만, Video-action 학습 효율화를 위해 압축·인코딩·chunking 등 일부 기능을 직접 변형한 것으로 밝혔습니다.
2. **1X Technologies** — 개발자 플랫폼에서 수집 데이터를 표준 MCAP 포맷으로 다운로드할 수 있도록 지원하며, 향후 외부 스토리지 연동도 계획하고 있습니다.

---

## 2. 성능 개선: Generalization v.s. Performance

### 2-1. Generalization을 위한 RFM 학습 레시피: 대규모 Pre-training + 소량 로봇 Teleoperation Post-training

범용 RFM을 제작하는데 있어, 주요 RFM 기업들의 발표에 따르면 **"확장 가능한 소스로 대규모 Pre-training을 수행하고, 로봇 데이터는 Post-training 단계에 소량만 사용한다"**는 공통된 RFM 학습 레시피를 확인할 수 있습니다.

| **기업명·모델** | Pre-training 데이터 | Post-training 데이터 | 메모 |
| --- | --- | --- | --- |
| **Dyna Robotics<br>Dyna-2 (2026.8)** | **Human Egocentric**: 1M시간 (일부 hand pose annotation) | **로봇 Teleoperation:** 소량 (몇 분 ~ 몇 시간) | • **Human Egocentric Video를 1K시간에서 1M hr로 늘릴수록 성능이 예측 가능하게 개선되는 Scaling Law 제시**<br>• 로봇 데이터를 전혀 포함하지 않은 조건에서도 물체 간 상호작용을 배우게 하여 사람 → 로봇으로의 전이가 가능함을 입증 |
| **Generalist AI<br>GEN-1.5 (2026.8)** | **UMI Gripper:** 500K시간 | **UMI Gripper:** 소량 약 5분 ~ 1시간 | • **Pre-training에 로봇 데이터를 전혀 쓰지 않고도 scaling만으로도 one-shot, few-shot, zero-shot이 가능함을 발표**<br>• 단, 배포 수준 성능을 위해, Task 별 1시간 규모의 로봇 데이터 필요 |
| **Sunday<br>ACT-2 (2026.7)** | **Glove:** 500개 환경, 10M개 영상 | **로봇 Teleoperation:** 단일 영상 수준으로 새 행동 학습 가능 | • **Pre-training 데이터 규모를 키울수록 학습 시 보지 않은 task, object, 환경 등 성능 격차가 줄어든다 입증**<br>• 같은 데이터 규모에서도 품질에 따라 성능이 크게 갈린다는 점 실험적으로 보임 |
| **Genesis AI<br>GENE 26.5 (2026.5)** | **Human Egocentric + Glove:** 10M시간 (최대 규모) | **Glove**(Task 당 수 시간) + **Simulation**(1M개) | • **LLM의 Pre-training → Post-training → 강화학습(RL) 구조를 로보틱스로 옮겨 Human Egocentric, Glove, Simulation 데이터를 활용한 학습 레시피 공개**<br>• 자체 Open Source Simulation 엔진을 강화학습(RL) 환경으로 활용 |
| **Physical Intelligence<br>π0.6(2025.11)<br>π0.7(2026.4)** | **Human Egocentric + 로봇 Teleoperation + 오픈소스 로봇 데이터** | **로봇 Teleoperation**(소량) + **실시간 로봇 수행 데이터**(RL) | • **유일하게 다양한 데이터를 혼합해 pre-training하며, 멀티모달 prompting을 통해 성능 향상 발견**<br>• 특히, Simulation 환경이 아닌 실시간 로봇 수행 데이터를 강화학습을 통해 학습하여 zero-shot generalization 발견 입증 |

이 표에서 두 가지를 확인할 수 있습니다.

1. **사전학습 단계에서 로봇 데이터를 사용하는 곳은 Physical Intelligence 한 곳뿐입니다.**
    1. 여러 데이터를 혼합하면 데이터 총량은 늘어나지만, 품질·속도·수행 전략이 제각각인 데이터가 함께 들어오므로 모델이 무엇을 기준으로 학습할지 모호해집니다. **Physical Intelligence가 멀티모달 prompt 방식을 채택한 것은 이 모호성을 해소하기 위한 영리한 아이디어로 해석할 수 있습니다.**
    2. 반면, **나머지 네 곳(Dyna Robotics, Generalist AI, Sunday, Genesis AI)은 풀스택을 지향하는 RFM 회사로 데이터 소스를 단일화/이원화하고, 수집 기구(UMI·Glove)를 로봇 End-effector와 동일한 형태로 제작함으로써 H/W 설계 자체를 변수 통제 수단으로 활용합니다.**
2. **Post-training에 사용되는 로봇 데이터의 규모가 극적으로 작아졌습니다.**
    1. Generalist AI는 1분 분량 데이터와 단 1회의 학습만으로 66.5%를, 학습 없이 3~12초 분량의 시연을 입력하는 것만으로 59%를 달성하였습니다. Sunday는 단 하나의 로봇 Teleoperation 영상으로 새로운 행동을 학습시켰고, Dyna Robotics는 13분 학습으로 병뚜껑 열기를 수행하였습니다.
    2. 과거 새로운 작업 하나를 학습시키기 위해 수백 시간의 로봇 Teleoperation 데이터가 필요했던 것과 비교하면 세 자릿수 이상의 감소에 해당합니다.
    3. 다만, 이 구조가 성립하려면 사전학습 데이터를 늘렸을 때 성능이 실제로, 그리고 예측 가능하게 향상된다는 것이 전제되어야 합니다. (Scaling Law)

#### 2-1-1. Generalist AI: Pre-Gen, GEN-0, GEN-1, GEN-1.5 발전

Generalist AI는 약 1년에 걸친 네 번의 모델 발표를 통해, **사전학습을 확대할수록 새로운 작업을 익히는 데 필요한 데이터와 학습량이 계속 줄어든다는 것**을 단계적으로 보여주었습니다.

1. **Pre-Gen (2025년 9월):** 레고 타일 조합을 복제하는 인-컨텍스트 학습을 시연하였습니다. 조합론적으로는 99,000가지 경우의 수가 가능했으나, 모델이 실제로 수행할 수 있는 작업 범위는 레고 3개 타일 조합에 한정되었습니다.
2. **GEN-0 (2025년 11월):** 데이터 규모를 늘릴수록 성능이 일정한 패턴으로 향상되는 예측 가능한 스케일링 법칙을 확인하였습니다. 개발 방향의 타당성을 확보한 시점입니다.
3. **GEN-1 (2026년 4월):** 특정 작업에서 99% 이상의 성공률과 즉흥적 지능(improvisational intelligence)의 초기 징후를 시연하였습니다. 이 시점의 태스크별 로봇 데이터 소요량은 약 1시간 수준으로 보고됩니다.
4. **GEN-1.5 (2026년 8월 19일):** 8개월 이상 연속 사전학습을 진행한 최신 모델로, one-shot·few-shot·zero-shot 역량을 통합적으로 구현하였습니다.

특히, **GEN-1.5의 One-shot In-context Learning은 단일 시연을 모델의 컨텍스트 창에 넣으면 별도 학습 없이 로봇이 즉시 해당 작업을 수행하도록 합니다.**

컨텍스트에 넣는 시연 데이터는 사람이 자신의 손으로 또는 UMI Gripper를 착용하여 기록한 것, 로봇 Teleoperation 또는 이전에 작업을 수행한 기록(rollout), Simulation 환경에서 기록한 것 모두 사용 가능합니다.

10개 태스크 기준 평균 59%의 성공률로 배포 가능한 수준은 아니나, ChatGPT의 Chain-of-Thought 발견으로 인한 기술 발전을 고려하였을 때 의미 있는 성과라고 평가 받고 있습니다.

> ### 💡 Generalist AI의 Emergent Capability
>
> **Emergent Capability**는 모델을 특정 능력을 갖도록 설계하거나 훈련하지 않았음에도, 규모가 커지는 과정에서 그 능력이 저절로 나타나는 현상을 의미합니다.
>
> LLM 분야에서는 모델 크기가 일정 수준을 넘어서면서 이전에 없던 추론이나 번역 능력이 갑자기 관찰된 사례들이 이에 해당합니다.
>
> Generalist AI는 1~5분 짜리 극소수 데이터 학습(Few-shot)이 RFM 모델 가중치를 0.15% 미만만 변경한다는 점을 근거로, fine-tuning이 새로운 표현을 만들어내는 것이 아니라 이미 보유한 지식을 미세하게 재배치하는 것이라고 해석합니다.
>
> **주목할 점은 평가에 사용한 task들이 pre-training 데이터에 포함되지 않았음에도, 학습한 적 없는 작업을 이미 수행할 수 있는 상태였고, fine-tuning은 그 능력을 꺼내는 역할에 그친다는 것입니다.**
>
> 이러한 능력이 별도 설계 없이 Pre-training 데이터의 Scaling만으로 나타났다는 점에서 Emergent Capability로 볼 수 있습니다.

#### 2-1-2. Dyna Robotics: 1M시간 Human Ego-centric Video만으로 이룬 Scaling Law

Dyna Robotics는 Dyna-2 발표에서 로봇 데이터 없이 Human Ego-centric Video만으로 사전학습했을 때 로봇 성능에 스케일링 법칙이 성립하는가라는 질문에 정량적으로 답하였습니다.

**Human Ego-centric Video를 1K시간에서 1M시간까지 늘려가며 성능을 측정한 결과, 데이터를 늘릴수록 성능이 일정한 패턴으로 향상되는 관계가 확인되었다고 하며, 이를 Human Ego-centric Video으로 로봇 성능이 개선된다는 것을 처음으로 보인 사례로 평가 받고 있습니다.**

Dyna는 데이터 유형을 세 가지로 나누어 비교하는 실험을 수행하였습니다. 직접 Annotation한 Action 레이블이 있는 Human Ego-centric Video만 사용한 경우, 여기에 Action 레이블이 없는 순수 영상을 함께 학습시킨 경우, 그리고 순수 영상만 사용한 경우입니다.

핵심 발견은 **Action 레이블이 없는 영상을 늘리는 것만으로도 로봇 성능이 향상된다**는 점이며, Human Ego-centric Video에서 전이되는 지식이 동작 방법이 아니라 물체가 반응하는 방식임을 강조하였습니다.

사람의 팔이 어떻게 움직였는지는 로봇과 신체 구조가 달라 그대로 쓸 수 없지만, 물체를 밀면 어떻게 밀리고 쥐면 어떻게 변형되는지는 로봇에도 동일하게 적용됩니다. 이 때문에 행동 레이블이 없어도 학습 가치가 있다는 설명입니다.

> ### 💡 Scaling의 전제 조건은 ML 인프라
>
> **Dyna Robotics는 1M 시간 규모 학습을 가능하게 하는 ML 인프라 구축이 핵심 병목이었다고 밝히며, 발표에서 "Iteration speed is king"을 반복 강조하였습니다.**
>
> 하루에도 여러 번 빠르게 실험을 반복할 수 있는 환경이 없으면 Scaling 신호 자체를 포착할 수 없다는 것입니다.
>
> **1M시간 데이터셋 규모에서는 데이터가 도착하기를 기다리는 시간(latency)이 수행 가능한 실험 개수를 제한하는 원인임을 분석하여, 저장 포맷을 Foxglove의 MCAP으로 전환하고, 압축·인코딩·chunking 등 일부 기능을 변형하여 최적화하였습니다.**
>
> | 항목 | 개선 전 | 개선 후 |
> | --- | --- | --- |
> | **카메라 1분당 저장 용량** | 80.3MB (JPEG) | 25.1MB (약 68% 감소) |
> | **샘플당 읽기 지연** | 27.0ms | 9.4ms (약 2.9배 개선) |
> | **데이터 수집 처리량** | 주당 14,000 episode-hr | 주당 440,000 (약 31배) |
> | **첫 학습 배치까지 소요 시간** | 약 48시간 | 1분 미만 |

### 2-2. 배포 수준 성능을 위한 강화학습

지금까지 정리한 내용은 Pre-training의 규모를 확대(Scaling)하면 새로운 작업과 환경에 대응하는 능력이 향상된다는 것이었습니다.

그러나 이번 컨퍼런스에서는 이 방향만으로는 **현장 배포하기에 부족하다**는 문제 제기도 함께 이루어졌습니다.

Eka Robotics의 공동 창업자 Pulkit Agrawal은 **로봇이 경제적으로 유용해지려면 성능(Performance)이 필요하며, 이를 신뢰성(Reliability)과 속도(Speed)의 결합으로 정의**하였습니다. 두 가지를 동시에 충족하지 못하면 기술이 아무리 뛰어나도 시장에서 채택되지 않는다는 것입니다.

**이 지점에서 각 기업이 택한 공통된 해법은 강화학습(Reinforcement Learning)입니다. 시연 데이터를 모방하는 방식으로는 시연자보다 잘하기 어렵지만, 강화학습은 반복 시도와 실패를 통해 학습하므로 시연 수준을 넘어설 수 있기 때문입니다.**

**Physical Intelligence는 실환경에서 로봇이 작업을 수행하며 온라인으로 학습하는 강화학습 방식을 택하였습니다.**

1. 로봇이 막힌 상황에 빠지면 사람이 원격 조작(intervention)으로 개입하여 복구 방법을 시연하고 조기 종료함으로써 불필요한 시간 낭비를 방지합니다.
2. 개별 작업마다 성공까지 남은 시간(time-to-success)을 예측하는 별도의 모델(Value Function)을 학습하여 다양한 작업에 걸쳐 좋고 나쁜 행동을 효율적으로 판별합니다.

이를 사전학습된 VLA 모델에 적용한 결과, 초콜릿 공장의 박스 조립 작업에서 처리량(속도)이 2배 향상되었으며, 에스프레소 제조 작업에서 13시간 연속 운용과 90% 이상의 성공률을 달성하였습니다.

**반면 Eka Robotics와 Genesis AI는 Simulation 환경에서 강화학습을 수행하는 방식을 택하였습니다.**

실환경에서 대규모 반복 시도를 하기에는 시간과 비용이 지나치게 크다는 판단 때문이며, Genesis AI는 100만 개 이상의 시뮬레이션 환경을 핵심 전략으로 제시하고 있으나, 업계에서는 Sim-to-Real Gap을 여전히 문제로 제기하고 있습니다.

---

## 3. VLA 다음 아키텍처: Interactive Loop와 Agentic System

지금까지 다룬 내용은 VLA 모델 자체를 어떻게 더 잘 학습시킬 것인가에 관한 것이었습니다.

이번 컨퍼런스에서는 이와 별개로 VLA 바깥에 무엇을 덧붙일 것인가에 대한 두 가지 방향이 제시되었습니다.

**하나는 VLA 학습 파이프라인 상에 물리 세계를 예측하는 World Model를 평가용 Simulator로써 두는 방향이고, 다른 하나는 VLA 모델 앞단에 고수준 추론을 담당하는 Agentic Layer를 두는 방향입니다.**

### 3-1. Veeda AI: World Model을 평가용 Simulator로 활용

Veeda AI는 2026년 7월 설립된 World Model 스타트업입니다. 공동창업자 겸 CEO인 Sanja Fidler는 University of Toronto 교수이자 NVIDIA에서 VP of AI Research로 재직하며 NVIDIA Toronto 연구소를 이끌었고, NVIDIA에서 실시간 Interactive World Model인 OmniDreams 연구를 주도한 인물입니다.

**Yann LeCun의 AMI, Fei-Fei Li의 World Labs에 이어 세 번째로 주목받는 World Model 스타트업**으로 평가되며, 앞선 두 곳이 범용 공간 지능을 지향하는 것과 달리 VLA 성능 개선을 위한 Interactive Loop 개선이라는 목적을 명확히 한다는 점에서 RFM 영역에 가장 가까이 있습니다.

Veeda AI의 문제의식은 현재 Physical AI가 모방 학습(Imitation Learning)에 지나치게 의존하고 있다는 점에서 출발합니다.

로봇 Teleoperation은 사람이 로봇을 조작해 시연 데이터를 모으고 이를 수백만 시간 규모로 확장하는 방식인데, 사람은 모방만으로 배우지 않고 같은 상황을 수천 번 반복 시도하며 상호작용을 기반으로 최적의 행동을 찾아내는 것을 Motivation으로 합니다.

![Physical AI Interaction Loop](/assets/img/notes/actuate-2026/interaction-loop.jpg)
*Actuate 2026 현장에서 촬영한 발표 슬라이드 — Physical AI Interaction Loop*

**Veeda AI가 개발하려는 것은 평가와 개선이 순환하는 폐쇄형 Interactive Loop로, VLA 모델이 Action을 생성하면 World Simulator가 그 결과를 예측하고, 작업 성공 여부를 판정하며, 그 결과가 다시 VLA 모델 개선에 반영되는 구조입니다.**

현재 Simulation 엔진이 이 역할을 하지 못하는 이유는 Graphic Rendering, 사람이 만든 물리법칙 Solver, 아티스트가 제작한 3D Asset에 의존하기 때문이며, 현실 세계의 다양성과 규모를 담아내지 못하기 때문에, 수집되지 않은 상황이나 복잡한 효과는 재현할 수 없음을 지적하고 있습니다.

이에 Veeda AI는 로봇이 취할 Action과 센서 관측값(state)을 학습에 사용하여 물리 세계의 반응(미래 센서 관측값)을 출력하는 World Model 기반의 Simulator(=**Physics World Model**로 명명)를 제작하여 평가와 개선의 순환을 완성하겠다고 발표하였습니다.

### 3-2. Google DeepMind: Agentic System으로 확장

**Google DeepMind는 기본 로봇이 새로운 환경·객체·task에 대응이 어려운 이유는 특정 작업에 특화되어 있기 때문이라는 문제의식에서 출발하여 VLA 위에 고수준 추론 레이어를 얹는 구조를 제시하였습니다.**

Gemini Robotics 2는 하나의 모델이 아니라 두 개의 모델로 구성된 Agentic System입니다.

1. **GR-ER 2 (Orchestrator):** 환경을 이해하고, 자연어 명령을 해석하며, 작업 단계를 계획하고, 필요한 도구를 호출하는 고수준 추론을 담당합니다.
2. **GR 2 (VLA 모델):** 자연어와 시각 입력을 받아 발끝부터 손끝까지 전체 관절을 직접 제어합니다. 네트워크 연결이 필요 없는 온디바이스 실행 버전을 지원하며, 다양한 로봇 형태에 대한 파인튜닝이 가능합니다.

**Figure AI, NVIDIA가 주장하던 계획(System2)과 제어(System1)를 분리한 구조를 Agentic System으로 발전시켰다**고 볼 수 있으며, 이를 통해 발견한 핵심 성과는 다음과 같습니다.

- **다중 로봇 Collaboration**: 두 대의 로봇이 각자 동일한 모델 스택을 실행하면서 개별적으로 추론한 뒤, Agentic Layer를 통해 서로 조율합니다. 각 로봇이 독립적으로 판단하고 역할을 분담하는 구조이기에 협업 과제에 있어 보다 정밀하고, 정확한 판단이 가능합니다.
- **Memory**: GR-ER 2 (Orchestrator)가 장면의 상태를 기억하고 이를 활용하는 기능입니다. 예를 들어 물건을 옮긴 뒤 초기 상태로 되돌리라는 명령을 수행할 수 있습니다.
- **Safety(harness):** 허용·금지 행동 목록 및 안전 규칙을 기반으로 위험한 명령을 거부하며, 로봇이 인간에게 충분히 근접했을때 정지하거나 안전 동작을 실행하는 물리적 Safety layer를 결합하여 안전한 로봇 Action을 출력하도록 합니다. 관련 벤치마크는 오픈소스로 공개되었습니다.

> ### ⚠️ Safety와 Reliability의 딜레마
>
> 로봇의 시야가 가려지는 상황에 대해 Google DeepMind와 Dyna Robotics가 서로 반대되는 대응을 제시하였습니다.
>
> Google DeepMind는 시야가 차단되면 작업을 중단하고 사람에게 상황을 알리는 것을 Safety 기능으로 제시한 반면, Dyna Robotics는 시야가 일부 가려진 상황에서도 작업을 계속 수행할 수 있음을 Reliability의 근거로 강조하였습니다.
>
> 중단하면 안전하지만 작업이 완료되지 않고, 계속 수행하면 작업은 끝나지만 예기치 못한 상황에서 사고 위험이 남습니다.
>
> 가정과 산업 현장 등 실제 배포 환경을 염두에 둘 때, 다양한 Edge Case에 로봇이 어떻게 대처해야 할지에 대한 사전 검토가 필요합니다.
>
> **중단과 지속 중 무엇을 정답으로 볼 것인지가 곧 학습 데이터의 레이블과 평가 기준을 결정하므로, 명확한 기준에 기반한 Task Design이 데이터 수집에 선행되어야 함**을 보여주는 사례로 판단됩니다.

---

## 마치며

이번 컨퍼런스에서 확인된 것은 주요 RFM 기업들이 데이터, 학습, 모델 구조 측면에서 유사한 방향으로 수렴하고 있다는 점입니다.

**데이터는 로봇 없이 모을 수 있는 소스로 이동하되 각 사가 자체 로봇 하드웨어에 맞는 수집 기구를 택하고 있고, 학습은 대규모 사전학습과 소량의 로봇 Teleoperation Post-training이라는 2단 구조로 정리되었으며, 아키텍처는 아래로는 World Model을 두어 평가 루프를 만들고 위로는 Agentic Layer를 두어 고수준 추론을 맡기는 방향으로 확장되고 있습니다.**

다만 발표자들이 공통적으로 강조한 것은 이제 실험실이 아니라 현장에서 검증할 단계라는 점이었습니다.

Physical Intelligence는 창고 물류(Ultra Robotics), 세탁물 폴딩(Weave Robotics), 굴착기 운용(Actor Labs) 등 파트너사와의 실제 배포를 진행 중이고, Google DeepMind는 Boston Dynamics, Agility Robotics, Apptronik과 협력하여 범용 휴머노이드에 자체 모델을 적용하고 있습니다.

파트너십을 통한 실증이 다음 경쟁 지점이 될 것으로 판단되며, **데이터 수집 설계, 배포 가능한 성능을 위한 학습 방식, 그리고 평가 기준의 정의라는 세 단계가 앞으로 협력의 실질적인 접점이 될 것으로 보입니다.**
