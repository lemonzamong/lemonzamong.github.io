---
layout: post
title:  "GNN-Transformer Task Planning"
date:   2025-10-01 11:03:36 +0900
categories: 논문 프로젝트 학회 로봇 AAAI
---

본 포스트에서는 2024년 6월 ~ 12월 동안 학부 연구생으로 한양대학교 로봇 AI Lab(Robots with Humans Lab)에 참여하며 진행했던 'GNN-Transformer Task Planning' 연구에 대해 기록하고자 합니다. 이 연구는 자연어 명령을 이해하고 실제 환경에서 복잡한 작업을 수행할 수 있는 로봇을 위한 새로운 태스크 플래닝(Task Planning) 기술 개발을 목표로 하였습니다.

## 서론 (Introduction)

### 필요성 및 목적
현실 세계에서 인간과 로봇이 자연스럽게 상호작용하기 위해서는, 로봇이 사람의 언어적 지시를 이해하고 그에 맞는 작업을 계획하고 실행하는 TAMP(Task and Motion Planning) 기술이 필수적입니다. 본 연구는 자연어 기반의 TAMP를 실제 로봇 환경에 적용하는 것을 목표로 합니다.

### 기존 연구의 한계
1.  **LLM 기반 플래너의 '그라운딩(Grounding)' 부재**: LLM은 그럴듯한 계획을 생성하지만, 물리 세계에 대한 구조적 이해(어포던스, 상태, 관계 등)가 부족하여 종종 실행 불가능한 계획을 만듭니다.
2.  **데이터 부족**: 로봇 학습에는 `<명령어, 상태, 행동>`으로 구성된 대규모 고품질 데이터가 필요하지만, 이를 제작하는 비용이 매우 높아 연구의 큰 걸림돌이었습니다.

### 핵심 아이디어
이러한 문제를 해결하기 위해 저희는 두 가지 핵심 아이디어를 제안했습니다.
*   데이터 병목 현상을 해결하기 위한 **새로운 데이터 자동 생성 파이프라인**
*   LLM보다 현실에 더 잘 기반하면서 효율적인 **그래프 기반 신경망 플래너(GTTP)**

## 1. Semantic-Driven Data Augmentation

[그림 1: Semantic-Driven Data Augmentation 프로세스]
![Data Augmentation Process](/assets/img/project-gnn-tamp/slide_9.png)

데이터 부족 문제를 해결하기 위해, 적은 수의 '시드 데이터'를 사람의 개입 없이 대규모 데이터셋으로 확장하는 파이프라인을 개발했습니다. 45개의 소규모 시드 데이터셋(`D_seed`)을 14,000개 이상의 다양한 데이터셋(`D'`)으로 확장했습니다.

**프로세스:**
1.  **Object Candidate Grouping**: 시드 데이터에 사용된 객체와 대체 가능한 후보 객체들을 그룹화합니다.
2.  **Semantic Filtering**: 후보 객체가 원래 객체와 같은 공간(room), 카테고리(category), 속성(properties)을 가질 때만 대체하여 계획의 의미적, 물리적 타당성을 보존합니다.
3.  **Combinatorial Plan Generation**: 필터링된 후보군에서 객체를 샘플링하여 수많은 새로운 계획을 조합적으로 생성합니다.
4.  **LLM-based Verification**: GPT-4와 CoT(Chain-of-Thought) 프롬프트를 사용하여 의미적으로는 맞지만 상식적으로 이상한 계획(예: 냉장고에 책 넣기)을 최종적으로 걸러냅니다.

## 2. GNN-Transformer Task Planner (GTTP)

[그림 2: GTTP 전체 아키텍처]
![GTTP Overall Architecture](/assets/img/project-gnn-tamp/slide_5.png)

저희가 제안하는 GTTP는 GNN과 Transformer를 결합하여 주어진 명령과 현재 환경 상태를 기반으로 다음 행동을 예측하는 모델입니다.

**작동 방식:**
1.  **Semantic Alignment**: 언어 명령어(Instruction)와 환경 상태(Scene Graph)를 Sentence-BERT를 이용해 동일한 잠재 공간으로 임베딩합니다.
2.  **State Token 생성**: GNN을 통해 Scene Graph의 시계열 데이터(Graph History)와 명령어를 처리하여, 현재 태스크에 가장 중요한 정보를 담은 '상태 토큰(State Tokens)'을 생성합니다.
    *   **Graph Feature Extractor**: GNN과 Cross-Attention을 통해 "이 명령어를 위해 지금 세상에서 무엇이 중요한가?"를 파악합니다. 예를 들어 "사과를 냉장고에 넣어"라는 명령이 주어지면, 초기에는 '사과'에, 후반에는 '냉장고'에 더 집중합니다.
3.  **Subgoal 예측**: Transformer 디코더가 상태 토큰 시퀀스를 입력받아, 다음 행동(action)과 대상 객체(object)의 확률 분포를 예측합니다.

**학습 전략:**
*   **Node Dropout**: 학습 중 태스크와 무관한 배경 객체들을 30~90% 확률로 무작위 제거합니다. 이를 통해 모델이 환경 변화에 강건해지고, 처음 보는 환경에 대한 일반화 성능을 크게 향상시킵니다.

## 3. Sim2Real: 실제 로봇 적용

[그림 3: 실제 로봇 시연]
![Real-World Demonstration](/assets/img/project-gnn-tamp/slide_11.png)

GTTP가 생성한 상징적 계획(`symbolic plan`, 예: `(Walk, kitchen)`)을 실제 로봇(Husky+Panda)에 적용하기 위해 End-to-End 시스템을 통합했습니다.

*   **인식 파이프라인 (Perception)**: SLAM(Cartographer)으로 로봇의 위치를 추정하고, ArUco 마커를 이용해 목표 객체의 6D Pose를 인식합니다. 이를 통해 '사과'라는 추상적 개념을 실제 세계의 물리적 좌표에 '그라운딩'합니다.
*   **제어 아키텍처 (Control)**: ROS2 기반의 계층적 제어 시스템을 구축하여 GTTP의 서브골을 로봇의 연속적인 모션 명령으로 변환합니다.
    *   `Walk(주방)` → Nav2 목표 지점 명령
    *   `Open(찬장)` → MoveIt2 경로 계획 목표

## 4. 실험 결과 (Key Results & Analysis)

[그림 4: 실험 결과표]
![Experiment Results](/assets/img/project-gnn-tamp/slide_12.png)

GTTP는 기존의 SOTA 모델들(GPT-4, GOALNET 등)과 비교하여 **Seen/Unseen 환경 모두에서 월등한 성능**을 보였습니다. 특히, 태스크가 길어질수록 다른 모델들은 성능이 급격히 저하된 반면, GTTP는 짧은, 중간, 긴 태스크 전반에서 일관된 우수성을 나타냈습니다.

**Ablation Study**를 통해 어텐션 기반 그라운딩 메커니즘과 Node Dropout 기법이 모델의 성능에 결정적인 역할을 한다는 것을 입증했습니다.

## 5. 결론 및 한계점

### 성과
1.  **Semantic-Driven Data Augmentation**: 최소한의 노력으로 고품질 대규모 학습 데이터를 자동 생성하는 방법을 개발했습니다.
2.  **GNN-Transformer Task Planner (GTTP)**: 확장성 높고 효율적이며, 실제 환경에서도 잘 동작하는 새로운 태스크 플래너를 개발하여 SOTA 성능을 달성하고 Sim-to-Real 검증에 성공했습니다.

### 한계점
*   데이터 증강 방식이 아직은 기존 태스크와 유사한 패턴을 벗어나기 어렵습니다.
*   로봇이 임무 중 실패했을 때, 스스로 계획을 수정하는 **재계획(Replanning) 기능**은 아직 부족합니다.

이 프로젝트를 통해 얻은 가장 큰 성과는 'GNN-Transformer Task Planning...' 이라는 제목으로 **AAAI 2025 학회에 공저자로 논문을 게재**하게 된 것입니다. 학부생으로서 아이디어를 내고, 데이터 파이프라인을 구축하며, 실제 로봇 시스템에 적용하는 전 과정에 참여하여 세계적인 학회에 성과를 인정받았다는 점에서 큰 보람을 느꼈습니다.

앞으로 이러한 한계점을 개선하여 더욱 지능적이고 강건한 로봇 시스템을 구축해나갈 계획입니다.