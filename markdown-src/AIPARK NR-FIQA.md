# No-Reference Face Image Quality Assessment

- 주최: 중소벤처기업진흥공단
- 수행 기간: September 23, 2022 → October 25, 2022

<br>

![https://user-images.githubusercontent.com/103119868/198959589-02cf3b7e-649b-4a35-8daa-faae508a8c6a.gif](https://user-images.githubusercontent.com/103119868/198959589-02cf3b7e-649b-4a35-8daa-faae508a8c6a.gif)

<br>

# 스타트업 기업 연계 프로젝트 (연계 기업: AIPARK)

## 1. 프로젝트 개요

### 목표
- 학습데이터 수집 시 낮은 품질 이미지를 제외하기 위한 얼굴 전용 평가 지표 생성

<br>

## 2. 주요 수행 내용

### 기업에서 요구하는 Task 구체화
- 아바타 제작할 때 필요한 고품질의 이미지란 무엇인가?   
    → 생성되는 아바타 품질을 높이는 데 영향을 줄 수 있는 input 이미지  
    → 얼굴을 인식할 수 있는 수준의 고품질 이미지

### 데이터
- YouTube에서 사람 얼굴 비중이 크고 여러 얼굴 각도를 보여주는 초고화질 영상을 찾아 얼굴 이미지 추출
- Input image size: 384x384

### IQA (Image Quality Assessment) 관련 논문 조사, 용어 정리
- No-Reference Metric 기술 발전 흐름 파악

### FIQA (Face Image Quality Assessment) 관련 논문 조사, 구현 시도
- 얼굴 이미지를 입력으로 받아 어떤 형태의 "품질” 추정치를 출력으로 생성하는 프로세스
- <a href="https://arxiv.org/abs/2003.09373" target="_blank" rel="noreferrer noopener">SER-FIQ : Unsupervised estimation of face image quality</a>를 base model로 설정하고 구현 시도

### 기존 IQA와 FIQA 단독 사용 시 한계점을 확인하고 SER-FIQ(FIQA)와 DBCNN(IQA)을 결합하는 방식 제안
- SER-FIQ (FIQA) 모델로 인식 가능한 수준의 얼굴 이미지를 필터링
- 얼굴에 대해서만 평가하기 위해 배경 요소를 제거하는 Making 작업 수행
- 이후 얼굴 부분만 존재하는 이미지에 IQA를 적용하여 최종 얼굴 이미지 품질 스코어 확인

<br>

## 3. 프로젝트를 통해 배운점

- 얼굴 이미지 품질 평가 (Face Image Quality Assessment)라는 개념을 처음 접하고 알게 되었습니다.
- 스타트업 기업 연계 프로젝트이므로 기업의 관점에서 필요한 것이 무엇인지 더욱 생각해 볼 수 있었고, 실제 현업에서 이루어지는 데이터 수집, 정제 과정에서 겪을 수 있는 어려움이 어떤 부분이 있는지 알 수 있었습니다.

---

### 🪩 프로젝트 GitHub Repository (Organization)
- <a href="https://github.com/yeardreamoff5/aipark" target="_blank" rel="noreferrer noopener">yeardreamoff5/aipark</a>

### 🗃️ 프로젝트 팀 노션 
- <a href="https://www.notion.so/AIPARK-c62dd9ad14534fb791992701a56143b2" target="_blank" rel="noreferrer noopener">(공유)스타트업 기업연계 프로젝트 - AIPARK</a>

### 🪧 최종 발표 자료
- <a href="https://github.com/nomaday/Portfolio/blob/main/markdown-src/final-presentation-aipark.pdf" target="_blank" rel="noreferrer noopener">[제출용] 25조_에이아이파크_최종발표.pdf</a>