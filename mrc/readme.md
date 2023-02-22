# AI CONNECT |  이어드림 경진대회
![Picture1](https://user-images.githubusercontent.com/103119868/220609056-92f55e77-1d5b-4a2d-9541-ab2b5ad3d512.png)


- 주최: 중소벤처기업진흥공단
- 주관: 마인즈앤컴퍼니  
- 수행 기간: November 14, 2022 → November 28, 2022
- <a href="https://aiconnect.kr/competition/detail/217" target="_blank">대회 홈페이지</a>

<br>

## 1. 프로젝트 개요
- 도서자료 검색 효율화를 위한 기계독해 
- 자연어 처리(NLP) | 개방형 문제 | Accuracy 

### 문제정의
- 주어진 지문 속 질문에 대한 답을 찾는 Machine Reading Comprehension

<br>

## 2. 주요 수행 내용
- 기계독해 관련 정보 조사
- Pre-trained Model 활용
    - `monologg/koelectra-base-v3-discriminator` 모델 적용
    - `KLUE-RoBERTa-large` 모델 적용 (대회 종료 후 확인한 Final Score 0.624331282258772로 가장 높음)
- HuggingFace library 활용
    - `ElectraForQuestionAnswering` `ElectraTokenizerFast`
    - `AutoModelForQuestionAnswering`, `AutoTokenizer`
- Parameters 조정
    - Learning rate, Batch size, Epoch
    - CosineAnnealingLR

<br>

## 3. 최종 결과
- Public 1위 / 139명
- Final 2위 / 139명 