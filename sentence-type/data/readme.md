## <a href="https://dacon.io/competitions/official/236037/data" target="_blank">Dataset Info.</a>

### train.csv [파일]
- ID : 샘플 문장 별 고유 ID
- 문장 : 샘플 별 한개의 문장
- 유형 : 문장의 유형 (사실형, 추론형, 대화형, 예측형)
- 극성 : 문장의 극성 (긍정, 부정, 미정)
- 시제 : 문장의 시제 (과거, 현재, 미래)
- 확실성 : 문장의 확실성 (확실, 불확실)
- label : 문장 별 유형, 극성, 시제, 확실성에 대한 Class (총 72개 종류의 Class 존재)
    - 예시) 사실형-긍정-현재-확실

### test.csv [파일]
- ID : 샘플 문장 별 고유 ID
- 문장 : 샘플 별 한개의 문장

### sample_submission.csv [파일] - 제출 양식
- ID : 샘플 문장 별 고유 ID
- label : 예측한 문장 별 유형, 극성, 시제, 확실성에 대한 Class
    - 예시) 사실형-긍정-현재-확실