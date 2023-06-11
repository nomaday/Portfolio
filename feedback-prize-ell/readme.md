# Kaggle | Feedback Prize - English Language Learning

<img width="960" alt="banner" src="https://github.com/nomaday/Portfolio/assets/103119868/65fd5550-d95b-4740-8422-1f0693e5b74c">


- Host: Vanderbilt University, The Learning Agency Lab
- Organizer: Kaggle
- Timeline: August 30, 2022 → November 29, 2022
- <a href="https://www.kaggle.com/competitions/feedback-prize-english-language-learning/overview" target="_blank" rel="noreferrer noopener" target="_blank" rel="noreferrer noopener">Competition site</a>

<br>
 
## 1. Project Overview

### Objective
- Predicting the scores of each of the six analytic measures (`cohesion`, `syntax`, `vocabulary`, `phraseology`, `grammar`, `conventions`) for a given essay to improve automated feedback tools that assesses the language proficiency of 8th-12th grade English Language learners (ELLs)

### Data
- The presented data set (ELLIPSE corpus) consists of argumentative essays written by English language learners (ELLs) in grades 8-12.
- Essays are scored according to six analysis indicators: cohesion, syntax, vocabulary, phraseology, grammar, and conventions. Each indicator represents a component of essay writing ability, and higher scores mean higher proficiency in that scale.

<br>

## 2. Key action

### Model Ensemble
- Weighted inference through the pre-trained ensemble 10 models

### Post-processing
- Average adjustment: Adjust the average of the inferred values and the average of the train data to be the same by adding a constant to the inference result for each indicator.
- Distribution adjustment: Adjust the distribution of the inferred values and the distribution of the train data to be the same by multiplying a constant to the inference result for each indicator.
- Adjustment by score unit (0.5): Adjust the inferred result for each indicator by 0.5 units, which is the actual score unit.

<br>

## 3. What I learned from the project

- In the meantime, if I had only thought of a data or model-oriented approach, I tried a post-processing approach in this project.
- I felt that it is important to not give up until the end in the line that can be done.

<br>

## 4. Final Result
- Got a Silver medal (Top 5%, 130th / 2654 teams)

    <img width="257" alt="medal" src="https://user-images.githubusercontent.com/103119868/236372435-f860c03d-94d2-439c-b6f1-62959f7fa034.png">