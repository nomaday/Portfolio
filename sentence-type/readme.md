# DACON |  Sentence Type Classification AI Competition  
<img width="1024" alt="banner" src="https://user-images.githubusercontent.com/103119868/229272028-6d04e7e7-5610-4df2-8aa9-6c7bab547b1e.png">

- Host: Sungkyunkwan University
- Organizer: DACON
- Timeline: December 12, 2022 → December 23, 2022
- <a href="https://dacon.io/competitions/official/236037/overview/description" target="_blank" rel="noreferrer noopener">Competition site</a>

<br>

## 1. Project Overview  
### Objective
- Sentence type classification AI model development  
    Receive the sentences as the input and create an AI classification model for `type`, `tense`, `polarity`, and `certainty` of the sentence  


### Data
<img width="1208" alt="Untitled 1" src="https://user-images.githubusercontent.com/103119868/218712977-fda1a19b-3121-40bb-81e1-2c3d5f12dcaa.png">

- ID: Unique ID by sample sentence
- Sentence: One sentence per sample
- Type: Type of sentence (factual, inferential, interactive, predictive)
- Polarity: The polarity of the sentence (positive, negative, undecided)
- Tense: The tense of the sentence (past, present, future)
- Certainty: The certainty of the sentence (certain, uncertain)
- Label: Class for each sentence - type, polarity, tense, and certainty (a total of 72 types of classes exist)  
    e.g, Factual-Positive-Present-Certain

<br>
  
## 2. Key action
![Untitled 2](https://user-images.githubusercontent.com/103119868/218712993-2f988986-0f76-4e58-99f5-3f17d7bb9e5b.png)


- While dealing with multi-label classification, survey the relevant reference paper for which model is appropriate for the weighted f1 score, which is the evaluation criteria for the competition
    - <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9826728" target="_blank" rel="noreferrer noopener">BigBird</a> model seemed to fit
- Try to several PLM application of Hugging Face (`monologg/kobigbird-bert-base`, `klue/roberta-large`, `tunib/electra-ko-base`)
    - Apply while adjusting various parameters such as KFold, batch size, max length, learning rate, etc.
- Team Notion management, creation and organization of experiment management template

<br>

## 3. What I learned from the project
- The effect of the ensemble is quite significant.
- Having many options does not necessarily improve performance.
- It would have been nice to try a little more various things during the two-week period.

<br>

## 4. Final Result
- 22<sup>nd</sup> / 333 teams (Top 7%)