# Start-up linked projects (Company: AIPARK)

- Host: Korea SMEs and Startups Agency
- Timeline: September 23, 2022 → October 25, 2022

<br>

# No-Reference Face Image Quality Assessment

<p style="text-align: center;"><img width="700" alt="CAFI Workflow" src="https://user-images.githubusercontent.com/103119868/198959589-02cf3b7e-649b-4a35-8daa-faae508a8c6a.gif"></p>

<br>

## 1. Project Overview

### Objective
- Creating face-only evaluation metric to exclude low-quality face images when collecting training data

<br>

## 2. Key action

### Clarification of tasks required by company
- What is the high-quality image needed to create an avatar?   
    → Input images that can affect positively the quality of generated avatars  
    → High-quality images that can recognize faces

### Data
- Extract face images by finding UHD videos on YouTube that show a large proportion of human faces and various facial angles
- Input image size: 384x384

### Research papers related to IQA (Image Quality Assessment), glossary of terms
- Understanding the flow of No-Reference Metric technology development

### Research papers related to FIQA (Face Image Quality Assessment), attempt to implement the thesis
- A process that takes an image of a face as input and produces some form of "quality" estimate as output
- Set "<a href="https://arxiv.org/abs/2003.09373" target="_blank" rel="noreferrer noopener">SER-FIQ : Unsupervised estimation of face image quality</a>" as a base model and attempt to implement

### Identify limitations when using existing IQA and FIQA alone and propose a combination of SER-FIQA (FIQA) and DBCNN (IQA) methods
- Filter face images recognizable levels with the SER-FIQ (FIQA) model
- Masking to remove background elements to evaluate only for face
- Afterwards, IQA is applied to images with only the face part to determine the final face image quality score

<br>

## 3. What I learned from the project

- I first encountered and learned about the concept of Face Image Quality Assessment.
- Since it is a start-up company-linked project, I was able to think more about what was needed from the point of view of the company, and I was able to know what difficulties might be encountered in the process of collecting and refining data in the actual field.

<br>

---

### 🪩 Project GitHub Repository (Organization)
- <a href="https://github.com/yeardreamoff5/aipark" target="_blank" rel="noreferrer noopener">yeardreamoff5/aipark</a>

### 🗃️ Project Team Notion 
- <a href="https://www.notion.so/AIPARK-c62dd9ad14534fb791992701a56143b2" target="_blank" rel="noreferrer noopener">(공유)스타트업 기업연계 프로젝트 - AIPARK</a>

### 🪧 Final Presentation Material
- <a href="https://github.com/nomaday/Portfolio/blob/main/markdown-src/final-presentation-aipark.pdf" target="_blank" rel="noreferrer noopener">[제출용] 25조_에이아이파크_최종발표.pdf</a>