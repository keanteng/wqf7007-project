﻿# WQF Natural Language Processing Project

![](https://img.shields.io/badge/Python-14354C?style=flat&logo=python&logoColor=white)
![](https://img.shields.io/badge/Shell_Script-121011?style=flat&logo=gnu-bash&logoColor=white)
![Static Badge](https://img.shields.io/badge/python-3.12-blue)
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keanteng/wqf7007-project)

In this course we will explore fine-tuning BERT models for climate sentiment analysis and topic classification using Twitter data. The project is structured to provide a comprehensive understanding of how to leverage Natural Language Processing (NLP) techniques to analyze climate change discourse on social media platforms.

The fine-tuning will be done using Google Colab, A100 GPUs, which are well-suited for training large language models. The project will cover the following key areas such as data collection, preprocessing, model fine-tuning, evaluation, and deployment

## Title

Advancing SDGs through Natural Language Processing: Insights from Twitter Climate Change Discourse

## Using this Repository

Load to your local machine:

```bash
git clone https://github.com/keanteng/wqf7007-project
```

Install the requirements:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
# without hot reload
python app.py

# with hot reload
bin/run.sh
```

## Model Hub & Space

Model /Space | Description | Link
--- | --- | ---
BERT-Base-Raw | Fine-tuned BERT base model for climate sentiment analysis (uncleaned data) | [Model](https://huggingface.co/keanteng/bert-base-raw-climate-sentiment-wqf7007) |
BERT-Base-Cleaned | Fine-tuned BERT base model for climate sentiment analysis (cleaned data) | [Model](https://huggingface.co/keanteng/bert-base-clean-climate-sentiment-wqf7007) |
BERT-Base-Generalized | Fine-tuned BERT base model for climate sentiment analysis (generalized tags) | [Model](https://huggingface.co/keanteng/bert-base-generalized-climate-sentiment-wqf7007) |
BERT-Base-Large-Raw | Fine-tuned BERT large model for climate sentiment analysis (uncleaned data) | [Model](https://huggingface.co/keanteng/bert-large-raw-climate-sentiment-wqf7007) |
Gradio Space | Streamlit app for climate sentiment analysis using BERT-Raw model for inference | [Space](https://huggingface.co/spaces/XIANZHIYI/wqf7007-project) |

## Your Contribution

How to contribute to this project:

```bash
# create a new folder on your local machine then clone the repository
git clone https://github.com/keanteng/wqf7007-project .

# create a new branch
git checkout -b <your_branch_name>

# make your changes
...

# add your changes
git add .

# commit your changes
git commit -m "your commit message"

# push your changes
git push origin <your_branch_name>

# merge your changes
git checkout main
git merge <your_branch_name>

# push your changes to the remote repository
git push origin main

# after you done can delete your branch
git branch -d <your_branch_name>

# delete the remote branch
git push origin --delete <your_branch_name>

# you can also delete the whole folder in your local machine after you done
# to update your local repository
git pull origin main
```

## Members
Name | Student ID
-----|----------
Xian Zhi Yi | 23122622
Zou Jingyi | 23103507
Loong Shih-Wai | 24059242
Khor Kean Teng | U2004763
Huang Lili | 23107324
