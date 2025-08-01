# MulCoT-RD

<p align="center">
    <img src="./assets/logo.png" width="400"/>
<p>

<p align="center">
        🤗 <a href="https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5">Demo (Developed based on Gradio)</a>
</p>

<h3 align="center">
  Resource-Limited Joint Multimodal Sentiment Reasoning and Classification <br/>
  via Chain-of-Thought Enhancement and Distillation
</h3>

## Overview
![overview](./assets/framework.png)

## Notice
- This repository is intended solely for **anonymous review** purposes. All identifying information has been removed.
- Only a portion of the core code has been uploaded at present. The complete codebase will be made publicly available upon acceptance of the paper. Nevertheless, the existing code supports reproduction of our main results.
- This demo focuses on the **MASC** task and utilizes the Qwen2.5-VL-3B model fine-tuned based on the MulCoT-RD framework. Due to local deployment and limited computational resources, response times may occasionally be slow. We appreciate your understanding.

## Requirements
### Runtime environment
Please use the following command to set up your runtime environment:
```
conda env create -f environment.yml
```

### Datasets
All datasets used in this work are publicly available and can be found at the links below:
- **MVSA**:  [[ MVSA-Single + MVSA-Multiple ]](https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/)
- **MASC**:  [[ Twitter-2015 + Twitter-2017 ]](https://github.com/jefferyYu/TomBERT)

If you wish to quickly reproduce our work, you may start by downloading the images of the Twitter-2015 dataset from the link above and placing them under the `.data/T15/Images` directory. We provide CoT-augmented data for the T15 dataset.

### Model Download
- **[[ Qwen/Qwen2.5-VL-7B-Instruct ]](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)**
- **[[ Qwen/Qwen2.5-VL-3B-Instruct ]](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)**
- **[[ sentence-transformers/all-MiniLM-L6-v2 ]](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)**

## Reproduction
Shell scripts are provided to facilitate a quick and easy start.

### Step1: Training the Assistant Model
```
bash shells/assistant_rd.sh
```

### Step2: Data Augmentation based on the Assistant Model
```
shells/assistant_augment.sh
```

### Step3: Training the Student Model
```
shells/student_rd.sh
```

You may also skip all the above steps and execute the following command for one-click setup:
```
shells/main.sh
```

### We appreciate your time and consideration, and hope this repository facilitates your review.
