# CritiqueFineTuning

This repo contains the code for [Critique Fine-Tuning: Learning to Critique is More Effective than Learning to Imitate](https://arxiv.org/abs/2501.17703). In this paper, we introduce Critique Fine-Tuning (CFT) - a paradigm shift in LLM training where models learn to critique rather than imitate!  

<a target="_blank" href="https://github.com/TIGER-AI-Lab/CritiqueFineTuning">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-black?style=flat&logo=github"></a>
<a target="_blank" href="https://arxiv.org/abs/2501.17703">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-green?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://tiger-ai-lab.github.io/CritiqueFineTuning">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/datasets/TIGER-Lab/WebInstruct-CFT">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/collections/TIGER-Lab/critiquefinetuning-679b25e1528e75180f55e5c4">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>
<br>

## Highlights
Our fine-tuning method can achieve on par results with RL training!

<img width="1432" alt="abs" src="https://tiger-ai-lab.github.io/CritiqueFineTuning/static/images/teaser.png">


## News
- **[2025/01/30]** ⚡️ The paper, code, data, and model for CritiqueFineTuning are all available online.

## Getting Started

### Installation

1. First install LLaMA-Factory:
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

2. Install additional requirements:
pip install -r requirements.txt

### Training Steps

1. First, clone the repository and download the dataset:
```bash
git clone https://github.com/TIGER-AI-Lab/CritiqueFineTuning.git
cd tools/scripts
bash download_data.sh
```

2. Configure model paths in train/scripts/train_qwen2_5-math-7b-cft/qwen2.5-math-7b-cft-webinstruct-50k.yaml

3. Start training:
```bash
cd ../../train/scripts/train_qwen2_5-math-7b-cft
bash train.sh
```

For training the 32B model, follow a similar process but refer to the configuration in train/scripts/train_qwen2_5-32b-instruct-cft/qwen2.5-32b-cft-webinstruct-4k.yaml.

Note: In our paper experiments, we used MATH-500 as the validation set to select the final checkpoint. After training is complete, run the following commands to generate validation scores:
```bash
cd train/Validation
bash start_validate.sh
```
This will create a validation_summary.txt file containing MATH-500 scores for each checkpoint. Select the checkpoint with the highest score as your final model.

## Evaluation

Fill in the model path and evaluation result save path in tools/scripts/evaluate.sh, then run:
```bash
cd tools/scripts
bash evaluate.sh
```
Hardware may have a slight impact on evaluation results based on our testing. To fully reproduce our results, we recommend testing on A6000 GPU with CUDA 12.4 and vllm==0.6.6. For more environment details, please refer to requirements.txt


Note: Our evaluation code is modified from [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) and [MAmmoTH](https://github.com/TIGER-AI-Lab/MAmmoTH).

## Construct Critique Data

To create your own critique data, you can use our data generation script:

```bash
cd tools/self_construct_critique_data
bash run.sh
```
Simply modify the model_name parameter in run.sh to specify which model you want to use as the critique teacher. The script will generate critique data following our paper's approach.


## Citation

Cite our paper as
```
@misc{wang2025critiquefinetuninglearningcritique,
      title={Critique Fine-Tuning: Learning to Critique is More Effective than Learning to Imitate},
      author={Yubo Wang and Xiang Yue and Wenhu Chen},
      year={2025},
      eprint={2501.17703},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.17703},
}
```
