
<h1 align="center"> Learning Active Perception via Self-Evolving Preference Optimization for GUI Grounding</a></h1>


<div align="center"> 

<!-- [![Notion](https://img.shields.io/badge/Notion-WebThinker-red?style=flat&logo=notion&logoColor=white)]() s -->
[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv)]()
[![Paper](https://img.shields.io/badge/Paper-Hugging%20Face-yellow?logo=huggingface)]()
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](https://opensource.org/licenses/MIT) 
</div>

<p align="center">
🤗 <a href="" target="_blank">Laser-7B</a> ｜
🤗 <a href="" target="_blank">Laser-7B-GTA1</a>
</p>


<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

## 📣 Latest News
<!-- 
- **[May 1, 2025]**: 🤗 **[WebThinker Model Collection](https://huggingface.co/collections/lixiaoxi45/webthinker-6812d5fd1287ee53d68f0557)** is now available on Hugging Face. You can deploy our optimized models for your deep research tasks.
- **[May 1, 2025]**: 📄 Our paper is now available on **[arXiv](https://arxiv.org/abs/2504.21776)** and **[Hugging Face](https://huggingface.co/papers/2504.21776)**.
- **[March 31, 2025]**: 🎉 **[Laser HomePage](https://wwfnb.github.io/Laser/)** launched with comprehensive project details. -->
- **[September 3, 2025]**: 🚀 Full codebase released. Laser now supports self-envloving pipeline with any models like Qwen2.5-VL-7B or GTA1-7B.


# Release Plans
- [x] Code
  - [x] Data Generation
  - [x] Training
  - [x] Evaluation
- [x] Model
  - [x][Laser(qwen2.5_vl-7b)]
  - [x][Laser(GTA1-7b)]
- [ ] Training Dataset(coming soon)


## 💡 Overview
<p align="center">
  <img src="./figures/performance.png" width="70%" />
</p>

**Laser** is a self-evolving optimization framework, which nables the model to bootstrap its active perception capabilities through rejection sampling–based SFT and region-wise preference learning, without relying on extensive human supervision.

### 📊 Overall Performance


<p align="center">
  <img src="./figures/performance1.png" height="250px" />
  <span style="display:inline-block; width:5%;"></span>
  <img src="./figures/performance2.png" height="250px" />
</p>
As shown above, s. The evaluation covers six GUI domains
and two task types (Text and Icon grounding). Our method,
LASER, consistently outperforms previous models in terms
of both overall grounding accuracy and generalization abil-
ity across different domains, demonstrating the effectiveness
and robustness of our self-evolving training strategy.

### ✨ The Laser Framework

<p align="center">
  <img src="./figures/overview.png" width="70%" />
</p>
<!-- ![Model Comparison](./figures/overview.png) -->

The framework of **Laser** is above. Given a user instruction and the original image, the trained MLASER
model progressively focuses on key regions through a multi-step reasoning process. At each step, the Visual CoT captures
critical cues (highlighted in red within the <think> tag) based on the current focus region. Below, we also illustrate the multi-
stage self-evolving optimization process that elicits LASER’s multi-step active perception capabilities

- **Eliciting Active Perception through Visual Cropping.** Given the paired training data, we prompt the VLM back-
bone M<sub>raw</sub> to predict a focused region. The correspond-
ing region is then cropped from the original image and
integrated into the CoT as visual context, guiding the
model toward accurate click-coordinate prediction. To
improve the quality of reasoning trajectories, we adopt
a STaR-style rejection sampling strategy to construct the
dataset D<sub>sft</sub>, which is used to finetune M<sub>sft</sub>.
- **Learning Focused Region Preferences.** We sample mul-
tiple reasoning trajectories from M<sub>sft</sub> and estimate
region-wise preferences using Monte Carlo Estimation.
An IoU-based filter is applied to remove low-quality can-
didates. The resulting preference pairs dataset D<sub>dpo</sub> are
used to train a stronger model M<sub>dpo</sub> via DPO.
- **Difficulty-Aware Multi-step Perception.** While M<sub>dpo</sub>
supports single-step perception, it is prone to failure in
complex scenarios that demand deeper reasoning. To
overcome this limitation, we allow M<sub>dpo</sub> to iteratively
generate multi-step reasoning trajectories, enabling the
construction of a diverse and difficulty-aware training
data. The final model is then trained on this multi-step
dataset D<sub>⟳</sub>, making it with the ability to dynamically ad-
just reasoning depth based on the difficulty of the query



# 🔧 Installation

##  Install LLaMA-Factory
```bash
conda create --name llama_factory python==3.11
conda activate llama_factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

## Install Laser
```bash
git clone https://github.com/wwfnb/Laser.git
conda create --name Laser python==3.11
conda activate Laser
cd laser
pip install qwen-vl-utils
pip install 'vllm>0.7.2'
pip install -e .
```
The two environments are used separately. Laser is used for data generation and evaluation, while LLaMA-Factory is used for model training.

# 🛠️ Data Generation

The project uses Laser for data generation. Before generating data, you need to download the raw dataset and preprocess it.
Make sure you are in the Laser environment.

## 1️⃣ Step 1: Preprocessing

### 📂 Download Dataset
The data used for generation comes from [GRPO for GUI Grounding Done Right](https://huggingface.co/blog/HelloKKMe/grounding-r1), available on Hugging Face: [grounding_dataset](https://huggingface.co/datasets/HelloKKMe/grounding_dataset/tree/main).Please download the dataset and place it under:

```bash
data/opensource_data
```

### ⚙️ Preprocess Dataset
We preprocess the dataset using the following script:

```bash
python src/laser/prodata_para.py
```
The processed dataset is stored in JSONL format, where each line corresponds to one sample.

Each sample contains:
<pre>
{
  "image_url": "image/dataset/Aria-UI_Data/web/images/screenshot_bb37986a-b810-44db-a28b-5cf5d5bd97cd_part_5.png",
  "instruction": "manage my information preferences.",
  "action_type": null,
  "coordinate": [854, 1034, 1062, 1068],
  "id": "47215f78-38f1-497a-8963-e3538ee32bd7",
  "source": "aria"
}</pre>
💡 Notes:

In the original [grounding_dataset](https://huggingface.co/datasets/HelloKKMe/grounding_dataset/tree/main), bounding box coordinates were normalized to [0, 1000].
During our preprocessing, they are converted into absolute pixel values based on the corresponding image resolution.

## 2️⃣ Step 2: Generation
We generate our dataset in **four stages**, each contributing to a specific training purpose.  
To make it easier to follow, we group the data generation steps by **purpose** and list the corresponding scripts.

### 🔍 **Eliciting Active Perception through Visual Cropping**
  - *Stage 1: Single-step SFT Data Generation* 
```bash
python src/laser/generator/single_step_sft_generator.py
```
### 🎯 **Learning Focused Region Preferences**
  - *Stage 2: Single-step DPO Data Generation* 
```bash
python src/laser/generator/single_step_dpo_generator.py
```
### 🧩 **Difficulty-Aware Multi-step Perception**
  - *Stage 3: Multi-step SFT Data Generation*  
```bash
python src/laser/generator/multi_step_sft_generator.py
```
  - *Stage 4: Multi-step SFT Data Generation*
```bash
python src/laser/generator/multi_step_dpo_generator.py
```

After running the scripts, the processed SFT and DPO datasets will be saved under:
```bash
data/llamafactory_training_data
```
They will follow the LLaMA-Factory training format, making them ready for immediate use in training.
# 🏋️‍♂️ Training
We train our models in **four stages**, using the datasets generated in the corresponding stages of the data generation process.  
Each stage focuses on a specific training purpose.

Make sure you are in the LLaMA-Factory training environment.
### 🔍 **Eliciting Active Perception through Visual Cropping**
  - *Stage 1: Single-step SFT Training* 
```bash
bash scripts/train/train_single_sft.sh
```
### 🎯 **Learning Focused Region Preferences**
  - *Stage 2: Single-step DPO Training* 
```bash
bash scripts/train/train_single_dpo.sh
```
### 🧩 **Difficulty-Aware Multi-step Perception**
  - *Stage 3: Multi-step SFT Training*  
```bash
bash scripts/train/train_multi_sft.sh
```
  - *Stage 4: Multi-step SFT Training*
```bash
bash scripts/train/train_multi_dpo.sh
```


# 📊 Evaluation
We evaluate **Laser** on two widely-used GUI grounding benchmarks:**[ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro)** and **[ScreenSpot-V2](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2)**. 
Make sure you are in the **Laser** environment and have **downloaded the datasets**.  We provide scripts for easy evaluation:

### Evaluate on **ScreenSpot-Pro**
```bash
bash scripts/eval/eval_sceenspot_pro.sh
```
### Evaluate on **ScreenSpot-V2**
```bash
## process the data
python data/benchmark/ScreenSpot-v2/transfer.py
## evaluate
bash scripts/eval/eval_screenspot_v2.sh
```

<!-- ## 📄 Citation

If you find this work helpful, please cite our paper:
```bibtex
@article{Li2025WebThinker,
  author       = {Xiaoxi Li and
                  Jiajie Jin and
                  Guanting Dong and
                  Hongjin Qian and
                  Yutao Zhu and
                  Yongkang Wu and
                  Ji{-}Rong Wen and
                  Zhicheng Dou},
  title        = {WebThinker: Empowering Large Reasoning Models with Deep Research Capability},
  journal      = {CoRR},
  volume       = {abs/2504.21776},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2504.21776},
  doi          = {10.48550/ARXIV.2504.21776},
  eprinttype    = {arXiv},
  eprint       = {2504.21776},
  timestamp    = {Sun, 25 May 2025 20:50:43 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2504-21776.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
``` -->

## 📄 License

This project is released under the [MIT License](LICENSE).

<!-- ## 📞 Contact

For any questions or feedback, please reach out to us at [wwf](). -->
