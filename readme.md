# TAIIE: Transformational AI Innovation Engine 🚀🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange)](https://pytorch.org/)

[cite_start]**TAIIE** (Transformational AI Innovation Engine) is a novel architecture operationalizing computational creativity theories[cite: 5]. Moving beyond standard generative AI, TAIIE systematically navigates high-dimensional concept spaces to discover novel-yet-valid scientific branches.

[cite_start]It integrates Bayesian information foraging, utility-guided reinforcement learning, and automated paradigm shifts to generate novel-yet-valid conceptual artifacts[cite: 6]. 

## 🔬 Core Philosophy

[cite_start]Current AI systems exhibit limitations in constrained creative generation[cite: 10]. [cite_start]TAIIE addresses these by optimizing the following objective function[cite: 11]:

$$topic^{*} = argmax_{topic}[Validity(t) + \beta Novelty(t)]$$

[cite_start]When the Kullback-Leibler divergence of the novelty distribution exceeds a threshold ($D_{KL} > \tau$), the system triggers a Kuhnian paradigm shift[cite: 22, 24]. [cite_start]This effectively resets the constraints, shifting the model from "Normal Science" to "Extraordinary Science" to explore entirely new semantic manifolds[cite: 21, 23, 24].

## 🏗️ Architecture & Features

This repository contains the full implementation of the TAIIE framework, including:

* **Custom GPT-2 Style Transformer (`ResearchTopicGPT`):** An 8-layer, 8-head self-attention architecture built from scratch in PyTorch to model concept embeddings.
* [cite_start]**The Creativity Engine (`CreativityEngine`):** Operates on three levels[cite: 14]:
    * [cite_start]*Combinatorial:* Token recombination[cite: 15].
    * [cite_start]*Exploratory:* Bayesian sampling within constraints[cite: 16].
    * [cite_start]*Transformational:* Constraint modification[cite: 17].
* **Human-in-the-Loop (HITL) RL Pipeline:** A dynamic feedback loop that evaluates generated topics for validity and scales a novelty threshold ($\tau$), penalizing semantic voids and rewarding genuine innovation.

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/YOUR_USERNAME/TAIIE-AI-Innovation-Engine.git](https://github.com/YOUR_USERNAME/TAIIE-AI-Innovation-Engine.git)
cd TAIIE-AI-Innovation-Engine
pip install -r requirements.txt

🚀 Quick Start
1. Train the Base Model
Train the foundational Transformer on your dataset of existing research topics:

Bash
python Main.py
Note: Ensure your data is formatted as a CSV with a Research Topic column.

2. Run the Creativity Cycles
The Main.py script automatically initializes the CreativityEngine after pre-training, running 5 distinct epochs of generation, human novelty scoring, and transformational fine-tuning.

3. Generate from Checkpoints
Use the Generator.py script to sample concepts from your transformed models:

Bash
python Generator.py --model_path final_creative_model.pth --data_path data.csv --start_words "Quantum" "Neural" --num_topics 5
📊 Evaluation & Limitations
Evaluated on domain-specific datasets (>10k examples), TAIIE achieves 96.2% validity and 37% higher novelty than baselines while dynamically evolving constraints.


Current Limitations: Constraints include vocabulary dependency and computational overhead. Balancing the reward function requires careful tuning. Future work will focus on cross-domain transfer and ethical governance.

📝 Citation
If you use this code or framework in your research, please refer to the corresponding paper:
(Insert link to your published paper or arXiv preprint here)

Built to explore the edges of what machines can conceptualize.


Would you like me to construct a quick `requirements.txt` file based on your `Main.py` imports so you can commit everything at once?
