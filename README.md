<div align=center>

# [AAAI 2026 Oral] PosterVerse: A Full-Workflow Framework for Commercial-Grade Poster Generation with HTML-Based Scalable Typography

</div>

## 🌟 Highlights
- **PosterVerse**
![Vis_1](fig/posterverse_pipeline.png)
- **PosterDNA**
![Vis_2](fig/dataset.png)

- We propose **PosterVerse**, a full-workflow method that integrates blueprint creation, graphical background generation, and unified layout-text rendering, enabling the creation of posters with aesthetically sophisticated layouts and text-dense designs for commercial-grade use.
- We propose **PosterDNA**, the first commercial-grade and text-dense poster generation dataset with fine-grained HTML-based specifications, designed to support modular training and validation with high-quality samples.
- PosterVerse allows users to generate commercial-grade posters solely from textual prompts.
- Extensive experiments demonstrate that PosterVerse can generate visually appealing posters with aesthetic designs, precise text, and well-crafted layouts, meeting the standards of commercial-grade posters.

## 📏 Evaluation Result
![Vis_7](fig/eval.png)

## 📅 News

- **2026.01.29**: Release the inference code.
- **2026.01.08**: Our [paper](https://arxiv.org/abs/2601.03993) is now available on arXiv.
- **2025.11.08**: 🎉🎉 Our [paper](https://arxiv.org/abs/2601.03993) is accepted by AAAI Oral.

## 🚧 TODO List

- [x] Release inference code
- [ ] Release pretrained model
- [ ] Release a WebUI
- [ ] Release dataset
- [ ] Upload pretrained model to Hugging Face

## 🔥 Model Zoo
| **Stage**    | **Checkpoint** | **Status** |
|----------------------------------------------|----------------|------------|
| Blueprint Creation   | -- | Coming soon  |
| Graphical Background Generation | -- | Coming soon  |
|Unified Layout-Text Rendering | -- | Coming soon  |

## 🚧 Installation

### Environment Setup
Clone this repo:
```bash
git clone https://github.com/wuhaer/PosterVerse.git
```

**Step 0**: Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1**: Create a conda environment and activate it.
```bash
conda create -n posterverse python=3.10 -y
conda activate posterverse
```

**Step 2**: Install the required packages.
```bash
pip install -r requirements.txt
```

## 📺 Inference

**Step 0**: Download all model files from the [Model Zoo](#-model-zoo) 

**Step 1**: Download the [Flux.1 dev](https://hf-mirror.com/black-forest-labs/FLUX.1-dev) model

**Step 2**: Using PosterVerse to generate posters:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python pipe_infer.py \
--stage1_model_path /path/to/Blueprint Creation/model \
--stage3_model_path /path/to/Unified Layout-Text Rendering/model \
--lora_local_path lora_model \
--request_prompt '能不能给我来一张以绿色为主色调、画布大小1080×1920的立春节气海报？风格清新唯美，画面要有树叶、樱花、古风人物等元素。主标题"立春"，配上古诗文案，营造春意盎然的感觉，体现出节气特点，展示春天的美好。'
```

## 💙 Acknowledgement
- [xflux](https://github.com/XLabs-AI/x-flux)
- [Qwen](https://github.com/QwenLM/Qwen3)
- [diffusers](https://github.com/huggingface/diffusers)

## ☎️ Contact
If you have any questions, feel free to contact [Junle Liu](https://github.com/wuhaer) at [junle_liu@foxmail.com](junle_liu@foxmail.com)



## 📜 License
The code and dataset should be used and distributed under [(CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/) for non-commercial research purposes.

## ⛔️ Copyright
- This repository can only be used for non-commercial research purposes.
- For commercial use, please contact Prof. Lianwen Jin (eelwjin@scut.edu.cn).
- Copyright 2026, [Deep Learning and Vision Computing Lab (DLVC-Lab)](http://www.dlvc-lab.net), South China University of Technology. 


## ✒️Citation
If you find PosterVerse helpful, please consider giving this repo a ⭐ and citing:
```latex
@inproceedings{Liu2026posterverse,
      title={PosterVerse: A Full-Workflow Framework for Commercial-Grade Poster Generation with HTML-Based Scalable Typography}, 
      author={Junle Liu, Peirong Zhang, Yuyi Zhang, Pengyu Yan, Hui Zhou, Xinyue Zhou, Fengjun Guo, Lianwen Jin},
      journal={Proceedings of the AAAI Conference on Artificial Intelligence},
      year={2026},
}
```
Thanks for your support!

## 🌄 Gallery
![Vis_3](fig/show.png)
![Vis_4](fig/show1.png)
![Vis_5](fig/show2.png)
![Vis_6](fig/show3.png)

## ⭐ Star Rising
[![Star Rising](https://api.star-history.com/svg?repos=wuhaer/PosterVerse&type=Timeline)](https://star-history.com/#wuhaer/PosterVerse&Timeline)