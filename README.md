# 视觉-文本实体链接使用说明

## 1. 算法描述
该算法用于视觉实体与文本实体之间的链接，构建起跨模态实体间的联系。该算法基于PyTorch框架开发，输入文本与图像，输出文本与图像之间有关系的概率。

## 2. 依赖及安装

CUDA版本: 11.7
其他依赖库的安装命令如下：

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

可使用如下命令下载安装算法包：
```bash
pip install -U mmkg-mm-entity-linking
```

## 3. 使用示例及运行参数说明

```python
from PIL import Image
from mmkg_mm_entity_linking import MMEntityLinking

image = Image.open("path/to/image")
probs = MMEntityLinking().inference(image, ["a dog", "a cat"])
```

## 4. 论文
```
@article{DBLP:journals/corr/abs-2103-00020,
  author    = {Alec Radford and
               Jong Wook Kim and
               Chris Hallacy and
               Aditya Ramesh and
               Gabriel Goh and
               Sandhini Agarwal and
               Girish Sastry and
               Amanda Askell and
               Pamela Mishkin and
               Jack Clark and
               Gretchen Krueger and
               Ilya Sutskever},
  title     = {Learning Transferable Visual Models From Natural Language Supervision},
  journal   = {CoRR},
  volume    = {abs/2103.00020},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.00020},
  eprinttype = {arXiv},
  eprint    = {2103.00020},
  timestamp = {Thu, 04 Mar 2021 17:00:40 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2103-00020.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
