# Blackgrass Benchmarks

## Dataset
Our blackgrass dataset is available [here](https://lcas.lincoln.ac.uk/wp/research/data-sets-software/eastern-england-blackgrass-dataset/) .

<div>
  <img src="/resources/BG_sparse_annotated.jpg" width="300" height="300" />
  <img src="/resources/BG_midseason_annotated.jpg" width="300" height="300" />
</div>

## Requirements
I use the [nvcr.io/nvidia/pytorch:22.06-py3](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-06.html#rel_22-06) docker container and install additional requirements.

## Train
To train, run `python train.py` with any options.

## Test
To test, run `python test.py` with any options.

## Paper
```
@article{darbyshire2025multispectral,
  title={Multispectral fine-grained classification of blackgrass in wheat and barley crops},
  author={Darbyshire, Madeleine and Coutts, Shaun and Hammond, Eleanor and Gokbudak, Fazilet and Oztireli, Cengiz and Bosilj, Petra and Gao, Junfeng and Sklar, Elizabeth and Parsons, Simon},
  journal={Computers and Electronics in Agriculture},
  volume={237},
  pages={110484},
  year={2025},
  publisher={Elsevier}
}
```
