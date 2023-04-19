# UniverSeg: Universal Medical Image Segmentation

### [Project Page](https://universeg.csail.mit.edu) | [Paper](http://arxiv.org/abs/2304.06131)

[![Explore UniverSeg in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19Sauvhyzae5qvVLguaZRCuH1vJ5oTuw-?usp=sharing)<br>

[Victor Ion Butoi](https://victorbutoi.github.io)\*,
[Jose Javier Gonzalez Ortiz](https://josejg.com)\*,
[Tianyu Ma](https://www.linkedin.com/in/tianyu-ma-472219174/),
[Mert R. Sabuncu](https://sabuncu.engineering.cornell.edu/),
[John Guttag](https://people.csail.mit.edu/guttag/),
[Adrian V. Dalca](http://www.mit.edu/~adalca/),
 \*denotes equal contribution

 This is the official implementation of the paper ["UniverSeg: Universal Medical Image Segmentation"](http://arxiv.org/abs/2304.06131).

![network](https://raw.githubusercontent.com/JJGO/UniverSeg/gh-pages/assets/images/network-architecture.png)

## Getting Started

The universeg architecture is described in the [`model.py`](https://github.com/JJGO/UniverSeg/blob/main/universeg/model.py#L125) file.
We provide pre-trained model weights a part of our [release](https://github.com/JJGO/UniverSeg/releases/tag/weights).

To instantiate the UniverSeg model (and optionally use  pre-trained weights):

```python
from universeg import universeg

model = universeg(pretrained=True)

# To perform a prediction (where B=batch, S=support, H=height, W=width)
prediction = model(
    target_image,        # (B, 1, H, W)
    support_images,      # (B, S, 1, H, W)
    support_labels,      # (B, S, 1, H, W)
) # -> (B, 1, H, W)

```

We provide a jupyter notebook with examples of how to do inference using UniverSeg: [Google colab](https://colab.research.google.com/drive/1TiNAgCehFdyHMJsS90V9ygUw0rLXdW0r?usp=sharing) | [Nbviewer](https://nbviewer.org/github/JJGO/UniverSeg/blob/gh-pages/jupyter/UniverSeg_demo.ipynb#).


## Installation

You can install `universeg` in two ways:

- **With pip**:

```shell
pip install git+https://github.com/JJGO/UniverSeg.git
```

- **Manually**: Cloning it and installing dependencies

```shell
git clone https://github.com/JJGO/UniverSeg
python -m pip install -r ./UniverSeg/requirements.txt
export PYTHONPATH="$PYTHONPATH:$(realpath ./UniverSeg)"
```


## Citation

If you find our work or any of our materials useful, please cite our paper:
```
 @article{butoi2023universeg,
  title={UniverSeg: Universal Medical Image Segmentation},
  author={Victor Ion Butoi* and Jose Javier Gonzalez Ortiz* and Tianyu Ma and Mert R. Sabuncu and John Guttag and Adrian V. Dalca},
  journal={arXiv:2304.02643},
  year={2023}
}
```

## Licenses

- **Code** is released under [Apache 2.0 license](LICENSE)
- **Model Weights** are released under [OpenRAIL++-M license](LICENSE-model). According to the usage restriction the model must be only used for research purposes.
