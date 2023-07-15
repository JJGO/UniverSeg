# UniverSeg: Universal Medical Image Segmentation

### [Project Page](https://universeg.csail.mit.edu) | [Paper](http://arxiv.org/abs/2304.06131)

[![Explore UniverSeg in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TiNAgCehFdyHMJsS90V9ygUw0rLXdW0r?usp=sharing)<br>

Official implementation of ["UniverSeg: Universal Medical Image Segmentation"](http://arxiv.org/abs/2304.06131) accepted at ICCV 2023.

[Victor Ion Butoi](https://victorbutoi.github.io)\*,
[Jose Javier Gonzalez Ortiz](https://josejg.com)\*,
[Tianyu Ma](https://www.linkedin.com/in/tianyu-ma-472219174/),
[Mert R. Sabuncu](https://sabuncu.engineering.cornell.edu/),
[John Guttag](https://people.csail.mit.edu/guttag/),
[Adrian V. Dalca](http://www.mit.edu/~adalca/),  
 \*denotes equal contribution

 

![network](https://raw.githubusercontent.com/JJGO/UniverSeg/gh-pages/assets/images/network-architecture.png)

Given a new segmentation task (e.g. new biomedical domain, new image type, new region of interest, etc), most existing strategies involve training or fine-tuning a segmentation model that takes an image input and outputs the segmentation map.    

This process works well in machine-learning labs, but is challenging in many applied settings, such as for scientists or clinical researchers who drive important scientific questions, but often lack the machine-learning expertiese and computational resources necessary.

UniverSeg enables users to tackle a new segmentation task **without the need to train or fine-tune a model**, removing the requirement for ML experience and computational burden. The key idea is to have a single global model which adapts to a new segmentation task at inference based on an input example set.



## Getting Started

The universeg architecture is described in the [`model.py`](https://github.com/JJGO/UniverSeg/blob/main/universeg/model.py#L125) file.
We provide model weights a part of our [release](https://github.com/JJGO/UniverSeg/releases/tag/weights).

To instantiate the UniverSeg model (and optionally use provided weights):

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

For all inputs ensure that pixel values are min-max normalized to the $[0,1]$ range and that the spatial dimensions are $(H, W) = (128, 128)$.

We provide a jupyter notebook with a tutorial and examples of how to do inference using UniverSeg: [Google colab](https://colab.research.google.com/drive/1TiNAgCehFdyHMJsS90V9ygUw0rLXdW0r?usp=sharing) | [Nbviewer](https://nbviewer.org/github/JJGO/UniverSeg/blob/gh-pages/jupyter/UniverSeg_demo.ipynb#).


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
  journal={International Conference on Computer Vision},
  year={2023}
}
```

## Licenses

- **Code** is released under [Apache 2.0 license](LICENSE)
- **Model Weights** are released under [OpenRAIL++-M license](LICENSE-model). According to the usage restriction the model must be only used for research purposes.
