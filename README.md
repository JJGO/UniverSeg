# UniverSeg: Universal Medical Image Segmentation

### [Project Page](https://universeg.csail.mit.edu) | [Paper](https://arxiv.org/abs/1809.05231)

[![Explore UniverSeg in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19Sauvhyzae5qvVLguaZRCuH1vJ5oTuw-?usp=sharing)<br>

[Victor Ion Butoi](https://victorbutoi.github.io)\*,
[Jose Javier Gonzalez Ortiz](https://josejg.com)\*
[Tianyu Ma](https://www.linkedin.com/in/tianyu-ma-472219174/),
[Mert R. Sabuncu](https://sabuncu.engineering.cornell.edu/),
[John Guttag](https://people.csail.mit.edu/guttag/),
[Adrian V. Dalca](http://www.mit.edu/~adalca/),
 \*denotes equal contribution

 This is the official implementation of the paper ["UniverSeg: Universal Medical Image Segmentation"](https://arxiv.org/abs/1809.05231).

![network](https://raw.githubusercontent.com/JJGO/UniverSeg/gh-pages/assets/images/network-architecture.png)

## Getting Started

The universeg architecture is described in the [`model.py`](https://github.com/JJGO/UniverSeg/blob/main/universeg/model.py) file.

We provide pre-trained model weights a part of our release [link](https://github.com/JJGO/UniverSeg/releases/tag/weights).


## Installation

You can install `universeg` in two ways:

- **With pip**:

```shell
$ pip install git+https://github.com/JJGO/UniverSeg.git
```

- **Manually**: Cloning it and installing dependencies

```shell
$ git clone https://github.com/JJGO/UniverSeg
$ python -m pip install -r ./UniverSeg/requirements.txt
$ export PYTHONPATH="$PYTHONPATH:$(realpath ./UniverSeg)"
```


## Citation

If you find our work or any of our materials useful, please cite our paper:
```
 @inproceedings{butoi2023universeg,
  title={UniverSeg: Universal Medical Image Segmentation},
  author={Victor Ion Butoi and Jose Javier Gonzalez Ortiz and Tianyu Ma and Mert R. Sabuncu and John Guttag and Adrian V. Dalca},
  booktitle={arxiv preprint},
  year={2023}
}
```

## Licenses

- **Code**: Code is released under [Apache 2.0 license](LICENSE)
- **Model Weights**: Model weights are released under [OpenRAIL++-M license](LICENSE-model). According to the usage restriction the model must be only used for research purposes.