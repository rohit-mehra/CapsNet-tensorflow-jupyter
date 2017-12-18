# CapsNet-tensorflow-jupyter

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=plastic)](https://opensource.org/licenses/Apache-2.0) ![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=plastic)

A Tensorflow implementation of [CapsNet](https://arxiv.org/abs/1710.09829), based on my understanding. This repository is built with an aim to simplify the concept, implement and understand it.

Any suggestions, or any mistake you find, I'll be happy to entertain.

Credits for this amazing comparison: <https://github.com/naturomics/CapsNet-Tensorflow> ![capsVSneuron](capsuleVSneuron.png)

## Requirements

- python3
- numpy==1.13.1
- tensorflow==1.3.0

## Usage

Clone this repository with `git`.

```
$ git clone https://github.com/rrqq/CapsNet-tensorflow-jupyter.git
$ cd CapsNet-tensorflow-jupyter-master
```

### Run Options

1. Use Jupyter Notebook (with visualizations)
2. From linux terminal:

  ```
  $ script -c "sudo python3 main.py" train_log.txt
  ```

3. For Dynamic Loop implementation:

  ```
  $ script -c "sudo python3 dynamic_main.py" train_log_dynamic.txt
  ```

  Test Accuracy (dynamic_main.py): 99.38 % First round with default parameters as in paper. The test set was the default provided by Tensorflow.

### Reference

[In depth analysis.](https://github.com/naturomics/CapsNet-Tensorflow)
