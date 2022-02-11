# Text-to-Text Extraction and Verbalization of Biomedical Event Graphs

This repository provides the source code & data of our paper: [Text-to-Text Extraction and Verbalization of Biomedical Event Graphs]().
We present a new event linearization approach based on formal grammar, reframing the extraction and verbalization of biomedical events (EE and EGV) as seq2seq tasks.
We introduce BioT2E and BioE2T, two large scale datasets for text- or graph-conditioned biomedical sequence modeling obtained by aggregating and preprocessing gold annotations coming from 10 popular EE benchmarks.
Based on these contributions, we propose baseline transformer model results according to several biomedical text mining benchmarks and NLG metrics, achieving greater or comparable state-of-the-art results than previous solutions.

* [Requirements](#Requirements)
* [Datasets](#Datasets)
  * [Original EE datasets](#Original-EE-datasets)
  * [Preprocessing](#Preprocessing)
  * [BioE2T](#BioE2T)
  * [BioT2E](#BioT2E)
* [Models](#Models)
  * [Training](#Training)
  * [Evaluation](#Evaluation)
* [Citation](#Citation)

## Requirements

General
- Python (verified on 3.8)
- CUDA (verified on 11.1)

Python Packages
- See requirements.txt

## Citation

If you found this repository useful, please consider citing the following paper:
...
