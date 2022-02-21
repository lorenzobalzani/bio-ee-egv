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

## Datasets

### Original EE datasets
Our BioE2T and BioT2E datasets are derived from 10 influential benchmarks originally designed for biomedical EE (BEE) and primarily released within BioNLP-ST competitions.
For ease of use, we include these freely accessible benchmarks directly within the repository.

<table>
  <tr>
   <th>Corpus</th>
   <th>Domain</th>
   <th>#Documents</th>
   <th>Annotation Schema</th>
  </tr>
  <tr>
   <th>Genia Event Corpus (GE08)</th>
   <th>Human blood cells transcription factors</th>
   <th>1,000 abstracts</th>
   <th>35 entity types, 35 event types</th>
  </tr>
  <tr>
   <th>Genia Event 2011 (GE11)</th>
   <th>See Genia08 </th>
   <th>1,210 abstracts, 14 full papers</th>
   <th>2 entity types, 9 event types, 2 modifiers</th>
  </tr>
  <tr>
   <th>Epigenetics and Post-translational Modification (EPI11)</th>
   <th>Epigenetic change and common protein post-translational modifications</th>
   <th>1,200 abstracts</th>
   <th>2 entity types, 14 event types, 2 modifiers</th>
  </tr>
  <tr>
   <th>Infectious Diseases (ID11)</th>
   <th>Two-component regulatory systems</th>
   <th>30 full papers</th>
   <th>5 entity types, 10 event types, 2 modifiers</th>
  </tr>
  <tr>
   <th>Multi-Level Event Extraction (MLEE)</th>
   <th>Blood vessel development from the subcellular to the whole organism</th>
   <th>262 abstracts</th>
   <th>16 entity types, 19 event types</th>
  </tr>
  <tr>
   <th>GENIA-MK</th>
   <th>See GE08</th>
   <th>1,000 abstracts</th>
   <th>35 entity types, 35 event types, 5 modifiers (+2 inferable)</th>
  </tr>
  <tr>
   <th>Genia Event 2013 (GE13)</th>
   <th>See GE08</th>
   <th>34 full papers</th>
   <th>2 entity types, 13 event types, 2 modifiers</th>
  </tr>
  <tr>
   <th>Cancer Genetics (CG13)</th>
   <th>Cancer biology</th>
   <th>600 abstracts</th>
   <th>18 entity types, 40 event types, 2 modifiers</th>
  </tr>
  <tr>
   <th>Pathway Curation (PC13)</th>
   <th>Reactions, pathways, and curation</th>
   <th>525 abstracts</th>
   <th>4 entity types, 23 event types, 2 modifiers</th>
  </tr>
  <tr>
   <th>Gene Regulation Ontology (GRO13)</th>
   <th>Human gene regulation and transcription</th>
   <th>300 abstracts</th>
   <th>174 entity types, 126 event types</th>
  </tr>
</table>

## Citation

If you found this repository useful, please consider citing the following paper:
...
