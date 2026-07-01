# Third-Party Notices

OmniMemEval code is licensed under the repository [LICENSE](./LICENSE). Benchmark
datasets keep their upstream licenses. The OmniMemEval code license does not
relicense external datasets.

## LoCoMo

- Source: https://github.com/snap-research/locomo
- Paper: https://arxiv.org/abs/2402.17753
- Data file used by OmniMemEval: `data/locomo/locomo10.json`
- License: Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0)
- Notes: the data file is not committed to this repository. Run
  `python data/locomo/prepare_locomo.py` to download it from upstream. Use of
  the data must remain non-commercial and include attribution to the LoCoMo
  authors.

## LongMemEval

- Source repository: https://github.com/xiaowu0162/LongMemEval
- Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
- Paper: https://arxiv.org/abs/2410.10813
- Data file used by default: `data/longmemeval/longmemeval_s_cleaned.json`
- License: MIT
- Notes: LongMemEval data is not committed to this repository. Run
  `python data/longmemeval/prepare_longmemeval.py` to download the default S
  variant.

## BEAM

- Source repository: https://github.com/mohammadtavakoli78/BEAM
- Dataset: https://huggingface.co/datasets/Mohammadta/BEAM and
  https://huggingface.co/datasets/Mohammadta/BEAM-10M
- Paper: https://arxiv.org/abs/2510.27246
- Data file used by default: `data/beam/beam_100k.json`
- License: Creative Commons Attribution-ShareAlike 4.0 International
  (CC BY-SA 4.0)
- Notes: BEAM data is not committed to this repository. Run
  `python data/beam/prepare_beam.py` to download the default 100K scale.

## PersonaMem v2

- Source repository: https://github.com/bowen-upenn/PersonaMem-v2
- Dataset: https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2
- Paper: https://arxiv.org/abs/2512.06688
- Data used by default: benchmark CSV and 32K chat histories referenced by the
  benchmark split
- License: MIT
- Notes: PersonaMem v2 data is not committed to this repository. Run
  `python data/personamem_v2/prepare_personamem.py` to download the default
  evaluation files.

## HaluMem

- Source repository: https://github.com/MemTensor/HaluMem
- Dataset: https://huggingface.co/datasets/IAAR-Shanghai/HaluMem
- Paper: https://arxiv.org/abs/2511.03506
- Data file used by default: `data/halumem/HaluMem-Medium.jsonl`
- License: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
  International (CC BY-NC-ND 4.0)
- Notes: HaluMem data is not committed to this repository. Run
  `python data/halumem/prepare_halumem.py` to download the default Medium
  variant.
