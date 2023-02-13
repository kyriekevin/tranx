# ML-TranX and ML-TranX'
Those two model are based on a general-purpose **Tran**sition-based abstract synta**X** parser [ACL '18 paper](https://arxiv.org/abs/1806.07832) and  
TRANX' is a variant of TRANX which decodes in a breadth-first manner. By adopting a mutual learning based model training framework, both models can fully absorb the knowledge from each other and thus could be improved simultaneously.


## System Architecture
The System Architecture is the same as TRANX [ACL '18 paper](https://arxiv.org/abs/1806.07832) which can be referred to https://github.com/pcyin/tranX .


### Evaluation Results

Here is a list of performance results on six datasets using pretrained models in `data/saved_models`

- **ML-TRANX** :
| Dataset | Results      | Metric             |
| ------- | ------------ | ------------------ |
| Django    | 79.3         | Accuracy           |
| ATIS     | 89.7         | Accuracy           |
| GEO      | 90.0         | Accuracy           |
| IFTTT     | 86.6         | Full Tree Accuracy |



- **ML-TRANX'** :
| Dataset | Results      | Metric             |
| ------- | ------------ | ------------------ |
| Django    | 78.8         | Accuracy           |
| ATIS     | 88.2         | Accuracy           |
| GEO      | 89.2         | Accuracy           |
| IFTTT     | 85.4        | Full Tree Accuracy |

## Usage

```bash
cd tranX
conda env create -f config/env/tranx.yml  # create conda Python environment.
./scripts/django/train-mutual-share-embedding.sh 0  # train on django code generation dataset  with random seed 0
./scripts/atis/train-mutual-share-embedding.sh 0  # train on ATIS semantic parsing dataset
./scripts/geo/train-mutual-share-embedding.sh 0  # train on GEO dataset
./scripts/ifttt/train-mutual-share-embedding.sh 0  # train on IFTTT dataset

./scripts/django/test_used.sh    # modify the configuration to test the model on django code generation dataset 
./scripts/atis/test_used.sh     # modify the configuration to test the model on atis code generation dataset 
./scripts/geo/test_used.sh      # modify the configuration to test the model on geo code generation dataset 
./scripts/ifttt/test_used.sh     # modify the configuration to test the model on ifttt code generation dataset 
```

