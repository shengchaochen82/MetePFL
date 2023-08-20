# Prompt Federated Learning for Weather Forecasting: Toward Foundation Models on Meteorological Data


This is the official repository for the paper [Prompt Federated Learning for Weather Forecasting: Toward Foundation Models on Meteorological Data](https://arxiv.org/abs/2301.09152) <ins> Accepted by IJCAI'23</ins> . 

## Datasets and experiments
All of the dataset utilized in this paper can be found in National Aeronautics and Space Administration [(NASA)](https://www.nasa.gov/)
### Abstract
To tackle the global climate challenge, it urgently needs to develop a collaborative platform for comprehensive weather forecasting on large-scale meteorological data. Despite urgency, heterogeneous meteorological sensors across countries and regions, inevitably causing multivariate heterogeneity and data exposure, become the main barrier. This paper develops a foundation model across regions capable of understanding complex meteorological data and providing weather forecasting. To relieve the data exposure concern across regions, a novel federated learning approach has been proposed to collaboratively learn a brand-new spatio-temporal Transformer-based foundation model across participants with heterogeneous meteorological data. Moreover, a novel prompt learning mechanism has been adopted to satisfy low-resourced sensors' communication and computational constraints. The effectiveness of the proposed method has been demonstrated on classical weather forecasting tasks using three meteorological datasets with multivariate time series.

![Alt](https://github.com/shengchaochen82/MetePFL/blob/main/Framework_MetePFL.png?raw=true)
### Code guideline

SPL in Our Paper:  base_module/pretrain_trans.py/[Novel_Prompting]

Optimization in Our Paper:GraphGenerator.py, aggregator.py

### Please cite our publication if you found our research to be helpful.

```bibtex
@article{chen2023prompt,
  title={Prompt Federated Learning for Weather Forecasting: Toward Foundation Models on Meteorological Data},
  author={Chen, Shengchao and Long, Guodong and Shen, Tao and Jiang, Jing},
  journal={arXiv preprint arXiv:2301.09152},
  year={2023}
}

```

### Contact
If you have any questions, please do not hesitate to contact me (Email: shengchao.chen.uts@gmail.com).

