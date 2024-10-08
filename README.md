# Prompt Federated Learning for Weather Forecasting: Toward Foundation Models on Meteorological Data


This is the official repository for the paper [Prompt Federated Learning for Weather Forecasting: Toward Foundation Models on Meteorological Data](https://arxiv.org/abs/2301.09152) <ins> Accepted by IJCAI'23</ins> . 


### Abstract
To tackle the global climate challenge, it urgently needs to develop a collaborative platform for comprehensive weather forecasting on large-scale meteorological data. Despite urgency, heterogeneous meteorological sensors across countries and regions, inevitably causing multivariate heterogeneity and data exposure, become the main barrier. This paper develops a foundation model across regions capable of understanding complex meteorological data and providing weather forecasting. To relieve the data exposure concern across regions, a novel federated learning approach has been proposed to collaboratively learn a brand-new spatio-temporal Transformer-based foundation model across participants with heterogeneous meteorological data. Moreover, a novel prompt learning mechanism has been adopted to satisfy low-resourced sensors' communication and computational constraints. The effectiveness of the proposed method has been demonstrated on classical weather forecasting tasks using three meteorological datasets with multivariate time series.

![Alt](https://github.com/shengchaochen82/MetePFL/blob/main/Framework_MetePFL.png?raw=true#pic_center)

### Code guideline

Path of the proposed SPL:  base_module/pretrain_trans.py/[Novel_Prompting]

Path of the framework optimization: GraphGenerator.py, aggregator.py

## Datasets and experiments
All of the dataset utilized in this paper can be found in National Aeronautics and Space Administration [(NASA)](https://www.nasa.gov/)
![Alt](https://github.com/shengchaochen82/MetePFL/blob/main/Experiment.png?raw=true#pic_center)

**Simple Instruction for Dataset Processing**: [(Dataset)](https://disc.gsfc.nasa.gov/datasets/M2C0NXASM_5.12.4/summary) To use the datasets in our paper, first select a subset from the NASA-provided dataset based on the client number and latitude/longitude ranges detailed in the Appendix of [this paper](https://arxiv.org/abs/2305.14244). Then, process it in array format.
*Self-processing ensures that the data is appropriate for your particular research scenario and complies with policies regarding data use and sharing.*

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

