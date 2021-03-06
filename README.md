# GraphACT: Accelerating GCN Training on CPU-FPGA Heterogeneous Platforms

Hanqing Zeng, Viktor Prasanna

Contact: 

Hanqing Zeng (zengh@usc.edu)

**Updates**

03/05/2021: We have released the IP cores for GraphACT at [this repository](https://github.com/GraphSAINT/GNN-ARCH). 
 * The IP cores improve upon the GraphACT design by supporting two computation orders of feature aggregation and weight transformation. See [our ASAP paper](https://ieeexplore.ieee.org/abstract/document/9153263) for description of the two orders.
 * The IP cores now support both the training and inference algorithms on FPGA. We will add in the current repo soon the complete training architecture with those IP cores as the building block. 

We will also soon release the C++ parallel implementation of the redundancy reduction algorithm in the current repo. 

**NOTE**

* The GCN training algorithm, together with the implementation is based on the paper ``Accurate, Efficient and Scalable Graph Embedding'' in IEEE/IPDPS '19.
  * Or, you can refer to our more recent [ICLR '20 paper](https://arxiv.org/abs/1907.04931) (and its [implementation](https://github.com/GraphSAINT/GraphSAINT)) for a better graph sampling based minibatch training algorithm. 
* The implementation for redundancy reduction algorithm, FPGA architecture and the performance model will be uploaded soon. 


**Citation**

```
@inproceedings{graphact,
  author = {Zeng, Hanqing and Prasanna, Viktor},
  title = {GraphACT: Accelerating GCN Training on CPU-FPGA Heterogeneous Platforms},
  year = {2020},
  isbn = {9781450370998},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3373087.3375312},
  doi = {10.1145/3373087.3375312},
  booktitle = {Proceedings of the 2020 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays},
  pages = {255–265},
  numpages = {11},
  location = {Seaside, CA, USA},
  series = {FPGA '20}
}
```
