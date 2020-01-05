### Redundancy Reduction

Currently, the (slow) python implementation of redundancy reduction is provided. The parallel C++ implementation will come soon. 

To run the python redundancy reduction, go into the `redundancy_reduction` directory and execute:

```
python rr.py --adj <path to subgraph adj> --round <number of rounds>
```

An example subgraph adj can be found at `../data/subgraphs/yelp_sub.npz`
