## Zincbase - study of tools #1

**ZincBase** is a state of the art knowledge base. It does the following:

1. Extract facts (aka triples and rules) from unstructured data/text
2. Store and retrieve those facts efficiently
3. Build them into a graph
4. Provide ways to query the graph, including via bleeding-edge graph neural networks.

Zincbase exists to answer questions like "what is the probability that Tom likes LARPing", or "who likes LARPing", or "classify people into LARPers vs normies", or simulations like "what happens if all the LARPers become normies".

It combines the latest in **neural networks** with **symbolic logic** (think expert systems and prolog), **graph search**, and **complexity theory**.


### How to install and run
```bash
python3 -m venv venv
source venv/bin/activate

pip3 install -r requirements.txt

python3 start.py
python3 countries.py
...
```

### How to run the WebUI tool
```bash
# running examples of webUI
python -m zincbase.web

# in another terminal
python3 basic.py
python3 gol.py
```

### References

* https://zincbase.readthedocs.io/en/latest/README.html#requirements
* https://github.com/complexdb/zincbase

