## Installation
```sh 
pip install -r requirements.txt
```
---

## Start the App
```sh 
python -m skactiveml_annotation
```
---

### Adding your own embedding methods
A new embedding method can be added by
implementing a new subclass of `embedding.base.EmbeddingBaseAdapter`.
This class has to be referenced by a Hydra config file
located at `config/embedding/<your-embedding-config-file>`.
Existing embedding configs can be used as an example for the required schema.

---

### Optional dependencies
To make use of the preconfigured embedding methods, additional dependencies
such as PyTorch may be needed.
The tool assumes that the user will create their own embedding methods as
described in
[Adding Your Own Embedding Methods](#adding-your-own-embedding-methods).
