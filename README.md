## Setup
### Recommended: Setup and active virtual environment (venv)
*Create and active virtual environment:*
```sh 
$ python -m venv .venv
$ .venv/Scripts/activate
```

*Install dependencies:*
```sh 
$ pip install -r requirements.md
```

*With Cuda support
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

*Without Cuda support
```sh
pip install torch torchvision torchaudio
```


## Start the App
*Make sure the venv is enabled, if used in setup and not already done so:*
```sh 
$ .venv/Scripts/activate
```

*Run the app:*
```sh 
$ python src/run.py
```


