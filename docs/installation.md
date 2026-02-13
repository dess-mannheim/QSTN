# Installation

QSTN supports both local and remote inference via API.

We support two type of installations:
    1. The base version, which only installs the dependencies neccessary to use the API.
    2. The full version, which supports both api and local inference via the vllm API.

To install both of these version you can use `pip` or `uv`.

The base version can be installed with the following command:

```bash
pip install qstn
```

The full version can be installed with this command:

```bash
pip install qstn[vllm]
```

You can install the project from source as well by cloning the repository:

```bash
git clone https://github.com/dess-mannheim/QSTN.git
cd QSTN
```

Then to install the base version:

```bash
pip install -e .
```

Or use this command to install the full version:

```bash
pip install -e .[vllm]
```
