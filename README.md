[![PyPI-Server](https://img.shields.io/pypi/v/wobbegong.svg)](https://pypi.org/project/wobbegong/)
![Unit tests](https://github.com/BiocPy/wobbegong/actions/workflows/run-tests.yml/badge.svg)

# wobbegong

wobbegong converts Bioconductor objects (like `BiocFrame`, `SummarizedExperiment`, and `SingleCellExperiment`) into a set of static files optimized for HTTP range requests.

It includes a native Python client that allows you to query these datasets remotely, fetching only the specific genes or metadata columns you need without downloading the entire file.

> [!NOTE]
>
> Check out the R version of this package [here](https://github.com/kanaverse/wobbegong-R).

## Install

To get started, install the package from [PyPI](https://pypi.org/project/wobbegong/)

```bash
pip install wobbegong
```

## Quick Start

### 1. Convert Data (`wobbegongify`)

Use `wobbegongify` to convert your objects into static files.

```python
from wobbegong import wobbegongify
from biocframe import BiocFrame
from singlecellexperiment import SingleCellExperiment
from scipy import sparse
import numpy as np

# 1. Create a SingleCellExperiment
counts = sparse.random(100, 20, density=0.1, format="csr")
pca = np.random.randn(20, 5)
sce = SingleCellExperiment(
    assays={"counts": counts},
    reduced_dims={"PCA": pca}
)

# 2. Convert to wobbegong format (defaults to zlib compression)
wobbegongify(sce, "output/my_study")

# OR use lz4 compression
wobbegongify(sce, "output/my_study", compression = "lz4")
```

### 2. Read Data (`load`)

Use `wobbegong.load()` to read data from a local path **or** a remote URL. The client automatically handles HTTP range requests for you.

```python
import wobbegong

# Load from a local directory (or URL)
sce = wobbegong.load("output/my_study")

# Access Assays (Row-wise access is optimized)
counts = sce.get_assay("counts")
gene_expression = counts.get_row(0)

# Access Reduced Dimensions
pca = sce.get_reduced_dim("PCA")
pc1 = pca.get_column(0)
```

<!-- biocsetup-notes -->

## Note

This project has been set up using [BiocSetup](https://github.com/biocpy/biocsetup)
and [PyScaffold](https://pyscaffold.org/).
