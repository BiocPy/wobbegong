# wobbegong

wobbegong converts Bioconductor objects (like `BiocFrame`, `SummarizedExperiment`, and `SingleCellExperiment`) into a set of static files optimized for HTTP range requests.

It includes a native Python client that allows you to query these datasets remotely, fetching only the specific genes or metadata columns you need without downloading the entire file.

## Contents

```{toctree}
:maxdepth: 2

Overview <tutorial>
Contributions & Help <contributing>
License <license>
Authors <authors>
Changelog <changelog>
Module Reference <api/modules>
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`

[Sphinx]: http://www.sphinx-doc.org/
[Markdown]: https://daringfireball.net/projects/markdown/
[reStructuredText]: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
[MyST]: https://myst-parser.readthedocs.io/en/latest/
