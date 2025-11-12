from setuptools import setup

setup(
    name="bttda",
    version="0.0.1",
    description="Block-Term Tensor Discriminant Analysis",
    url="https://github.com/arnevdk/bttda",
    author="Arne Van Den Kerchove",
    author_email="arne.vandenkerchove@kuleuven.be",
    install_requires=[
        "scikit-learn",
        "tensorly",
        "opt_einsum",
        "tqdm",
    ],
    extras_require={
        "experiments": [
            "moabb",
            "filelock",
            "pywavelets",
            "jupyter<4",
            "plotly",
        ],
        "hpc": [
            "dask",
            "dask-jobqueue",
            "bokeh",
        ],
    },
)
