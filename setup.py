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
            # "mne<=1.6.1",
            # "pyyaml<5.4",
            # "moabb==1.1.0",
            # "matplotlib<3.9",
            # "jupyter",
            # "seaborn",
            # "numpy<1.24",
            "moabb",
            "pywavelets",
            "jupyter<4",
            "plotly",
            "dask",
            "dask-jobqueue",
        ]
    },
)
