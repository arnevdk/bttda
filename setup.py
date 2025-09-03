from setuptools import setup

setup(
    name="hoda",
    version="0.0.1",
    description="Reguralized Higher Order Discriminant Analysis",
    url="https://gitlab.kuleuven.be/compneuro/hoda-bci",
    author="Arne Van Den Kerchove",
    author_email="arne.vandenkerchove@kuleuven.be",
    install_requires=["scikit-learn", "tensorly", "opt_einsum", "tqdm", "kneed"],
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
        ]
    },
)
