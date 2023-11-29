from setuptools import setup

setup(
    name="hoda",
    version="0.0.1",
    description="Reguralized Higher Order Discriminant Analysis",
    url="https://gitlab.kuleuven.be/compneuro/hoda-bci",
    author="Arne Van Den Kerchove",
    author_email="arne.vandenkerchove@kuleuven.be",
    install_requires=[
        "pyyaml<5.4",
        "tensorly",
        "tqdm",
        "statsmodels",
    ],
    extras_require={
        "notebook": [
            "numpy<1.24",
            "moabb",
            "jupyter",
            "toeplitzlda",
        ]
    },
)
