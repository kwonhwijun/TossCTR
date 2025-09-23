import setuptools

setuptools.setup(
    name="fuxictr",
    version="2.0.2.colab",
    author="FESeq Implementation",
    description="FuxiCTR for Colab execution",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "pandas", "numpy", "h5py", "PyYAML>=5.1", 
        "scikit-learn", "tqdm", "torch", "pyarrow"
    ]
)
