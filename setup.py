import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

with open(HERE / "requirements.txt", encoding="utf-8") as f:
    install_requires = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#") and not line.startswith("pyvips")
    ]

setup(
    name="GenericTiffHandler",
    version="0.2.0",
    description="Lazy-loading wrapper around TIFF-family whole-slide images (SVS, NDPI, SCN, TIFF)",
    long_description=(HERE / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Maya Silva",
    author_email="mayahptsilva@gmail.com",
    python_requires=">=3.10",
    py_modules=["GenericTiffHandler"],
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Intended Audience :: Science/Research",
    ],
)
