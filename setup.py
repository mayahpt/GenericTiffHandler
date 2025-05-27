# setup.py
import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

# Automatically read the pip-installable dependencies from requirements.txt
with open(HERE / "requirements.txt", encoding="utf-8") as f:
    install_requires = [line.strip() for line in f if line.strip()]

setup(
    name="GenericTiffHandler",
    version="0.1.0",
    description="A simple handler for generic TIFF files",
    long_description=(HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="you@example.com",
    python_requires=">=3.10",
    py_modules=["GenericTiffHandler"],
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)