from setuptools import setup, find_packages

setup(
    name="muf-clust",
    version="0.0.1",
    description="MUF-Clust preprocessing and clustering tools",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
)