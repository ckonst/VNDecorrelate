from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy", "scipy", "matplotlib"]

setup(
    name="VNDecorrelate",
    version="0.0.1",
    author="Christian Konstantinov",
    author_email="christian.konstantinov98@gmail.com",
    description="A Velvet-Noise Decorrelatora",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/ckonst/VNDecorrelate",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
)