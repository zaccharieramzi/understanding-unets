import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="understanding-unets",
    version="0.0.1",
    author="Zaccharie Ramzi",
    author_email="zaccharie.ramzi@gmail.com",
    description="A project to test whether we can learn wavelet filters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zaccharieramzi/understanding-unets",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
