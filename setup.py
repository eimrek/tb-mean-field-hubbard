import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tb-mean-field-hubbard",
    version="2.0.0",
    author="Kristjan Eimre",
    author_email="kristjaneimre@gmail.com",
    description="Package to run tight-binding mean field hubbard calculations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eimrek/tb-mean-field-hubbard",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "ase",
        "pythtb",
        'igor-tools @ git+https://git@github.com/nanotech-empa/igor-tools@main#egg=igor-tools',
    ],
)
