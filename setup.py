from setuptools import find_packages, setup

setup(
    name="dfusion",
    version="0.0.2",
    description="Easy to use library for various implementtaions of denoising diffusion models",
    author="Kitsunetic",
    author_email="jh.shim.gg@gmail.com",
    url="https://github.com/Kitsunetic/DFusion.git",
    # packages=find_packages(include=["src", "src.*"]),
    packages=find_packages(),
    # install_requires=["PyYAML", "pandas==0.23.3", "numpy>=1.14.5"],
    # extras_require={"plotting": ["matplotlib>=2.2.0", "jupyter"]},
    # setup_requires=["pytest-runner", "flake8"],
    # tests_require=["pytest"],
    install_requires=[],
    extras_require={},
    setup_requires=["torch", "einops"],
    tests_require=[],
    # entry_points={
    #     "console_scripts": ["my-command=exampleproject.example:main"]
    # },
    # package_data={"exampleproject": ["data/schema.json"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3",
)
