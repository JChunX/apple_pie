from setuptools import setup, find_packages

setup(
    name="apple_pie",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "mani_skill",
        "streamlit",
        "numpy",
        "torch"
    ],
    description="A package for robotic manipulation policies",
    author="",
    author_email="",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
