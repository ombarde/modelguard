from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="modelguard",
    version="0.1.0",
    author="Om Barde",
    author_email="your@email.com",
    description="Git diff for neural networks — compare, debug, and track ML model changes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/modelguard",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "modelguard": ["templates/*.html"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "jinja2>=3.0.0",
        "tqdm>=4.62.0",
    ],
    keywords="ml model diff debug compare neural network drift",
)