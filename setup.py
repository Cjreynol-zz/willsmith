from setuptools import setup, find_packages

from main import __version__


with open("README.md") as f:
    long_description = f.read()

setup(  
    name = "willsmith",
    version = __version__,
    description = "A framework for creating, testing, and comparing AI agents.",
    long_description = long_description,
    long_description_content_type='text/markdown',
    author = "John Bourassa, Chad Reynolds",
    author_email = "cjreynol13@aol.com",
    url = "https://github.com/Cjreynol/willsmith",
    project_urls = {
        "Source" : "https://github.com/Cjreynol/willsmith"
        },
    license = "MIT",
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3 :: Only"
        ],
    keywords = "AI artificial intelligence agent game mdp",
    python_requires = ">=3",
    py_modules = ["main"],
    packages = find_packages(),
    test_suite = "tests"
)
