from setuptools import setup, find_packages

setup(
    name="recommendation_lib",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Add your project dependencies here
        "numpy",
        "torch>=1.6",
        "pandas",
        "scikit-learn"
        # Add any other dependencies
    ],
    author="Sai Preethi Pantangi",
    author_email="spantang@gmu.com",
    description="A PyTorch-based implicit recommendation library",
    url="https://github.com/PreethiPantangi/CS782_Final_Project",
)
