from setuptools import setup, find_packages

setup(
    name="crop_classification",
    version="1.0.0",
    author="Your Name",
    description="Machine Learning based Crop Classification System",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "joblib"
    ],
    python_requires=">=3.8",
)