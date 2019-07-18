import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="imbalance-xgboost",
    version="0.7.2",
    author="Chen Wang",
    author_email="chen.wang.cs@rutgers.edu",
    description="XGBoost for label-imbalanced data: XGBoost with weighted and focal loss functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jhwjhw0123/Xgboost-With-Imbalance-And-Focal-Loss",
    download_url="https://github.com/jhwjhw0123/Xgboost-With-Imbalance-And-Focal-Loss",
    packages=['imxgboost'],
    scripts=['imxgboost/imbalance_xgb.py', 'imxgboost/focal_loss.py', 'imxgboost/weighted_loss.py'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    include_package_data = True,
    license = "MIT",
    install_requires = ["numpy>=1.11", 'scikit-learn>=0.19', 'xgboost>=0.4a30'],
)