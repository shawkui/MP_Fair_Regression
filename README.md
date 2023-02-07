# Mean Parity Fair Regression in RKHS

The official implementation of the paper ***Mean Parity Fair Regression in RKHS***. This implementation is built upon https://github.com/steven7woo/fair_regression_reduction.

In this fair_reg.py, we implement three methods used in the paper including:
- MP_Fair_regression: Our method for achieving MP fairness or Covariance-based fairness
- MP_Penalty_regression: A penalty-based method for achieving MP fairness or Covariance-based fairness
- Fair_kernel_learning: A penalty-based method for achieving Covariance-based fairness

## Steps to run the code:
1. Download the preprocessed dataset from https://github.com/steven7woo/fair_regression_reduction by running the following code in your terminal
    
    ```bash collect_data.sh```

2. Install the dependencies by running the following code in your terminal
    
    ```pip install -r requirements.txt```

3. Run our demo notebook in your Jupyter Notebook Environment.

For other baselines used in these paper, we refer users to their official implementations:
- Nonconvex Regression with Fairness Constraints: https://github.com/jkomiyama/fairregresion
- Reduction Based Algorithm: https://github.com/steven7woo/fair_regression_reduction

If you have any problem, please contact [Shaokui Wei](mailto::shaokuiwei@link.cuhk.edu.cn).