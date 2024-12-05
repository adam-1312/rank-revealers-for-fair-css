# rank-revealers-for-fair-css

## Note on implementation
This is the github repository accompanying my bachelor thesis on Rank-revealer methods for Fair Column Subset Selction which was completed in the institute of Numerical Mathematics at the University of Ulm in 2024. For the implementations used in this work we used Python and Matlab. Further, we used a mix of our own original work and implementations from the work of [Damle et. al.](https://arxiv.org/abs/2405.04330). In particular, in the repository you will find the following implementations:

- CPQR.m: Implementation of CPQR by Damle et. al.
- fairCPQR.m: Our implementation of Fair CPQR
- fairLowQRforCSS.m: Our implementation of Fair Low QR specific to CSS for memory efficiency (based on Q-less QR)
- gammaQR.m: Implementation of computing $\mu_B$ by Damle et. al.
- givens_rot.m, givensr.m: Implementation of Givens transformations by Damle et. al.
- lowQR.m: Our implementation of Low QR
- lowQRforCSS.m: Our implementation of Low QR specific to CSS for memory efficiency (based on Q-less QR)
- mu_B_computation.m: Our script to generate random matrices, compute $\mu_B$ and plot result in histogram
- RRQR.m: Implementation of RRQR by Damle et. al.
- CSSlib.py: Our module with functions we regularly used to perform CSS on the MNIST dataset
- plot.py: Our module with functions we used to create plots specific to the CSS experiments
