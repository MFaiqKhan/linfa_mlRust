# A Model that can predict whether a student will be accepted into a university or not
## It is based on two test scores and written in rust

It's using logistic regression , it is similar to linear regression but logistic regression predicts if something is 
true or false , example, he fails the test or he passed the test, it makes S curve on a scale of 0 to 1. It uses
maximum likelihood estimation to find the best parameters for the model.

OpenBlas feature allow  you to use openblas library to speed up the matrix multiplication and other low level operations 
on matrices.
using ndarray to implement n dimensional arrays
ndarray-csv  to read csv files and load datasets into the application and converted to an ndarray

using plotlib crate to visualize scatter plot of the data and the decision boundary


Crates USED
- linfa : toolkit to build machine learning application in rust.
- linfa-logistic : logistic regression algorithm.
- ndarray : n dimensional arrays
- ndarray-csv : read csv files and load datasets into the application and converted to an ndarray
- plotlib : crate to visualize scatter plot of the data and the decision boundary