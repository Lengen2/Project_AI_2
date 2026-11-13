Using the SVM Programs
======================

These programs use the CVXOPT solvers.  You may install 
CVXOPT in your Python/conda setup by following the instructions
on https://cvxopt.org.

I recommend setting things up in a separate conda environment,
called, say "cvxopt".

You will need python, numpy, matplotlib and jupyterlab in addition
to cvxopt to use the code in this folder.

   "CVXOPT Intro.ipynb"   A Jupyter Lab notebook with an introduction
                          to CVXOPT and a walkthrough of setting up an
                          SVM problem manually using this package.


   linsvm.py              A Python program (usage described in 
                          094-linear-svm.pdf) that solves linear
                          SVM problems.

   kernelsvm.py           A Python program (usage described in 
                          095-nonlinear-svm.pdf) that solves nonlinear
                          SVM problems using "kernels".  This program
                          supports quadratic and exponential kernels.


N.B., Th command shown below sets up a conda environment will all required 
pacakges.  Note that you only have to do this once.


    conda create -n cvxopt -c conda-forge python cvxopt numpy scipy matplotlib jupyterlab


Type this in at the shell prompt of a terminal window (linux/Mac) or a (Type 
this in at the shell prompt of a terminal window (linux/Mac) or a command 
prompt (windows).

Thereafter, every time you need the environment, you have to "activate" it, 
i.e. enter the command below at the shell prompt.


    conda activate cvxopt


CF
19/08/2025
