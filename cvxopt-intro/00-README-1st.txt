You will need to have installed CVXOPT before you can run the Jupyter notebook
in this folder.  If you have installed Anaconda Python, I recommend creating 
a separate environment for CVXOPT like so:


    conda create -n cvxopt -c conda-forge python cvxopt numpy scipy matplotlib jupyterlab


Type this at the prompt of a shell window (Linux/Mac) or in a command-prompt
window (Windows).  (N.B., You only need to do this once, to download and install 
the libraries.)

Then activate the environment:


    conda activate cvxopt


And start the Jupyter Lab server:


    jupyter lab


Once the Jupyter Lab server has started, a browser window should open with the
Jupyter interface in it, you can start the notebook "CVXOPT Intro.ipynb" from
there and walk through it.

CF
19/08/25
