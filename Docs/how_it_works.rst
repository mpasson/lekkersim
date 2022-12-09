How it works
-------------


Single block representation
================================

.. math::
   :nowrap:

    \[
    \left[
    \begin{array}{c}
    o_0 \\ o_1 \\ o_2 \\ \vdots \\o_n
    \end{array}
    \right]=
    \left[
    \begin{array}{ccccc}
        s_{00} & s_{01} &s_{02} & \hdots & s_{0n} \\ 
        s_{10} & s_{11} &s_{12} & \hdots & s_{1n} \\
        s_{20} & s_{21} &s_{22} & \hdots & s_{2n} \\
        \vdots & \vdots &\vdots & \ddots & \vdots \\ 
        s_{n0} & s_{n1} &s_{n2} & \hdots & s_{nn} \\ 
    \end{array}
    \right]
    \left[
    \begin{array}{c}
    i_0 \\ i_1 \\ i_2 \\ \vdots \\i_n
    \end{array}
    \right]
    =S
    \left[
    \begin{array}{c}
    i_0 \\ i_1 \\ i_2 \\ \vdots \\i_n
    \end{array}
    \right]
    \]


Recursion algorithm
================================



.. math::
   :nowrap:
    
    \[
    \begin{array}{ccc}
    S1[nxn] =
    \left[ \begin{array}{cc}
        S1_{00} & S1_{01} \\
        S1_{10} & S1_{11} \\
    \end{array} \right]
    &
    \quad &
    S1[mxm] =
    \left[ \begin{array}{cc}
        S1_{00} & S1_{01} \\
        S1_{10} & S1_{11} \\
    \end{array} \right]
    \\
    \end{array}    
    \]