import streamlit as st
from BAS import BAS
import UIs

st.sidebar.header('Welcome to BAS')
optimizers = ['BAS', 'QBAS', 'QIBAS']

func = st.sidebar.text_input('Enter Numpy Function', "np.sum(np.power(x, 2))")
optimizer = st.sidebar.radio('Select Optimizer', optimizers, index=0)

if optimizer == optimizers[0]:
    #UIs.bas('Algorithm')
    dim, T, iter, d, a = UIs.bas()
    optimize = st.sidebar.button('Optimize')
    deflt, pub = st.sidebar.beta_columns(2)
    algorithm = deflt.button('Algorithm')
    publications = pub.button('Publications')
    if optimize:
        bas = BAS(T, iter, dim, d, a, func)
        f_avg, x_avg = bas.optimize()
        bas.plotting(f_avg, x_avg)
    elif publications:
        UIs.bas('Publications')
    elif algorithm:
        UIs.bas('Algorithm')
    else:
        UIs.bas('Algorithm')
else:
    st.title('Heuristic Optimizers')
