import streamlit as st
from BAS import BAS
import UIs
#from Function import func


optimizers = ['BAS', 'QBAS', 'QIBAS']

func = st.sidebar.text_input('Enter Numpy Function', "np.sum(np.power(x, 2))")
optimizer = st.sidebar.radio('Select Optimizer', optimizers, index=1)

if optimizer == optimizers[0]:
    dim, T, iter, d, a = UIs.bas()
    opt, deflt, pub = st.sidebar.beta_columns((4.6, 5, 6))
    optimize = opt.button('Optimize')
    algorithm = deflt.button('Algorithm')
    publications = pub.button('Publications')
    if optimize:
        bas = BAS(T, iter, dim, d, a, func)
        f_avg, x_avg = bas.optimize()
        bas.plotting(f_avg, x_avg)
        # col1, col2 = st.beta_columns(2)
        # col1.text_input("F_min= ", f_avg[0])
        # col2.text_input("X_min= ", x_avg[:])
    elif algorithm:
        UIs.bas('Algorithm')
    elif publications:
        UIs.bas('Publications')
else:
    st.title('Heuristic Optimizers')


