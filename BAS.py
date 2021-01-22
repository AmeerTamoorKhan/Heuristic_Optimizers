import numpy as np
import streamlit as st
import pandas as pd
class BAS:
    def __init__(self, T=0, iter=0, dim=0, d=0.9, a=0.95, func=None):
        self.T = int(T)
        self.iter = int(iter)
        self.dim = int(dim)
        self.d_int = float(d)
        self.d = float(d)
        self.a = float(a)
        self.f = func
        self.f_best = np.array([])
        self.x_best = np.array([])
        self.F_best = np.array([])
        self.X_best = np.empty((1, self.dim))

    def optimize(self):

        for i in range(self.T):
            x = np.random.random((1, self.dim))
            self.f_best = np.array([eval(self.f, {'x': x, 'np': np})])
            self.x_best = x
            for j in range(self.iter):
                b = np.random.random((1, self.dim))
                xr = x + self.d*b
                xl = x - self.d*b
                df = [eval(self.f, {'x': xr, 'np': np}) - eval(self.f, {'x': xl, 'np': np})]
                x = x - self.d*b*np.sign(df)
                fnc = np.array([eval(self.f, {'x': x, 'np': np})])

                if fnc < self.f_best[-1]:
                    self.f_best = np.append(self.f_best, fnc)
                    self.x_best = np.append(self.x_best, x, axis=0)

                self.d = self.a*self.d + 0.01

            self.F_best = np.append(self.F_best, self.f_best[-1])
            self.X_best = np.append(self.X_best, np.array([self.x_best[-1, :]]), axis=0)
            self.d = self.d_int

        f_avg = np.array([self.F_best/self.T])
        x_avg = np.array(self.X_best/self.T)

        return f_avg.round(decimals=4), x_avg.round(decimals=4).tolist()

    def plotting(self, f_avg, x_avg):
        # Display the F_min and X_min
        col1, col2 = st.beta_columns(2)
        col1.text_input("F_min= ", f_avg[-1, -1])
        col2.text_input("X_min= ", x_avg[-1])

        # Display the dataframe of F_avg and X_avg
        col1.markdown('''<h4>F_best</h4>''', unsafe_allow_html=True)
        dfF = pd.Series(self.f_best, name='F_best')
        col1.dataframe(dfF, height=200)
        cols = ['X_'+str(i) for i in range(np.shape(self.x_best)[1])]
        dfX = pd.DataFrame(self.x_best, columns=cols)
        col2.markdown('''<h4>X_best</h4>''', unsafe_allow_html=True)
        col2.dataframe(dfX, height=200)

        st.markdown('''<h3>Plots And Charts</h3><hr>''', unsafe_allow_html=True)
        plt1, plt2 = st.beta_columns(2)
        # Plot of F_avg
        plt1.markdown('''<h3>Plot of F_best</h3>''', unsafe_allow_html=True)
        plt1.line_chart(dfF, width=200, height=200)
        # Plot of X_avg
        plt2.markdown('''<h3>Plot of X_best</h3>''', unsafe_allow_html=True)
        plt2.line_chart(dfX, width=200, height=200)


if __name__ == '__main__':
    bas = BAS('10', '1000', '2', func='np.sum(np.power(x, 3))')
    f_avg, x_avg = bas.optimize()
    bas.plotting(f_avg, x_avg)








