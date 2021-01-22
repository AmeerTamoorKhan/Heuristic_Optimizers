import streamlit as st

base = '/Users/ameertamoorkhan/PycharmProjects/HeuristicOptmizers'


def bas(option='Optimize'):
    if option == 'Optimize':
        st.empty()
        img, tit = st.beta_columns((1, 5))
        img.image(base+'/Images/beetle.png', width=100)
        tit.title('BAS: Beetle Antennae Search')
        #st.video('animations/BAS.mp4')
        dim = st.sidebar.text_input('X Dimensions (dim)', 2)
        col1, col2 = st.sidebar.beta_columns(2)
        with col1:
            T = st.text_input('Total Trials (T)', 2)
        with col2:
            iter = st.text_input('Itertions (iter)', 1000)

        st.sidebar.write('BAS Parameters')
        d = st.sidebar.slider('Antenna Length (d)', value=0.9, min_value=0.0, max_value=1.0, step=0.01)
        a = st.sidebar.slider('Antenna Decay Factor (a)', value=0.95, min_value=0.0, max_value=1.0, step=0.01)
        return dim, T, iter, d, a
    elif option == 'Algorithm':
        st.empty()
        st.header('BAS Background')
        st.markdown('''
        BAS is a nature inspired heuristic algorithm. It mimics the food searching nature of the beetle. Beetle has two
        antennae, i.e., right ($\mathbf{x_r}$) and left ($\mathbf{x_l}$) to register the smell of food and based
        on the intensity of the smell it moves either left or right. Beetle iteratively repeats this until it reaches
        food. BAS algorithm has been employed in numerous real-world applications and have evolved through different
        variants.<hr>
        ''', unsafe_allow_html=True)
        st.header('Beetle Searching Nature')
        st.video(base+'/animations/BAS.mp4')
        st.markdown('''<hr>''', unsafe_allow_html=True)

        col1, col2 = st.beta_columns(2)
        col1.header('BAS Formulation')
        col1.markdown('''
                <h3>Objective: </h3>\n
                $\\min_{\mathbf{x}} f(\mathbf{x})$ \n
                $\mathbf{x_r} = \mathbf{x} + d\\times \mathbf{b}$ \n 
                $\mathbf{x_l} = \mathbf{x} - d\\times \mathbf{b}$ \n
                where: \n
                $\mathbf{x}=$ Current position of serching particle (beetle) \n
                $d=$ Length of antenna \n
                $\mathbf{b}=$ Direction vector (Random Vector) \n
                <h3>Update particle position: </h3> \n
                $\mathbf{x}=\mathbf{x}-d\\times b\\times sgn(f(\mathbf{x_r}) - f(\mathbf{x_l}))$ \n
                $d = a \\times d + 0.01$ \n
                where: \n
                $a=$ Antenna decay factor            
                ''', unsafe_allow_html=True)
        col2.header('BAS Algorithm')
        col2.markdown('''
                $f(\mathbf{x})$ = define_function($\mathbf{x}$)\n
                <strong>Initialize</strong> \n
                $T, iter, dim, d, a$ \n
                $\mathbf{x} =$ rand$(1, dim)$ \n
                $f_{best} = f(\mathbf{x}),$ $\mathbf{x_{best}}=\mathbf{x}$ \n
                <strong>For</strong> $i=1:T$ \n
                <strong style="text-indent: 50px;">For</strong> $j=1:iter$ \n
                Compute the direction vector $\mathbf{b}$ \n
                Compute $\mathbf{x_r},$ $\mathbf{x_l}$ and $\mathbf{x}$\n  
                Compute $fnc = f(\mathbf{x})$ \n
                <strong>If</strong> $fnc < f_{best}$ \n
                $f_{best} = fnc,$  $\mathbf{x_{best}} = \mathbf{x}$\n
                <strong>End If</strong>
                Update $d$ 
                <strong>For End</strong> \n
                <strong>For End</strong>
                ''', unsafe_allow_html=True)

        # *********************************************************************************************
        # *************************************** MATLAB Code *****************************************
        # *********************************************************************************************
        st.markdown('''<hr><h3>MATLAB Code</h3>''', unsafe_allow_html=True)
        st.code(
            '''
# Algorithm Execution Parameters
T = 1000;                   %Total Number of Trials
iter = 1;                   %Iterations each Trial

% Define Problem
f = @(x) def_Function(x);   %Define Function
D = 30;                     %Problem Dimension

% Beetle Parameters
a = 0.9                     %Antenna Decay Factor

# Vectors To Store Function And Variables
f_best = []
x_best = []
F = [];
Y = [];

for i = 1:T
    d = 0.95;                   %Initialize Antenna Length
    x = rand(1, D);             %Initialize x
    f_best = f(x);              %Initialize f_best
    x_best = x;                 %Initialize x_best
    for j = 1:iter 
        b = rand(1, D);
        
        x_r = x + d*b;
        x_l = x - d*b;

        x = x - d*b.*(f(x_r) - f(x_l));
        
        fnc = f(x)

        if fnc < f_best[end]
            f_best = [f_best; fnc];
            x_best = [x_best; x];
        end
        
        d = a*d + 0.01;

    end

F = [Y; f_best(end)];
X = [X; x_best(end, :)];
end

f_avg = mean(F)
x_avg = mean(X)


            ''', language='MATLAB'
        )

        # *********************************************************************************************
        # *************************************** Python Code *****************************************
        # *********************************************************************************************
        st.markdown('''<hr><h3>PythonCode</h3>''', unsafe_allow_html=True)
        st.code(
            '''
import numpy as np

# Algorithm Execution Parameters
T = int(T)
iter = 1000

# Define Problem
f = lambda x: def_func(x)
dim = 2

# Define Beetle Parameters
a = 0.9

# Vectors To Store Function And Variables
f_best = np.array([])
x_best = np.array([])
F_best = np.array([])
X_best = np.empty((1, dim))


for i in range(T):
    d = 0.95                            #Initialize Antennae Length         
    x = np.random.random((1, dim))      #Initialize x
    f_best = f(x)                       
    x_best = x
    for j in range(iter):
        b = np.random.random((1, dim))
        xr = x + d*b
        xl = x - d*b
        x = x - d*b*np.sign(f(xr) - f(xl))
        fnc = f(x)

        if fnc < f_best[-1]:
            f_best = np.append(f_best, fnc)
            x_best = np.append(x_best, x, axis=0)

        d = a*d + 0.01

    F_best = np.append(F_best, f_best[-1])
    X_best = np.append(X_best, np.array([x_best[-1, :]]), axis=0)

f_avg = np.array([F_best/T])
x_avg = np.array(X_best/T)
            ''', language='python'
        )

    elif option == 'Publications':
        st.empty()
        paper1 = '''<a href="https://arxiv.org/abs/1710.10724" target="_blank"> Jiang, X., & Li, S. BAS: beetle antennae search algorithm for optimization problems (2017). arXiv preprint arXiv:1710.10724. </a>'''
        paper2 = '''<a href="https://arxiv.org/abs/1904.02397" target="_blank"> Zhang, Y., Li, S., & Xu, B. (2019). Convergence analysis of beetle antennae search algorithm and its applications. arXiv preprint arXiv:1904.02397. </a>'''
        paper3 = '''<a href="https://www.mdpi.com/1424-8220/19/8/1758" target="_blank"> Wu, Q., Shen, X., Jin, Y., Chen, Z., Li, S., Khan, A. H., & Chen, D. (2019). Intelligent beetle antennae search for UAV sensing and avoidance of obstacles. Sensors, 19(8), 1758. </a>'''
        paper4 = '''<a href="https://ieeexplore.ieee.org/abstract/document/8717631/" target="_blank"> Wu, Q., Ma, Z., Xu, G., Li, S., & Chen, D. (2019). A novel neural network classifier using beetle antennae search algorithm for pattern classification. IEEE access, 7, 64686-64696. </a>'''
        paper5 = '''<a href="https://www.sciencedirect.com/science/article/pii/S0029801819306766" target="_blank"> Xie, S., Chu, X., Zheng, M., & Liu, C. (2019). Ship predictive collision avoidance method based on an improved beetle antennae search algorithm. Ocean Engineering, 192, 106542. </a>'''
        paper6 = '''<a href="https://arxiv.org/abs/2002.10090" target="_blank"> Zhang, J., Huang, Y., Ma, G., & Nener, B. (2020). Multi-objective beetle antennae search algorithm. arXiv preprint arXiv:2002.10090. </a>'''


        st.markdown(
            f'''
            <h3> Publications: </h3>
            <ol>
                <li>{paper1} </li>
                <li>{paper2} </li>
                <li>{paper3} </li>
                <li>{paper4} </li>
                <li>{paper5} </li>
                <li>{paper6} </li>
            </ol>
            '''
        , unsafe_allow_html=True)

