import numpy as np

model_v_matrix = []
model_p_matrix = []

with open('model_1.obj') as file:
    for s in file:
        s_splt = s.split()
        
        if s_splt[0] == 'v':
            model_v_matrix.append(tuple(map(lambda x: float(x), s_splt[1:])))
        elif s_splt[0] == 'f':
            s_splt_dash = []
            for ss in s_splt[1:]:
                s_splt_dash.append(tuple(map(lambda x: int(x), ss.split('/'))))
            model_p_matrix.append([model_v_matrix[s_splt_dash[0][0] - 1], model_v_matrix[s_splt_dash[1][0] - 1], model_v_matrix[s_splt_dash[2][0] - 1]])
