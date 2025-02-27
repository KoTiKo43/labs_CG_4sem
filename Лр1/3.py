model_matrix = []

with open('model_1.obj') as file:
    for s in file:
        s_splt = s.split()
        
        if s_splt[0] == 'v':
            model_matrix.append(list(map(lambda x: float(x), s_splt[1:])))
