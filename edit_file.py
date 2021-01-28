
from .general import *


'''
edit freeze num in train.py

replace 
    freeze = []
to
    freeze = ['model.%s.' % x for x in range({freeze_num})]
'''
def freeze_layer_in_train_py(freeze_num):

    with open('train.py', 'r') as f:
        train_py_lines = f.read().splitlines()
    

    #replace line to freeze
    previous_freeze = "null"
    new_freeze = "null"
    new_train_py_lines = []
    
    for line in train_py_lines:
        if 'freeze = [' in line:
            previous_freeze = line
            #unfreeze all if freeze_num is 0 else freeze by num
            if freeze_num == 0:
                line = f"    freeze = []"

            else:
                line = f"    freeze = ['model.%s.' % x for x in range({freeze_num})]"
            
            new_freeze = line
        
        new_train_py_lines.append(line+'\n')

    
    #update train.py
    with open('train.py', 'w') as f:
        f.writelines(new_train_py_lines)
        
        
    print(f"overwrite freeze : {previous_freeze} to \n {new_freeze} in train.py")   


'''
set evolve num by
    for _ in range({evolve_num})

becareful as the sentence is general
'''
def set_evolve_num_in_train_py(evolve_num):
    with open('train.py', 'r') as f:
        train_py_lines = f.read().splitlines()


    #replace line to freeze
    previous_line = "null"
    new_line = "null"
    new_train_py_lines = []

    for line in train_py_lines:
        if 'for _ in range(' in line:
            previous_line = line
            #unfreeze all if freeze_num is 0 else freeze by num
            line = f"        for _ in range({evolve_num}):"
            new_line = line

        new_train_py_lines.append(line+'\n')


    #update train.py
    with open('train.py', 'w') as f:
        f.writelines(new_train_py_lines)


    print(f"overwrite freeze : {previous_line} to \n {new_line} in train.py")  
    
    
    
def replace_line_in_file(replace_dict: dict, file='train.py'):
    with open(file, 'r') as f:
        lines = f.read().splitlines()
        
    previous_line = "null"
    replaced_line = None
    
    new_lines = []
    for line in lines:
        
        for key in replace_dict.keys():
            if key in line:
                replaced_line = f"{replace_dict[key]}" 
                
        if replaced_line:
            previous_line = line
            line = replaced_line
            print(f"overwrite")
            print("-"*10)            
            print(f"{previous_line} \nto\n {replaced_line} \nin <{file}>")
        new_lines.append(line+'\n')
        
        replaced_line = None #reset
    #update hyp_yaml file
    with open(file, 'w') as f:
        f.writelines(new_lines)

def replace_line_in_hyp(replace_dict: dict, hyp_yaml='data/hyp_custom.yaml'):
    with open(hyp_yaml, 'r') as f:
        hyp_lines = f.read().splitlines()
        
    previous_line = "null"
    replaced_line = None
    
    new_hyp_lines = []
    for line in hyp_lines:
        
        for key in replace_dict.keys():
            if key in line:
                replaced_line = f"{str(key)} {replace_dict[key]}" # i.e.  hsv_h: 0.1
            
        if replaced_line:
            previous_line = line
            line = replaced_line
            print(f"overwrite <<{previous_line}>> to <<{replaced_line}>> in {hyp_yaml}")
        new_hyp_lines.append(line+'\n')
        
        replaced_line = None #reset
    #update hyp_yaml file
    with open(hyp_yaml, 'w') as f:
        f.writelines(new_hyp_lines)
