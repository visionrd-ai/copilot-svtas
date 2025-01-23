import random 
for i in range(5):
    for j in range(3):
        while True: 
            i_test = [random.randint(0,5) for _ in range(32)] 
            j_test = [random.randint(0,3) for _ in range(32)]
            
            import pdb; pdb.set_trace()