gt = ['bg', 'bg', 'bg', 'clip', 'clip', 'clip', 'tape']
tt = ['bg', 'bg', 'clip', 'clip', 'clip', 'clip']


def get_preprocessed(liszt):
    lbl_dict = {}
    last_label = None
    for lbl_idx, lbl in enumerate(liszt):
        if lbl not in lbl_dict:
            lbl_dict[lbl] = {'start':lbl_idx, 'end':None}
            if last_label is not None: 
                lbl_dict[last_label]['end'] = lbl_idx 



# def get_preprocessed(liszt):

#     label_dict = {
#                     lbl:{'start':None, 'end':None} for lbl in set(liszt)
#                 }
    
#     for unique_label in label_dict.keys():
#         for lbl_idx, lbl in enumerate(liszt):

#             if lbl == unique_label:

#                 if label_dict[unique_label]['start'] is None: 
#                     label_dict[unique_label]['start'] = lbl_idx
            
#                 label_dict[unique_label]['end'] = lbl_idx
#     return label_dict 

# gt_dict = 
import pdb; pdb.set_trace()
