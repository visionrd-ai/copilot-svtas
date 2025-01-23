import cv2
from PIL import Image
import torch
from torchvision import transforms
from vfuseACT import vfuseACT
import numpy as np
import threading
import queue
import time 
import json 
import copy 

sample_rate = 4
clip_length = 32
vid_path = 'IMG_3367.mp4'
mapping = {int(elem.strip().split(' ')[0]): elem.strip().split(' ')[-1] for elem in open('../data/thal_noCLSWISE/mapping_tasks.txt', 'r').readlines()}
output_path = 'IMG_3367_out.mp4'

model = vfuseACT(clip_seg_num=clip_length//sample_rate)
_weights = torch.load('/home/multi-gpu/amur/SVTAS-fresh/output/ResnetMemTCN_Thal_45_4_32SW_resume/ResnetMemTCN_Thal_45_4_32SW_resume_best.pkl')['model_state_dict']
# _weights = {key.replace('conv.net', 'conv'): val for key, val in _weights.items()}
model.load_state_dict(_weights)
model.eval()
model = model.cuda()

capture = cv2.VideoCapture(vid_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_fps = int(capture.get(cv2.CAP_PROP_FPS) / sample_rate)
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_writer = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.Resize((256, 320)), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    frame = Image.fromarray(frame.astype('uint8')[:, :, ::-1])
    return transform(frame)

def postprocess_frame(frame_list, wh=(1280, 720)):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    original_frames = []
    for frame in frame_list:
        unnormalized_frame = frame * std[:, None, None] + mean[:, None, None]
        uint8_frame = (unnormalized_frame.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        original_frames.append(cv2.resize(uint8_frame[:, :, ::-1], wh))
    
    return original_frames

frame_queue = queue.Queue(maxsize=32)

def frame_reader():
    while True:
        ret, frame = capture.read()
        if not ret:
            frame_queue.put(None) 
            break
        frame_queue.put(frame)

def frame_processor():
    stack = []
    frame_count = 0
    states = [
                {'area1':
                    {
                        'tasks':[], 
                        'status':False
                    }
                }
            ]
    
    prev_area = None 
    areas_gt = [lbl.strip().split(': ')[-1] for lbl in open('areas.txt', 'r').readlines()]
    states = json.load(open('gt_states.json', 'r'))
    states_pipeline = copy.deepcopy(states)

    while True:
        ts_st = time.time()
        frame = frame_queue.get()
        if frame is None:  # End of video
            break

        curr_area = areas_gt[frame_count]

        if prev_area is None and curr_area != 'background': # first area seen 
            
            if curr_area == list(states[0].keys())[0]: # check if first area matches 
                states_pipeline[0][curr_area]['status'] = 'IN PROGRESS'
                states_pipeline[0][curr_area]['duration'] = 0
                prev_area = curr_area 
            else: 
                print(15*"*"+"\nWRONG AREA DETECTED\n"+15*"*")
                states_pipeline[0][curr_area]['status'] = 'INCOMPLETE'
        
        if prev_area == curr_area: 
            [idx for idx, area in enumerate(states_pipeline) if area['']]
            
        stack.append(preprocess_frame(frame))
        frame_count += 1
        to_it = time.time()

        if frame_count % clip_length == 0:
            input_data = {'imgs': torch.stack(stack[::sample_rate]).unsqueeze(0).cuda(),
                          'masks': torch.ones((1, clip_length)).cuda()}
            with torch.no_grad():
                ts = time.time()
                outs = model(input_data)    
                # print("infer_time: ", time.time()-ts)
            outs = np.argmax(outs[-1, :].detach().cpu().numpy().copy(), axis=-2)[0]
            outs_post = [mapping[elem] for elem in outs]
            for out_frame, out in zip(postprocess_frame(stack), outs_post):
                cv2.putText(out_frame, out, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                video_writer.write(out_frame)
            stack = []
            frame_count = 0
            t_inf = time.time()
            # print("One second infer time: ", t_inf-ts_st)
        if (to_it - ts_st) < 1/30:
            # print("rcvd frame in: ",to_it - ts_st)
            # print("sleeping for: ",1/30-(to_it - ts_st))

            time.sleep(1/30 - (to_it-ts_st))
        print("Iter FPS: ", 1/(time.time() - ts_st))

reader_thread = threading.Thread(target=frame_reader)
processor_thread = threading.Thread(target=frame_processor)

reader_thread.start()
processor_thread.start()

reader_thread.join()
processor_thread.join()

capture.release()
video_writer.release()
print(f"Processed video saved to {output_path}")
