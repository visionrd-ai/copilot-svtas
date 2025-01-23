import cv2 
from PIL import Image
import torch 
from torchvision import transforms

sample_rate = 4 
clip_length = 128

vid_path = '/home/multi-gpu/amur/SVTAS-fresh/data/thal/Videos/out.mp4'
capture = cv2.VideoCapture(vid_path)


def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    frame = Image.fromarray(frame.astype('uint8'))
    return transform(frame)


stack = []
frame_count = 0
while True: 

    ret, frame = capture.read()
    if not ret: 
        break 

    stack.append(preprocess_frame(frame))
    frame_count += 1

    if frame_count % clip_length == 0:
        seg_scores[-1, :].detach().cpu().numpy().copy()
        
        input_data = {'imgs': torch.stack(stack[::sample_rate]).unsqueeze(0),
                      'masks': torch.ones((1, clip_length))
                     }
        
        import pdb; pdb.set_trace()
        
        stack = []
        frame_count = 0
    
    