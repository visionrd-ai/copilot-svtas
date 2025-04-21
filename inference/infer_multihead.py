import cv2 
from PIL import Image
import torch 
from torchvision import transforms
import albumentations as A
from vfuseACT import vfuseACT_multihead_small
import numpy as np 

def load_heads(model, weights):
    action_path = weights.replace('_latest_best', '_best_action')
    branch_path = weights.replace('_latest_best', '_best_branch')
    
    branch_weight = torch.load(branch_path)
    branch_weight = branch_weight['model_state_dict']
    branch_head_weights = {}
    for key, param in branch_weight.items():
        if 'branch_head' in key:
            branch_head_weights[key.replace('branch_head.', '')]   = param
    model.branch_head.load_state_dict(branch_head_weights)
    
    action_weight = torch.load(action_path)
    action_weight = action_weight['model_state_dict']
    action_head_weights = {}
    for key, param in action_weight.items():
        if 'action_head' in key:
            action_head_weights[key.replace('action_head.', '')]   = param
    model.action_head.load_state_dict(action_head_weights)

    return model

sample_rate = 4 
clip_length = 32

vid_path = '/home/multi-gpu/amur/copilot/copilot-svtas/data/thal_dija/Videos/2.mp4'
mapping_actions = {int(elem.strip().split(' ')[0]): elem.strip().split(' ')[-1] for elem in open('/home/multi-gpu/amur/copilot/copilot-svtas/data/thal/mapping_tasks.txt','r').readlines()}#{int(elem.strip().split(' ')[0]): elem.strip().split(' ')[-1] for elem in open('mapping_tasks.txt', 'r').readlines()}
mapping_branch  = {int(elem.strip().split(' ')[0]): elem.strip().split(' ')[-1] for elem in open('/home/multi-gpu/amur/copilot/copilot-svtas/data/thal/mapping_branches.txt','r').readlines()}#open('../data/thal/mapping_branches.txt','r')#{int(elem.strip().split(' ')[0]): elem.strip().split(' ')[-1] for elem in open('mapping_branches.txt', 'r').readlines()}
 
capture = cv2.VideoCapture(vid_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
output_fps = int(capture.get(cv2.CAP_PROP_FPS) / sample_rate)
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = 'infer.mp4'
video_writer = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

path = '../output/thal_production_cloud/thal_production_all_data_latest_best.pkl' 

model = vfuseACT_multihead_small(clip_seg_num=clip_length//sample_rate)
weights = torch.load(path)['model_state_dict']
model.load_state_dict(weights)

model = load_heads(model, path)

model.eval()
model = model.cuda()


def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    frame = Image.fromarray(frame.astype('uint8')[:,:,::-1])
    return transform(frame)


def postprocess_frame(frame_list, wh =  (1280, 720)):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    original_frames = []
    for frame in frame_list:
        unnormalized_frame = frame * std[:, None, None] + mean[:, None, None]
        uint8_frame = (unnormalized_frame.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        original_frames.append(cv2.resize(uint8_frame[:,:,::-1], wh))
    return original_frames



def postprocess_outputs(outs, mapping_actions, mapping_branch, action_threshold=0.8, branch_threshold=0.8):
    def process_scores(scores, mapping, default_idx, threshold):
        probs = torch.nn.functional.softmax(torch.tensor(scores), dim=-2).numpy()
        predicted_classes = np.argmax(probs, axis=-2)[0]
        # max_probs = np.max(probs, axis=-2)[0]
        
        # predicted_classes[max_probs < threshold] = default_idx
        mapped_preds = [mapping[elem] for elem in predicted_classes]
        
        return mapped_preds
    
    with torch.no_grad():
        action_scores = outs['action_score'][-1, :].detach().cpu().numpy().copy()
        branch_scores = outs['branch_score'][-1, :].detach().cpu().numpy().copy()
        
    outs_action_post = process_scores(action_scores, mapping_actions, default_idx=4, threshold=action_threshold)
    outs_branch_post = process_scores(branch_scores, mapping_branch, default_idx=14, threshold=branch_threshold)
    
    return outs_action_post, outs_branch_post


stack = []
frame_count = 0
while True: 

    ret, frame = capture.read()
    if not ret: 
        break 
    
    stack.append(preprocess_frame(frame))
    frame_count += 1

    if frame_count % clip_length == 0:
        input_data = {'imgs': torch.stack(stack[::sample_rate]).unsqueeze(0).cuda(),
                      'masks': torch.ones((1, clip_length)).cuda()
                     }
        with torch.no_grad():
            outs = model(input_data) 

        outs_action_post, outs_branch_post = postprocess_outputs(outs, mapping_actions, mapping_branch, action_threshold=0.8, branch_threshold=0.8)

        for out_frame, out_action, out_branch in zip(postprocess_frame(stack, (frame_width, frame_height)), outs_action_post, outs_branch_post):
            cv2.putText(out_frame, 'Branch: '+ out_branch, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(out_frame, 'Action: '+ out_action, (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            video_writer.write(out_frame) 
        
        stack = []
        frame_count = 0
    
capture.release()
video_writer.release()
print(f"Processed video saved to {output_path}")