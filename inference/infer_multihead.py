import cv2 
from PIL import Image
import torch 
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from vfuseACT import vfuseACT_multihead
import numpy as np 

sample_rate = 4 
clip_length = 32

vid_path = '../data/thal/Videos/IMG_4743_counter.mp4'
mapping_actions = {int(elem.strip().split(' ')[0]): elem.strip().split(' ')[-1] for elem in open('mapping_tasks.txt', 'r').readlines()}
mapping_branch = {int(elem.strip().split(' ')[0]): elem.strip().split(' ')[-1] for elem in open('mapping_branches.txt', 'r').readlines()}

capture = cv2.VideoCapture(vid_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
output_fps = int(capture.get(cv2.CAP_PROP_FPS) / sample_rate)
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = 'IMG_4743_counter_nocent.mp4'
video_writer = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))


# path128 = '/home/multi-gpu/amur/SVTAS-fresh/output/ResnetMemTCN_Thal_45_4_128SW_resume/ResnetMemTCN_Thal_45_4_128SW_resume_best.pkl'
# path32 = '/home/multi-gpu/amur/SVTAS-fresh/output/ResnetMemTCN_Thal_45_4_32SW_resume/ResnetMemTCN_Thal_45_4_32SW_resume_best.pkl'

# if clip_length == 128: 
#     path = path128 
# else:
path = '../output/ThalMultiHead_Colored_224nocenter/ThalMultiHead_Colored_224nocenter_best.pkl' 

model = vfuseACT_multihead(clip_seg_num=clip_length//sample_rate)
weights = torch.load(path)['model_state_dict']
model.load_state_dict(weights)

# weights = {key.replace('conv.net', 'conv'):val for key, val in _weights.items()}

model.eval()
model = model.cuda()

# gt_in = torch.load('/home/multi-gpu/amur/SVTAS-fresh/imgs.pt')
# gt_ma = torch.load('/home/multi-gpu/amur/SVTAS-fresh/masks.pt')
# gt_out = torch.load('/home/multi-gpu/amur/SVTAS-fresh/out.pt')

# tt_out = model({'imgs':gt_in, 'masks':gt_ma})

def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        # transforms.CenterCrop(224),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    frame = Image.fromarray(frame.astype('uint8')[:,:,::-1])
    return transform(frame)
    # transform = A.Compose([
    #                         A.Resize(256,320),
    #                         A.CenterCrop(224, 224),
    #                         A.Normalize(
    #                         mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225],
    #                         ),
    #                         ToTensorV2(),
    #                     ])
    # frame = transform(image=frame)
    # return frame['image']

def postprocess_frame(frame_list, wh =  (1280, 720)):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    original_frames = []
    for frame in frame_list:
        unnormalized_frame = frame * std[:, None, None] + mean[:, None, None]
        uint8_frame = (unnormalized_frame.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        original_frames.append(cv2.resize(uint8_frame[:,:,::-1], wh))
    return original_frames

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

        outs_action = np.argmax(outs['action_score'][-1, :].detach().cpu().numpy().copy(), axis=-2)[0]
        outs_action_post = [mapping_actions[elem] for elem in outs_action]

        outs_branch = np.argmax(outs['branch_score'][-1, :].detach().cpu().numpy().copy(), axis=-2)[0]
        outs_branch_post = [mapping_branch[elem] for elem in outs_branch]

        # for i, frame in enumerate(postprocess_frame(torch.stack(stack[::sample_rate]))): cv2.imwrite(f'ins/test_{i}.png',frame)

        for out_frame, out_action, out_branch in zip(postprocess_frame(stack, (frame_width, frame_height)), outs_action_post, outs_branch_post):
            cv2.putText(out_frame, out_action+out_branch, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            video_writer.write(out_frame) 
        
        stack = []
        frame_count = 0
    
capture.release()
video_writer.release()
print(f"Processed video saved to {output_path}")