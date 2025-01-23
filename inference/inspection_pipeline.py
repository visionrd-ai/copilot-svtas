import cv2
import threading
import queue
import copy
import torch
from PIL import Image
from vfuseACT import vfuseACT 
from torchvision import transforms
import numpy as np 
from collections import Counter
import time 

class FrameReaderThread(threading.Thread):
    def __init__(self, video_path, frame_queue, queue_size=100):
        super(FrameReaderThread, self).__init__()
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.queue_size = queue_size
        self.stop_signal = threading.Event()
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

    def run(self):
        while not self.stop_signal.is_set():
            ret, frame = self.cap.read()
            if not ret: 
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def stop(self):
        self.stop_signal.set()
        self.cap.release()

class Frame:
    def __init__(self, idx, frame, branch_ids, corners, preprocessed):
        self.idx = idx 
        self.frame = frame
        self.branch_ids = branch_ids
        self.corners = corners
        self.preprocessed_frame = preprocessed
    def get_unique_branches(self):
        return list(set(self.branch_ids))
        

class FrameProcessorThread(threading.Thread):
    def __init__(self, frame_queue, branches, detector):
        super(FrameProcessorThread, self).__init__()
        self.frame_queue = frame_queue
        self.branches = copy.deepcopy(branches)
        self.detector = detector
        self.stop_signal = threading.Event()
        self.clip_length = 32
        self.sample_rate = 4
        self.smoothen_ratio = 0.30
        self.path = '/home/multi-gpu/amur/SVTAS-fresh/output/ResnetMemTCN_Thal_45_4_32SW_resume/ResnetMemTCN_Thal_45_4_32SW_resume_best.pkl'
        self.model = vfuseACT(clip_seg_num=self.clip_length//self.sample_rate)
        self.weights = torch.load(self.path)['model_state_dict']
        self.model.load_state_dict(self.weights)
        self.model.eval()
        self.model = self.model.cuda()
        self.mapping = {int(elem.strip().split(' ')[0]): elem.strip().split(' ')[-1] for elem in open('mapping.txt', 'r').readlines()}
        self.previous_task = None 

    def preprocess_frame(self, frame):
        transform = transforms.Compose([
            transforms.Resize((256, 320)), 
            transforms.CenterCrop(224),
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])
        frame = Image.fromarray(frame.astype('uint8')[:,:,::-1])
        return transform(frame)
    
    def show_frames(self, frame_stack, predictions, plot_text):
        
        for frame, pred in zip(frame_stack, predictions):
            plot = cv2.putText(frame.frame, plot_text+pred, (25, 25),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Frame',plot)
            cv2.waitKey(1)
            time.sleep(1/30)
        cv2.destroyAllWindows()
    
    def smooth_majority(self, lst, x, clip_length):
        """
        Smooth a list by replacing chunks of non-majority strings with the majority string.

        Parameters:
            lst (list of str): List of strings to smoothen.
            x (int): Chunk size to consider for replacement.

        Returns:
            list of str: Smoothed list.
        """
        # Determine the majority string
        x = int(x * clip_length)
        if x <= 0:
            raise ValueError("Chunk size x must be greater than 0.")

        # Determine the majority string
        majority = Counter(lst).most_common(1)[0][0]

        # Log calculation details
        # print(f"Chunk size (x): {x}, Clip length: {clip_length}, Majority: {majority}")

        # Iterate through the list in chunks of size x
        smoothed_list = lst[:]
        for i in range(0, len(lst), x):
            chunk = lst[i:min(i+x, len(lst))]
            # Count occurrences of the majority string in the chunk
            majority_count = sum(el == majority for el in chunk)
            if majority_count < len(chunk):  # If the majority does not dominate the chunk
                smoothed_list[i:i+len(chunk)] = [majority] * len(chunk)

        return smoothed_list

    def validate_branches(self, curr_branch, frames):
        lost = 0 
        for frame in frames: 
            if frame.branch_ids is None or curr_branch not in frame.branch_ids:
                lost += 1 
        return lost <= self.smoothen_ratio*self.clip_length

    def run(self):
        while not self.stop_signal.is_set():
            if not self.frame_queue.empty():
                for branch_idx, current_branch in enumerate(self.branches):
                    current_branch_name = current_branch['num']
                    branch_tasks_mapped = [task['mapping'] for task in current_branch['tasklist']]
                    task_counter = Counter(branch_tasks_mapped)

                    for task_idx, current_task in enumerate(current_branch["tasklist"]):
                        current_task_name = current_task["task_name"]

                        task_complete = False 
                        branch_lost = False
                        while True:


                            duration = current_task["duration"]
                            ids = None 
                            curr_frames = [] # use a frame stack class with special methods that return all preprocessed frames, ids, etc.

                            for frame_idx in range(self.clip_length):
                                # while ids is None: 
                                frame = self.frame_queue.get()
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                corners, ids, rejected = detector.detectMarkers(gray)
                                cv2.imwrite(f'frames/{frame_idx}.png', frame)

                                curr_frames.append(Frame(idx=frame_idx, frame=frame, branch_ids=ids, corners=corners, preprocessed=self.preprocess_frame(frame)))
                            
                            branch_valid = self.validate_branches(curr_branch=current_branch['num'], frames=curr_frames)
                            
                            if branch_valid:

                                stack = [pre.preprocessed_frame for pre in curr_frames]
                                
                                input_data = {  'imgs': torch.stack(stack[::self.sample_rate]).unsqueeze(0).cuda(),
                                                'masks': torch.ones((1, self.clip_length)).cuda()
                                            }
                                
                                outs = self.model(input_data)
                                outs = np.argmax(outs[-1, :].detach().cpu().numpy().copy(), axis=-2)[0]
                                outs_post = [self.mapping[elem] for elem in outs]
                                # print("INFO: Before smoothing: ", list(set(outs_post)))
                                outs_post = self.smooth_majority(outs_post, self.smoothen_ratio, self.clip_length)
                                # print("INFO: After smoothing: ", list(set(outs_post)))
                                
                                # plot_text = f'Current Branch: {current_branch_name} | Current Task: {current_task_name} | Prediction: '
                                # self.show_frames(curr_frames, outs_post, plot_text)

                                pred_count = Counter(outs_post)
                                preds_ordered = list(set(outs_post))

                                print(150*'*')
                                print(f"INFO: Current Branch: {current_branch_name} | Current Task: {current_task_name} | Predictions (in order): {preds_ordered}")
                                if preds_ordered[0] == current_task['mapping']: # the prediction is correct (order is correct), add the duration
                                    print(f"INFO: Current Branch: {current_branch_name} | Current Task: {current_task_name} | The first prediction was in order with current task: '{current_task['mapping']}', adding duration ({task_counter[current_task['mapping']]+pred_count[current_task['mapping']]}/{current_task['duration']})")
                                    task_counter += pred_count 
                                    # check if task has passed its duration 
                                    if task_counter[current_task['mapping']] >= current_task['duration']:
                                        self.branches[branch_idx]['tasklist'][task_idx]['status'] = True
                                        print(f"INFO: Current Branch: {current_branch_name} | Current Task: {current_task_name} | {current_branch['name']}: {current_task['task_name']} was completed, going to next task")
                                        self.previous_task = current_task
                                        break
                                    else: # if task hasn't passed its duration and there are leftover predictions
                                        # if there are leftover predictions in this second and the current task hasn't been matched, they are wrong predictions.
                                        if len(preds_ordered[1:]):
                                            print(f"WARNING: Current Branch: {current_branch_name} | Current Task: {current_task_name} | Encountered wrong leftover predictions ({preds_ordered[1:]}) instead of {current_task['mapping']}") # blink red 

                                else: #  
                                    print(f"INFO: Current Branch: {current_branch_name} | Current Task: {current_task_name} | The first prediction was out of order with current task '{current_task['mapping']}'") # blink red 
                                    if self.previous_task is not None and preds_ordered[0] == self.previous_task['mapping']:
                                        print(f"INFO: Current Branch: {current_branch_name} | Current Task: {current_task_name} | Still getting previous task '{self.previous_task['task_name']}', waiting for '{current_task_name}'") # blink red 
                                    else: 
                                        print(f"WARNING: Current Branch: {current_branch_name} | Current Task: {current_task_name} | Encountered wrong prediction {preds_ordered[0]} instead of {current_task['mapping']}") # blink red 
                                    pass 
                            else: 
                                print(f"WARNING: Could not validate current branch ({current_branch['num']})")


                            ###### SINGLE MULTIPLE SEPARATE ########
                            # if len(preds_ordered) == 1: # there is only one prediction for this second
                            #     print(f"INFO: Current Branch: {current_branch_name} | Current Task: {current_task_name} | There is only one prediction in this second: '{preds_ordered[0]}'")
                            #     if preds_ordered[0] == current_task['mapping']: # the prediction is correct (order is correct), add the duration
                            #         print(f"INFO: Current Branch: {current_branch_name} | Current Task: {current_task_name} | The single prediction was in order with current task: '{current_task['mapping']}', adding duration ({task_counter[current_task['mapping']]}/{current_task['duration']})")
                            #         task_counter += pred_count 
                            #         if task_counter[current_task['mapping']] >= current_task['duration']:
                            #             self.branches[branch_idx]['tasklist'][task_idx]['status'] = True
                            #             print(f"INFO: Current Branch: {current_branch_name} | Current Task: {current_task_name} | {current_branch['name']}: {current_task['task_name']} was completed, breaking")
                            #             self.previous_task = current_task
                            #             break
                            #         # check if task has passed its duration 
                            #     else: #  
                            #         print(f"INFO: Current Branch: {current_branch_name} | Current Task: {current_task_name} | The single prediction was out of order with current task '{current_task['mapping']}'") # blink red 
                            #         if self.previous_task is not None and preds_ordered[0] == self.previous_task['mapping']:
                            #             print(f"INFO: Current Branch: {current_branch_name} | Current Task: {current_task_name} | Still getting previous task '{self.previous_task['task_name']}', waiting for '{current_task_name}'") # blink red 
                            #         else: 
                            #             print(f"WARNING: Current Branch: {current_branch_name} | Current Task: {current_task_name} | Encountered wrong prediction! Seeing {preds_ordered[0]} instead of {current_task_name}'") # blink red 
                            #         pass 
                            # else: # there were multiple predictions, check their order
                            #     print(f"INFO: Current Branch: {current_branch_name} | Current Task: {current_task_name} | There are multiple predictions in this second: '{preds_ordered}'")
                            #     pass
                            ########################################

                            # check if current task has completed 
                            # if self.branches[branch_idx]['tasklist'][task_idx]['task_status']: 
                            #     break 
                        if self.branches[branch_idx]['tasklist'][task_idx]['task_status']: 
                            print(f"INFO: Current Branch: {current_branch_name} | Current Task: {current_task_name} | {current_branch_name}: {current_task_name} was completed")
                            import pdb; pdb.set_trace()
                             #### REFACTOR THIS AS FOLLOWS: 
                            # check if first pred matches, if it does, add duration and check duration for completion 
                            # if first pred doesnt match, blink red 
                            # if it matches and the length of predictions is greater than 1, check if duration for first prediction has completed
                            # if it hasnt, blink red 
                            # if it has blink green, check if the next predictions belong in the tasklist of this branch and add them to the task counter



                                    

                            # check if there are any predictions that dont belong to this branch
                            # extra_preds = pred_count.keys()-branch_tasks_mapped
                            # if pred_count.keys()-branch_tasks_mapped:
                                # it is a wrong step e.g. checking tag of b3 while b1 is incomplete, blink yellow 
                                # or the model is predicting incorrectly 
                                # import pdb; pdb.set_trace()
                            
                            # if there are 

                            
                        #     print(f"Waiting for {task_name} in {current_branch['name']} (duration: {duration}s)")
                        #     branch_lost = any(map(lambda x: current_branch['num'] not in x.branch_ids, curr_frames))
                            
                        #     ##### check tasks here and set unchecked_branch['status'] before checking lost_without completion ####
                            
                        #     lost_without_completion = branch_lost and not current_branch['status']

                        #     if lost_without_completion:
                        #         print("Branch is lost without completion")
                        #         break
                            

                        # if lost_without_completion: 
                        #     # branch was lost without completion i.e. person changed area without completing 
                        #     # the branch, make it blink red and retry
                        #     print("Breaking out of branch")
                        #     #### 
                        #     break 

    def stop(self):
        self.stop_signal.set()

if __name__ == "__main__":
    input_video_path = "aruco_test.mp4"
    frame_queue = queue.Queue(maxsize=100)  

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    branches = [
        {
        'num':1, 
        'name': 'Branch 1', 
        'tasklist': [{'task_name': 'Check Tape', 'duration': 60, 'task_status':False, 'mapping':'tape'},
                     {'task_name': 'Check Connectors', 'duration': 100, 'task_status':False, 'mapping':'clip'}],
        'status': False
        },
        {
        'num':2, 
        'name': 'Branch 2', 
        'tasklist': [{'task_name': 'Check Tape', 'duration': 30, 'task_status':False, 'mapping':'tape'},
                     {'task_name': 'Check Connectors', 'duration': 75, 'task_status':False, 'mapping':'clip'}],
        'status': False
        },
    ]

    # Start the threads
    frame_reader = FrameReaderThread(input_video_path, frame_queue)
    frame_processor = FrameProcessorThread(frame_queue, branches, detector)

    frame_reader.start()
    frame_processor.start()

    try:
        while True:
            pass  # Main thread can handle other tasks
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        frame_reader.stop()
        frame_processor.stop()
        frame_reader.join()
        frame_processor.join()
        cv2.destroyAllWindows()
