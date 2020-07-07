
"""
SMART QUEUE SYSTEM USING INTEL DEVCLOUD
Created on Mon Jun 22 17:11:05 2020

@author: Abdul Basit
"""
# IMPORTING REQUIRED LIBRARIES FOR THE PROJECT
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2 # Importing opencv
import argparse
import sys


class Queue:
    '''
    Class for dealing with queues
    Performs operations for queues like adding , getting the queues 
    and checking the coordinates
    '''
    def __init__(self):
        # A list contains the queues data
        self.queues=[]

    def add_queue(self, points):
        # Add points to queues
        self.queues.append(points)

    def get_queues(self, image):
        # Get queue from iamges
        # yield frame of image after extracting coordinates
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords, frame):
        # Check coordinates for queues
        # Return frame
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
                   
        return d, frame


class PersonDetect:
    '''
    Class for the Person Detection Model.
    Performs prediction, draw outputs, preprocessed inputs and outputs
    '''

    def __init__(self, model_name, device, threshold=0.60):
        # Set threshold to 0.6, it can be changed as per application requirement
        ''' Inits PersonDetect class with model_name (weights and structure), device, threshold, initial width and height'''
        self.model_weights=model_name+'.bin'
        # String contains model weights path i.e. .bin
        self.model_structure=model_name+'.xml'
        # String contains model structure path i.e. .xml
        self.device=device
        # String contains device name
        self.threshold=threshold
        # String contains threshold value as floating point
        self.initial_w = ''
        # String contains initial width that will be used in draw_output function to extract boundaries
        self.initial_h = ''
        # String contains initial height that will be used in draw_output function to extract boundaries
        
        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        
        # A tuple of the input shape : input_shape
        # A list of output name : soutput_name
        # A tuple of the output shape : output_shape
       
        
        self.input_name=next(iter(self.model.inputs))
        # Get the name of the input node
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        # Get the name of the output node
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        ''' Load the model
        '''
        self.core = IECore()
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        # IECore object : core
        # Loaded net object:  net
        # raise NotImplementedError
        
        
    def predict(self, image):
        ''' Make asynchronous predictions from images
        List of image data: image
        '''
        input_img = self.preprocess_input(image)
        # Running Inference in a loop on the same image
        input_dict = {self.input_name:input_img}
        
        # Start asynchronous inference for specified request.
        self.net.start_async(request_id=0,inputs=input_dict)
        infer_status = self.net.requests[0].wait(-1)
        if infer_status == 0:
            results = self.net.requests[0].outputs[self.output_name]
            image,coords = self.draw_outputs(results, image)
        return coords,image
        # returns coords and image
        #raise NotImplementedError
    
    def draw_outputs(self, results, frame):
        initial_point=0 
        # Represent top left corner of rectangle
        ending_point=0 
        # Represent bottom right corner of rectangle
        det=[]
        # Set initial value i.e. coordinates list, two points/coordinates for rectangle
        
        """
        It Draws outputs (predictions) on image.
        It takes coords/results : The coordinates of predictions.
            and image/ frame: The image on which boxes need to be drawn.
        It will return
            1) the frame
            2) bounding boxes above threshold
        """
        # Rectangle need two coordinates, one is top left corner and second one is bottom right
        # Top left corner will be (xmin,ymin)
        # Bottom right corner will be (xmax,ymax)
        
        # Loop through detections and determine what and where the objects are in the image
        # For each detection , it has 7 values i.e. [image_id,label,conf,x_min,y_min,x_max,y_max]
        # image_id - ID of the image in the batch
        # label - predicted class ID
        # conf - confidence for the predicted class
        # (x_min, y_min) - coordinates of the top left bounding box corner
        # (x_max, y_max) - coordinates of the bottom right bounding box corner
        for obj in results[0][0]:
            # Draw bounding box for object when it's probability is more than the specified threshold
            
            if obj[2] > self.threshold: # Extract the confidence and compare with threshold value
                xmin = int(obj[3] * self.initial_w)
                ymin = int(obj[4] * self.initial_h)
                xmax = int(obj[5] * self.initial_w)
                ymax = int(obj[6] * self.initial_h)
                initial_point = (xmin,ymin)
                ending_point = (xmax,ymax)
                # Use cv2.rectangle() method to draw a rectangle around detection 
                # Draw a rectangle with colored line (can be changed as per requirement) borders of thickness of 1 px
                # cv2. rectangle(img, pt1, pt2, color, thickness)
                cv2.rectangle(frame, initial_point, ending_point, (255, 0, 0), 2)
                rec_points = [xmin,ymin,xmax,ymax]
                # Can also be written as rec_points=[initial_points,ending_points]
                det.append(rec_points)
               
        return frame,det
        #raise NotImplementedError

    def preprocess_outputs(self, outputs):
        """
        Preprocess the outputs.
        It takes the output from predictions i.e outputs
        It will return preprocessed dictionary.
        """
        out_dict = {}
        for output in outputs:
            output_name = self.output_name
            output_img = output
            out_dict[output_name] = output_img
        
        return out_dict
    
        return output
        #raise NotImplementedError
        
    def preprocess_input(self, image):
        ''' It preproprocessed input
        An input image in the format [BxCxHxW], where:

            B - batch size (here we use n)
            C - number of channels
            H - image height
            W - image width
        '''
        n, c, h, w = self.input_shape
        # Extracting n,c,h and w from input image
        image = cv2.resize(image, (w, h),interpolation = cv2.INTER_AREA)
        # We used INTER_AREA for interpolation as i has resamping using pixel area relation and preferred method for image decimation
        
        # Change image from HWC to CHW
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        
        return image
        #raise NotImplementedError
        ''' Here we can also use these code instead of above code:
            image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            image = image.transpose(2, 0, 1)
            image = image.reshape(1, *image.shape)
            return image
        OR
            image = cv2.resize(image, (w, h))
            pp_image = image.transpose((2, 0, 1))
            pp_image = pp_image.reshape(1, *pp_image.shape)
            return pp_image
        '''


def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path
    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    pd.initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pd.initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # In opencv we can replace CAP_PROP_FRAME_WIDTH with (3)
    # In the same maner we can replace CAP_PROP_FRAME_HEIGHT with (4)
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (pd.initial_w, pd.initial_h), True)
    
    counter=0
    start_inference_time=time.time()
    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            coords, image= pd.predict(frame)
            num_people, image= queue.check_coords(coords,image)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text=""
            y_pixel=45
            
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)


#The argparse module makes it easy to write user-friendly command-line interfaces
#Add required  groups
#Create the arguments
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)