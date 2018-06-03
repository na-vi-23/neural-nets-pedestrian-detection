from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os.path
import re
import time
import math
import logging
from munkres import Munkres,print_matrix
from pykalman import KalmanFilter
from imutils.video import FPS
from movingobject import MovingObject

## get parameters from arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
        help="path to input video file")
ap.add_argument("-o", "--output",
        help="output video file name")
ap.add_argument("-b", "--bg", type=int, default=0,
        help="1: save background, 0: no saving")
args = vars(ap.parse_args())

## get input video file
##videofile = '../video/Camera65_13_00_09272017_0.mp4'
videofile = args["input"]

##get the input video file name without extension
## ex. Camera65_13_00_09272017_0
m = re.match(r".*/([^/]*)(\.[^/]*)",videofile)
vfname = m.group(1)

###output video file
out_file = vfname + '_out.avi'
if args["output"] is not None:
    out_file = args["output"]
##background
QUICK_BACKGROUND_SAVE = False 
if args["bg"] == 1:
    QUICK_BACKGROUND_SAVE = True

m = re.match(r"([a-zA-Z0-9_]+)_([^_].*)",vfname)
prefix = m.group(1)
num = m.group(2)
num_next = int(num) + 1
#background = 'config/background_camera65.npy'
background = 'config/' + vfname + '_bg.npy'
background_next = 'config/' + prefix + '_' + str(num_next) + '_bg.npy'

##logfile
##'log/tracking_kalman_hungarian.log'
logfile = 'log/' + vfname + '.log'

###for logging
def mylogger(name):
    logger = logging.getLogger(name)
    if not len(logger.handlers):
        logger = logging.getLogger(name)
        hdlr = logging.FileHandler(logfile)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)

    return logger

my_logger = mylogger('log_tracking_kalman')

savedir = "camera65_detect/"
savedir_blur = "camera65_bgremoved/"
savedir_bg = "camera65_bg/"


##Configuration for  threasholds
##threashold for decide if x and y and their predictions are the same
POSITION_THREASHOLD= 30

##contours lower than this threshold will be ignored, because either they are not pedestrian
## or they are too far away
HIGHT_THRESHOLD = 50

##interest area threshold, we can use this threshold to limit detecting area
LIMIT_AREA = True 
X_MG = 80 
Y_MG = 0
X1_MG = 220
Y1_MG = 240

##missing frame numbers to dismiss the object from the current_tracks
MISSING_THREASHOLD = 90 
##minimum contour area to consider
CONTOUR_THREASHOLD_MIN = 300
#CONTOUR_THREASHOLD_MIN = 20
##maximum contour area to consider
CONTOUR_THREASHOLD_MAX = 8000
#CONTOUR_THREASHOLD_MAX = 1000

COST_THREASHOLD = 80

##pedestrian counter
COUNTER_m = 0
COUNTER_p = 0
OVERLAP_THREASHOLD = 0.30
DISTANCE_THREASHOLD = 20

##init array to hold current objects
current_tracks = []

## To resize the frame by 2, then RESIZEFRAME = 2
## No resize, the RESIZEFRAME = 0
#RESIZEFRAME = 0.75
RESIZEFRAME = 360

##debug
debug = True 
PEDESTRIAN_DEBUG = True


Pedestrians ={}

def track_new_object(position, tracks, counter):
    new_mObject = MovingObject(counter,position)
    new_mObject.add_position([position])
    if debug:
        my_logger.info("create new object, id: %d", new_mObject.id)
    new_mObject.init_kalman_filter()
    filtered_state_means, filtered_state_covariances = new_mObject.kf.filter(new_mObject.position)
    new_mObject.set_next_mean(filtered_state_means[-1])
    new_mObject.set_next_covariance(filtered_state_covariances[-1])

    ##add to current_tracks
    tracks.append(new_mObject)
    
#function overlap calculate how much 2 rectangles overlap
def overlap_area(boxes):
    if(len(boxes) == 0):
        return 0
    
    xx1 = max(boxes[0,0], boxes[1,0])
    yy1 = max(boxes[0,1], boxes[1,1])
    xx2 = min(boxes[0,2], boxes[1,2]) 
    yy2 = min(boxes[0,3], boxes[1,3])
    
    w = max(0, xx2 - xx1 + 1)
    h = max(0, yy2 - yy1 + 1)
    
    ##box1 area
    area1 = (boxes[0,2]-boxes[0,0]) * (boxes[0,3]-boxes[0,1])
    ##box2 area
    area2 = (boxes[1,2]-boxes[1,0]) * (boxes[1,3]-boxes[1,1])
    if area1 > area2:
        area = area2
    else:
        area = area1
        
    overlap = (w * h) / area
    
    return overlap

##calculate distance between 2 points, pos: [0,0], points: [[1,1],[2,2],[3,3]]
def get_costs(pos,points):
    distances = [math.floor(math.sqrt((x2-pos[0])**2+(y2-pos[1])**2)) for (x2,y2) in points]
    return distances

def update_skipped_frame(frame,bgrmv,fname,tracks):
    
    if debug:
        my_logger.info("skipped")
    cv2.imshow("Frame", frame)
    cv2.imshow('bgrmv', bgrmv)
    #if PEDESTRIAN_DEBUG:
        #cv2.imshow('pedestrian',ped_debug)
    
    ##savepath = savedir + fname
    ##cv2.imwrite(savepath, frame)
    ##blurred_file_path = savedir_blur + fname
    ##cv2.imwrite(blurred_file_path, bgrmv)

    h,w = frame.shape[:2]
    print(fname)
    counter = "Pedestrians: " +  str(COUNTER_p) + " frame " + fname
    cv2.putText(frame,counter,(int(w/4),int(h-20)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,0),1)   

    ##write out to video
    out.write(frame)
        
    cv2.waitKey(1)
    fps.update()

    ##update tracking objects
    for obj in tracks:
        obj.frames_since_seen += 1
        if debug:
            my_logger.info("disappeared object id %d", obj.id)
            my_logger.info("frames_since_seen %d", obj.frames_since_seen)
    new_tracks = removeTrackedObjects(tracks,frame)

    return new_tracks

def removeTrackedObjects(tracking_arr,frame):
    for index, obj in enumerate(tracking_arr):
        ##if a moving object hasn't been updated for 10 frames then remove it
        if obj.frames_since_seen > MISSING_THREASHOLD:
            del tracking_arr[index]
            if debug:
                my_logger.info("frames_since_seen %d", obj.frames_since_seen)
                my_logger.info("Remove tracking object %d", obj.id)
        ## if the object is out of the scene then remove from current tracking right away
        h,w = frame.shape[:2]
        if (obj.position[-1][0] < 0 or obj.position[-1][0] > w):
            if debug:
                my_logger.info("Remove out of scene tracking object %d", obj.id)
            del tracking_arr[index]
        elif (obj.position[-1][1] < 0 or obj.position[-1][1] > h):
            if debug:
                my_logger.info("Remove out of scene tracking object %d", obj.id)
            del tracking_arr[index]

    return tracking_arr

##Main program starts here
n = 0
avg = None

if os.path.exists(background):
    avg = np.load(background)
elif os.path.exists(background_next):
    avg = np.load(background_next)

##hog for pedestrian detection
hog = cv2.HOGDescriptor((48,96), (16,16), (8,8), (8,8), 9)
hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())
hogParams = {'winStride': (8, 8), 'padding': (16, 16), 'scale': 1.05}

##Loop through each frame and find the contours
##then:
## 
##  if no movingObject existing yet, init the first one
##    run Kalman filter to predict next position
##  else, in the following frame:
##    save all the detected contours corrdinations and calculate the cost to form a matrix
##      --> distance from each assigned movingObject to the detected coutours in this frame
##    use Hungarian algorithm to calculate assignment
##      in case: more contours than detected object, means unassigned moving object exists
##      find the index and start a new movingObject and track it
##      another case: tracked movingObjects index not shown in the assignment, means it's not updated in this frame
##    Use the assignment to update Kalman filter
if debug:
    my_logger.info("Start detection...")
    my_logger.info(videofile)
    ##if (not os.path.exists(savedir) or (not os.path.exists(savedir_blur))):
        ##print(savedir + " or " + savedir_blur + " does not exist! Please provide valid directory. Exiting...")

##Main: loop through the video frames
cap = cv2.VideoCapture(videofile)

##for write out videos
FOURCC = cv2.VideoWriter_fourcc('M','J','P','G')
output_fps = 30
out = None

fps = FPS().start()

while(True):
    ret, frame_ori = cap.read()
    if ret == True:
        cv2.imshow("Frameori", frame_ori)

        h,w = frame_ori.shape[:2]
        ##if resize the frame
        if(RESIZEFRAME > 0 and RESIZEFRAME <1):
            frame = imutils.resize(frame_ori,width=int(RESIZEFRAME*w))
        elif (RESIZEFRAME > 1):
            frame = imutils.resize(frame_ori,width=int(RESIZEFRAME))
        else:
            frame = frame_ori.copy() 
            
        h,w = frame.shape[:2]
        if out is None:
            out = cv2.VideoWriter(out_file,FOURCC, output_fps, (w,h))
        
        fname = "%05d.jpg" % n
        if debug:
            my_logger.info("%s ---------------------------------", fname)
        ##remove background and find contour
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if avg is None:
            avg = np.float32(gray)
        cv2.accumulateWeighted(gray, avg, 0.01)
        framediff = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        thresh = cv2.threshold(framediff, 25, 255, cv2.THRESH_BINARY)[1]
        blurred = cv2.dilate(thresh, None, iterations=1)
        
        if QUICK_BACKGROUND_SAVE:
            np.save(background, avg)

        ##detect pedestrian
        ##run detection against blurred image
        (ped_rects, weights) = hog.detectMultiScale(frame, **hogParams)

        ##find contour in background removed frames
        (_, cnts, _) = cv2.findContours(blurred, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        ##mark pedestrians with write bounding box in backgroud
        if PEDESTRIAN_DEBUG:
            ped_debug = frame.copy()
            for (ped_x, ped_y, ped_w, ped_h) in ped_rects:
                cv2.rectangle(ped_debug, (ped_x, ped_y), (ped_x + 40, ped_y + 70), (255, 0, 0), 2)
        
        if(len(current_tracks) > 0):
            for index, obj in enumerate(current_tracks):
                obj.counted = 0

        ##loop through all detected contours and save them into array contours
        contours = np.zeros((0,2))
        contours_orig = np.zeros((0,4))
        for ct in cnts:
            ##skip contours too small or too large
            if cv2.contourArea(ct) < CONTOUR_THREASHOLD_MIN or cv2.contourArea(ct) > CONTOUR_THREASHOLD_MAX:
                if(cv2.contourArea(ct) > CONTOUR_THREASHOLD_MAX):
                    if debug:
                        my_logger.info("skip too large area %f")
                continue
            (x, y, w, h) = cv2.boundingRect(ct)
            ##skip contour has very low hight
            if(h < HIGHT_THRESHOLD):
                continue
            ## limit the contour to this area of interest
            if((x < X_MG or x > X1_MG or y < Y_MG or y>Y1_MG) and LIMIT_AREA == True ):
                ##print("skipt this contour", x,y)
                continue
            ##threashold  to exclude cars
            if(w > 2*h):
                if debug:
                    print("skip ct w > 2h",x, y, w, h)
                continue
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (240, 0, 159), 2)
            ##center of the bounding rectangle
            cx = x + int(w/2)
            cy = y + int(h/2)

            ##draw rectangle around contour
            cv2.rectangle(frame, (x, y), (x + w, y + h), (240, 0, 159), 2)
            cv2.rectangle(blurred, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
            ##location for Kalman filter to track
            tdata = [cx,cy]
            
            ##add positions to contours array
            contours_orig = np.append(contours_orig,[[x,y,w,h]],axis=0)
            contours = np.append(contours, [tdata],axis=0)

        ##done with saving all the detected contours

        #############################################
        ##For first frame, add all detected contours as new MovingObjects
        if(len(current_tracks) == 0):
            for cont in contours:

                COUNTER_m += 1
                ##create new movingObject
                new_mObject = MovingObject(COUNTER_m,tdata)
                new_mObject.add_position([tdata])
                if debug:
                    my_logger.info("create object, id: %d ", new_mObject.id)
                    
                #cv2.putText(frame,str(new_mObject.id),(int(tdata[0]),int(tdata[1]-40)),cv2.FONT_HERSHEY_SIMPLEX,0.8, (255,0,0),2)

                new_mObject.init_kalman_filter()
                filtered_state_means, filtered_state_covariances = new_mObject.kf.filter(new_mObject.position)
                new_mObject.set_next_mean(filtered_state_means[-1])
                new_mObject.set_next_covariance(filtered_state_covariances[-1])

                ##add to current_tracks
                current_tracks.append(new_mObject)
                if debug:
                    my_logger.info("COUNTER_m++ %d", COUNTER_m)

        ##from the 2nd frame, calculate cost using predicted position and new contour positions
        else:
            ##save all the positions to a matrix and calculate the cost
            ##initiate a matrix for calculating assianment by Hungarian algorithm
            if(len(contours) == 0):
                if debug:
                    my_logger.info("No contour found!")
                #savepath = savedir + fname
                #cv2.imwrite(savepath, frame)

                #blurred_file_path = savedir_blur + fname
                #cv2.imwrite(blurred_file_path, blurred)
                n = n + 1
                current_tracks = update_skipped_frame(frame,blurred,fname,current_tracks)
                continue

            #matrix_h = np.zeros((0,len(contours)))
            matrix_h =[]
            remove_obj = []
            #if debug:
                #print("number of current objects", len(current_tracks))

            ##loop through existing tracked movingObjects
            ## find movingObjects which are disappeared already and add to remove_obj array
            
            ## caclulate the cost to each contour and form a matrix
            ## use the available_tracks list to remove tracking object which has distance larger than threshold
            ## so it won't mess up the rest
            available_tracks = []
            for index, obj in enumerate(current_tracks):
                ##calculate costs for each tracked movingObjects using their predicted position
                costs = get_costs(obj.predicted_position[-1], contours)
                
                ## if tracking object to all contours distances are too large, then not to consider it at all
                if all(c > COST_THREASHOLD for c in costs):
                    ##update it with KF predicted position
                    obj.kalman_update_missing(obj.predicted_position[-1])
                    ##skip this tracking object
                    continue
                    
                matrix_h.append(costs)
                ##only valid tracking objects are added to available_tracks
                available_tracks.append(obj)

            ## matrix_h: 
            ##
            ##       |contour1 | contour2 |...
            ##-----------------------------------
            ## mObj1 | cost11   | cost12  |...
            ## mObj2 | cost21   | cost22  |...
            ##
            ##calculate assignment with the matrix_h
            ## a missing column means new track
            ## a missing row means missing track in the frame

            munkres = Munkres()
            ## when matrix is empty, skip this frame
            if(len(matrix_h) < 1):
                n = n + 1
                current_tracks = update_skipped_frame(frame,blurred,fname,current_tracks)
                if debug:
                    my_logger.info("matrix_h < 1, skip this frame")
                continue
            indexes = munkres.compute(matrix_h)

            total = 0
            for row, column in indexes:
                value = matrix_h[row][column]
                total += value
                
            ## next : loop through the indexes got from the assignment and update Kalman filter
            ## find untracked MovingObjects and track them
            ## the movingObjects not being updated, set last_since_seen += 1

            ## loop through the contours and update Kalman filter for each contour
            indexes_np = np.array(indexes)
            for index_c,cont in enumerate(contours):
                ## found contour index, then update this contour position with the tracked object
                if index_c in indexes_np[:,1]:
                    contour_index_list = indexes_np[:,1].tolist()
                    ##find index of the movinbObject
                    ## ex. [(0, 2), (1, 1), (3, 0), (4, 3)] --> 2nd element with contour[0], index_m=2
                    index_m = contour_index_list.index(index_c)

                    ##find index in current_tracks
                    index_track = indexes_np[index_m,0]
                    
                    ##check if cost is bigger than threashold then track it as a new one
                    if matrix_h[index_track][index_c] > COST_THREASHOLD:
                        if debug:
                            my_logger.info("Too much distance in between, cannot be the same object")
                        ##create new MovingObject of this contour
                        COUNTER_m += 1
                        track_new_object(cont, current_tracks, COUNTER_m)
                        continue

                    ##get the object from the index and update it's Kalman filter
                    obj_m = available_tracks[index_track]
                    ##get corresponding contour position, update kalman filter
                    position_new = contours[index_c]
                    obj_m.kalman_update(position_new)
                    
                    obj_m.counted = 1
                    ## mark the moving object with the id
                    prx = position_new[0]
                    pry = position_new[1]
                    #label_points = np.array([[[position_new[0], position_new[1]],[position_new[0]+40, position_new[1]],[position_new[0]+40, position_new[1]-20], [position_new[0], position_new[1]-20]]])
                    #cv2.fillPoly(frame,label_points,(0,255,0))
                    #cv2.putText(frame,str(obj_m.id),(int(position_new[0]),int(position_new[1]-20)),cv2.FONT_HERSHEY_SIMPLEX,0.8, (255,0,0),2)
                    ##show pedestrian counting number
                    if obj_m.id in Pedestrians:
                        cv2.putText(frame,str(Pedestrians[obj_m.id]),(int(position_new[0]),int(position_new[1]+30)),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,255,0),2)

                    ##get the original contour x,y,w,h, not the center corrdination
                    cont_x,cont_y,cont_w,cont_h = contours_orig[index_c]

                    ##add pedestrian detection
                    for (prx,pry,prw,prh) in ped_rects:
                        c_x = prx + int(prw/2)
                        c_y = pry + int(prh/2)

                        ##compare by bounding box overlap
                        boxes_2compare = np.array([[cont_x,cont_y,cont_x+cont_w,cont_y+cont_h],[prx,pry,prx+40,pry+70]])

                        o_rate = overlap_area(boxes_2compare)
                        if(o_rate > OVERLAP_THREASHOLD):
                            obj_m.detection_increase()
                            if debug:
                                my_logger.info("enough overlap to be counted as pedestrian.")
                            
                            ##mark overlapped pedestrian and moving object in blue box
                            #if debug:
                                #cv2.rectangle(frame, (prx, pry), (prx + 40, pry + 70), (0, 0, 255), 2)

                            ##for pedestrian detection
                            if(obj_m.detection >= 5 ):
                                ##check if this is a new pedestrian we detected by checking
                                ##if the object id is in Pedestrians dict or not
                                if (not obj_m.id in Pedestrians.keys()):
                                    
                                    COUNTER_p += 1
                                    Pedestrians[obj_m.id] = COUNTER_p
                                    if debug:
                                        my_logger.info("Pedestrian No " + str(COUNTER_p) + " frame: " + fname)
                                ##if added already then draw notation in the frame
                                else:
                                    ##mark the pedestrian in the image
                                    pr_points = np.array([[[prx, pry],[prx+prw, pry],[prx+40, pry-20], [prx, pry-20]]])
                                    #cv2.fillPoly(frame,pr_points,(0,255,0))
                                    #cv2.putText(frame,str(Pedestrians[obj_m.id]),(int((prx+w)/2),int(pry-5)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,0,0))

                ## not found in columns, means not being tracked, so start tracking it
                else:
                    position_new = contours[index_c]
                    COUNTER_m += 1
                    new_mObject = MovingObject(COUNTER_m,position_new)
                    new_mObject.add_position([position_new])
                    new_mObject.init_kalman_filter()
                    if debug:
                        my_logger.info("create new object, id: %d", new_mObject.id)
                        
                    filtered_state_means, filtered_state_covariances = new_mObject.kf.filter(new_mObject.position)
                    new_mObject.set_next_mean(filtered_state_means[-1])
                    new_mObject.set_next_covariance(filtered_state_covariances[-1])

                    ##add to current_tracks
                    current_tracks.append(new_mObject)
                    #cv2.putText(frame,str(new_mObject.id),(int(position_new[0]),int(position_new[1]-20)),cv2.FONT_HERSHEY_SIMPLEX,0.8, (255,0,0),2)

                    if debug:
                        my_logger.info("counter %d", COUNTER_m)

            ##these are tracks missed either because they disappeared 
            ## or because they are temporarily invisable 
            for index,obj in enumerate(available_tracks):
                if index not in indexes_np[:,0]:
                    ## not update in this frame, increase frames_since_seen
                    obj.frames_since_seen += 1
                    ##but we update KF with predicted location
                    obj.kalman_update_missing(obj.predicted_position[-1])
                    
                    if debug:
                        my_logger.info("disappeard object id %d", obj.id)
                        my_logger.info("frames_since_seen %d", obj.frames_since_seen)
                    
            ##remove movingObj not updated for more than threasholds numbers of frames  
            for index, obj in enumerate(current_tracks):
                ##if a moving object hasn't been updated for 10 frames then remove it
                if obj.frames_since_seen > MISSING_THREASHOLD:
                    del current_tracks[index]
                    
                    if debug:
                        my_logger.info("disappeard object id %d", obj.id)
                        my_logger.info("frames_since_seen %d", obj.frames_since_seen)
                        my_logger.info("Remove tracking object %d", obj.id)
                ## if the object is out of the scene then remove from current tracking right away
                h,w = frame.shape[:2]
                if (obj.position[-1][0] < 0 or obj.position[-1][0] > w):
                    if debug:
                        my_logger.info("Remove out of scene tracking object %d", obj.id)
                    del current_tracks[index]
                    
                elif (obj.position[-1][1] < 0 or obj.position[-1][1] > h):
                    if debug:
                        my_logger.info("Remove out of scene tracking object %d", obj.id)
                    del current_tracks[index]
                
            ##remove movingObj not updated for more than threasholds numbers of frames  
            if debug:
                for index, obj in enumerate(current_tracks):
                    my_logger.info("current tracking objects %d", obj.id)
                

        h,w = frame.shape[:2]
        print(fname)
        counter = "Pedestrians: " +  str(COUNTER_p) + " frame " + fname 
        cv2.putText(frame,counter,(int(w/4),int(h-20)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,0),1)
        n = n + 1

        #if debug:
            #blurred_file_path = savedir_blur + fname
            #cv2.imwrite(blurred_file_path, blurred)

            #savepath = savedir + fname
            #cv2.imwrite(savepath, frame)
        
        cv2.imshow('Frame',frame)
        cv2.imshow('bgrmv',blurred)
        #if PEDESTRIAN_DEBUG:
            #cv2.imshow('pedestrian',ped_debug)
        
        ##write to output video
        out.write(frame)
        
        fps.update()
        cv2.waitKey(1) 
        
        if 0xFF == ord('q'):
            break
    else:
        break 

        
print("Counted " , COUNTER_p , " pedestrians")
if debug:
    my_logger.info("Counted Pedestrians %d", COUNTER_p)

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

np.save(background, avg)
np.save(background_next, avg)

cap.release()
cv2.destroyAllWindows()
