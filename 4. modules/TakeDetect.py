'''
if there exists '** (Frame:1825): WARNING **: Error retrieving accessibility bus address: org.freedesktop.DBus.Error.ServiceUnknown:
The name org.a11y.Bus was not provided by any .service files'

$ sudo apt-get install at-spi2-core

Install picamera
    $ source activate cv
    (cv)-> $ python -m pip install -U --force-reinstall setuptools picamera

Install dlib
    (cv)-> $ python -m pip install -U --force-reinstall dlib
    (cv)-> $ python -m pip install -U --force-reinstall imutils
    
Install gpio
    (cv)-> $ python -m pip install -U --force-reinstall rpi.gpio

Function:

'''
import time
import datetime

DI0Pin = 32   # GPIO12
DI0Value = 0

xDim = 1024
yDim = 768
frameRate = 32

smpPeriod = 0.1     #sample period in sec
waitPeriod = 1
TsEnd = False       #collector finished
maxTDtime = 30      #total time for taking picture and detecting target 

#--------------------------------------------------------------------------------------------
# Enable GPIO
#--------------------------Take------------------------------------------------------------------            
import RPi.GPIO as GPIO  #not supported by windows
def initHW(pin=DI0Pin):
    # Use physical pin numbers
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

#--------------------------------------------------------------------------------------------
# Enable a stable time to triger data collection
#--------------------------------------------------------------------------------------------            
import threading
class TM(object): # periodily read GPIO data 
    
    def __init__(self, period):
        self.pd = period
    
    def run(self):
        global TsEnd
        global DI0Value 

        while TsEnd == False:
            DI0Value = GPIO.input(DI0Pin) # access the DI value
            time.sleep(self.pd)
            
#--------------------------------------------------------------------------------------------
# take picture from pi camera
#--------------------------------------------------------------------------------------------
from picamera.array import PiRGBArray
from picamera import PiCamera
def takeShot(): 

    # allow the camera to warmup
    time.sleep(0.1)
     
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp 
        # and occupied/unoccupied text
        image = frame.array
     
        # show the frame
        cv2.imshow("Taking Photo..", image)
        key = cv2.waitKey(1) & 0xFF
     
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
     
        # if the `q` key was pressed, break from the loop
        if key == ord('q') or DI0Value != 0:
            break
    
    cv2.destroyAllWindows()    
    return(key, image)

# preprocess images 
#def preProcess(img):
    

#--------------------------------------------------------------------------------------------
# detect face from image
#--------------------------------------------------------------------------------------------
import dlib
import cv2
import imutils
def detectFace(img):    
    # resize
    image = imutils.resize(img, width=1280)

    # call face detector
    detector = dlib.get_frontal_face_detector()

    # detect face
    face_rects = detector(image, 0)

    # extrract results
    for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()

      # tag face
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

    cv2.imshow("Face Detection", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return(image)
    
#--------------------------------------------------------
if __name__=='__main__':
    
    print('1. Initial GPIO with timer')
    initHW() # initial input
        # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (xDim, yDim)
    camera.framerate = frameRate
    rawCapture = PiRGBArray(camera, size=(xDim, yDim))
    
    tm = TM(smpPeriod)  # new timer object
    tmRcv = threading.Thread(target = tm.run,  args=()) # new timer trhread

    tmRcv.start()  # start the triger timer
    
    TsEnd = False
    tStart = datetime.datetime.now() # record start time
    while TsEnd == False:      #wait for finish  
        print('2. Take picture from camera: q to exit ')
        key, img = takeShot()  #

        detectFace(img)        # try to detect face
        
        
        time.sleep(waitPeriod) # reduce cpu loading by sleep
        TDtime = datetime.datetime.now() - tStart  # calculate the time difference
        if TDtime.total_seconds() > maxTDtime:  # Exit when the total time is expired
            TsEnd  = True
    
    print('3. Finished!')  