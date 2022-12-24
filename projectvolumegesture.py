import cv2
import mediapipe as mep
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume 
import numpy as num

capture=cv2.VideoCapture(0)

Hand=mep.solutions.hands
Handes=Hand.Hands()
Draw=mep.solutions.drawing_utils

speaker=AudioUtilities.GetSpeakers()
process=speaker.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volumevalue=cast(process,POINTER(IAudioEndpointVolume))

volmini,volmaxi=volumevalue.GetVolumeRange()[:2]

while True:
    aim,image=capture.read()
    RGB=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    point=Handes.process(RGB)
    
    lispoint=[]
    if point.multi_hand_landmarks:
        for handlandmark in point.multi_hand_landmarks:
            for id,lm in enumerate(handlandmark.landmark):
                h,w,_ = image.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                lispoint.append([id,cx,cy])
            Draw.draw_landmarks(image,handlandmark,Hand.HAND_CONNECTIONS)
    
    if lispoint != []:

        x1,y1 = lispoint[4][1],lispoint[4][2]
        x2,y2 = lispoint[8][1],lispoint[8][2] 
        cv2.circle(image,(x1,y1),13,(0,0,255),cv2.FILLED) 
        cv2.circle(image,(x2,y2),13,(0,0,255),cv2.FILLED) 
        cv2.line(image,(x1,y1),(x2,y2),(255,255,255),3) 
 
        length = ((x2-x1)**2+(y2-y1)**2)**0.5
        vol = num.interp(length,[30,200],[volmini,volmaxi]) 
        volbar=num.interp(length,[30,200],[400,150])
        volper=num.interp(length,[30,200 ],[0,100])
        
        
        print(vol,int(length))
        volumevalue.SetMasterVolumeLevel(vol, None)
        
 
        cv2.rectangle(image,(50,150),(85,400),(0,0,0),4) 
        cv2.rectangle(image,(50,int(volbar)),(85,400),(255,255,255),cv2.FILLED)
        cv2.putText(image,f"{int(volper)}%",(10,40),cv2.FONT_ITALIC,1,(0,0,0),3)

    cv2.imshow('Image',image)
    if cv2.waitKey(1) & 0xff==ord(' '):
        break
        
capture.release()     
cv2.destroyAllWindows()

