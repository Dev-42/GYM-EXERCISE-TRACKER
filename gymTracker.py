import cv2
import numpy as np
import mediapipe as mp

mp_drawing=mp.solutions.drawing_utils
#Importing the pose estimation model from mediapipe
mp_pose=mp.solutions.pose

#Calculate angles

def calculate_angle(a,b,c):
    a=np.array(a) #First
    b=np.array(b) #Mid
    c=np.array(c) #End

    radians=np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle=360-angle

    return angle

cap = cv2.VideoCapture(0)

#Curl Counter variables
counter=0
stage=None

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #Extraxt Landmarks
        try:
            landmarks=results.pose_landmarks.landmark
            #Left Arm Angle
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            #Calculating left hand angle
            l_angle=calculate_angle(l_shoulder,l_elbow,l_wrist)


            #Visualize left
            cv2.putText(image, str(int(l_angle)), 
                           tuple(np.multiply(l_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

            #Curl counter logic
            if l_angle >160:
                stage="Down"
            if l_angle <30 and stage=="Down":
                stage="Up"
                counter+=1
                print(counter)

        except:
            pass

        #Render curl counter
        #Setup status box
        cv2.rectangle(image,(0,0),(225,73),(245,117,16),-1)

        #Rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

         # Stage data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

len(landmarks)
for lndmrk in mp_pose.PoseLandmark:
    print(lndmrk)

landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility

landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]

landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]



   

