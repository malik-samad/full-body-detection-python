import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture(0)
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

# face Draw specs
face_line_spec = mp_draw.DrawingSpec((50,255,0),1,1)
face_point_spec = mp_draw.DrawingSpec((200,255,50),1,1)

# hands Draw specs
hand_line_spec = mp_draw.DrawingSpec((250,0,100),2,2)
hand_point_spec = mp_draw.DrawingSpec((250,200,100),3,3)

# pose Draw specs
pose_line_spec = mp_draw.DrawingSpec((0,0,255),2,2)
pose_point_spec = mp_draw.DrawingSpec((200,100,255),2,2)


ptime = 0
with mp_holistic.Holistic() as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_img)

        ctime = time.time()
        fps = int(1/(ctime-ptime))
        ptime=ctime

        cv2.putText(frame, f"FPS: {fps}",(30,30), cv2.FONT_ITALIC,0.8,(50,255,50),1)
        # face landmarks drawing
        mp_draw.draw_landmarks(frame,results.face_landmarks,mp_holistic.FACE_CONNECTIONS , face_point_spec,face_line_spec)

        # body landmarks drawing
        mp_draw.draw_landmarks(frame,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS , pose_point_spec,pose_line_spec)

        # left-hand landmarks drawing
        mp_draw.draw_landmarks(frame,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS , hand_point_spec,hand_line_spec)

        # right-hand landmarks drawing
        mp_draw.draw_landmarks(frame,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS , hand_point_spec,hand_line_spec)

        cv2.imshow('Output window',frame)

        cv2.waitKey(2)
