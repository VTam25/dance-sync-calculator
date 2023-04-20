import cv2
import mediapipe as mp
import time
import numpy as np

#https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md

class poseDetector():
    def __init__(self, mode = False, complex = 1, smooth = True, segmentation = False, smooth_segment = True, detectionCon = 0.5, trackCon = 0.5):
        #   def __init__(self,
        #        static_image_mode=False,
        #        model_complexity=1,
        #        smooth_landmarks=True,
        #        enable_segmentation=False,
        #        smooth_segmentation=True,
        #        min_detection_confidence=0.5,
        #        min_tracking_confidence=0.5):

        self.mode = mode
        self.complex = complex
        self.smooth = smooth
        self.smooth_segment = smooth_segment
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complex, self.smooth, self.smooth_segment, self.detectionCon, self.trackCon)
    
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to make capatible w/ pose
        self.results = self.pose.process(imgRGB)

        #draw landmarks and connections
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return img
    
    def findPosition(self, img, draw=True):
        lmList = []
        #index of lm corresponds to documentation picture
        if self.results.pose_landmarks:      
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                print(id, lm)
                #get actual pixel value of landmarks
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy]) 

                #put blue dot over
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)   

        return lmList   



def main():
    cap = cv2.VideoCapture('orig_videos/Fearless.mp4')
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        #print(lmList)

        #show fps on video
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()