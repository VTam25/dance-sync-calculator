import cv2
import time
import math
import poseDetector as pd
import numpy as np

def processVideo(cap, outputTitle):
    detector = pd.poseDetector()
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(outputTitle, fourcc, fps, (frame_width, frame_height))
    poses = []
    while cap.isOpened():
        success, img = cap.read()
        if success == True:
            img = detector.findPose(img)
            lmList = detector.findPosition(img)
            poses.append(lmList)

            cv2.imshow("Image", img)
            # write the frame to the output file
            output.write(img) 
            if cv2.waitKey(20) == ord('q'):
                break
        else:
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()

    return poses

def comparePoses(posesOrig, posesSelf):
    smaller = min(len(posesOrig),  len(posesSelf))
    posesOrig = posesOrig[0:smaller, :, :]
    posesSelf = posesSelf[0:smaller, :, :]
    avgUnsync = 0
    unsyncFrames = []
    scores = []
    for f in range(smaller):
        print("Frame:", f)
        pairs = []
        shoulders0 = (posesOrig[f,12,2] - posesOrig[f,11,2])/(posesOrig[f,12,1] - posesOrig[f,11,1])
        shoulders1 = (posesSelf[f,12,2] - posesSelf[f,11,2])/(posesSelf[f,12,1] - posesSelf[f,11,1])
        pairs.append([shoulders0, shoulders1])

        hips0 = (posesOrig[f,23,2] - posesOrig[f,24,2])/(posesOrig[f,23,1] - posesOrig[f,24,1])
        hips1 = (posesSelf[f,23,2] - posesSelf[f,24,2])/(posesSelf[f,23,1] - posesSelf[f,24,1])
        pairs.append([hips0, hips1])

        rForearm0 = (posesOrig[f,22,2] - posesOrig[f,14,2])/(posesOrig[f,22,1] - posesOrig[f,14,1])
        rForearm1 = (posesSelf[f,22,2] - posesSelf[f,14,2])/(posesSelf[f,22,1] - posesSelf[f,14,1]) 
        pairs.append([rForearm0, rForearm1])

        lForearm0 = (posesOrig[f,13,2] - posesOrig[f,15,2])/(posesOrig[f,13,1] - posesOrig[f,15,1])
        lForearm1 = (posesSelf[f,13,2] - posesSelf[f,15,2])/(posesSelf[f,13,1] - posesSelf[f,15,1])  
        pairs.append([lForearm0, lForearm1]) 
        
        rUpperArm0 = (posesOrig[f,14,2] - posesOrig[f,12,2])/(posesOrig[f,14,1] - posesOrig[f,12,1])
        rUpperArm1 = (posesSelf[f,14,2] - posesSelf[f,12,2])/(posesSelf[f,14,1] - posesSelf[f,12,1]) 
        pairs.append([rUpperArm0, rUpperArm1])
        
        lUpperArm0 = (posesOrig[f,13,2] - posesOrig[f,11,2])/(posesOrig[f,13,1] - posesOrig[f,11,1])
        lUpperArm1 = (posesSelf[f,13,2] - posesSelf[f,11,2])/(posesSelf[f,13,1] - posesSelf[f,11,1])
        pairs.append([lUpperArm0, lUpperArm1])

        rThigh0 = (posesOrig[f,26,2] - posesOrig[f,24,2])/(posesOrig[f,26,1] - posesOrig[f,24,1])
        rThigh1 = (posesSelf[f,26,2] - posesSelf[f,24,2])/(posesSelf[f,26,1] - posesSelf[f,24,1])
        pairs.append([rThigh0, rThigh1])

        lThigh0 = (posesOrig[f,25,2] - posesOrig[f,23,2])/(posesOrig[f,25,1] - posesOrig[f,23,1])
        lThigh1 = (posesSelf[f,25,2] - posesSelf[f,23,2])/(posesSelf[f,25,1] - posesSelf[f,23,1])
        pairs.append([lThigh0, lThigh1]) 

        rCalf0 = (posesOrig[f,26,2] - posesOrig[f,28,2])/(posesOrig[f,26,1] - posesOrig[f,28,1])
        rCalf1 = (posesSelf[f,26,2] - posesSelf[f,28,2])/(posesSelf[f,26,1] - posesSelf[f,28,1])
        pairs.append([rCalf0, rCalf1])

        lCalf0 = (posesOrig[f,25,2] - posesOrig[f,27,2])/(posesOrig[f,25,1] - posesOrig[f,27,1])
        lCalf1 = (posesSelf[f,25,2] - posesSelf[f,27,2])/(posesSelf[f,25,1] - posesSelf[f,27,1])
        pairs.append([lCalf0, lCalf1])

        rFoot0 = (posesOrig[f,30,2] - posesOrig[f,32,2])/(posesOrig[f,30,1] - posesOrig[f,32,1])
        rFoot1 = (posesSelf[f,30,2] - posesSelf[f,32,2])/(posesSelf[f,30,1] - posesSelf[f,32,1])
        pairs.append([rFoot0, rFoot1])   

        lFoot0 = (posesOrig[f,29,2] - posesOrig[f,31,2])/(posesOrig[f,29,1] - posesOrig[f,31,1])
        lFoot1 = (posesSelf[f,29,2] - posesSelf[f,31,2])/(posesSelf[f,29,1] - posesSelf[f,31,1])
        pairs.append([lFoot0, lFoot1])

        threshold = 100
        scorePerFrame = 0
        for p in pairs:
            if abs(p[0]) == float('inf') or abs(p[1]) == float('inf') or np.isnan(abs(p[0])) or np.isnan(abs(p[1])):
                break
            dif = abs(abs(p[0]) - abs(p[1]))
            scorePerFrame += dif

        if scorePerFrame > threshold: 
            unsyncFrames.append((f, scorePerFrame))
        
        print("Score Per Frame:", scorePerFrame)
        if np.isnan(scorePerFrame) == False:
            avgUnsync += scorePerFrame
            scores.append(scorePerFrame)
        else:
            print("Caught nan")
    
    avgUnsync = avgUnsync/smaller

    return scores, unsyncFrames, avgUnsync

def showScores(cap, scores, outputTitle):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(outputTitle, fourcc, fps, (frame_width, frame_height))

    count = 0
    while cap.isOpened():
        success, img = cap.read()
        if success == True:
            if scores[count] == 0 or scores[count] < 50:
                cv2.putText(img, "Score:" + str(int(scores[count])), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            elif scores[count] > 50 and scores[count] <= 100:
                cv2.putText(img, "Score:" + str(int(scores[count])), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 165, 255), 3)
            else:
                cv2.putText(img, "Score:" + str(int(scores[count])), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            cv2.imshow("Image", img)
            # write the frame to the output file
            output.write(img) 
            if cv2.waitKey(20) == ord('q'):
                break
            count += 1
        else:
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()




# cap_orig = cv2.VideoCapture('orig_videos/Fearless_orig3.mp4')
# cap_self = cv2.VideoCapture('self_videos/Fearless_Self.mp4')

cap_orig = cv2.VideoCapture('orig_videos/Case_orig2.mp4')
cap_self = cv2.VideoCapture('self_videos/Case_Self.mp4')

# posesOrig = processVideo(cap_orig, "Fearless_Orig_Pose.mp4")
# posesSelf = processVideo(cap_self, "Fearless_Self_Poses.mp4")

posesOrig = processVideo(cap_orig, "Case_Orig_Pose.mp4")
posesSelf = processVideo(cap_self, "Case_Self_Poses.mp4")

scores, unsyncFrames, avgUnsync = comparePoses(np.array(posesOrig), np.array(posesSelf))
print("Unsync Frames:", unsyncFrames)
print("Avg Unsynchronization:", avgUnsync)

# cap_combined = cv2.VideoCapture('Fearless_Combined.mp4')
# showScores(cap_combined, scores, "Fearless_Output.mp4")

cap_combined = cv2.VideoCapture('Case_Combined.mp4')
showScores(cap_combined, scores, "Case_Output.mp4")
