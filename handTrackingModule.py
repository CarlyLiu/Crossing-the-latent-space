import cv2
import mediapipe as mp
import math
import try1GUIvisualizer

class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    pinch_detected = False
    def positionFinder(self,image, handNo=0, draw=True):
        lmlist = []
        thumb_tip = None
        index_tip = None
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])
                # if draw and (id == 4 or id == 8):
                #     cv2.circle(image,(cx,cy), 15 , (255,0,255), cv2.FILLED)
                # cv2.putText(image, f'({cx}, {cy})', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                #             cv2.LINE_AA)

                # 检查食指指尖和拇指指尖的坐标
                if id == 4:  # 拇指指尖
                    thumb_tip = (cx, cy)
                    if draw:
                        cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                elif id == 8:  # 食指指尖
                    index_tip = (cx, cy)
                    if draw:
                        cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                        cv2.putText(image, f'({cx}, {cy})', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                                    cv2.LINE_AA)

                if thumb_tip is not None and index_tip is not None:
                    distance = math.sqrt((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2)
                    if distance <= 20 :
                        print(f"pinch detected")
        return lmlist

def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()

    while True:
        success,image = cap.read()
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)
        if len(lmList) != 0:
            print(lmList[0])

        cv2.imshow("Video",image)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()