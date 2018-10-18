import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
poop = cv2.imread('pooper/poop.png', -1)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
labels = {"person_name":1}

with open("labels.pickle",'rb') as f:
    
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image
 
    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src
 

while(True):
    
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:

               

        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf>=45:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color= (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y),font,1,color,stroke,cv2.LINE_AA)
            if id_==1:
                #POOPERSONIA
                if h > 0 and w > 0:
                    poop_symin = int(y +1 * h / 4)
                    poop_symax = int(y + 5.5 * h / 6)
                    sh_poop = poop_symax - poop_symin 
                    face_poop_roi_color = frame[poop_symin:poop_symax,x:x+w]
                    poopidy = cv2.resize(poop, (w, sh_poop),interpolation=cv2.INTER_CUBIC)
                    transparentOverlay(face_poop_roi_color,poopidy) 
                

        img_item = "7.png"
        cv2.imwrite(img_item, roi_color)

        color = (255, 0, 0) 
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame,(x,y),(end_cord_x, end_cord_y), color, stroke)
    	
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
