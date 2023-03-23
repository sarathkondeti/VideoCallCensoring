import face_recognition
import cv2
import numpy as np
import time
unknown = "Unknown"

video_capture = cv2.VideoCapture(0)
bodyCascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")

#  =======================================================
# add authorized personnel here
barack_image = face_recognition.load_image_file("Barack.jpg")
barack_face_encoding = face_recognition.face_encodings(barack_image)[0]

# Optional for better identification - add unwanted faces here and put their name as unknown in known_face_names.
donald_image = face_recognition.load_image_file("Donald.jpg")
donald_face_encoding = face_recognition.face_encodings(donald_image)[0]

# =========================================================

known_face_encodings = [
    sarath_face_encoding,
    keshav_face_encoding
]
known_face_names = [
    "Barack", # can be anything for wanted faces
    unknown   # this face should be anonmyzed.
]
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
last_unknown_detected    = time.time()
last_authorized_detected = time.time()

# this function outputs blurred img of input, use factor to agjust blur intensity.
def blur_img(img, factor = 20):
    kW = int(img.shape[1] / factor)
    kH = int(img.shape[0] / factor)
    #ensure the shape of the kernel is odd
    if kW % 2 == 0: kW = kW - 1
    if kH % 2 == 0: kH = kH - 1
    blurred_img = cv2.GaussianBlur(img, (kW, kH), 0)
    return blurred_img

def isPresent(s,arr):
    for x in arr:
        if x==s:
            return True
    return False

def backgroundMovement(frame):
    global last_unknown_detected
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = bodyCascade.detectMultiScale(
        gray,
        scaleFactor = 1.4,
        minNeighbors = 5,
        minSize = (50, 100), # Min size for valid detection, changes according to video size or body size in the video.
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    if len(bodies)!=0:
        last_unknown_detected    = time.time()
        return True
    return False

def extract_indexes(length, step_size):
    indexes = []
    cycles = int(length / step_size)
    for i in range(cycles):
        begin = i * step_size; end = i * step_size+step_size
        #print(i, ". [",begin,", ",end,")")
        index = []
        index.append(begin)
        index.append(end)
        indexes.append(index)
        if begin >= length: break
        if end > length: end = length
    if end < length:
        #print(i+1,". [", end,", ",length,")")
        index = []
        index.append(end)
        index.append(length)
        indexes.append(index)
    return indexes

process_this_frame = True
# this loop processes frames from webcam.
while True:
    ret, frame = video_capture.read()

    # process alternate frames
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = unknown # default name
            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
            if name is unknown:
                last_unknown_detected = time.time()
            else:
                last_authorized_detected=time.time()

    process_this_frame = not process_this_frame
#####################################################################################################################################################################
    #testing purpose
    # last_authorized_detected=time.time()
    #last_unknown_detected=time.time()

    movement_detected = backgroundMovement(frame)
    if time.time() - last_authorized_detected > 3:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (1280, 720), (0,0,0), cv2.FILLED)
        alpha = 0.7  # Transparency factor.
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.putText(frame, "VIDEO FEED OFF : No authorised personnel detected", (100, 100), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
    elif movement_detected or time.time() - last_unknown_detected < 3 or isPresent(unknown,face_names) :
        blurred_frame = blur_img(frame, factor = 15)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top    *= 4
            right  *= 4
            bottom *= 4
            left   *= 4
            detected_face = frame[int(top):int(bottom), int(left):int(right)]

            if name == unknown:
                pixelated_face = detected_face.copy()
                width = pixelated_face.shape[0]
                height = pixelated_face.shape[1]
                step_size = 80
                for wi in extract_indexes(width, step_size):
                    for hi in extract_indexes(height, step_size):
                        detected_face_area = detected_face[wi[0]:wi[1], hi[0]:hi[1]]

                        if detected_face_area.shape[0] > 0 and detected_face_area.shape[1] > 0:
                            detected_face_area = blur_img(detected_face_area, factor = 0.5)
                            pixelated_face[wi[0]:wi[1], hi[0]:hi[1]] = detected_face_area

                blurred_frame[top:bottom, left:right] = pixelated_face
            else:
                blurred_frame[top:bottom, left:right] = detected_face
                # Draw a box around the face
                #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                #cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        frame = blurred_frame
#######################################################################################################################################################################
    # Display the resulting image
    cv2.imshow('Video', frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
