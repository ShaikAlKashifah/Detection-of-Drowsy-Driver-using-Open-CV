import tkinter as tk
from tkinter import messagebox, PhotoImage
import pickle
import os
from playsound import playsound
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords

def validate_login():
    username = username_entry.get()
    password = password_entry.get()

    if os.path.exists('users.pkl'):
        with open('users.pkl', 'rb') as f:
            users = pickle.load(f)
    else:
        users = {}

    if username in users and users[username] == password:
        messagebox.showinfo("Login Successful", f"Welcome, {username}!")
        root.destroy()  # Close the login window
        run_detection_system()  # Run the detection system
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

def signup():
    def save_user():
        username = new_username_entry.get()
        password = new_password_entry.get()
        confirm_password = confirm_password_entry.get()

        if password != confirm_password:
            messagebox.showerror("Error", "Passwords do not match!")
            return

        if os.path.exists('users.pkl'):
            with open('users.pkl', 'rb') as f:
                users = pickle.load(f)
        else:
            users = {}

        if username in users:
            messagebox.showerror("Error", "Username already exists!")
        else:
            users[username] = password
            with open('users.pkl', 'wb') as f:
                pickle.dump(users, f)
            messagebox.showinfo("Success", "User registered successfully!")
            signup_window.destroy()

    signup_window = tk.Toplevel(root)
    signup_window.title("Sign Up")
    signup_window.geometry("800x600")
    signup_window.configure(bg='lightblue')

    # Add college name and department name above the logo
    tk.Label(signup_window, text="SJC Institute of Technology", bg='lightblue', font=('Helvetica', 18, 'bold')).pack(pady=5)
    tk.Label(signup_window, text="Department of Computer Science and Engineering", bg='lightblue', font=('Helvetica', 16)).pack(pady=5)

    # Add logo to signup window
    logo_label = tk.Label(signup_window, image=logo, bg='lightblue')
    logo_label.pack(pady=10)

    # Add photos at the top corners
    left_photo_label = tk.Label(signup_window, image=left_photo, bg='lightblue')
    left_photo_label.place(x=10, y=10)
    right_photo_label = tk.Label(signup_window, image=right_photo, bg='lightblue')
    right_photo_label.place(x=1100, y=10)  # Adjusted x-coordinate to fit within 800x600 window

    tk.Label(signup_window, text="New Username", bg='lightblue', font=('Helvetica', 16)).pack(pady=5)
    new_username_entry = tk.Entry(signup_window, font=('Helvetica', 14))
    new_username_entry.pack(pady=5)

    tk.Label(signup_window, text="New Password", bg='lightblue', font=('Helvetica', 16)).pack(pady=5)
    new_password_entry = tk.Entry(signup_window, show="*", font=('Helvetica', 14))
    new_password_entry.pack(pady=5)

    tk.Label(signup_window, text="Confirm Password", bg='lightblue', font=('Helvetica', 16)).pack(pady=5)
    confirm_password_entry = tk.Entry(signup_window, show="*", font=('Helvetica', 14))
    confirm_password_entry.pack(pady=5)

    save_button = tk.Button(signup_window, text="Sign Up", command=save_user, font=('Helvetica', 16))
    save_button.pack(pady=10)

    close_button = tk.Button(signup_window, text="Close", command=signup_window.destroy, font=('Helvetica', 16))
    close_button.pack(pady=10)

    # Add your name and guide's name below the login text area
    tk.Label(root, text="By: Shaik Al Kashifah", bg='lightblue', font=('Helvetica', 14)).pack(pady=5)
    tk.Label(root, text="Under the guidance of: Dr. Seshaiah Merikapudi", bg='lightblue', font=('Helvetica', 14)).pack(pady=5)

def run_detection_system():
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('C:/Users/basha/Downloads/Driver-Drowsiness-Detection-master/dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

    print("[INFO] initializing camera...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    image_points = np.array([
        (359, 391),     # Nose tip 34
        (399, 561),     # Chin 9
        (337, 297),     # Left eye left corner 37
        (513, 301),     # Right eye right corner 46
        (345, 465),     # Left Mouth corner 49
        (453, 469)      # Right mouth corner 55
    ], dtype="double")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    EYE_AR_THRESH = 0.25
    MOUTH_AR_THRESH = 0.79
    EYE_AR_CONSEC_FRAMES = 3
    COUNTER = 0

    (mStart, mEnd) = (49, 68)

    while True:
        frame = vs.read()
        if frame is None:
            print("[ERROR] Could not capture frame.")
            break

        frame = imutils.resize(frame, width=1024)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape

        rects = detector(gray, 0)

        if len(rects) > 0:
            text = "{} face(s) found".format(len(rects))
            cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for rect in rects:
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "Eyes Closed!", (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    playsound('C:/Users/basha/Downloads/sound.wav')  # Beep sound when eyes are closed
            else:
                COUNTER = 0

            mouth = shape[mStart:mEnd]
            mouthMAR = mouth_aspect_ratio(mouth)
            mar = mouthMAR
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if mar > MOUTH_AR_THRESH:
                cv2.putText(frame, "Yawning!", (800, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                playsound('C:/Users/basha/Downloads/sound.wav')  # Beep sound when yawning

            for (i, (x, y)) in enumerate(shape):
                if i == 33:
                    image_points[0] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 8:
                    image_points[1] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 36:
                    image_points[2] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 45:
                    image_points[3] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 48:
                    image_points[4] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 54:
                    image_points[5] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.stop()
    cv2.destroyAllWindows()

root = tk.Tk()
root.title("Login Page")
root.geometry("800x600")
root.configure(bg='lightblue')

# Load logo image
logo = PhotoImage(file="C:/Users/basha/Desktop/1.png")  # Update the path to your logo image
logo = logo.subsample(3, 3)  # Resize the logo to be smaller

# Load additional images for the corners
left_photo = PhotoImage(file="C:/Users/basha/Desktop/bs.png")  # Update with your left corner image path
right_photo = PhotoImage(file="C:/Users/basha/Desktop/nns.png")  # Update with your right corner image path

# Resize the corner images to passport size
left_photo = left_photo.subsample(4, 4)
right_photo = right_photo.subsample(4, 4)

# Add college name and department name above the logo
tk.Label(root, text="SJC Institute of Technology", bg='lightblue', font=('Helvetica', 18, 'bold')).pack(pady=5)
tk.Label(root, text="Department of Computer Science and Engineering", bg='lightblue', font=('Helvetica', 16)).pack(pady=5)

# Add logo to login window
logo_label = tk.Label(root, image=logo, bg='lightblue')
logo_label.pack(pady=10)

# Add photos at the top corners
left_photo_label = tk.Label(root, image=left_photo, bg='lightblue')
left_photo_label.place(x=10, y=10)
right_photo_label = tk.Label(root, image=right_photo, bg='lightblue')
right_photo_label.place(x=1100, y=10)  # Adjusted x-coordinate to fit within 800x600 window

tk.Label(root, text="Username", bg='lightblue', font=('Helvetica', 16)).pack(pady=5)
username_entry = tk.Entry(root, font=('Helvetica', 14))
username_entry.pack(pady=5)

tk.Label(root, text="Password", bg='lightblue', font=('Helvetica', 16)).pack(pady=5)
password_entry = tk.Entry(root, show="*", font=('Helvetica', 14))
password_entry.pack(pady=5)

login_button = tk.Button(root, text="Login", command=validate_login, font=('Helvetica', 16))
login_button.pack(pady=10)

signup_button = tk.Button(root, text="Sign Up", command=signup, font=('Helvetica', 16))
signup_button.pack(pady=10)

# Add your name and guide's name below the login text area
tk.Label(root, text="By: Shaik Al Kashifah", bg='lightblue', font=('Helvetica', 14)).pack(pady=5)
tk.Label(root, text="Under the guidance of: Dr. Seshaiah Merikapudi", bg='lightblue', font=('Helvetica', 14)).pack(pady=5)

root.mainloop()
