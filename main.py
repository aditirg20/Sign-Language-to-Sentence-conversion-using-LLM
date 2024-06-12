import cv2
import mediapipe as mp
import numpy as np
import pickle
import google.generativeai as genai
from api import Gemini_API_KEY as api
from PIL import Image, ImageTk
import tkinter as tk
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Global variables
capturing = False
captured_signs = []

# Load ASL model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)

# Configure and initialize the Generative AI model
genai.configure(api_key=api)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

# Function to capture and append sign
def capture_sign():
    global capturing
    capturing = True

# Function to stop capturing and display the complete sentence
def stop_capture():
    global capturing, captured_signs
    capturing = False
    if captured_signs:
        complete_word = ''.join(captured_signs)
        print("Recognized Words:", complete_word)

        # Remove stopwords and generate POS tags
        tokens = word_tokenize(complete_word.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

        # Get POS tags
        pos_tags = pos_tag(filtered_tokens)

        # Send recognized words to the Generative AI API
        response = chat.send_message(str(complete_word + "MAKE CONVERSATIONAL SENTENCE USING THESE WORDS "))
        complete_sentence = response.text
        print("Complete Sentence:", complete_sentence)

        # Reset captured signs for the next sentence
        captured_signs = []

# Function to add space after the word
def add_space():
    global captured_signs
    if captured_signs:
        captured_signs.append(' ')
        print("Space added.")

# Tkinter GUI setup
root = tk.Tk()
root.title("ASL and Gemini")

# Canvas for displaying video feed
canvas = tk.Canvas(root, width=1000, height=500)
canvas.pack()

# Label to display the recognized class
class_label = tk.Label(root, text="", font=("Helvetica", 16))
class_label.pack(side=tk.BOTTOM)

# Button to capture sign
capture_button = tk.Button(root, text="Capture Sign", command=capture_sign)
capture_button.pack(side=tk.LEFT)

# Button to stop capturing and display the complete sentence
stop_button = tk.Button(root, text="Stop and Create Sentence", command=stop_capture)
stop_button.pack(side=tk.RIGHT)

# Button to add space after the word
space_button = tk.Button(root, text="Add Space", command=add_space)
space_button.pack(side=tk.RIGHT)

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.resize(image, (500, 500))
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                )

            
            without_garbage = []
            clean = []
            data = results.multi_hand_landmarks[0]

            data = str(data).strip().split('\n')
            garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

            for i in data:
                if i not in garbage:
                    without_garbage.append(i)

            for i in without_garbage:
                i = i.strip()
                clean.append(i[2:])

            for i in range(0, len(clean)):
                clean[i] = float(clean[i])

            Class = svm.predict(np.array(clean).reshape(-1, 63))
            Class = Class[0]
           # print(Class)
            cv2.putText(image, str(Class), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
            if capturing:
                # Append recognized sign to the list
                captured_signs.append(str(Class))

                # Display the recognized class on the label
                class_label.config(text=f"Recognized Class: {Class}")
                print(Class)

                # Reset capturing to False after appending the sign
                capturing = False
        #cv2.putText(image, str(Class), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        # Display the video feed and GUI
        img = Image.fromarray(image)
        img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        root.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.mainloop()
