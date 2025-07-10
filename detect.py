import cv2
from deepface import DeepFace
import traceback
import time
import sys

# Constants
EMOTION_COLORS = {
    'happy': (0, 255, 255),    # Yellow
    'sad': (255, 0, 0),        # Blue
    'angry': (0, 0, 255),      # Red
    'surprise': (255, 255, 0), # Cyan
    'fear': (0, 165, 255),     # Orange
    'disgust': (0, 255, 0),    # Green
    'neutral': (255, 255, 255) # White
}
FONT = cv2.FONT_HERSHEY_SIMPLEX
RESOLUTION = (640, 480)  # Optimized for performance

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)  # Try alternate camera index
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    return cap

def analyze_emotion(frame):
    try:
        results = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',  # Faster than default
            silent=True  # Disables unnecessary logs
        )
        return results[0]['emotion'], results[0]['dominant_emotion']
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return None, None

def display_results(frame, emotions, dominant_emotion):
    if emotions and dominant_emotion:
        # Display dominant emotion
        cv2.putText(frame, f"Dominant: {dominant_emotion}", (10, 40), 
                    FONT, 0.8, EMOTION_COLORS.get(dominant_emotion, (255,255,255)), 2)
        
        # Display all emotion scores
        y_offset = 70
        for emotion, score in emotions.items():
            cv2.putText(frame, f"{emotion}: {score:.1f}%", (10, y_offset), 
                        FONT, 0.5, EMOTION_COLORS.get(emotion, (255,255,255)), 1)
            y_offset += 25
    else:
        cv2.putText(frame, "Analyzing...", (10, 40), 
                    FONT, 0.8, (255,255,255), 2)

def main():
    cap = None
    try:
        cap = initialize_camera()
        last_analysis_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture error")
                break
            
            # Throttle analysis to 2 FPS for better performance
            if time.time() - last_analysis_time > 0.5:  # 0.5 second interval
                emotions, dominant_emotion = analyze_emotion(frame)
                last_analysis_time = time.time()
            
            display_results(frame, emotions, dominant_emotion)
            cv2.imshow('Emotion Detection (Press Q to quit)', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Critical error: {str(e)}")
        traceback.print_exc()
    finally:
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

if __name__ == "__main__":
    main()