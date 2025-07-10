import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)  # Change to 1 if webcam not detected

while True:
    ret, frame = cap.read()
    if not ret: break
    
    try:
        # Analyze frame for ALL 7 emotions
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Get dominant emotion and all emotion scores
        dominant_emotion = results[0]['dominant_emotion']
        emotion_scores = results[0]['emotion']  # Dictionary of all 7 emotions
        
        # Display dominant emotion (like before)
        cv2.putText(frame, f"Dominant: {dominant_emotion}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display all 7 emotion scores (below dominant)
        y_offset = 80
        for emotion, score in emotion_scores.items():
            cv2.putText(frame, f"{emotion}: {score:.1f}%", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
            
    except Exception as e:
        print("Error:", e)
    
    cv2.imshow('Press Q to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()