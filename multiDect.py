import cv2


def detect_faces(frame, cascade_path):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the trained cascade classifier
    cascade = cv2.CascadeClassifier(cascade_path)

    # Detect faces in the frame
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Return the coordinates of the detected faces
    return faces


def draw_faces(frame, faces, names):
    # Draw rectangles around the detected faces and display their names
    for (x, y, w, h), name in zip(faces, names):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with faces
    cv2.imshow("Real-time Face Detection", frame)


def main():
    # Specify the paths to the cascades
    cascade_paths = [
        "path_to_cascade1.xml",
        "path_to_cascade2.xml",
        "path_to_cascade3.xml"
    ]

    # Specify the names corresponding to the cascades
    names = [
        "Person 1",
        "Person 2",
        "Person 3"
    ]

    # Load the trained cascades
    cascades = [cv2.CascadeClassifier(cascade_path) for cascade_path in cascade_paths]

    # Initialize the video capture
    video_capture = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video capture
        ret, frame = video_capture.read()

        if not ret:
            break

        # Detect faces using each cascade
        all_faces = []
        for cascade in cascades:
            faces = detect_faces(frame, cascade)
            all_faces.extend(faces)

        # Draw faces on the frame
        draw_faces(frame, all_faces, names * len(all_faces))

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
