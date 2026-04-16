import cv2
from ultralytics import YOLO
import argparse

def run_realtime_inference(model_path, video_source):
    # 1. Load your trained model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # 2. Open the video source
    # This can be a file path ("test_video.mp4") or a camera index (0, 1, etc.)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    print("Starting inference... Press 'q' to quit.")

    # 3. Process the video stream frame-by-frame
    while True:
        ret, frame = cap.read()
        
        # Break the loop if the video ends
        if not ret:
            print("Video stream ended.")
            break

        # 4. Run YOLO11 inference on the frame
        # conf: Minimum confidence threshold (e.g., 0.5 means 50% confident)
        # iou: Intersection Over Union threshold for Non-Maximum Suppression (removes duplicate boxes)
        # verbose: False keeps your terminal clean from per-frame printouts
        results = model.predict(source=frame, conf=0.6, iou=0.5, verbose=False)

        # 5. Visualize the results
        # results[0].plot() automatically draws the bounding boxes and labels onto the frame
        annotated_frame = results[0].plot()

        # 6. Display the frame in a window
        cv2.imshow("YOLO11 Live Inference - NIC & SC Detection", annotated_frame)

        # 7. Listen for the 'q' key to stop the script
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Inference stopped by user.")
            break

    # 8. Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Run YOLO11 real-time inference on a video.")
    parser.add_argument("--weights", type=str, required=True, help="Path to your best.pt file")
    parser.add_argument("--source", type=str, required=True, help="Path to video file or camera index (e.g., '0')")
    
    args = parser.parse_args()
    
    # Check if the source is meant to be a webcam index (integer)
    source = int(args.source) if args.source.isdigit() else args.source

    run_realtime_inference(args.weights, source)