import os
import cv2
import torch
import torch.nn.functional as F
import time
import argparse
import os
from modules.SignLanguageProcessor import RealtimeSignLanguageProcessor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time Sign Language Recognition')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model file')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS (default: 30)')
    parser.add_argument('--sequence_length', type=int, default=3, 
                        help='Number of frames to process for prediction (default: 3)')
    
    parser.add_argument('--width', type=int, default=640, help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Camera height (default: 480)')
    args = parser.parse_args()
    
    # Initialize the processor
    processor = RealtimeSignLanguageProcessor(
        model_path=args.model,
        sequence_length=args.sequence_length
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print(f"Starting real-time sign language recognition...")
    if args.model:
        print(f"Model loaded from: {args.model}")
    else:
        print("No model loaded. Running in visualization-only mode.")
    
    # Calculate target frame interval
    target_interval = 1.0 / args.fps
    
    try:
        while True:
            start_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame from camera.")
                break
            
            # Flip frame horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Process the frame
            processed_frame = processor.process_frame(frame)
            
            # Display FPS
            processing_time = time.time() - start_time
            fps = 1.0 / max(processing_time, 0.001)  # Avoid division by zero
            cv2.putText(
                processed_frame, 
                f"FPS: {fps:.1f}", 
                (processed_frame.shape[1] - 120, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            # Display instructions
            cv2.putText(
                processed_frame,
                "Press 'q' to quit",
                (10, processed_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Show the frame
            cv2.imshow('Real-time Sign Language Recognition', processed_frame)
            
            # Check for key press
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
            # Control frame rate
            elapsed = time.time() - start_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
    
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")



if __name__ == "__main__":
    main()