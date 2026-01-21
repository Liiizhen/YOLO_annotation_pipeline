import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

def main():
    # --- Part 1: Query and List All Connected Devices ---
    print("--- Scanning for RealSense Devices ---")
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if devices.size() == 0:
        print("Error: No Intel RealSense cameras detected.")
        return

    # Automatically select the first device found
    dev = devices[0]
    target_serial = dev.get_info(rs.camera_info.serial_number)
    
    print(f"--- Device Found ---")
    print(f"Using Serial Number: {target_serial}")

    # --- Part 2: Configure and Start the Pipeline ---
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable the specific device
    config.enable_device(target_serial)
    # Configure Color Stream: 640x480, BGR format, 30 fps
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    print("Starting pipeline...")
    pipeline.start(config)

    # Creating a named window for GUI display
    window_name = "RealSense Capture (Press 's' to Save, 'q' to Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("Camera started.")
    print("Controls:")
    print("  [s] - Save current frame to ../data/")
    print("  [q] - Quit program")

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue

            # Convert image to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Display the image in the GUI window
            cv2.imshow(window_name, color_image)

            # Wait for key press (1ms delay)
            key = cv2.waitKey(1) & 0xFF

            # Press 'q' to quit
            if key == ord('q'):
                break
            
            # Press 's' (or Enter) to save image
            elif key == ord('s') or key == 13:
                # Ensure save directory exists
                save_dir = "../data/"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_dir, f"capture_{timestamp}.png")
                
                cv2.imwrite(filename, color_image)
                print(f"Saved: {filename}")
                
                # Visual feedback: flash the window title or draw on image (optional)
                print("--- Capture Successful ---")

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped. Exiting.")

if __name__ == "__main__":
    main()
