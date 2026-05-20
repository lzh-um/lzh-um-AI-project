import cv2
import os
import glob
from pathlib import Path
from ultralytics import RTDETR

def run_inference(weights_path, source_path, project_name='uav_inference', name='exp', save_video=True, save_mot=True):
    """
    Run inference using RT-DETR model and save results in video and MOT format.
    """
    # Load the model
    model = RTDETR(weights_path)
    
    # Run inference
    results = model.predict(
        source=source_path,
        project=project_name,
        name=name,
        save=True,          # Save annotated images
        save_txt=True,      # Save detection labels
        conf=0.25,
        exist_ok=True
    )
    
    output_dir = os.path.join(project_name, name)
    
    # 1. Save results in MOT format
    if save_mot:
        mot_file = os.path.join(output_dir, f"{name}_mot.txt")
        with open(mot_file, 'w') as f:
            for i, res in enumerate(results):
                # frame is 1-indexed in MOT format
                frame_idx = i + 1
                boxes = res.boxes
                for box in boxes:
                    # box.xywh is [x_center, y_center, w, h]
                    # We need [left, top, w, h]
                    # box.xyxy is [left, top, right, bottom]
                    xyxy = box.xyxy[0].cpu().numpy()
                    left, top, right, bottom = xyxy
                    width = right - left
                    height = bottom - top
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                    # Since it's detection, id is set to -1
                    line = f"{frame_idx},-1,{left:.2f},{top:.2f},{width:.2f},{height:.2f},{conf:.4f},-1,-1,-1\n"
                    f.write(line)
        print(f"MOT format results saved to: {mot_file}")

    # 2. Generate video
    if save_video:
        # Get list of saved images
        img_files = sorted(glob.glob(os.path.join(output_dir, "*.jpg")))
        if not img_files:
            print("No images found to create video.")
            return
            
        # Read the first image to get dimensions
        first_img = cv2.imread(img_files[0])
        h, w, _ = first_img.shape
        
        video_path = os.path.join(output_dir, f"{name}_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
        
        print(f"Generating video: {video_path}")
        for img_path in img_files:
            img = cv2.imread(img_path)
            out.write(img)
        out.release()
        print(f"Video saved to: {video_path}")

    print(f"Inference and post-processing completed. Results in: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    weights = "/mnt/gemlab_data_3/User_database/xiejiefeng/PT/1600/runs/train/uav-benchmark-m-rtdetr-l3/weights/best.pt"
    source = "/mnt/gemlab_data_3/User_database/xiejiefeng/PT/1600/UAV-benchmark-M/M0101"
    
    if os.path.exists(weights) and os.path.exists(source):
        run_inference(weights, source, name='exp_v2')
    else:
        print("Check if weights and source paths are correct.")
