from ultralytics import YOLO

def resume_training():
    # 1. Load the 'last.pt' weights from your interrupted run
    # Update this path if your project/name folders were different
    model_path = "C:\\Users\\SKADWG95\\Downloads\\Blender Stuff\\runs\\detect\\aic_detector\\aic_run1\\weights\\last.pt"
    
    print(f"Loading interrupted model from {model_path}...")
    model = YOLO(model_path)

    # 2. Resume training
    # Setting workers=0 bypasses the Windows multiprocessing DLL bug
    print("Resuming YOLO11 training from Epoch 90...")
    results = model.train(resume=True, workers=0)
    
    print(f"Training fully complete! Best weights saved to: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    resume_training()