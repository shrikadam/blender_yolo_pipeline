from ultralytics import YOLO

def train_detector():
    # 1. Load a pre-trained YOLO11 Nano model
    # Starting from a pre-trained model speeds up convergence massively
    model = YOLO("yolo11m.pt") 

    # 2. Start Training
    print("Starting YOLO11 training...")
    results = model.train(
        data="./yolo_dataset/dataset.yaml",      # Path to your dataset config
        epochs=100,               # 100 is a safe start; it will auto-stop if it stops learning
        patience=20,              # Early stopping if no improvement for 20 epochs
        imgsz=1024,               # Your images are 1152x1024. 1024 keeps high resolution for small SC ports
        batch=16,                 # Lower this to 8 or 4 if your GPU runs out of memory (OOM)
        device="0",               # "0" uses your primary GPU. Use "cpu" if you don't have a GPU
        optimizer="auto",         # AdamW or SGD chosen automatically based on dataset characteristics
        project="aic_detector",# Folder name for your training runs
        name="aic_run2_yolo11m",       # Sub-folder for this specific run
        save=True,                 # Ensure the best.pt weights are saved
        workers=0
    )
    
    print(f"Training complete! Best weights saved to: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_detector()