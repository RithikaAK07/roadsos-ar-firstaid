import argparse
import os
import subprocess
import sys


MODELS = {
    "cardiac_arrest": {
        "yaml":    "data/cardiac.yaml",
        "output":  "models/chest_detector.pt",
        "classes": ["chest", "person_lying"],
    },
    "severe_bleeding": {
        "yaml":    "data/bleeding.yaml",
        "output":  "models/wound_detector.pt",
        "classes": ["wound_minor", "wound_moderate", "wound_severe", "tourniquet_point"],
    },
    "burns": {
        "yaml":    "data/burns.yaml",
        "output":  "models/burn_classifier.pt",
        "classes": ["burn_1st", "burn_2nd", "burn_3rd"],
    },
    "first_aid_kit": {
        "yaml":    "data/kit.yaml",
        "output":  "models/kit_detector.pt",
        "classes": ["first_aid_kit", "aed", "fire_extinguisher"],
    },
}

YAML_TEMPLATE = (
    "path: datasets/{name}\n"
    "train: images/train\n"
    "val:   images/val\n\n"
    "nc: {nc}\n"
    "names: {names}\n"
)


def generate_yaml(injury_name, config):
    os.makedirs("data", exist_ok=True)
    content = YAML_TEMPLATE.format(
        name=injury_name,
        nc=len(config["classes"]),
        names=config["classes"]
    )
    with open(config["yaml"], "w") as f:
        f.write(content)
    print(f"[Train] Generated {config['yaml']}")


def train_model(injury_name, config, epochs=50):
    print(f"\n{'='*55}")
    print(f"[Train] {injury_name}")
    print(f"{'='*55}")

    generate_yaml(injury_name, config)
    os.makedirs("models", exist_ok=True)

    cmd = [
        sys.executable, "yolov5/train.py",
        "--img", "640",
        "--batch", "16",
        "--epochs", str(epochs),
        "--data", config["yaml"],
        "--weights", "yolov5s.pt",
        "--name", f"roadsos_{injury_name}",
        "--project", "runs/train",
        "--patience", "15",
        "--cache",
    ]

    print(f"[Train] Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    best = f"runs/train/roadsos_{injury_name}/weights/best.pt"
    if os.path.exists(best):
        import shutil
        shutil.copy(best, config["output"])
        print(f"[Train] Saved model to {config['output']}")
    else:
        print(f"[Train] Training finished but best.pt not found at {best}")


def setup_yolov5():
    if not os.path.exists("yolov5"):
        print("[Setup] Cloning YOLOv5...")
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "yolov5/requirements.txt"], check=True)
    print("[Setup] Done.")


def main():
    parser = argparse.ArgumentParser(description="RoadSoS Model Trainer")
    parser.add_argument("--injury", choices=list(MODELS.keys()))
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--setup",  action="store_true")
    args = parser.parse_args()

    if args.setup:
        setup_yolov5()
        return

    if args.all:
        for name, config in MODELS.items():
            train_model(name, config, args.epochs)
    elif args.injury:
        train_model(args.injury, MODELS[args.injury], args.epochs)
    else:
        parser.print_help()
        print("\nAvailable:")
        for name, cfg in MODELS.items():
            print(f"  {name:20s} -> {cfg['output']}")


if __name__ == "__main__":
    main()
