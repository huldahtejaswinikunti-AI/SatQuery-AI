import argparse
def train_classifier(data_dir: str = "data/processed/bigearthnet_subset", epochs: int = 5):
    print(f"Training ResNet-18 head on {data_dir} for {epochs} epochs.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/bigearthnet_subset")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    train_classifier(args.data_dir, args.epochs)
