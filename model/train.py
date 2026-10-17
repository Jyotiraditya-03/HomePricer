"""Run this once to train and save the model artifacts."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.pipeline import train

if __name__ == "__main__":
    csv = os.path.join(os.path.dirname(__file__), "../data/housing.csv")
    metrics = train(csv, n_trials=60)
    print("\n✅ Training complete!")
    print(f"   R²   = {metrics['r2']}")
    print(f"   CV R²= {metrics['cv_r2']}")
    print(f"   MAE  = ₹{metrics['mae']:,.0f}")
    print(f"   RMSE = ₹{metrics['rmse']:,.0f}")
