""" Train ffn model.
"""
from mist import train_ffn_binned
import time

if __name__=="__main__":
    start_time = time.time()
    train_ffn_binned.run_training()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
