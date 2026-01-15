from model import Model
import torch

def main():
    model = Model()
    model.eval()

    sentence = "When the sun sets, the sky turns orange."

    with torch.no_grad():
        translation = model([sentence])[0]

    print("EN:", sentence)
    print("ES:", translation)

if __name__ == "__main__":
    main()
