import numpy as np
import argparse
from sklearn.datasets import make_regression

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--samples", type=int , required=True)        #Αριθμός των δειγμάτων που θα δημιουργηθούν (Ν)
    parser.add_argument("--features", type=int , required=True)       #Αριθμός των χαρακτηριστικών που θα δημιουργηθούν (D)
    parser.add_argument("--dtype", type=str , default="float64")  #Ο τύπος δεδομένων που θα χρησιμοποιηθεί για τα χαρακτηριστικά και τις ετικέτες (πχ float64, float32)
    parser.add_argument("--output", type=str , required=True)         #Το όνομα του αρχείου εξόδου, όπου θα αποθηκευτούν τα παραγόμενα δεδομένα
    parser.add_argument("--seed", type=int , default=42)             #Το seed της γεννήτριας τυχαίων αριθμών για την αναπαραγωγή των αποτελεσμάτων

    #Ανάγνωση των ορισμάτων από τη γραμμή εντολών
    args = parser.parse_args()

    #Ορισμός του seed για την αναπαραγωγή των αποτελεσμάτων
    np.random.seed(args.seed)

    X, _ = make_regression(n_samples=args.samples, n_features=args.features, noise=0.1, random_state=args.seed)

    datatype_map = {
        "float64": np.float64,
        "float32": np.float32,
        "int64": np.int64,
        "int32": np.int32
    }
    datatype = datatype_map[args.dtype]

    X = X.astype(datatype)
    X.tofile(args.output)

    print("Dataset generated successfully!")
    print(f"Shape: {X.shape}")
    print(f"Datatype: {X.dtype}")
    print(f"Output: {args.output}")
    print(f"Size: {X.nbytes / (1024**2):.2f} MB")
    print(f"Expected size: {args.samples * args.features * X.itemsize} bytes")
    print(f"Actual size: {X.nbytes} bytes")


if __name__ == "__main__":
    main()
