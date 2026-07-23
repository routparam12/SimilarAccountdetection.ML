from src.prediction.panelist_search import find_similar_panelists
from src.prediction.prediction import predict_similarity
from src.training.train_model import train


if __name__ == "__main__":
    print("1. Train Model")
    print("2. Predict (two emails)")
    print("3. Find similar panelists (one email vs panelists.csv)")

    choice = input("Select: ")

    if choice == "1":
        train()

    elif choice == "2":
        email1 = input("Enter Email 1: ")
        email2 = input("Enter Email 2: ")
        print(predict_similarity(email1, email2))

    elif choice == "3":
        input_email = input("Enter email to search: ")
        results = find_similar_panelists(input_email)

        if results.empty:
            print("No similar emails found above threshold.")
        else:
            print(results.to_string(index=False))
