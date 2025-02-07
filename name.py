if _name_ == "_main_":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = load_model()
    texts = input("Enter the text to predict: ")
    predictions = predict(texts, model, tokenizer)
    for ouput in predictions:
        print("Output:", ouput)