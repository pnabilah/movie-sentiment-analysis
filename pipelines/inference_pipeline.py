import yaml
from movie_sentiment_analysis.infer_sentiment import analyze

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = config["paths"]["best_model_path"]
    texts = [
        "I loved this movie! Amazing story and acting.",
        "Worst movie I've seen this year. Boring plot.",
        "It was okay, some parts were good, some were bad.",
        "Absolutely fantastic! Highly recommend it.",
        "Terrible. I want my time back."
    ]

    labels = analyze(model, texts)
    print(labels)

if __name__ == "__main__":
    main()