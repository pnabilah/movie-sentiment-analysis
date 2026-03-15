import yaml
from movie_sentiment_analysis.preprocess import preprocess
from movie_sentiment_analysis.models import build_gru_lang_binary
from movie_sentiment_analysis.train import train
from movie_sentiment_analysis.utils.visualization import plot_loss_history, plot_accuracy_history

def training_pipeline():
    with open("config.yaml", "r") as f:
        config = yaml.safeload(f)
    # Load preprocessed dataset
    train_set, val_set, test_set, n_tokens, text_vec_layer = preprocess(dataset_path=config["dataset"]["raw_path"])
    # Create model
    model = build_gru_lang_binary(text_vec_layer, n_tokens, embed_dim=config["model"]["embed_dim"], gru_units=config["model"]["gru_units"])
    # Train model
    history = train(model, train_set, valid_set=val_set, epochs=config["training"]["epochs"],
                    best_model_path=config["paths"]["best_model_path"],
                    final_model_path=config["paths"]["final_model_path"],
                    history_path=config["paths"]["history_path"])
    # Plot metrics
    plot_loss_history(history, save_path=config["paths"]["loss_curve_path"])
    plot_accuracy_history(history, save_path=config["paths"]["accuracy_curve_path"])

if __name__ == "__main__":
    training_pipeline()