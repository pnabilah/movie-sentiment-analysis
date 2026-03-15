import json
from movie_sentiment_analysis.utils.helper import prepare_save_path
from tensorflow.keras.callbacks import ModelCheckpoint

def train(model, train_set, valid_set=None, epochs=10,
          best_model_path="models/best_val_model.keras",
          final_model_path="models/final_epoch_model.keras",
          history_path="models/training_history.json"):
    
    best_model_path = prepare_save_path(best_model_path)
    final_model_path = prepare_save_path(final_model_path)
    history_path = prepare_save_path(history_path)

    callbacks = []

    if valid_set:
        checkpoint = ModelCheckpoint(best_model_path, 
                                    monitor="val_accuracy",
                                    save_best_only=True)
        callbacks.append(checkpoint)
    
    history = model.fit(train_set, 
                        validation_data=valid_set, 
                        epochs=epochs, 
                        callbacks=callbacks)
    
    model.save(final_model_path)
    
    with open(history_path, "w") as f:
        json.dump(history.history, f)
        
    print("Model and training history has been saved.")
    return history.history