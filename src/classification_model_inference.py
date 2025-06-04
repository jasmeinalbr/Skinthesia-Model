import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras import backend as K

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * K.pow(y_pred, gamma)
        loss = weight * cross_entropy

        return K.mean(K.sum(loss, axis=1))  # sum over classes, mean over batch
    return focal_loss_fixed

def predict_ingredients(skin_type_list, skin_concern_list, skin_goal_list, model, mlb, threshold):
    """
    Fungsi untuk prediksi ingredients berdasarkan input user.

    Parameters:
        skin_type_list: list of str
        skin_concern_list: list of str
        skin_goal_list: list of str
        model: model TensorFlow yang sudah dilatih
        mlb: MultiLabelBinarizer yang sudah dilatih
        threshold: float

    Returns:
        List of predicted ingredients
    """
    # Gabungkan input menjadi satu string seperti pada preprocessing
    text_input = ' '.join(skin_type_list + skin_concern_list + skin_goal_list)

    # Konversi jadi array string NumPy
    text_array = np.array([text_input], dtype=object)

    # Prediksi probabilitas
    pred_probs = model.predict(text_array)

    # Threshold → multi-label biner
    pred_binary = (pred_probs >= threshold).astype(int)

    # Decode kembali ke label ingredients
    predicted_labels = mlb.inverse_transform(pred_binary)[0]

    return predicted_labels

def main():
    skin_type_input = ['dry']
    skin_concern_input = ['acne','irritation','redness']
    skin_goal_input = ['glowing','hydrating', 'moisturizing']

    print("\nLoading model and MultiLabelBinarizer...")
    try :
        model = tf.keras.models.load_model(
            'models/ingredients_classification_model.keras',
            custom_objects={'focal_loss_fixed': focal_loss(alpha=0.25, gamma=2)}
        )
        with open('models/ingredients_mlb.pkl', 'rb') as f:
            mlb = pickle.load(f)
    except Exception as e:
        print(f"Error loading model or MultiLabelBinarizer: {e}")
        return
    print("Model and MultiLabelBinarizer loaded successfully.")

    print("\nInput parameters:")
    print("Input Skin Type:", skin_type_input)
    print("Input Skin Concern:", skin_concern_input)
    print("Input Skin Goal:", skin_goal_input)

    # Prediksi ingredients dengan threshold awal 0.4
    print("\nPredicting ingredients...")
    predicted = predict_ingredients(
        skin_type_input,
        skin_concern_input,
        skin_goal_input,
        model=model,
        mlb=mlb,
        threshold=0.4
    )
    if not predicted:
        print("No ingredients found with threshold 0.4, trying lower thresholds...")
        print("changing treshold to 0.35...")
        predicted = predict_ingredients(
            skin_type_input,
            skin_concern_input,
            skin_goal_input,
            model=model,
            mlb=mlb,
            threshold=0.35
        )
        if not predicted:
            print("No ingredients found with threshold 0.35, trying lower thresholds...")
            print("changing treshold to 0.3...")
            predicted = predict_ingredients(
                skin_type_input,
                skin_concern_input,
                skin_goal_input,
                model=model,
                mlb=mlb,
                threshold=0.3
            )
            if not predicted:
                print("No ingredients found, try different input")
                return
    
    print("\nIngredients predicted successfully.")
    print("Predicted Ingredients:", predicted)

if __name__ == "__main__":
    main()