import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd # Import pandas to potentially load classes from a CSV

# Define the WeedClassifier model class exactly as it was defined in the notebook
class WeedClassifier(nn.Module):
    def __init__(self, num_classes):
        super(WeedClassifier, self).__init__()
        # Use ResNet50 with pre-trained weights
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # Use weights parameter for pretrained
        # Modify the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Define the image transformation pipeline for inference (same as validation)
# Note: Using transforms from torchvision
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the path to the saved best model file in Google Drive
# Make sure this path is accessible from where the Streamlit app will run
# If running locally, ensure the file is in the correct location or provide the full path
# If running in Colab, the path should be the one where you saved the model in Drive
model_path = "/content/drive/MyDrive/weed_project/best_model.pth"

# Define the list of class names in the same order as they were indexed during training.
# It's best to load this from the cleaned training CSV if possible to ensure correctness.
# Assuming the cleaned training CSV is available at this path:
cleaned_train_csv_path = "weed25/train_subset1_cleaned.csv"
try:
    train_df = pd.read_csv(cleaned_train_csv_path)
    classes = sorted(train_df['label'].unique())
    print("✅ Class names loaded successfully from CSV.")
except FileNotFoundError:
    # Fallback: If the CSV is not available, manually define the classes.
    # This list MUST match the classes in your training data in sorted order.
    print(f"❗ Cleaned training CSV not found at {cleaned_train_csv_path}. Using hardcoded class list.")
    classes = [
        'Alligatorweed', 'Barnyard grass', 'Bidens', 'Billygoat weed', 'Black nightshade',
        'Broadleaf plantain', 'Canada thistle', 'Cocklebur', 'Common dayflower', 'Common ragweed',
        'Crabgrass', 'Dandelion', 'Field thistle', 'Goosefoots', 'Green foxtail',
        'Horseweed', 'Indian', 'Lambsquarters', 'Mock strawberry', 'Morningglory',
        'Nutsedge', 'Palmer amaranth', 'Pigweed', 'Plantian', 'Purslane',
        'Redroot pigweed', 'Sedge', 'Shepherds purse', 'Sowthistle', 'Velvetleaf',
        'Viola', 'White clover', 'White smart weed', 'Yellow nutsedge', 'Yellow woodsorrel'
    ]
    # Note: The number of classes in the model must match the actual number of unique classes in your training data.
    # If you used a subset with fewer classes, adjust this list accordingly.
    # Based on the previous notebook output, there are 20 classes. Let's try to match that.
    # You might need to adjust this list based on the actual classes in your cleaned subset.
    # For this example, let's assume the 20 classes from the classification report:
    classes = [
        'barnyard grass', 'bidens', 'billygoat weed', 'black nightshade', 'cocklebur',
        'common dayflower', 'crabgrass', 'field thistle', 'goosefoots', 'green foxtail',
        'horseweed', 'indian', 'mock strawberry', 'pigweed', 'plantian',
        'purslane', 'sedge', 'velvetleaf', 'viola', 'white smart weed'
    ]
    # It is highly recommended to load classes from the cleaned CSV to avoid errors.


# Determine the number of classes based on the loaded or hardcoded list
num_classes = len(classes)

# Load the trained model state dictionary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WeedClassifier(num_classes=num_classes).to(device)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    st.success("✅ Model loaded successfully.")
except FileNotFoundError:
    st.error(f"❗ Model file not found at {model_path}. Please ensure the path is correct and the file exists.")
except Exception as e:
    st.error(f"❗ An error occurred while loading the model: {e}")
    model = None # Set model to None if loading fails

# Create the Streamlit application layout
st.title("Weed Species Classifier")

st.write("Upload an image of a weed to get a prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    if model is not None:
        try:
            # Preprocess the image
            img_tensor = inference_transform(image).unsqueeze(0).to(device) # Add batch dimension and move to device

            # Perform inference
            with torch.no_grad():
                output = model(img_tensor)
                # Apply softmax to get probabilities
                probabilities = torch.softmax(output, dim=1)
                # Get the predicted class index and probability
                pred_prob, pred_class_idx = torch.max(probabilities, 1)
                pred_class_idx = pred_class_idx.item()
                pred_prob = pred_prob.item()

            # Map index to class name
            if 0 <= pred_class_idx < len(classes):
                pred_class_name = classes[pred_class_idx]
                st.success(f"✅ Predicted class: **{pred_class_name}** (Confidence: {pred_prob:.2f})")
            else:
                st.warning(f"❗ Predicted class index {pred_class_idx} is out of bounds for the defined classes.")

        except Exception as e:
            st.error(f"❗ An error occurred during prediction: {e}")
    else:
        st.warning("❗ Model is not loaded. Cannot perform prediction.")
