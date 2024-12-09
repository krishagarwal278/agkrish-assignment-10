import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from PIL import Image
from open_clip import create_model_and_transforms
import torch
import os
import shutil  
import open_clip

# Load the embeddings file as a pandas DataFrame
EMBEDDINGS_FILE = "all_images/image_embeddings.pickle"
df = pd.read_pickle(EMBEDDINGS_FILE)

# Convert the DataFrame to a dictionary for easier processing
IMAGE_EMBEDDINGS = df.set_index('file_name')['embedding'].to_dict()

# Initialize the CLIP model and preprocessing transforms
MODEL_NAME = "ViT-B-32-quickgelu"  # Use the correct model name
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and preprocess functions
model, _, preprocess = create_model_and_transforms(MODEL_NAME, pretrained="openai")
model = model.to(DEVICE)
model.eval()

def get_image_embeddings(image_path):
    """
    Extract embeddings from an image using the CLIP model.

    Args:
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Embedding vector of the image.
    """
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        # Extract embeddings
        with torch.no_grad():
            embedding = model.encode_image(image_tensor).cpu().numpy()

        return embedding
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return np.random.rand(1, 512)  # Fallback for testing

def get_text_embeddings(text_query):
    """
    Generate embeddings for a text query using the CLIP model.

    Args:
        text_query (str): The input text query.

    Returns:
        np.ndarray: The embedding vector of the text.
    """
    try:
        # Tokenize and encode the text
        text_tokens = open_clip.tokenize([text_query]).to(DEVICE)
        with torch.no_grad():
            text_embedding = model.encode_text(text_tokens).cpu().numpy()
        return text_embedding
    except Exception as e:
        print(f"Error processing text query '{text_query}': {e}")
        return np.random.rand(1, 512)  # Fallback for testing

def search_images(text_query, image_path, query_type, weight):
    """
    Search the database for relevant images based on a text or image query.

    Args:
        text_query (str): Text query string.
        image_path (str): Path to the uploaded image query.
        query_type (str): Query type ('text', 'image', or 'hybrid').
        weight (float): Hybrid query weight for text (0.0 to 1.0).

    Returns:
        List[dict]: Top 5 results with similarity scores.
    """
    results = []
    uploads_folder = "uploads"

    # Generate a random embedding for the text query (placeholder)
    text_embedding = get_text_embeddings(text_query) if text_query else None

    # Generate an embedding for the image query
    image_embedding = get_image_embeddings(image_path) if image_path else None

    # Extract embeddings as a matrix and corresponding image names
    embeddings_matrix = np.array(list(IMAGE_EMBEDDINGS.values()))
    image_names = list(IMAGE_EMBEDDINGS.keys())

    # Perform the similarity search based on the query type
    if query_type == "text" and text_embedding is not None:
        similarities = cosine_similarity(text_embedding, embeddings_matrix)
    elif query_type == "image" and image_embedding is not None:
        similarities = cosine_similarity(image_embedding, embeddings_matrix)
    elif query_type == "hybrid" and text_embedding is not None and image_embedding is not None:
        hybrid_embedding = weight * text_embedding + (1 - weight) * image_embedding
        similarities = cosine_similarity(hybrid_embedding, embeddings_matrix)
    else:
        return {"error": "Invalid query type or missing inputs."}

    # Extract the top 5 results based on similarity scores
    top_indices = np.argsort(similarities[0])[::-1][:5]
    for idx in top_indices:
        src_image_path = f"all_images/coco_images_resized/coco_images_resized/{image_names[idx]}"
        dest_image_path = os.path.join(uploads_folder, image_names[idx])

        # Copy the image to the uploads folder
        try:
            shutil.copy(src_image_path, dest_image_path)
        except Exception as e:
            print(f"Error copying image {src_image_path} to {dest_image_path}: {e}")
            continue

        # Append the result
        results.append({
            "image": f"uploads/{image_names[idx]}",  # Path in the uploads folder
            "similarity": float(similarities[0][idx])  # Convert to Python float for JSON compatibility
        })

    return results