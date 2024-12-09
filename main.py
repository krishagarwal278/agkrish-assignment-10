import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load data and CLIP model
DATA_PATH = 'image_embeddings.pickle'
STATIC_FOLDER = 'coco_images_resized'

with open(DATA_PATH, 'rb') as f:
    image_data = pickle.load(f)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Prepare data
embeddings = np.array(image_data['embedding'].tolist())
file_names = image_data['file_name'].tolist()


def get_text_embedding(text):
    """Generate text embedding using CLIP."""
    inputs = clip_processor(text=[text], return_tensors="pt", truncation=True)
    return clip_model.get_text_features(**inputs).detach().numpy()


def get_image_embedding(image_path):
    """Generate image embedding using CLIP."""
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt", truncation=True)
    print("ed")
    return clip_model.get_image_features(**inputs).detach().numpy()


def perform_search(query_embedding, embeddings, top_k=5):
    """Perform cosine similarity search."""
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = np.argsort(similarities[0])[::-1][:top_k]
    return [(file_names[idx], float(similarities[0][idx])) for idx in top_indices]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    text_query = request.form.get('text_query')
    image_query = request.files.get('image_query')
    hybrid_weight = float(request.form.get('hybrid_weight', 0.5))
    query_type = request.form.get('query_type', 'text')  # text, image, or hybrid
    use_pca = request.form.get('use_pca') == 'true'

    query_embedding = None

    # Apply PCA if enabled
    if use_pca:
        pca = PCA(n_components=5)
        reduced_embeddings = pca.fit_transform(embeddings)

        # Ensure the query embedding is reduced with the same PCA
        if query_type == 'text' and text_query:
            query_embedding = pca.transform(get_text_embedding(text_query))
        elif query_type == 'image' and image_query:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_query.filename)
            image_query.save(image_path)
            query_embedding = pca.transform(get_image_embedding(image_path))
        elif query_type == 'hybrid' and text_query and image_query:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_query.filename)
            image_query.save(image_path)
            text_embedding = get_text_embedding(text_query)
            image_embedding = get_image_embedding(image_path)
            hybrid_embedding = hybrid_weight * text_embedding + (1 - hybrid_weight) * image_embedding
            query_embedding = pca.transform(hybrid_embedding)
    else:
        reduced_embeddings = embeddings

        if query_type == 'text' and text_query:
            query_embedding = get_text_embedding(text_query)
        elif query_type == 'image' and image_query:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_query.filename)
            image_query.save(image_path)
            query_embedding = get_image_embedding(image_path)
        elif query_type == 'hybrid' and text_query and image_query:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_query.filename)
            image_query.save(image_path)
            text_embedding = get_text_embedding(text_query)
            image_embedding = get_image_embedding(image_path)
            query_embedding = hybrid_weight * text_embedding + (1 - hybrid_weight) * image_embedding

    if query_embedding is None:
        return jsonify({"error": "Invalid query type or missing inputs."}), 400

    # Perform the search
    results = perform_search(query_embedding, reduced_embeddings)

    # Return results with file paths for static serving
    results_with_paths = [(f"static/images/{os.path.basename(image)}", score) for image, score in results]
    return jsonify(results_with_paths)


if __name__ == '__main__':
    app.run(debug=True)