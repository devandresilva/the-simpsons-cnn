from PIL import Image
from flask import Flask, request, jsonify
from joblib import load
import wandb
import os
from dotenv import load_dotenv
from keras.models import load_model
import platform
import tensorflow as tf
import numpy as np

load_dotenv()  # Load variables from .env file

wandb_api_key = os.getenv("WANDB_API_KEY")

model = None

classes = ['abraham_grampa_simpson', 'agnes_skinner','apu_nahasapeemapetilon', 'barney_gumble', 'bart_simpson', 'carl_carlson', 'charles_montgomery_burns', 'chief_wiggum', 'cletus_spuckler', 'comic_book_guy', 'disco_stu', 'edna_krabappel', 'fat_tony', 'gil', 'groundskeeper_willie', 'homer_simpson', 'kent_brockman', 'krusty_the_clown', 'lenny_leonard', 'lionel_hutz', 'lisa_simpson', 'maggie_simpson', 'marge_simpson', 'martin_prince', 'mayor_quimby', 'milhouse_van_houten', 'miss_hoover', 'moe_szyslak', 'ned_flanders', 'nelson_muntz', 'otto_mann', 'patty_bouvier', 'principal_skinner', 'professor_john_frink', 'rainier_wolfcastle', 'ralph_wiggum', 'selma_bouvier', 'sideshow_bob', 'sideshow_mel', 'snake_jailbird', 'troy_mcclure', 'waylon_smithers']

if wandb_api_key:
    wandb.login(key=os.getenv("WANDB_API_KEY"))
else:
    print("WANDB_API_KEY not found. Please set the environment variable.")

api = wandb.Api()
artifact = api.artifact('flateam/the_simpsons_characters/the_simpsons_character_model_pipeline:latest', type='pipeline')
arquivo = artifact.file()
artifact2 = api.artifact('flateam/the_simpsons_characters/image_processor_class:latest', type='python')
artifact_dir = artifact2.file()

dest_path = './'

if platform.system() == "Windows":
    os.system(f'copy "{artifact_dir}" "{dest_path}"')
else:
    os.system(f'cp {artifact_dir} {dest_path}')

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    image_bytes = request.data
    image = Image.frombytes("RGBA", (300, 300), image_bytes)

    p = model.predict([image])
    i = classes[np.argmax(p)]

    print("Predição realizada com sucesso!")
    print(i)
    return jsonify({'message': i})


@app.route("/")
def index():
    return '''<h1>Bem vindo ao servidor de Predição</h1> <p>Aqui você pode predizer imagens dos iconicos personagens dos simpsons </p>'''


if __name__ == "__main__":
    try:
        model = load(arquivo)
    except Exception as e:
        print(f"Error loading model: {e}")
    app.run(debug=True)