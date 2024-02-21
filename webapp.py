
import gradio as gr
from PIL import Image
import requests
import io
import json
import pandas
import os
import numpy as np
from methods import create_df
from methods import extract_features_from_image


def similar_poster(image):
    #df = create_df('./extracted_data//movielens-20m-posters-for-machine-learning/MLP-20M')
    df = pandas.read_pickle("Fichiers_pkl/df.pkl")
    #index_movie = 15
    image_features = extract_features_from_image(image)
    #dico = {'nb_reco': 5,
    #        'vec': df['features'].iloc[index_movie].tolist()}
    method="image"
    dico = {'nb_reco': 5, 'method': method,'vec': image_features.tolist()}
    jsonData = json.dumps(dico)
    headers = {'Content-Type': 'application/json'}

    response = requests.post("http://api:5000/predict", data=jsonData, headers=headers)
    pred = df.iloc[response.json()["prediction"]]
    paths = pred.path
    paths = [path.replace("\\", "/")for path in paths]
    paths = [path.replace("//", "/") for path in paths]
    paths = [path.replace('movielens-20m-posters-for-machine-learning/','') for path in paths]

    print(paths)
    if response.status_code == 200:
        similar_images = [Image.open(path) for path in paths]
        all_images =  similar_images

        return all_images[0], all_images[1], all_images[2], all_images[3],all_images[4]


def similar_poster_txt(method,texte):
    
    df = pandas.read_csv("./movies_metadata.csv")
    #df2 = pandas.read_pickle("df.pkl")
    #print("TAILLE DF2",df2.shape)
    
    dico = {'nb_reco': 5, 'method':method,'vec': texte}
    jsonData = json.dumps(dico)
    headers = {'Content-Type': 'application/json'}

    response = requests.post(f'http://api:5000/{method}', data=jsonData, headers=headers)
    indices = response.json()["prediction"]
    #print("indices AAAAAAAAAAA",indices)
    
    temp=[]
    similar_movies=[]
    # Afficher les noms des films associés aux indices
    for i in indices:
        temp.append(i)
        temp.append(df['title'].iloc[i])
        temp.append(df['overview'].iloc[i])
        
        similar_movies.append(temp.copy())
        temp.clear()
        
        

  
    return similar_movies


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
        """
        # Choisssez une méthode 
    
        """)
       
        gr.Markdown(
        """
        # Chargez une image
    
        """)

    with gr.Row():
      with gr.Column(scale=1):
        a = gr.Radio(["Bag_of_words", "Bow_tfidf","Glove"], label="Choisissez la méthode utilisée")
        b = gr.Textbox(label="Taper une description de film")
      c=gr.Image(type="pil")



    with gr.Row():
      b1 = gr.Button(value="Submit")
      b2 = gr.Button(value="Submit")

    with gr.Row():
        with gr.Column():
          board = gr.Dataframe(headers=["idx", "titre", "description"],datatype=["number", "str", "str"],row_count=5,col_count=(3, "fixed"))
         
          
        with gr.Column():
          outputs=gr.Image()
          outputs2=gr.Image()
          outputs3=gr.Image()
          outputs4=gr.Image()
          outputs5=gr.Image()
            
            
        

    #b1.click(traiter_texte, inputs=a, outputs=text)
    b1.click(similar_poster_txt, inputs=[a,b], outputs=[board])
    b2.click(similar_poster, inputs=c,outputs=[outputs,outputs2,outputs3,outputs4,outputs5])
    


demo.launch(debug=True,server_name="0.0.0.0")


