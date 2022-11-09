from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import gradio as gr
import pandas as pd
import json
import os
import glob

# === READ AND LOAD FILES ===

folder = '.'

data = pd.read_csv(os.path.join(folder, 'species_info.csv'))

with open(os.path.join(folder, 'translation.json'), 'r') as f:
  translation = json.load(f)

# Load the model
model = load_model(os.path.join(folder, 'keras_model.h5'))
# Load label file
with open(os.path.join(folder, 'labels.txt'),'r') as f:
    labels = f.readlines()

# === GLOBAL VARIABLES ===
language = ''
article = ""


def format_label(label):
  """
  From '0 rùa khác\n' to 'rùa khác'
  """
  try:
    int(label.split(' ')[0])
    return label[label.find(" ")+1:-1]
  except:
    return label[:-1]

def get_name(scientific_name, lan):
  """
  Return name in Vietnamese
  """
  return data[data[f'scientific_name'] == scientific_name][f'name_{lan}'].to_list()[0]

def get_fun_fact(scientific_name, lan):
  """
  Return fun fact of the species
  """
  return data[data[f'scientific_name'] == scientific_name][f'fun_fact_{lan}'].to_list()[0]

def get_law(scientific_name):
  cites = data[data['scientific_name'] == scientific_name]['CITES'].to_list()[0]
  nd06 = data[data['scientific_name'] == scientific_name]['ND06'].to_list()[0]
  return cites, nd06

def get_habitat(scientific_name, lan):
  return data[data['scientific_name'] == scientific_name][f'habitat_{lan}'].to_list()[0]

def get_conservation_status(scientific_name, lan):
  status_list = ['NE', 'DD', 'LC', 'NT', 'VU', 'EN', 'CR', 'EW', 'EX']
  status = data[data['scientific_name'] == scientific_name]['IUCN'].to_list()[0]
  for s in status_list:
    if s in status:
      return translation['conservation_status'][s][lan]

def get_language_code(lan):
  global language
  if lan == "Tiếng Việt":
    language = 'vi'
  if lan == "English":
    language = 'en'

  return language

def get_species_list():
  """
  Example:
  ['Indotestudo elongata',
  'Cuora galbinifrons',
  'Cuora mouhotii',
  'Cuora bourreti']
  """
  return [format_label(s) for s in labels]

def get_species_abbreviation(scientific_name):
  return "".join([s[0] for s in scientific_name.split()])

def get_species_abbreviation_list():
  """
  Example:
  ['Ie', 'Cg', 'Cm', 'Cb']
  """
  return [get_species_abbreviation(s) for s in get_species_list()]

def get_description(language):
  num_class = len(labels)
  num_native = 0
  num_non_native = 0

  native_list = ''
  non_native_list = ''

  for i in labels:
    label = format_label(i)
    if label in data[data.native == 'y'].scientific_name.values:
      num_native += 1
      native_list += f"({num_native}) {get_name(label, language)}, "
    else:
      num_non_native += 1
      non_native_list += f"({num_non_native}) {get_name(label, language)}, "

  if language=='vi':
    description=f"""
    VNTurtle nhận diện các loài rùa Việt Nam. Mô hình này có thể nhận diện **{num_class}** loại rùa thường xuất hiện ở VN gồm
    - **{num_native}** loài bản địa: {native_list} \n\n
    - **{num_non_native}** loài ngoại lai: {non_native_list}
    """
  if language=='en':
    description=f"""
    VNTurtle can recognize turtle species in Vietnam. This model can identify {num_class} common turtles in Vietnam including **{num_native}** native species \n\n
    {native_list} \n\n
    and **{num_non_native}** non-native species \n\n
    {non_native_list}
    """
  return description

def update_language(language):
  language = get_language_code(language)
  return get_description(language), \
          translation['label']['label_run_btn'][language], \
          translation["accordion"]["fun_fact"][language], \
          translation["accordion"]["status"][language], \
          translation["accordion"]["law"][language], \
          translation["accordion"]["info"][language]

def predict(image):
  # Create the array of the right shape to feed into the keras model
  # The 'length' or number of images you can put into the array is
  # determined by the first position in the shape tuple, in this case 1.
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
  #resize the image to a 224x224 with the same strategy as in TM2:
  #resizing the image to be at least 224x224 and then cropping from the center
  size = (224, 224)
  image = ImageOps.fit(image, size, Image.ANTIALIAS)

  #turn the image into a numpy array
  image_array = np.asarray(image)
  # Normalize the image
  normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
  # Load the image into the array
  data[0] = normalized_image_array

  # run the inference
  pred = model.predict(data)
  pred = pred.tolist()

  return pred 

result = {}
best_prediction = ''

def interpret_prediction(prediction):
  global result

  sorted_index = np.argsort(prediction).tolist()[0]

  display_index = []
  for i in sorted_index[::-1]:
    if prediction[0][i] > 0.01:
      display_index.append(i)
  
  # best_prediction = format_label(labels[sorted_index[-1]]).strip()

  result = {format_label(labels[i]): round(prediction[0][i],2) for i in display_index}

  # return best_prediction

def run_btn_click(image):
  global best_prediction 
  best_prediction = None
  global article
  article = translation["info"]["ATP_contact"][language]

  interpret_prediction(predict(image))

  visible_result = [
      False, 
      False,
      False,
      False,
      False
  ] 

  image_result = [
      os.path.join(folder, 'examples', 'empty.JPG'),
      os.path.join(folder, 'examples', 'empty.JPG'),
      os.path.join(folder, 'examples', 'empty.JPG'),
      os.path.join(folder, 'examples', 'empty.JPG'),
      os.path.join(folder, 'examples', 'empty.JPG')
  ]

  percent_result = [
      "", 
      "", 
      "", 
      "", 
      ""
  ] 

  species_result = [
      "", 
      "", 
      "", 
      "", 
      ""
  ] 

  for i, (species, percent) in enumerate(result.items()):
    print(species, result)
    visible_result[i] = True
    image_result[i] = os.path.join(folder, 'examples', f'test_{get_species_abbreviation(species)}.JPG')
    percent_result[i] = f'{round(percent*100)}%'
    species_result[i] = species

  return  gr.Accordion.update(open=True, visible=True), \
          gr.Image.update(value=image_result[0], visible=visible_result[0]), \
            gr.HighlightedText.update(value=[('', percent_result[0])], label=species_result[0], visible=visible_result[0]), \
            gr.Button.update(visible=visible_result[0]), \
            \
          gr.Image.update(value=image_result[1], visible=visible_result[1]), \
            gr.HighlightedText.update(value=[('', percent_result[1])], label=species_result[1], visible=visible_result[1]), \
            gr.Button.update(visible=visible_result[1]), \
            \
          gr.Image.update(value=image_result[2], visible=visible_result[2]), \
            gr.HighlightedText.update(value=[('', percent_result[2])], label=species_result[2], visible=visible_result[2]), \
            gr.Button.update(visible=visible_result[2]), \
            \
          gr.Image.update(value=image_result[3], visible=visible_result[3]), \
            gr.HighlightedText.update(value=[('', percent_result[3])], label=species_result[3], visible=visible_result[3]), \
            gr.Button.update(visible=visible_result[3]), \
            \
          gr.Image.update(value=image_result[4], visible=visible_result[4]), \
            gr.HighlightedText.update(value=[('', percent_result[4])], label=species_result[4], visible=visible_result[4]), \
            gr.Button.update(visible=visible_result[4]), \
            gr.Accordion.update(visible=False), \
            []
            # gr.Accordion.update(visible=False), \
            # gr.Accordion.update(visible=False), \
            # gr.Accordion.update(visible=False), \
            # gr.Accordion.update(visible=False), \
            # gr.Markdown.update(value=percent_result[4], visible=visible_result[4]), \


def get_image_gallery_species_1():
  global best_prediction
  for i, name in enumerate(result):
    if i == 0:
      best_prediction = name
      return glob.glob(os.path.join(folder, 'gallery', name, '*'))

def get_image_gallery_species_2():
  global best_prediction
  for i, name in enumerate(result):
    if i == 1:
      best_prediction = name
      return glob.glob(os.path.join(folder, 'gallery', name, '*'))

def get_image_gallery_species_3():
  global best_prediction
  for i, name in enumerate(result):
    if i == 2:
      best_prediction = name
      return glob.glob(os.path.join(folder, 'gallery', name, '*'))

def get_image_gallery_species_4():
  global best_prediction
  for i, name in enumerate(result):
    if i == 3:
      best_prediction = name
      return glob.glob(os.path.join(folder, 'gallery', name, '*'))

def get_image_gallery_species_5():
  global best_prediction
  for i, name in enumerate(result):
    if i == 4:
      best_prediction = name
      return glob.glob(os.path.join(folder, 'gallery', name, '*'))

def display_info():  
  cites, nd06 = get_law(best_prediction)

  fun_fact = f"{get_fun_fact(best_prediction, language)}."

  status = f"{get_conservation_status(best_prediction, language)}"
  
  law = f'CITES: {cites}, NĐ06: {nd06}'

  info = ""

  if str(nd06) != "":
    law_protection = translation["info"]["law_protection"][language]    
    report = translation["info"]["report"][language]
    deliver = translation["info"]["deliver"][language]
    release = translation["info"]["release"][language] + f" **{get_habitat(best_prediction, language)}**"
  
  info = f"- {law_protection}\n\n- {report}\n\n- {deliver}\n\n- {release}"

  return gr.Accordion.update(visible=True), \
          gr.Accordion.update(open=False), \
          gr.Accordion.update(visible=True), \
          gr.Accordion.update(visible=True), \
          gr.Accordion.update(visible=True), \
          gr.Accordion.update(visible=True), \
          fun_fact, status, law, info

default_lan = 'Tiếng Việt'

with gr.Blocks() as demo:
  gr.Markdown("# VNTurtle")
  radio_lan = gr.Radio(choices=['Tiếng Việt', 'English'], value=default_lan, label='Ngôn ngữ/Language', show_label=True, interactive=True)
  md_des = gr.Markdown(get_description(get_language_code(default_lan)))

  with gr.Row():
    inp = gr.Image(type="pil", show_label=True, label='Ảnh tải lên', interactive=True)
    gallery = gr.Gallery(show_label=True, label='Ảnh đối chiếu')
  with gr.Row():
    run_btn = gr.Button(translation['label']['label_run_btn'][get_language_code(default_lan)])
    result_verify_btn = gr.Button(translation['label']['label_verify_btn'][get_language_code(default_lan)], visible=True)

  accordion_result_section = gr.Accordion(translation["accordion"]["result_section"][get_language_code(default_lan)], open=True, visible=False)

  with accordion_result_section:
    with gr.Row() as display_result:
      with gr.Column(scale=0.2, min_width=150) as result_1:
        result_percent_1 =  gr.HighlightedText(show_label=True, visible=False).style(color_map={f'{i}%': 'green' for i in range(101)})
        # result_percent_1 = gr.Markdown("", visible=False)
        result_img_1 = gr.Image(interactive=False, visible=False, show_label=False)
        result_view_btn_1 = gr.Button(translation['label']['label_check_btn'][get_language_code(default_lan)], visible=False)
      with gr.Column(scale=0.2, min_width=150) as result_2:
        result_percent_2 = gr.HighlightedText(show_label=True, visible=False).style(color_map={f'{i}%': 'yellow' for i in range(101)})
        result_img_2 = gr.Image(interactive=False, visible=False, show_label=False)
        result_view_btn_2 = gr.Button(translation['label']['label_check_btn'][get_language_code(default_lan)], visible=False)
      with gr.Column(scale=0.2, min_width=150) as result_3:
        result_percent_3 = gr.HighlightedText(show_label=True, visible=False).style(color_map={f'{i}%': 'orange' for i in range(101)})
        result_img_3 = gr.Image(interactive=False, visible=False, show_label=False)
        result_view_btn_3 = gr.Button(translation['label']['label_check_btn'][get_language_code(default_lan)], visible=False)
      with gr.Column(scale=0.2, min_width=150) as result_4:
        result_percent_4 = gr.HighlightedText(show_label=True, visible=False).style(color_map={f'{i}%': 'chocolate' for i in range(101)})
        result_img_4 = gr.Image(interactive=False, visible=False, show_label=False)
        result_view_btn_4 = gr.Button(translation['label']['label_check_btn'][get_language_code(default_lan)], visible=False)
      with gr.Column(scale=0.2, min_width=150) as result_5:
        result_percent_5 = gr.HighlightedText(show_label=True, visible=False).style(color_map={f'{i}%': 'grey' for i in range(101)})
        result_img_5 = gr.Image(interactive=False, visible=False, show_label=False)
        result_view_btn_5 = gr.Button(translation['label']['label_check_btn'][get_language_code(default_lan)], visible=False)
  
  accordion_info_section = gr.Accordion(translation['accordion']['info_section'][get_language_code(default_lan)], visible=False, open=True)

  with accordion_info_section:
    accordion_fun_fact = gr.Accordion(translation["accordion"]["fun_fact"][get_language_code(default_lan)], open=False, visible=False)
    accordion_status = gr.Accordion(translation["accordion"]["status"][get_language_code(default_lan)], open=False, visible=False)
    accordion_law = gr.Accordion(translation["accordion"]["law"][get_language_code(default_lan)], open=False, visible=False)
    accordion_info = gr.Accordion(translation["accordion"]["info"][get_language_code(default_lan)], open=False, visible=False)

    with accordion_fun_fact:
      md_fun_fact = gr.Markdown()
    with accordion_status:
      md_status = gr.Markdown()
    with accordion_law:
      md_law = gr.Markdown()
    with accordion_info:
      md_info = gr.Markdown()

  gr.Markdown("---")
  with gr.Accordion("🌅 Ảnh thử nghiệm", open=False):
    gr.Examples(
      examples=[[os.path.join(folder, 'examples', f'test_{get_species_abbreviation(s)}.JPG'), get_name(s, language)] for s in get_species_list()],
      inputs=[inp],
      label=""
      )
  radio_lan.change(fn=update_language, inputs=[radio_lan], outputs=[
      md_des, 
      run_btn, 
      accordion_fun_fact, 
      accordion_status, 
      accordion_law, 
      accordion_info
      ])
  run_btn.click(fn=run_btn_click, inputs=inp, outputs= [
      accordion_result_section,
      # md_fun_fact, md_status, md_law, md_info,
      result_img_1, result_percent_1, result_view_btn_1,
      result_img_2, result_percent_2, result_view_btn_2,
      result_img_3, result_percent_3, result_view_btn_3,
      result_img_4, result_percent_4, result_view_btn_4,
      result_img_5, result_percent_5, result_view_btn_5,
      # accordion_fun_fact, accordion_status, accordion_law, accordion_info, 
      accordion_info_section,
      gallery
  ], show_progress=True, scroll_to_output=True)

  result_view_btn_1.click(fn=get_image_gallery_species_1, outputs=gallery)
  result_view_btn_2.click(fn=get_image_gallery_species_2, outputs=gallery)
  result_view_btn_3.click(fn=get_image_gallery_species_3, outputs=gallery)
  result_view_btn_4.click(fn=get_image_gallery_species_4, outputs=gallery)
  result_view_btn_5.click(fn=get_image_gallery_species_5, outputs=gallery)

  result_verify_btn.click(fn=display_info, outputs=[
      accordion_info_section,
      accordion_result_section,
      accordion_fun_fact,
      accordion_status,
      accordion_law,
      accordion_info,
      md_fun_fact,
      md_status,
      md_law,
      md_info,
  ], scroll_to_output=True)

demo.launch(debug=False)
