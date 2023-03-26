#                   SYNTHIA
#   The AI system to accelerate knowledge 

##########
#LIBRARIES
##########

import streamlit as st

st.set_page_config(page_title="medieval charter analyze", 
                   page_icon=":robot_face:",
                   layout="wide",
                   initial_sidebar_state="expanded"
                   )


import time
from googletrans import Translator
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import readtime
import textstat
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from io import StringIO
from textattack.augmentation import EmbeddingAugmenter
from textattack.augmentation import WordNetAugmenter
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.models import TextClassifier

from kraken import pageseg
from kraken import blla
from kraken.lib import vgsl
from kraken import rpred
from kraken.lib import models
import json
import time
import glob

from PIL import Image

from kraken import binarization

import numpy as np
import codecs

# loading the model


def WORD2HTML(sentence):
  CONLL_html=[str(token).split("Token: ")[1].split()[1] for token in sentence]
  tokenized_text=[str(token).split("Token: ")[1].split()[1] for token in sentence]

  index_entities=["O"]*len(tokenized_text)

  dict_colors={"PERS": "255FD2", "ORG": "EE4220", "MISC":"19842E", "LOC":"7A1A7A"}


  for entity in sentence.get_spans('ner'):
    x=" ".join([(str(y).split("Token: ")[1]) for y in entity]).split()[::2]
    x=[int(x[0])-1, int(x[-1])]

    type_ent=str(entity).split("− Labels: ")[1].split()[0]
    
    if x[1]-x[0]>1: #if we are leading with a more one-unit entity. vg: Madame de Parme
      index_entities[x[0]:x[1]]=["B-"+type_ent]+["I-"+type_ent]*(x[1]-x[0]-1)
      if x[1]-x[0]>2:
        CONLL_html[x[0]:x[1]]=['<span style="background-color: #'+dict_colors[type_ent]+'; padding:1px">'+CONLL_html[x[0]]]+[x for x in CONLL_html[x[0]+1:x[1]-1]]+[CONLL_html[x[1]-1]+'</span>']
      else:
        CONLL_html[x[0]:x[1]]=['<span style="background-color: #'+dict_colors[type_ent]+'; padding:1px">'+CONLL_html[x[0]]]+[CONLL_html[x[1]-1]+'</span>']
    else: #single entities
      index_entities[x[0]:x[1]]=["B-"+type_ent]*(x[1]-x[0])
      CONLL_html[x[0]:x[1]]=['<span style="background-color: #'+dict_colors[type_ent]+'; padding:1px">'+CONLL_html[x[0]:x[1]][0]+'</span>']

  CONLL=list(zip(tokenized_text, index_entities))
  return CONLL_html


@st.cache()
def ner(sentence):
  # make a sentence
  sentence = Sentence(sentence)

  # load the NER tagger
  tagger = SequenceTagger.load("models/FLAT_model_31_01_2022.pt")
  tagger.predict(sentence)

  tagged_sent=WORD2HTML(sentence)

  return " ".join(tagged_sent)


@st.cache()
def parts_dis(sentence):
  DIS_model = SequenceTagger.load('models/discours_parts_05_02_2022.pt')

  DIS_sentence= Sentence(sentence)
  entidades=WORD2HTML(DIS_sentence)
  
  DIS_model.predict(DIS_sentence)
  
  
  
  
  tokenized_text=[str(token).split("Token: ")[1].split()[1] for token in DIS_sentence]
  parts_discours=[]
  for x in DIS_sentence.get_spans('ner'):

    index=" ".join([(str(y).split("Token: ")[1]) for y in x]).split()[::2]#captura los index
    index=[int(index[0])-1, int(index[-1])]
    part=str(x).split("[− Labels: ")[1].replace("]", "")

    parts_discours.append([part, " ".join(entidades[index[0]:index[1]])])
    
  html="<table>"
  for x in parts_discours:
    #html+="<tr><th>Part of discourse</th"
    html+="<tr>"
    html+="<td>"+x[0]+"</td>"
    html+="<td>"+x[1]+"</td>"
    html+="</tr>"
  html+="</table>"
    
  return html


def read_image(image_name):
  #img=Image.open(image_name)
  img = cv2.imread(image_name)
  #Carga del modelod e segmentacion
  
  return img


def segmentation(image):
  model_path = 'models/blla.mlmodel'
  model = vgsl.TorchVGSLModel.load_model(model_path)
  #segmentación de la imagen
  baseline_seg = blla.segment(image, model=model)
  print("final de segmentación")
  
  return baseline_seg
  
  
@st.cache()
def transcript(img, baseline_seg):
  #aplicación del modelo de reconocimiento
  rec_model_path = 'models/model_36.mlmodel'
  modelito = models.load_any(rec_model_path)
  
  pred_it = rpred.rpred(network=modelito, im=img, bounds=baseline_seg)
  #obtención de las predicciones
  pred_char=[]
  for record in pred_it:
    #print(record)
    pred_char.append(record.prediction)
    
  return " ".join(pred_char)


@st.cache()
def class_acta(sentence):
  class_model = TextClassifier.load('models/charters_class_05_02_2022.pt')
  
  sentence = Sentence(sentence)
  class_model.predict(sentence)
  # print sentence with predicted labels
  #print('document above is: ', sentence.labels)
  return "Most probably type : "+str(sentence.labels[0])
  

#############
#PAGE SET UP
#############


def p_title(title):
    st.markdown(f'<h3 style="text-align: left; color:#F63366; font-size:28px;">{title}</h3>', unsafe_allow_html=True)

#########
#SIDEBAR
########

st.sidebar.header('Analyze medieval charter')
nav = st.sidebar.radio('',['Go to homepage', 'medieval charter analyze', 'Handrwritten text recognition', 'OCR enriching engine'])
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')

#CONTACT
########
expander = st.sidebar.expander('Contact')
expander.write("I'd love your feedback :smiley: Want to collaborate? Develop a project? Find me on [LinkedIn] (https://www.linkedin.com/in/lopezyse/), [Twitter] (https://twitter.com/lopezyse) and [Medium] (https://lopezyse.medium.com/)")

#######
#PAGES
######

#HOME
#####

if nav == 'Go to homepage':

    st.markdown("<h1 style='text-align: center; color: white; font-size:28px;'>Easy charter analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; font-size:56px;'<p>&#129302;</p></h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: grey; font-size:20px;'>Summarize, paraphrase, analyze text & more. Try our models, browse their source code, and share with the world!</h3>", unsafe_allow_html=True)
    """
    [![Star](https://img.shields.io/github/stars/dlopezyse/Synthia.svg?logo=github&style=social)](https://gitHub.com/dlopezyse/Synthia)
    &nbsp[![Follow](https://img.shields.io/twitter/follow/lopezyse?style=social)](https://www.twitter.com/lopezyse)
    &nbsp[![Buy me a coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee--yellow.svg?logo=buy-me-a-coffee&logoColor=orange&style=social)](https://www.buymeacoffee.com/lopezyse)
    """
    st.markdown('___')
    st.write(':point_left: Use the menu at left to select a task (click on > if closed).')
    st.markdown('___')
    st.markdown("<h3 style='text-align: left; color:#F63366; font-size:18px;'><b>What is this App about?<b></h3>", unsafe_allow_html=True)
    st.write("Learning happens best when content is personalized to meet our needs and strengths.")
    st.write("For this reason I created SYNTHIA :robot_face:, the AI system to accelerate and design your knowledge in seconds! Use this App to summarize and simplify content. Paste your text or upload your file and you're done. We'll process it for you!")     
    st.markdown("<h3 style='text-align: left; color:#F63366; font-size:18px;'><b>Who is this App for?<b></h3>", unsafe_allow_html=True)
    st.write("Anyone can use this App completely for free! If you like it :heart:, show your support by sharing :+1: ")
    st.write("Are you into NLP? Our code is 100% open source and written for easy understanding. Fork it from [GitHub] (https://github.com/dlopezyse/Synthia), and pull any suggestions you may have. Become part of the community! Help yourself and help others :smiley:")

#-----------------------------------------

#SUMMARIZE
##########

if nav == 'medieval charter analyze':    
    st.markdown("<h4 style='text-align: center; color:grey;'>Accelerate knowledge with streamlit &#129302;</h4>", unsafe_allow_html=True)
    st.text('')
    p_title('Analyze charter')
    st.text('')

    source = st.radio("How would you like to start? Choose an option below",
                          ("I want to input some text", "I want to upload a file"))
    st.text('')
    
    s_example = "In nomine sancte et indiuidue Trinitatis , Amen . Ludovicus dei gratia Francorum Rex . Quoniam vita hominum brevis et memoria labilis est dignum est ut controversie que mediantibus sapientibus viris seu transactione seu pacifica compositione sopita sunt , scriptis insinuentur , ne in posterum recidivas pariant questiones . Notum igitur facimus universis praesentibus et futuris quod controversia que inter ecclesiam Beati Dionysii et dominum Paganum de Praellis super nemoribus de Roseai divitius agitata fuerat , aspirante divina gratia hunc finem sortita est . De universis nemoribus de Roseai , medietatem habebit ecclesia beati Dionysii videlicet de Broces , Galentrudis , de Chesnai , sicut tota vallis in se continet , de Busboum , de Buslapidis , de Cuble engarri , de Broca Ade , de Vico , de Fraibouce , sicut salvueres continet usque ad fontem de Buhi , de Brouisses , de Leaele , de Bus , iuxta culturam de nemoribus Theobaldi de magno Roseai , et siqua alia sunt que pertineant ad nemus de Roseai , omnia communia erunt excepto nemore de Fai quod totum est Ecclesie beati Dionysii . Reliquam vero medietatem omnium predictorum nemorum tenebit praenominatus Paganus de ecclesia beati Dionysii , in feodi illius quem ab ipso tenebat incrementum . Si vero abbas qui pro tempore fuerit sive Paganus predictus , aut successor eius , nemus vendere voluerit , alter alterum prohibere non poterit , ipsa tamen venditio ab utroque communiter fiet . Precium vero quod inde percipietur aequaliter inter se divident , uterque suum servientem ponet qui nemus custodiat : Ita quod serviens ecclesie Domino Pagano et econverso serviens Domini Pagani ecclesie fidelitatem iurabit . Si vero serviens ecclesie aliquem in forisfacto nemoris ceperit , quicquid ab eo acceperit , ad villas Maflers portabit . Similiter servienti domini Pagani quod inde acceperit ad Francovillam ferre licebit . Ita tamen quod serviens Ecclesie domino Pagano et vicissim serviens domini Pagani , preposito beati Dionysi forisfacta omnia nunciabit et si de forisfactis querela aliqua orta fuerit , ambo pariter eam tractabunt . Si vero aliqua pars nemorum eradicata fuerit , et in culturam redacta eadem cultura communis erit salvo iure ecclesie beati Dionysii in decima . Iuravit itaque dominus Paganus quod hanc compositionem firmam tenebit et contra omnes de cognatione sua et contra alios de iure ad ipsum calumniantes ecclesiam immunem faciet et illesam custodiet et super hoc feodum suum obligavit . Iuraverunt quoque duo fratres sui videlicet Adam et Petrus se pretaxatam compositionem firmiter observaturos . Agnes etiam mater atque Richoldis soror eorumdem prescripte transactioni assensum prebuerunt . Et ut hoc ratum deinceps et inconcussum permaneat , scribi et sigilli nostri auctoritate precepimus communiri . Actum Parisius anno ab Incarnatione Domini M C LXX III . Astantibus in palatio nostro quorum nomina supposita sunt et signa . Signum Theobaldi comitis Dapiferi nostri . Signum Matthaei Camerarii . Signum Guidonis Buticularii . Signum Radulfi Constabularii. Vacante Cancelaria. "
    if source == 'I want to input some text':
        input_su = st.text_area("Use the example below or input your own text in Latin, French or Spanish (between 1,000 and 10,000 characters)", value=s_example, max_chars=10000, height=330)
        if st.button('medieval charter analyze'):
            if len(input_su) < 700:
                st.error('Please enter a text of minimum 1,000 characters')
            else:
                with st.spinner('Processing...'):
                    time.sleep(2)
  
                    t_r=("cochino")
                    st.markdown('___')
                    st.write("Named entities in Flat mode")
                    st.write(ner(input_su), unsafe_allow_html=True)
                    
                    
                    st.markdown('___')
                    st.success(class_acta(input_su))
                    text = input_su
                    
                    
                    
                    st.markdown('___')
                    st.write('Diplomatics parts')
                    st.write(parts_dis(input_su), unsafe_allow_html=True)
                    
              
                    

    if source == 'I want to upload a file':
        file = st.file_uploader('Upload your file here',type=['jpg'])
        if file is not None:
            with st.spinner('Processing...'):
                    time.sleep(2)
                    #image = file.read()
                    #resultado=transcript(read_image(file), segmentation(read_image(file)))
                    st.image(read_image(file),width=250)
                    st.write(segmentation(read_image(file)))
                    st.success("mamita 3")
                    

#-----------------------------------------

#PARAPHRASE
###########

if nav == 'Paraphrase text':
    st.markdown("<h4 style='text-align: center; color:grey;'>Accelerate knowledge with SYNTHIA &#129302;</h4>", unsafe_allow_html=True)
    st.text('')
    p_title('Paraphrase')
    st.text('')
    
    p_example = 'Health is the level of functional or metabolic efficiency of a living organism. In humans, it is the ability of individuals or communities to adapt and self-manage when facing physical, mental, or social challenges. The most widely accepted definition of good health is that of the World Health Organization Constitution.'
   
    input_pa = st.text_area("Use the example below or input your own text in English (maximum 500 characters)", max_chars=500, value=p_example, height=160)

    if st.button('Paraphrase'):
        if input_pa =='':
            st.error('Please enter some text')
        else:
            with st.spinner('Wait for it...'):
                    time.sleep(2)
                    translator = Translator()
                    mid = translator.translate(input_pa, dest="fr").text
                    mid2 = translator.translate(mid, dest="de").text
                    back = translator.translate(mid2, dest="en").text
                    st.markdown('___')
                    st.write('Back Translation Model')
                    st.success(back)
                    e_augmenter = EmbeddingAugmenter(transformations_per_example=1, pct_words_to_swap=0.3)
                    e_a = e_augmenter.augment(input_pa)
                    st.markdown('___')
                    st.write('Embedding Augmenter Model')
                    st.success(e_a)
                    w_augmenter = WordNetAugmenter(transformations_per_example=1, pct_words_to_swap=0.3)
                    w_a = w_augmenter.augment(input_pa)
                    st.markdown('___')
                    st.write('WordNet Augmenter Model')
                    st.success(w_a)
                    st.balloons()

#-----------------------------------------
   
#ANALYZE
########
       
if nav == 'Analyze text':
    st.markdown("<h4 style='text-align: center; color:grey;'>Accelerate knowledge with SYNTHIA &#129302;</h4>", unsafe_allow_html=True)
    st.text('')
    p_title('Analyze text')
    st.text('')
    
    a_example = "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by humans or animals. Leading AI textbooks define the field as the study of 'intelligent agents': any system that perceives its environment and takes actions that maximize its chance of achieving its goals. Some popular accounts use the term 'artificial intelligence' to describe machines that mimic cognitive functions that humans associate with the human mind, such as learning and problem solving, however this definition is rejected by major AI researchers. AI applications include advanced web search engines, recommendation systems (used by YouTube, Amazon and Netflix), understanding human speech (such as Siri or Alexa), self-driving cars (such as Tesla), and competing at the highest level in strategic game systems (such as chess and Go). As machines become increasingly capable, tasks considered to require intelligence are often removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology."

    source = st.radio("How would you like to start? Choose an option below",
                          ("I want to input some text", "I want to upload a file"))
    st.text('')

    if source == 'I want to input some text':
        input_me = st.text_area("Use the example below or input your own text in English (maximum of 10,000 characters)", max_chars=10000, value=a_example, height=330)
        if st.button('Analyze'):
            if len(input_me) > 10000:
                st.error('Please enter a text in English of maximum 1,000 characters')
            else:
                with st.spinner('Processing...'):
                    time.sleep(2)
                    nltk.download('punkt')
                    rt = readtime.of_text(input_me)
                    tc = textstat.flesch_reading_ease(input_me)
                    tokenized_words = word_tokenize(input_me)
                    lr = len(set(tokenized_words)) / len(tokenized_words)
                    lr = round(lr,2)
                    n_s = textstat.sentence_count(input_me)
                    st.markdown('___')
                    st.text('Reading Time')
                    st.write(rt)
                    st.markdown('___')
                    st.text('Text Complexity: from 0 or negative (hard to read), to 100 or more (easy to read)')
                    st.write(tc)
                    st.markdown('___')
                    st.text('Lexical Richness (distinct words over total number of words)')
                    st.write(lr)
                    st.markdown('___')
                    st.text('Number of sentences')
                    st.write(n_s)
                    st.balloons()

    if source == 'I want to upload a file':
        file = st.file_uploader('Upload your file here',type=['txt'])
        if file is not None:
            with st.spinner('Processing...'):
                    time.sleep(2)
                    stringio = StringIO(file.getvalue().decode("utf-8"))
                    string_data = stringio.read()
                    if len(string_data) > 10000:
                        st.error('Please upload a file of maximum 10,000 characters')
                    else:
                        nltk.download('punkt')
                        rt = readtime.of_text(string_data)
                        tc = textstat.flesch_reading_ease(string_data)
                        tokenized_words = word_tokenize(string_data)
                        lr = len(set(tokenized_words)) / len(tokenized_words)
                        lr = round(lr,2)
                        n_s = textstat.sentence_count(string_data)
                        st.markdown('___')
                        st.text('Reading Time')
                        st.write(rt)
                        st.markdown('___')
                        st.text('Text Complexity: from 0 or negative (hard to read), to 100 or more (easy to read)')
                        st.write(tc)
                        st.markdown('___')
                        st.text('Lexical Richness (distinct words over total number of words)')
                        st.write(lr)
                        st.markdown('___')
                        st.text('Number of sentences')
                        st.write(n_s)
                        st.balloons()

#-----------------------------------------
