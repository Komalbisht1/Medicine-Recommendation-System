from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

#Load database
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

#Load model
svc = pickle.load(open('models/svc.pkl','rb'))

app = Flask(__name__)

#Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']
    return desc,pre,med,die,wrkout

symptoms_dict = {'itching': 0, 'skinrash': 1, 'nodalskineruptions': 2, 'continuoussneezing': 3, 'shivering': 4, 'chills': 5, 'jointpain': 6, 'stomachpain': 7, 'acidity': 8, 'ulcersontongue': 9, 'musclewasting': 10, 'vomiting': 11, 'burningmicturition': 12, 'spottingurination': 13, 'fatigue': 14, 'weightgain': 15, 'anxiety': 16, 'coldhandsandfeets': 17, 'moodswings': 18, 'weightloss': 19, 'restlessness': 20, 'lethargy': 21, 'patchesinthroat': 22, 'irregularsugarlevel': 23, 'cough': 24, 'highfever': 25, 'sunkeneyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowishskin': 32, 'darkurine': 33, 'nausea': 34, 'lossofappetite': 35, 'painbehindtheeyes': 36, 'backpain': 37, 'constipation': 38, 'abdominalpain': 39, 'diarrhoea': 40, 'mildfever': 41, 'yellowurine': 42, 'yellowingofeyes': 43, 'acuteliverfailure': 44, 'fluidoverload': 45, 'swellingofstomach': 46, 'swelledlymphnodes': 47, 'malaise': 48, 'blurredanddistortedvision': 49, 'phlegm': 50, 'throatirritation': 51, 'rednessofeyes': 52, 'sinuspressure': 53, 'runnynose': 54, 'congestion': 55, 'chestpain': 56, 'weaknessinlimbs': 57, 'fastheartrate': 58, 'painduringbowelmovements': 59, 'paininanalregion': 60, 'bloodystool': 61, 'irritationinanus': 62, 'neckpain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollenlegs': 68, 'swollenbloodvessels': 69, 'puffyfaceandeyes': 70, 'enlargedthyroid': 71, 'brittlenails': 72, 'swollenextremeties': 73, 'excessivehunger': 74, 'extramaritalcontacts': 75, 'dryingandtinglinglips': 76, 'slurredspeech': 77, 'kneepain': 78, 'hipjointpain': 79, 'muscleweakness': 80, 'stiffneck': 81, 'swellingjoints': 82, 'movementstiffness': 83, 'spinningmovements': 84, 'lossofbalance': 85, 'unsteadiness': 86, 'weaknessofonebodyside': 87, 'lossofsmell': 88, 'bladderdiscomfort': 89, 'foulsmellofurine': 90, 'continuousfeelofurine': 91, 'passageofgases': 92, 'internalitching': 93, 'toxiclook(typhos)': 94, 'depression': 95, 'irritability': 96, 'musclepain': 97, 'alteredsensorium': 98, 'redspotsoverbody': 99, 'bellypain': 100, 'abnormalmenstruation': 101, 'dischromicpatches': 102, 'wateringfromeyes': 103, 'increasedappetite': 104, 'polyuria': 105, 'familyhistory': 106, 'mucoidsputum': 107, 'rustysputum': 108, 'lackofconcentration': 109, 'visualdisturbances': 110, 'receivingbloodtransfusion': 111, 'receivingunsterileinjections': 112, 'coma': 113, 'stomachbleeding': 114, 'distentionofabdomen': 115, 'historyofalcoholconsumption': 116, 'fluidoverload.1': 117, 'bloodinsputum': 118, 'prominentveinsoncalf': 119, 'palpitations': 120, 'painfulwalking': 121, 'pusfilledpimples': 122, 'blackheads': 123, 'scurring': 124, 'skinpeeling': 125, 'silverlikedusting': 126, 'smalldentsinnails': 127, 'inflammatorynails': 128, 'blister': 129, 'redsorearoundnose': 130, 'yellowcrustooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

#Model prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

#ROUTES
@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        symptoms=request.form.get('symptoms')
        user_symptoms = [s.strip() for s in symptoms.split(',')]
        user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
        predicted_disease = get_predicted_value(user_symptoms)
        desc, pre, med, die, wrkout = helper(predicted_disease)

        my_pre=[]
        for i in pre[0]:
            my_pre.append(i)

        return render_template('index.html',predicted_disease=predicted_disease,dis_des=desc,dis_pre=my_pre,dis_med=med,dis_diet=die,dis_wrkout=wrkout)

if __name__=="__main__":
    app.run(debug=True)

