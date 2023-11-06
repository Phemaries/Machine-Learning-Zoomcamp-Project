from flask import request


url = 'http://localhost:9696/predict'

patient_id = 'P54'

patient = {
    "systemic_illness": "Fever",
    "rectal_pain": "True",
    "sore_throat": "False",
    "penile_oedema": "True",
    "oral_lesions": "False",
    "solitary_lession": "False",
    "swollen_tonsils": "True",
    "hiv_infection": "False",
    "sexually_transmitted_infection": "True",
}


response = requests.post(url, json=patient).json()
print(response)

if response['concern'] == True:
    print(f'Patient {patient_id} has a high chance of being positive with Monkey pox')
else:
    print(f'Patient {patient_id} is not under threat')