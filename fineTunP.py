from openai import OpenAI
client = OpenAI(api_key='')

#Carga los archivos de los datasets a la plataforma de OpenAI entrenamiento y validación

client.files.create(
  file=open("/home/frb/Escritorio/ReadmeCodes/IA/reducJsonl/data_train.jsonl", "rb"),
  purpose="fine-tune"
)

client.files.create(
  file=open("/home/frb/Escritorio/ReadmeCodes/IA/reducJsonl/data_val.jsonl", "rb"),
  purpose="fine-tune"
)
#Crea un proceso de fine-tuning.
#Necesitas el id de tus dos archivos de datasets de entrenamiento y validación.
#Además del nombre del modelo base.

client.fine_tuning.jobs.create(
  training_file="INGRESA EL FILE ID DEL ARCHIVO DE TRAIN", 
  validation_file='INGRESA EL FILE ID DEL ARCHIVO DE VAL',
  model="gpt-3.5-turbo-1106" #Puedes cambiar el modelo base según lo necesites.
)

#Listar modelos en tu organización
client.fine_tuning.jobs.list(limit=10)

#Eliminar modelos de tu organización
client.models.delete("INGRESA EL NOMBRE DEL MODELO A ELIMINAR")

