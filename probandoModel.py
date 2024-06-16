from openai import OpenAI
import os
import pandas as pd
import matplotlib.pyplot as plt

apiKey = os.getenv('OPENAI_API_KEY') #importar la variablee de entorno 

client = OpenAI(api_key = apiKey)


#1 utilizando elmodelo entrenado desde la página de openai
""" response = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-1106:personal::9KPQzjkg",
  messages=[
    {
      "role": "system",
      "content": "Eres un asistente de atención a clientes y estudiantes de la plataforma de educación online en tecnología, inglés y liderazgo llamada Platzi"
    },
    {
      "role": "user",
      "content": "¿que curso tomar para aprender redes neuronales?"
    },
    {
      "role": "assistant",
      "content": "Te recomiendo tomar el Curso de Introducción a las Redes Neuronales en https://platzi.com/cursos/redes-neuronales/. Aprenderás los fundamentos de las redes neuronales y cómo aplicarlas en diferentes escenarios."
    }
  ],
  temperature=0,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
print(response.choices[0].message.content) """


#2 ver la informcion de un modelo mediante el id que nos genera al entrenar el modelo 

#print(client.fine_tuning.jobs.retrieve('ftjob-eNM3VcxjH5LK5Vh6rEpHPfE3'))

# nos sirve para obtener el archivo y las metricas del modleo en un futuro





#3 obtener archivo de resultados de fien-tuning que obtenemos en el resultado  result_files=['file-qhzlYowRq1TRV9R9SwQcYIeX']
# es decir cada paso que tuvo de entrenamiento el modelo 
content = client.files.content('file-qhzlYowRq1TRV9R9SwQcYIeX')
#print(content.text)

#4 interpretación de los resultados anteriores con pandas 
metrics_str = content.text

metrics_list = [line.split(',') for line in metrics_str.split('\n')]

#metricas desde la primer fila       nombres de las columnas 
df = pd.DataFrame(metrics_list[1:], columns=metrics_list[0])
df = df.apply(pd.to_numeric, errors='coerce')
df.tail()

#print(df) 
#segun los datos obtenido el train_loss en el último paso fue de 0.57255 que es la perdida de datos
# y el train_accuracy fue de 0.81579 que es bastante bueno que es casi uno 
#podemos ver esto datos en grafica para ver en cada paso la perdida  de entrenamiento o train_loss y la precisión o train_accuracy

#5 datos a números en gráfica con matplotlib
df = df.apply(pd.to_numeric, errors='coerce')
df.tail()

# gráfia de precisión 
plt.figure(figsize=(7,4))
plt.plot(df['step'], df['train_accuracy'])
plt.title('Training Accuracy over Steps')
plt.xlabel('Step')
plt.ylabel('Training Accuracy')
plt.show()

# gráfia de pérdida 
plt.figure(figsize=(7,4))
plt.plot(df['step'], df['train_loss'])
plt.title('Training Loss over Steps')
plt.xlabel('Step')
plt.ylabel('Training Accuracy')
plt.show()
