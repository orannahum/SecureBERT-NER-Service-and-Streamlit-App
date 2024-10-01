
# Sleek ML Engineer Assignment


## 0. Mapping

![alt text](map_image.png)

## 1.notebooks




## 2.fastapi API service

#### pull image and run locally in 0.0.0.0:80
docker run -d -p 80:80 --name ner-fastapi-service oranne5/ner-fastapi-service:v2 uvicorn main:app --host 0.0.0.0 --port 80


#### example of SecureBERT-NER model for post request to"http://localhost:80/ner/" as json with test {"text": "bla bla bla"}

curl -X POST "http://localhost:80/ner/" -H "Content-Type: application/json" -d '{"text": "We observed DDKONG in use between February 2017 and the present, while PLAINTEE is a newer addition with the earliest known sample being observed in October 2017. The RANCOR campaign represents a continued trend of targeted attacks against entities within the South East Asia region."}'

#### same for SecureBERT-CyNER
curl -X POST "http://localhost:80/cyner/" -H "Content-Type: application/json" -d '{"text": "bla bla bla"}'

## 3.streamlit app

#### run on this up on port 8501 on image :  oranne5/ner-streamlit-app:v1

docker run -p 8501:8501 oranne5/ner-streamlit-app:v1

#### open browser

go to http://localhost:8501