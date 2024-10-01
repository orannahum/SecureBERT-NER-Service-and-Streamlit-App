

|----------------------------------------------------------------------|--------------|
| DNRTI            | SecureBERT-NER                                    | Category     |
|----------------------------------------------------------------------|--------------|
| HackOrg          | APT                                               | APT          |
| SecTeam          | SECTEAM                                           | SECTEAM      |
| Idus, Org        | IDTY                                              | IDTY         |
| OffAct, Way      | ACT, OS, TOOL                                     | ACT_OS_TOOL  |
| Exp              | VULID, VULNAME                                    | VULID_VULNAME|
| Tool             | MAL                                               | MAL          |
| SamFile          | FILE                                              | FILE         |
| -                | DOM, ENCR, IP, URL, MD5, PROT, EMAIL, SHA1, SHA2  | OTHERS       |
| Time             | TIME                                              | TIME         |
| Area             | LOC                                               | LOC          |
| Purp, Features   | -                                                 | NULL         |
|----------------------------------------------------------------------|--------------|

Create a NER (named entity recognition) service
Create an architecture that allows users to send text to an API and receive the
extracted entities and their class.
Build a Docker container, as you see fit, to run this HTTP-based service. Ensure that
you allow offline usage of your work (meaning your work should run entirely
on-prem).
Verify the functionality of the Docker container locally.
Bonus: Web UI Development using Streamlit / React
Develop a web interface using the Streamlit framework, that includes a file upload
feature for text files.
Implement backend functionality to process uploaded files using the NER model.
Only one file should be processed at a time.
Generate a results table in the web UI containing two columns: class names and their
identified entities.
Deliverables
A benchmarking report and code for evaluation of SecureBert-NER model based on






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