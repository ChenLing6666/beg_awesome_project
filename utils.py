import json

from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
from google.cloud import speech
import vertexai
from vertexai.language_models import TextGenerationModel
from langchain.llms import VertexAI
from google.cloud import aiplatform
from google.oauth2 import service_account
from google.cloud import speech_v1p1beta1 as speech

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp3'}


def get_project_and_location():
    """
    Do what it says
    """    
    return ("ambient-mystery-292420", "us-east1")


def load_json_config(config_path):
    with open(
       config_path
    ) as f: 
        service_account_info = json.load(f)

    return service_account_info


def setup_credential():


    project_id, location = get_project_and_location()

    service_account_info = load_json_config("./credential_test.json")
     
    my_credentials = service_account.Credentials.from_service_account_info(
        service_account_info
    )


    aiplatform.init(
        project=project_id,
        location=location,
        staging_bucket='gs://awesome_ai_project_test_07_09',
        credentials=my_credentials,
        experiment='my-experiment',
        experiment_description='my experiment decsription'
    )


def transcribe_gcs(gcs_uri):
    """
    Transcribe Audio to Text
    """
    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,  # adjust as per your audio encoding
        sample_rate_hertz=16000,  # adjust as per your audio sample rate
        language_code="en-US",  # adjust as per your audio language
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    response = operation.result()

    res = [result.alternatives[0].transcript for result in response.results]
    res = " ".join(res)
    return res 


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file to the bucket
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def engineer_prompt(context):
    summary_prompt = f'Given this paragraph: <{context}>, could you write a summary to convey the key information:'
    bp_prompt = f'Given this paragraph: <{context}>, could you give me 5 bulletpoints takeaway that best summarize the content?'
    kw_prompt = f'Given this paragraph: <{context}>, could you give me 5 keywords that best describe the content?'
    
    return [summary_prompt, bp_prompt, kw_prompt]


def get_llm_response(prompt):
    project_id, location = get_project_and_location()
    vertexai.init(project=project_id, location=location)
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40
    }
    model = VertexAI(
        model_name="text-bison@001",
        **parameters,
    )

    response = model(
       prompt
    )
    return f"Response from Model: {response}"


def get_ai_help_from_text(text, summary_type):
    
    setup_credential()

    prompt_type = summary_type
    customized_param = "short and concise"
    if len(customized_param) == 0:
        customized_prompt = ''
    else:
        customized_prompt = f'Please follow my specific needs of the summary: give me {customized_param}'
    [summary_prompt, bp_prompt, kw_prompt] = engineer_prompt(text)
    if prompt_type == 'summary':
        prompt = summary_prompt + customized_prompt
    if prompt_type == 'bullet-point':

        prompt = bp_prompt + customized_prompt
    if prompt_type == 'keywords':
        prompt = kw_prompt + customized_prompt
 
    res = get_llm_response(prompt)
    return res
