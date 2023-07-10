import os
import json
import datetime

from flask import (
    Flask, 
    flash, 
    request, 
    redirect, 
    url_for,
    render_template,
)
from werkzeug.utils import secure_filename
from google.cloud import aiplatform
from google.auth import credentials
from google.oauth2 import service_account
import google.cloud.aiplatform as aiplatform
from vertexai.preview.language_models import ChatModel, InputOutputTextPair
import vertexai

from lib.utils import transcribe_gcs
from lib.utils import storage
from lib.utils import allowed_file
from lib.utils import upload_blob
from lib.utils import get_ai_help_from_text

UPLOAD_FOLDER = './uploads'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def welcome():
    return render_template("welcome.html")


@app.route('/submit', methods=['GET', 'POST'])
def upload_file_and_transcribe():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        bucket_name = "awesome_ai_project_beg"
        upload_blob(
            bucket_name, 
            f"./uploads/{filename}", 
            filename,
        )
        summary_type = request.form.get('summary_type')

        fetch_url = (
            f"transcribe_audio?filename={filename}"
        )
        return render_template(
            "loading.html", 
            fetch_url=fetch_url,
            summary_type=summary_type
        )
    # render_template("transcribe_res.html", input=res)
    
    return render_template("submit.html")


@app.route('/transcribe_audio')
def transcribe_audio():
    bucket_name = "awesome_ai_project_beg"
    file_name = request.args.get('filename')


    res = transcribe_gcs(f"gs://{bucket_name}/{file_name}")

    return res


@app.route('/transcribe_output')
def show_summarized_res():

    res = request.args.get('input')
    summary_type = request.args.get('summary_type')

    # prompt a summary fron res text & output 
    summary = get_ai_help_from_text(res, summary_type)

    return render_template("transcribe_res.html", input=summary)


if __name__ == "__main__":
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host="127.0.0.1", port=8080, debug=True)

