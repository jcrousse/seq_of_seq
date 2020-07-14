FROM tensorflow/tensorflow:2.2.0rc0-gpu

COPY additional_requirements.txt /app/additional_requirements.txt

RUN pip install --no-cache-dir -r /app/additional_requirements.txt && \
    python -m spacy download en_trf_bertbaseuncased_lg && \
    python -m spacy download en_core_web_sm
