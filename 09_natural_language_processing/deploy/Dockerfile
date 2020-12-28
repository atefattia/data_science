FROM python:3.6.3-slim

LABEL maintainer="Atef Attia <attia.atef92@gmail.com>" \
	description="Data science: model deployment"

WORKDIR /app

COPY Dockerfile app.py requirements.txt saved_model/ templates/ /app

RUN pip install --upgrade pip
RUN pip --no-cache-dir install -r requirements.txt

CMD ["python", "app.py"]