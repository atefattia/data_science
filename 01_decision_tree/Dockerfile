FROM python:3.6.3-slim

LABEL maintainer="Atef Attia <attia.atef92@gmail.com>" \
	description="Data science: example with docker using tree decision"

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip --no-cache-dir install cython scikit-learn pandas jupyter

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip='*'", "--port=8888","--no-browser", "--allow-root"]
