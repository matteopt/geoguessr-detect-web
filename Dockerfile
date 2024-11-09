FROM rocm/pytorch:latest
RUN pip install transformers
RUN mkdir /app
WORKDIR /app
RUN git clone https://github.com/matteopt/geoguessr-detect-web
WORKDIR /app/geoguessr-detect-web
CMD ["python", "server.py"]
