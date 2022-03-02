FROM 589hero/stylized-neural-painting-renderer:1.1.0

COPY . ./app
WORKDIR ./app
 
RUN pip install -r requirements.txt

CMD ["python", "server.py"]