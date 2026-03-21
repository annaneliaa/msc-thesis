# start minimal Linux image with Python 3.11 installed
FROM python:3.11-slim

# set /app as current dir inside the container so all below commands run from there
WORKDIR /app

# copy entire project to /app inside the container (src, configs, etc.)
COPY . .

# install project as package + dependencies from .toml file
RUN pip install --no-cache-dir -e .

# expose container on port 8000
EXPOSE 8000

# start FastAPI app when container is running
# api listens on port 8000
CMD ["uvicorn", "thesis.api.main:app", "--host", "0.0.0.0", "--port", "8000"]