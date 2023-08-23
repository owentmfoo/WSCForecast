FROM public.ecr.aws/lambda/python:3.8

# Copy road file
COPY RoadFile-LatLon-2021.dat ${LAMBDA_TASK_ROOT}

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Copy function code
COPY combine_forecast.py ${LAMBDA_TASK_ROOT}

# Install Git using the package manager (for Amazon Linux)
RUN yum install -y git

# Install the specified packages
RUN pip install -r requirements.txt

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "combine_forecast.main" ]