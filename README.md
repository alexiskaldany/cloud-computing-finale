# cloud-computing-finale

Alexis, Nate, Suhas

Games on April 7th

## Building Dagster as Service

Gameplan

- Create example job
- Work through [tutorial](https://github.com/dagster-io/dagster/tree/0.14.8/examples/deploy_ecs)


```
# Dagster
FROM python:3.7-slim as dagster

RUN apt-get update && apt-get upgrade -yqq
RUN apt-get install git -y
ENV DAGSTER_HOME=/opt/dagster/dagster_home/
RUN mkdir -p $DAGSTER_HOME
WORKDIR $DAGSTER_HOME
COPY dagster.yaml workspace.yaml $DAGSTER_HOME
RUN git clone https://github.com/dagster-io/dagster.git

# Install:
# - dagster so we can run `dagster-daemon run`
# - dagster-aws so we can use EcsRunLauncher
# - dagster-postgres so we can use PostgresEventStorage,
#   PostgresRunStorage, and PostgresScheduleStorage
COPY requirements-dagster.txt $DAGSTER_HOME
RUN pip install -r requirements-dagster.txt

# Dagit
FROM dagster as dagit
COPY requirements-dagit.txt $DAGSTER_HOME
RUN pip install -r requirements-dagit.txt

# User Code gRPC Server
# You can either include all of your repositories in this
# stage or you can create multiple stages that each use
# the same base - one for each repository.
FROM dagster as user_code
COPY repo.py $DAGSTER_HOME
```

# For setting up and connecting to front end EC2

- port = 8050
- `chmod 600 /Users/alexiskaldany/school/cloud-dash.pem`
-
- ```
   ssh -i "/Users/alexiskaldany/school/cloud-dash.pem" ubuntu@44.200.50.123 # dash
   ssh -i "/Users/alexiskaldany/school/cloud-dash.pem" ubuntu@52.23.176.8 # daggit
   cd cloud-computing-finale
   source venv/bin/activate
   python3 Cloud_dash.py
   ```

- `sudo apt-get update`
- `sudo apt install python3-pip`
- `sudo apt install ruby`
- `sudo pip install virtualenv`
- `git clone https://github.com/alexiskaldany/cloud-computing-finale.git`
- `cd cloud-computing-finale`
- `sudo apt install python3.8-venv`
- `python3 -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`

### To run app

- `python3 app.py`

link will be the IP of the EC2 with :8050

- `http://44.200.50.123:8050`

- `pip install wget`
- `wget https://aws-codedeploy-us-east-1.s3.us-east-1.amazonaws.com/latest/install`
- 'chmod +x ./install`
- `sudo ./install auto`
- `sudo service codedeploy-agent start`

### IAM user

`

### In local terminal

Copy github creds

- scp -i "/Users/alexiskaldany/school/cloud-dash.pem" "/Users/alexiskaldany/.ssh/id_ed25519" ubuntu@54.208.41.106:~/
- `scp -i "/Users/alexiskaldany/school/cloud-dash.pem" -r "/Users/alexiskaldany/school/cloud-computing-finale/" ubuntu@54.208.41.106:cloud-computing-finale/`

pip3 freeze > requirements.txt
