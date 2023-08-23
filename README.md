# WSC Forecast Integration

Collections of scripts that runs on AWS to collect the latest forecast and 
compile it into SolarSim Weather files. 

The file will be combined as a docker image and be push to AWS ECR and to be 
run on lambda.
 
## Deploying to AWS Lambda
This section is mostly based off [this article](https://docs.aws.amazon.com/lambda/latest/dg/python-image.html#python-image-instructions).

Before you run the commands:
* replace the account ID
* replace role arn 

For the first time:
```shell
docker build --platform linux/amd64 -t combine-forecast:test .
aws ecr create-repository --repository-name combine-forecast --image-scanning-configuration scanOnPush=true --image-tag-mutability MUTABLE
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 111122223333.dkr.ecr.us-east-1.amazonaws.com
docker tag combine-forecast:test 111122223333.dkr.ecr.us-east-1.amazonaws.com/combine-forecast
docker push 111122223333.dkr.ecr.us-east-1.amazonaws.com/combine-forecast:latest
aws lambda create-function   --function-name combine-forecast   --package-type Image  --code ImageUri=111122223333.dkr.ecr.us-east-1.amazonaws.com/combine-forecast:latest --role arn:aws:iam::111122223333:role/service-role/my-service-role
```

To rebuild and push dock image to ECR: 
```shell
docker build --platform linux/amd64 -t combine-forecast:test .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 111122223333.dkr.ecr.us-east-1.amazonaws.com
docker tag combine-forecast:test 111122223333.dkr.ecr.us-east-1.amazonaws.com/combine-forecast
docker push 111122223333.dkr.ecr.us-east-1.amazonaws.com/combine-forecast:latest
```
Then you'll need to update the image on lambda