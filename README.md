# WSC Forecast Integration

Collections of scripts that runs on AWS to collect the latest forecast and 
compile it into SolarSim Weather files. 

The file will be combined as a docker image and be push to AWS ECR and to be 
run on lambda.
 
## Combining forecast
`combine_forecast.py` takes the forecast obtained from solcast and tomorrow and
combines them.

To deal with the fact that the two different forecast operates in different 
resolution, the code will do an outer merge and then linearly interpolate to 
fill in the gaps.

The values from Solcast are period averages, so we will shift the timestamps \
forward by half the period so the values correspond to the centre of the period. 

## Deploying to AWS Lambda
### Automatic deployment
Github action have been set up to automatically deploy to AWS by publishing a
new docker image to AWS ECR and update the Lambda to use the latest image.

### Manual deployment
This section is mostly based off [this article](https://docs.aws.amazon.com/lambda/latest/dg/python-image.html#python-image-instructions).

Before you run the commands:
* replace the account ID
* replace role arn 
* add road file to the top level directory 

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