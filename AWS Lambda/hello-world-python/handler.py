import json


def hello(event, context):
    print('hi !')

    return 'hello-world'



'''serverless config credentials --provider aws --key AKIA27PZKPA5WH2HROFO --secret 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 --profile serverless-admin

sls create --template aws-python --path hello-world-python

sls deploy -v





'''