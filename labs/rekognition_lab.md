## Automate Image Labeling with Amazon Rekognition

### Object Detection Context and Limitations

Object and Scene Detection is the set of technologies and algorithms that allow you to identify objects and their context within an image. In most cases, Object Detection aims at finding actual instances and positions of particular objects such as humans, cars, buildings, etc.

In the context of Amazon Rekognition, Object Detection can be compared to the simpler problem of Image Labeling, where the goal is to extract a finite set of semantic labels (or tags) from the given image.

Such a formulation of the problem makes it more suitable for scene detection as well, since the system would detect both single objects and scene-related attributes. For example, the system may extract specific labels related to individual objects (i.e. “person”, “bike”, “hat”), as well as more abstract labels related to the overall scene (i.e. “beach”, “party”, “sport”).

#### What labels can be detected?

Amazon Rekognition is based on deep learning models, which have been trained to identify specific objects and labels. As with many other supervised learning applications, such machine learning models can only identify the set labels used during the training phase. This means that we cannot ask Rekognition to identify the specific objects we’re looking for, nor provide a custom set of labels to use. We can only map and adapt Rekognition’s labels to our own needs and scenarios.

The hope for Amazon’s AI services is that they improve over time from a precision perspective and the total number of labels as well. For the time being, there is not an exhaustive reference list of available labels. However, we can get a quick idea of the most common labels by using the service for a little while (e.g. “Human”, “Child”, “Water”, etc.).

Furthermore, every label usually comes with a confidence level, so that you can decide whether to use a given label or not depending on each use case.

#### What does Object Detection look like in AWS?

We can perform Object Detection with Amazon Rekognition by invoking its *DetectLabels* API. We can provide the image as a binary string or as a S3 Object reference. The advantages of submitting images as S3 Objects is that we will not waste time uploading the same image multiple times, and that we can submit images up to 15MB, instead of 5MB.

The response will look similar to the following JSON structure:

```JSON
{
    "Labels": [
        {
            "Confidence": 98,
            "Name": "Person"
        },
        {
            "Confidence": 95,
            "Name": "Beach"
        },
        ...
    ]
}
```

The API returns an ordered list of labels, starting from the highest confidence level.

The number of labels depends on two parameters:

1. *MaxLabels*: The API will never return more than *MaxLabels* labels.
2. *MinConfidence*: The API will not return labels with a confidence score lower than *MinConfidence*.

It’s important to realize that low values of *MaxLabels* combined with high values of *MinConfidence* might generate empty responses.

During the laboratory, we will store images in a S3 bucket and use this API to extract labels from every newly uploaded image. We will store every image-label pair in a DynamoDB table.

*Note*: The same setup may be used to store labels into CloudSearch, or to handle more advanced situations such as deleting an image if it contains undesired labels.

### Create an S3 Bucket

1. Select the S3 service from the *AWS Management Console* under the **Storage** section.
2. From the S3 console, click the blue **Create Bucket** button.
The Create bucket dialog box appears.
3. Next you will have to enter the **Bucket name** and select a **Region** from the selection box.
  - Enter the following bucket name: `ca-labs-images` and then click **Create**.
    - Used "ca-lab-images-82"
  - If you receive a "*The requested bucket name is not available*" error message, click **Previous** a few times in order to return to the original screen of the dialog, and append a unique number to the bucket name in order to guarantee its uniqueness.
  - *Warning*: Because of the final validation checks, make sure that your bucket name starts with "calab" **or "ca-lab"?**.
4. Select anywhere in the row that shows your bucket (but not on the actual name itself) in order to see additional information about your empty S3 bucket.

**Warning popped up about error in permissions and properties...**

### Create a new Lambda Function with an S3 trigger

We are ready to create a new Lambda Function that will process each S3 object and extract labels using Amazon Rekognition.

We will use the default AWS Lambda creation wizard to configure the S3 trigger.

First, select AWS Lambda in the Console and click **Get Started Now** (or **Create a function**).

Your exposure up to this point to Lambda through Courses and Labs may include using *Author from Scratch* exclusively. However, there is a very helpful blueprint for using Rekognition, so you don't have to code everything from scratch.

Select **Use a blueprint**.

You can now choose the `rekognition-python` blueprint. This blueprint is a good starting point for all the Amazon Rekognition features, including object detection. You can find the blueprint by entering *rekognition* in the search filter.

Click on the blueprint and on **configure**, then fill out the following:

- **Function Name**: *ObjectDetectionS3*
- **Role**: *Choose existing role*
- **Existing role**: *lambda_s3_rekognition_dynamo*

By choosing the correct blueprint, the AWS Console will automatically selects S3 as a Trigger for the function.

Configure the S3 trigger parameters as follows:

- **Bucket**: *ca-labs-images-82*
- **Event Type**: *Object Created (All)* All object create events - This is the trigger part!
- **Prefix**: *images/*
- **Suffix**: *.jpg*
- **Enable Trigger**: *Checked*

Do not modify the Blueprint code for now, we will customize it slightly during the next step. For now, simply have a quick look at the code and understand its basic structure.

The lambda_handler function will take care of extracting the S3 Object information from the given event data. Our Python code will not need to read the S3 object's body, as Amazon Rekognition will read it directly from S3. Even if we could pass the image binary data directly to Amazon Rekognition, using an S3 Object reference is a good practice to reduce network latency and simplify IAM policies.

Click **Create Function**.

Lastly, go to the **Configuration** tab and increase the **Timeout** to *10* seconds in the **Basic Settings** panel.

Then save the whole configuration by clicking **Save**.

### Implement the Object Detection logic

We have created a new Lambda Function and configured it to react to S3 events.
In this step, we will implement the Python code to extract its labels with Amazon Rekognition and save them into DynamoDB for later use. For example, we could query DynamoDB to find the labels of a specific image (without invoking Rekognition again), or query DynamoDB by label in order to find all the matching images.

Since we used the official Lambda Blueprint for AWS Rekognition, most of the code should be in good shape to proceed with our customization. We will focus on the *detect_labels* and *lambda_handler* functions. Return to the Code tab of your new Lambda function. The following tasks should be carried out to customize the Python code:

-  Uncomment the invocation of *detect_labels* in *lambda_handler* and remove or comment out the rest
- In *detect_labels*, enable the DynamoDB *put_item* operation and customize the table name (*images*, in our case)
- Read at most 10 labels (*MaxLabels*) with confidence score above 80% (*MinConfidence*)

Here is the resulting code:

```py
import boto3, urllib

rekognition = boto3.client('rekognition')
table = boto3.resource('dynamodb').Table('images')

def detect_labels(bucket, key):
    response = rekognition.detect_labels(
        Image={"S3Object": {"Bucket": bucket, "Name": key}},
        MaxLabels=10,
        MinConfidence=80,
    )

    labels = [label_prediction['Name'] for label_prediction in response['Labels']]

    table.put_item(Item={
        'PK': key,
        'Labels': labels,
    })

    return response


def lambda_handler(event, context):
    data = event['Records'][0]['s3']
    bucket = data['bucket']['name']
    key = urllib.unquote_plus(data['object']['key'].encode('utf8'))
    try:
        response = detect_labels(bucket, key)
        print(response)
        return response
    except Exception as e:
        print(e)
        raise e
```

Be sure to click the **Save** button to save your code changes.

As you can see in the code snippet above, we are going to save a new DynamoDB Item for each image, using the S3 Object Key as the primary key. Each item will be associated to the corresponding labels, stored as a list of strings (or a Strings Set, in DynamoDB terminology). This way, we will be able to retrieve the labels of a given image by querying on the primary key, and we will also able to retrieve images with a given label of “X” by scanning the table with a conditional CONTAINS filter.

Note: A SCAN operation is very costly and therefore not recommended for large datasets of images. As a valid managed alternative, you may want to consider Amazon CloudSearch. For the purposes of this lab, a SCAN is fine.

**Full Blueprint Code**

```python
from __future__ import print_function

import boto3
from decimal import Decimal
import json
import urllib

print('Loading function')

rekognition = boto3.client('rekognition')


# --------------- Helper Functions to call Rekognition APIs ------------------


def detect_faces(bucket, key):
    response = rekognition.detect_faces(Image={"S3Object": {"Bucket": bucket, "Name": key}})
    return response


def detect_labels(bucket, key):
    response = rekognition.detect_labels(Image={"S3Object": {"Bucket": bucket, "Name": key}})

    # Sample code to write response to DynamoDB table 'MyTable' with 'PK' as Primary Key.
    # Note: role used for executing this Lambda function should have write access to the table.
    table = boto3.resource('dynamodb').Table('images')
    labels = [{'Confidence': Decimal(str(label_prediction['Confidence'])), 'Name': label_prediction['Name']} for label_prediction in response['Labels']]
    table.put_item(Item={'PK': key, 'Labels': labels})
    return response


def index_faces(bucket, key):
    # Note: Collection has to be created upfront. Use CreateCollection API to create a collecion.
    #rekognition.create_collection(CollectionId='BLUEPRINT_COLLECTION')
    response = rekognition.index_faces(Image={"S3Object": {"Bucket": bucket, "Name": key}}, CollectionId="BLUEPRINT_COLLECTION")
    return response


# --------------- Main handler ------------------


def lambda_handler(event, context):
    '''Demonstrates S3 trigger that uses
    Rekognition APIs to detect faces, labels and index faces in S3 Object.
    '''
    #print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.unquote_plus(event['Records'][0]['s3']['object']['key'].encode('utf8'))
    try:
        # Calls rekognition DetectFaces API to detect faces in S3 object
        # response = detect_faces(bucket, key)

        # Calls rekognition DetectLabels API to detect labels in S3 object
        response = detect_labels(bucket, key)

        # Calls rekognition IndexFaces API to detect faces in S3 object and index faces into specified collection
        #response = index_faces(bucket, key)

        # Print response to console.
        print(response)

        return response
    except Exception as e:
        print(e)
        print("Error processing object {} from bucket {}. ".format(key, bucket) +
              "Make sure your object and bucket exist and your bucket is in the same region as this function.")
        raise e
```

### Test the labeling system with new images

It’s time to test our automated labeling system.

We are going to upload a new image on S3 and verify that a new Item has been created in DynamoDB. We will also show how to search by image and by label in the DynamoDB Console.

First, select Amazon S3 in the Console menu. Select your bucket and create a new images folder (**Actions > Create Folder**).

Then select and enter the images folder and click **Upload**.

Here we can upload one or more images to our S3 bucket. Click **Add Files** and select one or more jpg files from your local file system. Once ready, start the upload by clicking **Start Upload**. By uploading more than one file at once, our Lambda Function will be triggered for each new file and subsequently add multiple records in DynamoDB.

In case something goes wrong after the upload, we can always use the Lambda Testing functionality to simulate the trigger by providing the same S3 Object Bucket and Key in the sample event.

If everything completed correctly, we should find one DynamoDB Item for each uploaded image.

Switch to **DynamoDB** in the Console. Click on **Tables** in the left pane to see the images table already created for your account. Select images and then the **Items** tab to see the entries in the images table.

The DynamoDB Items tab should look similar to the following:

![Dynambo DB Image](images/dynamo_db_image.png)

Here you can select the **Scan** operation and add a conditional filter on the **Labels** field by clicking on the **Add filter** link. Enter *Labels* and select the **Contains** operator. For the images you’ve uploaded to S3, you can try to filter items based on a common label. Time and interest permitting, spend a few minutes running different searches. For example, try Contains, Not Contains, change the string you match against based on the labels Rekognition assigned to the images, etc.

*Note*: To view all items in the DynamoDB again, delete your filter (“x”) and click the Start search button again.

Please note that this configuration will generate a new record for each image uploaded to S3. Filtering on the Labels field is possible because we stored labels as a set of strings. Remember that our search by label on DynamoDB requires an expensive **Scan** operation. Production implementations may want to use CloudSearch to reduce costs and eventually achieve full-text search.

**Response Example**

![Guinness](images/guinness.jpg)

```JSON
[
  { "Confidence" : "98.6580123901", "Name" : "Dog" },    
  { "Confidence" : "98.6580123901", "Name" : "Pet" },    
  { "Confidence" : "98.6580123901", "Name" : "Mammal" },    
  { "Confidence" : "98.6580123901", "Name" : "Animal" },    
  { "Confidence" : "98.6580123901", "Name" : "Canine" },    
  { "Confidence" : "88.4169082642", "Name" : "Terrier" },    
  { "Confidence" : "72.4511260986", "Name" : "Alcohol" },    
  { "Confidence" : "72.4511260986", "Name" : "Beer" },    
  { "Confidence" : "72.4511260986", "Name" : "Beverage" },    
  { "Confidence" : "72.4511260986", "Name" : "Drink" },    
  { "Confidence" : "67.4517364502", "Name" : "Machine" },    
  { "Confidence" : "60.2033653259", "Name" : "Wheel" }  
]
```

### Summary

To wrap up, we have configured Amazon S3 to invoke AWS Lambda every time a new png image is uploaded to it. Lambda will invoke Rekognition to extract labels and store the results in a DynamoDB table. This serverless configuration is quite flexible and we could achieve interesting results with very little effort by customizing the Lambda Function for more sophisticated scenarios.
