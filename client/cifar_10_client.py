import json
import requests
import sys
from PIL import Image
import numpy as np
 
 
def get_rest_url(model_name, host='spark-linux1.conygre.com',
                 port='4040', task='predict', version=None):
    """ This function takes hostname, port, task (b/w predict and classify)
    and version to generate the URL path for REST API"""
    # Our REST URL should be http://spark-linux1:8501/v1/models/cifar_10/predict
    url = "http://{host}:{port}/v1/models/{model_name}".format(host=host,
     port=port, model_name=model_name)
    if version:
        url += 'versions/{version}'.format(version=version)
    url += ':{task}'.format(task=task)
    return url
 
 
def get_model_prediction(model_input, model_name='cifar_10',
                         signature_name='serving_default'):
    """ This function sends request to the URL and get prediction
    in the form of response"""
    url = get_rest_url(model_name)
    image = Image.open(model_input)
    # convert image to array
    im =  np.asarray(image)
    # add the 4th dimension
    im = np.expand_dims(im, axis=0)
    im= im/255
    print("Image shape: ",im.shape)
    data = json.dumps({"signature_name": "serving_default",
                       "instances": im.tolist()})
    headers = {"content-type": "application/json"}
    # Send the post request and get response   
    rv = requests.post(url, data=data, headers=headers)
    print(rv.json())
    return rv.json()['predictions']
 
if __name__ == '__main__':
    class_names =["airplane","automobile","bird","cat","deer"
    ,"dog","frog","horse", "ship","truck"]
    print("\nGenerate REST url ...")
    url = get_rest_url(model_name='cifar_10')
    print(url)
     
    while True:
        print("\nEnter the image path [:q for Quit]")
        if sys.version_info[0] >= 3:
            path = str(input())
        if path == ':q':
            break
        try:
            model_prediction = get_model_prediction(path)
            print(model_prediction)
            print("The model predicted ...")
            print(class_names[np.argmax(model_prediction)])
        except FileNotFoundError as ex:
            print('Invalid File [', ex, ']')