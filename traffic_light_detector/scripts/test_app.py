from flask import Flask, jsonify
from arithmatic import Project
from traffic import *

app = Flask(__name__)

# label_sample=[  
#         {
#             'type':'Green-up',
#             'cordinates':{
#                 "xmin":100,
#                 "ymin":100,
#                 "xmax":1000,
#                 "ymax":1000
#             }
#         },
#         {
#             'type':'Red',
#             'cordinates':{
#                 "xmin":100,
#                 "ymin":200,
#                 "xmax":1500,
#                 "ymax":1600
#             }
#         },
#         {
#             'type':'Yellow',
#             'cordinates':{
#                 "xmin":100,
#                 "ymin":100,
#                 "xmax":1000,
#                 "ymax":1000
#             }
#         }
# ]
P1=Project("Traffic light detection", "II")
label_sample=P1.label_sample

@app.route('/')
def home():
    # P1=Project("Traffic light detection", "II")
    out = test_app()
    return out
    # return P1.myfunc()
    
   
    

@app.route('/tab1', methods=['GET'])
def get_method_plain1():
    print(label_sample[0])

    return jsonify(label_sample)



@app.route('/tab1/<string:label>', methods=['GET'])
def update_collection(label):
    print('done')
    for obj in label_sample:
        print(obj)
        if obj['type']==label:
            return jsonify(obj)
    return jsonify({'label':'Not Found'}) #comment add

@app.route('/tab1', methods=['POST'])
def post_method_plain1():
    new_obj={"Red-Yellow":{
                "xmin":100,
                "ymin":100,
                "xmax":1000,
                "ymax":1000
            }
            }
    label_sample.append(new_obj)
    return jsonify([new_obj])

app.run(debug=True)
app.run(host='https://tld-rt.herokuapp.com/')
# app.run(port=5000)
