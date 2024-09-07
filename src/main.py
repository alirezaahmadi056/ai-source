from flask import Flask , request , jsonify
import json

server = Flask(__name__)

@server.route("/",methods=["GET"])
def mainRoute():
    return "hello , welcome to amozeshgam ai service"
    
@server.route("/ai/planning",methods=["POST"])
def aiPlanningRoute():
    return "planning ai "
    
if __name__ == "__main__":
    mainRoute()
    aiPlanningRoute()
    server.run(host='0.0.0.0',port=8000 , debug=True)
    