from flask import Flask, request, jsonify, make_response
import json
from utils import CalculateMd5
from artificialIntelligence import LieanerPlanningAI, NeuralPlanningAI

server = Flask(__name__)

neural_planning_ai = NeuralPlanningAI.NeuralPlanningAI()
linear_planning_ai = LieanerPlanningAI.LinearPlanningAI()
neural_planning_ai.create_and_train_model()


@server.route("/", methods=["GET"])
def main_route():
    return "hello , welcome to amozeshgam ai service"


@server.route("/ai/planning", methods=['POST'])
def ai_planning_route():
    model = request.headers.get("model")
    api_key = request.headers.get("x-api-key")
    if CalculateMd5.calculate_hash(api_key) != "2d469051f1224fd252ba28b9d84c966d":
        return make_response("error-z", 500)
    data = request.get_json()
    dump_data = json.dumps(data)
    load_data = json.loads(dump_data)
    ai_data = [int(load_data["age"]), int(load_data["educationalStatus"]), int(load_data["fieldOfStudy"]),
               int(load_data["maritalStatus"]), int(load_data["gender"]), int(load_data["militaryStatus"]),
               int(load_data["freeTime"]), int(load_data["targetIncome"]), int(load_data["intentionToMigrate"]),
               int(load_data["interestInMathematics"]), int(load_data["computerExperience"]),
               int(load_data["whichOneDoYouLikeMore"]), int(load_data["whichCaseIsmoreRelevant"]),
               int(load_data["doYouWorkOnHolidays"]), int(load_data["disability"]),
               int(load_data["addictionred"])]
    if model == 'oogway':
        result = neural_planning_ai.predict(ai_data).tolist()
        return jsonify({"result": eval(result[0])})

    elif model == 'shifu':
        try:
            result = linear_planning_ai.predict(ai_data).tolist()
            return jsonify({"result": eval(result[0])})
        except Exception as e:
            return make_response("error-0", 500)
    else:
        return make_response("error12", 500)


@server.route("/ai/change/data", methods=["GET"])
def change_data():
    global neural_planning_ai, linear_planning_ai
    api_key = request.headers.get('x-api-key')
    if api_key == "":
        return make_response("error", 500)
    if CalculateMd5.calculate_hash(api_key) != "2d469051f1224fd252ba28b9d84c966d":
        return make_response("error", 500)
    neural_planning_ai = NeuralPlanningAI.NeuralPlanningAI()
    neural_planning_ai.create_and_train_model()
    linear_planning_ai = LieanerPlanningAI.LinearPlanningAI()
    return jsonify({"message": "data changed successfully"})


if __name__ == "__main__":
    server.run(host='0.0.0.0', port=8000, debug=True)
