from flask import Flask, request, jsonify
from model_service import cause_group_code_predictor, generate_cause_text

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Flask App!"

@app.route('/v2/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    damage_group = data.get('damage_group')
    damage_code = data.get('damage_code')
    damage_text = data.get('damage_text')

    response = {}

    if damage_group and damage_code:
        predicted_cause_group, probability_cause_group, predicted_cause_code, probability_cause_code = cause_group_code_predictor(damage_group, damage_code)
        response.update({
            'predicted_cause_group': predicted_cause_group,
            'probability_cause_group': probability_cause_group.item(),
            'predicted_cause_code': predicted_cause_code,
            'probability_cause_code': probability_cause_code.item()
        })
    if damage_text:
        generated_text = generate_cause_text(damage_text)
        response['full_text'] = generated_text[0]
        response['cut_down_text'] = generated_text[1]
    if not response:
        return jsonify({'error': 'Invalid input. Provide either damage_group and damage_code, or damage_text, or both.'})

    return jsonify(response)