# app.py
import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from mistralai import Mistral
from http.client import responses
import asyncio
from openai import OpenAI
#import timeout_decorator

app = Flask(__name__)

class VehicleDataManager:
    def __init__(self, car_file='cars_dataset.csv', bike_file='bikesCleaned.csv'):
        self.car_data = pd.read_csv(car_file)
        self.bike_data = pd.read_csv(bike_file)
        self.car_brands = self.car_data['Make'].unique().tolist()
        self.bike_brands = self.bike_data['Make'].unique().tolist()
        self.car_models = self._get_models_by_brand(self.car_data)
        self.bike_models = self._get_models_by_brand(self.bike_data)
        
    def _get_models_by_brand(self, data):
        models = {}
        for brand in data['Make'].unique():
            models[brand] = data[data['Make'] == brand]['Model'].unique().tolist()
        return models

    def get_comparison_specs(self, vehicle_type):
        if vehicle_type == 'car':
            return [
                'Make', 'Model', 'Price', 'Year', 'Kilometer', 'Fuel Type', 'Transmission', 
                'Location', 'Color', 'Owner', 'Seller Type', 'Engine', 'Max Power', 
                'Max Torque', 'Drivetrain', 'Length', 'Width', 'Height', 'Seating Capacity', 
                'Fuel Tank Capacity'
            ]
        return [
            'Model', 'Make', 'Price', 'Max power', 'Max torque', 'Cooling System', 'Transmission', 
            'Transmission type', 'displacement', 'cylinders', 'bore', 'stroke', 
            'valves per cylinder', 'spark plugs', 'gear shifting pattern', 'clutch', 
            'Fuel Tank Capacity', 'Mileage - arai', 'Mileage', 'Top speed', 'Braking system', 
            'front brake type', 'front brake size', 'rear tyre size', 'tyre type', 
            'radial tyres', 'rear brake type', 'rear brake size', 'wheel type',
            'front wheel size', 'rear wheel size', 'front tyre size', 
            'front tyre pressure (rider)', 'rear tyre pressure (rider)', 
            'front tyre pressure (rider & pillion)', 'rear tyre pressure (rider & pillion)', 
            'Kerb weight', 'overall length', 'overall width', 'wheelbase', 'ground clearance', 
            'seat height', 'overall height', 'chassis type', 'Location', 'Kilometer Driven', 
            'Seller Type'
        ]

    def get_vehicle_data(self, vehicle_type, brand, model):
        data = self.car_data if vehicle_type == 'car' else self.bike_data
        return data[(data['Make'] == brand) & (data['Model'] == model)]



# Initialize managers
vehicle_manager = VehicleDataManager()


@app.route('/')
def index():
    
    return render_template('index.html',
                         car_brands=vehicle_manager.car_brands,
                         bike_brands=vehicle_manager.bike_brands,
                         car_models=vehicle_manager.car_models,
                         bike_models=vehicle_manager.bike_models)

@app.route('/get_models', methods=['POST'])
def get_models():
    brand = request.form.get('brand')
    vehicle_type = request.form.get('vehicle_type')
    
    if not brand:
        return jsonify({'error': 'Brand not provided'}), 400
    
    models = (vehicle_manager.car_models if vehicle_type == 'car' else vehicle_manager.bike_models).get(brand)
    if not models:
        return jsonify({'error': 'Models not found for brand'}), 404
    
    return jsonify({'models': models})

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if request.method != 'POST':
        return render_template('index.html',
                             car_brands=vehicle_manager.car_brands,
                             bike_brands=vehicle_manager.bike_brands,
                             car_models=vehicle_manager.car_models,
                             bike_models=vehicle_manager.bike_models)

    vehicle_type = request.form.get('vehicle_type','car')
    brand1 = request.form.get('selected_brand1')
    model1 = request.form.get('selected_model1')
    brand2 = request.form.get('selected_brand2')
    model2 = request.form.get('selected_model2')

    # Get vehicle data
    vehicle1_df = vehicle_manager.get_vehicle_data(vehicle_type, brand1, model1)
    vehicle2_df = vehicle_manager.get_vehicle_data(vehicle_type, brand2, model2)

    if vehicle1_df.empty or vehicle2_df.empty:
        return render_template('index.html',
                            error='Selected brand and model combination not found in dataset.',
                            car_brands=vehicle_manager.car_brands,
                            bike_brands=vehicle_manager.bike_brands,
                            car_models=vehicle_manager.car_models,
                            bike_models=vehicle_manager.bike_models)

    # Create comparison table
    specs = vehicle_manager.get_comparison_specs(vehicle_type)
    vehicle1 = vehicle1_df.iloc[0]
    vehicle2 = vehicle2_df.iloc[0]

    comparison_table = []
    for spec in specs:
        value1 = vehicle1[spec]
        value2 = vehicle2[spec]

        # Color coding for numerical comparisons
        if spec in ['Price', 'Kilometer', 'Kilometer Driven']:
            if value1 < value2:
                value1 = f'<span style="color: green">{value1}</span>'
            elif value2 < value1:
                value2 = f'<span style="color: green">{value2}</span>'

        comparison_table.append({'Spec': spec, 'Vehicle 1': value1, 'Vehicle 2': value2})

    return render_template('index.html',
                         comparison_table=comparison_table,
                         car_brands=vehicle_manager.car_brands,
                         bike_brands=vehicle_manager.bike_brands,
                         car_models=vehicle_manager.car_models,
                         bike_models=vehicle_manager.bike_models,
                         selected_brand1=brand1,
                         selected_model1=model1,
                         selected_brand2=brand2,
                         selected_model2=model2,
                         vehicle_type=vehicle_type)

@app.route('/get_ai_suggestions', methods=['POST'])
def get_ai_suggestions():
    try:
        brand1 = request.form.get('brand1')
        model1 = request.form.get('model1')
        brand2 = request.form.get('brand2')
        model2 = request.form.get('model2')

        if not all([brand1, model1, brand2, model2]):

            return jsonify({'suggestion': 'Please select both vehicles to compare'}), 400

        # Initialize Mistral client
        client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

        # Prepare the structured comparison prompt
        message = f"""Provide a structured comparison between {brand1} {model1} and {brand2} {model2} with the following format:

Key Features:
• List key features of {brand1} {model1}
• List key features of {brand2} {model2}

Performance Comparison:
• Power and acceleration
• Fuel efficiency
• Handling and comfort

Value Proposition:
• Price comparison
• Cost-effectiveness
• Target audience

Final Recommendation:
• Clear recommendation based on different use cases

Please provide a detailed comparison without using any markdown symbols like ** or ###."""

        # Get the response from Mistral
        
        completion = client.chat.completions.create(
  model="nvidia/llama-3.1-nemotron-70b-instruct",
  messages=[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":message}],
  temperature=0.5,
  top_p=0.7,
  max_tokens=1024,
  stream=True
)

        # Collect the response
        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
                

        # Format the response with proper HTML structure
        def format_response(text):
            # Remove any remaining ** markers
            formatted = text.replace("**", "")
            
            # Replace ### markers
            formatted = formatted.replace("###", "")
            
            # Split into sections
            sections = ["Key Features:", "Performance Comparison:", "Value Proposition:", "Final Recommendation:"]
            
            # Add HTML formatting for sections and content
            for section in sections:
                # Replace section headers with styled h3 tags
                formatted = formatted.replace(
                    section,
                    f'<h3 class="text-xl font-bold text-black mt-6 mb-3 border-b pb-2">{section}</h3>'
                )
            
            # Format bullet points and content structure
            lines = formatted.split('\n')
            formatted_lines = []
            
            for line in lines:
                line = line.strip()
                if line:
                    if line.startswith('•'):
                        # Convert bullet points to styled list items
                        line = f'<li class="ml-6 mb-2">{line[1:].strip()}</li>'
                    elif any(line.startswith(section) for section in sections):
                        # Section headers are already handled above
                        formatted_lines.append(line)
                    else:
                        # Regular text gets paragraph styling
                        line = f'<p class="mb-3">{line}</p>'
                    formatted_lines.append(line)
            
            formatted = '\n'.join(formatted_lines)
            
            # Wrap in container with styling
            formatted = f'''
                <div class="ai-suggestion bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-2xl font-bold text-center mb-6 border-b pb-3">AI Analysis: {brand1} {model1} vs {brand2} {model2}</h2>
                    <div class="text-gray-800 space-y-4">
                        {formatted}
                    </div>
                </div>
            '''
            
            return formatted

        formatted_response = format_response(response)
        return jsonify({'suggestion': formatted_response})

    except Exception as e:
        app.logger.error(f"Error in get_ai_suggestions: {str(e)}")
        return jsonify({'suggestion': 'An error occurred while getting the AI suggestion. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
