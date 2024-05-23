from flask import Flask, render_template, request, jsonify

from chat_app import ChatApp





app = Flask(__name__)
instance = ChatApp()

@app.route('/')
def index():
    instance.get_response("")
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json['message']
    # Here you would process the user message and get a response from your bot logic
    # For now, let's just echo the user's message
    bot_response = instance.get_response(user_message)

    bot_response = "Bot: " + str(bot_response)
    return jsonify({'message': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
