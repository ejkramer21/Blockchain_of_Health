from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/upload', methods=['POST'])
def fileUpload():
    response="Whatever you wish too return"
    return response

if __name__ == "__main__":
    app.run()