from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h1>Flask Application in Docker</h1>
    <p>The Flask application is running successfully.</p>
    """

@app.route("/student")
def student():
    return {
        "name": "Student Name",
        "course": "Docker Practical",
        "status": "Completed"
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
