from flask import Flask

app = Flask(__name__)


@app.route("/")
def home():
    return """
    <h1>Student Details</h1>
    <p><strong>Name:</strong> Saptanshu Wanjari</p>
    <p><strong>Roll Number:</strong> 18 </p>
    <p><strong>Branch:</strong> CSE - AIML</p>
    """


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
