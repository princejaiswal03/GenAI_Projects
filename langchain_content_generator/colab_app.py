from flask import Flask, render_template, request
from langchain_community.llms import  OpenAI, HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pyngrok import ngrok, conf

conf.get_default().ngrok_path = "/usr/local/bin/ngrok"

# <NgrokTunnel: "https://<public_sub>.ngrok.io" -> "http://localhost:80">
ngrok_tunnel = ngrok.connect("https://localhost:81")

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["GET", "POST"])
def generate():
    if request.method == "POST":
        prompt = PromptTemplate.from_template("Generate a blog on title {title}?")
        # llm = OpenAI(temperature=0.3)
        llm = HuggingFacePipeline.from_model_id(
            model_id="microsoft/Phi-3-mini-4k-instruct",
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 100,
                "top_k": 50,
                "temperature": 0.1,
            },
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        prompt = request.json.get("prompt")
        output = chain.run(prompt)
        return output


app.run(host="0.0.0.0", port=5000, debug=True)