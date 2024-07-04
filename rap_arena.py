
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama

from dotenv import load_dotenv, find_dotenv
import pyaudio
from openai import OpenAI
import os

_ = load_dotenv(find_dotenv())
client = OpenAI()

# llm = ChatGoogleGenerativeAI(model="gemini-pro")

model1 = "ChatGPT"
# model1 = "Gemini"
model2 = "Gemini"
model2 = "Llama3"

llm1 = ChatOpenAI(temperature=0, model='gpt-4o')

# llm2 = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0, max_tokens=500)
# llm1 = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-pro-latest", max_tokens=1024)
# llm2 = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-pro-latest", max_tokens=1024)
# llm2 = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-pro-latest", max_tokens=2048)
llm2 = ChatGroq(temperature=0, model_name="llama3-70b-8192")
# llm2 = ChatOpenAI(
#     temperature=0, 
#     model="Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF",
#     # model="PrunaAI/dolphin-2.9-llama3-8b-256k-GGUF-smashed",
#     base_url="http://localhost:1234/v1", 
#     api_key="not-needed")

# llm2 = ChatOllama(
#     # model="dolphin-llama3:8b"
#     # model="llama3:8b-instruct-q8_0"
#     model="phi3:14b"
#     )

memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")
memory2 = ConversationBufferMemory(memory_key="chat_history", input_key="input")

# Definindo os prompts para as LLMs
system_template = """
    Você é um rapper profissional e está participando de uma batalha de rimas.
    Para vencer, você deve ser capaz de produzir as melhores rimas, 
    desmoralizando seu adversário.

    Você é o {who_iam} e está duelando com {opponent}.
    Lembre-se que se rimar usando o nome do oponente ganhará pontos 
    com o público.
    
    Seja agressivo, fale baixarias, desmoralize o oponente, xingue a mãe!
    Se você não receber nenhuma entrada, deve iniciar rimando.

    Você deve pegar a última frase do seu oponente e rimar a partir daí. 
    Não é necessário citar a frase literalmente, mas usá-la 
    como inspiração, ponto de partida.

    Sua rima deve ter 1 estrofe apenas, com 4 frases.
    Use frases curtas.
"""

base_prompt = """
    Histórico da batalha até agora: {chat_history}

    Última entrada: {input}    
    """



prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template.format(who_iam=model1, opponent=model2)), 
    ("human", base_prompt)])
prompt_template2 = ChatPromptTemplate.from_messages([
    ("system", system_template.format(who_iam=model2, opponent=model1)), 
    ("human", base_prompt)])

# base_prompt = PromptTemplate.from_template(template=template)
# Criando os LLMChains
chain1 = LLMChain(llm=llm1, prompt=prompt_template, memory=memory)
chain2 = LLMChain(llm=llm2, prompt=prompt_template2, memory=memory2)

os.system("clear")

def speak(text, voice):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, 
                                           channels=1, rate=24000, 
                                           output=True)
    stream_start = False
    with client.audio.speech.with_streaming_response.create(model="tts-1", 
                                                            voice=voice, 
                                                            response_format="pcm", 
                                                            input=text) as response:
        silence_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            else:
                if max(chunk) > silence_threshold:
                    player_stream.write(chunk)
                    stream_start = True


def invoke_with_prompt(chain, user_input):
    variables = {"input": user_input}
    prompt = chain.prompt.format(**variables, chat_history=memory.load_memory_variables({})["chat_history"])
    # Imprima o prompt completo
    print("Prompt completo:\n", prompt)
    
    # Execute a chain
    response = chain.run(user_input)
    return response


last_response = {"text": "Sua vez de rimar."}
rounds = 2
models = {
        model1: chain1,
        model2: chain2,
        }
# models["Claude3.5"].invoke({"input": last_response["text"]})

for i in range(rounds):
    # for model, voice in zip(models.keys(), ["fable", "echo"]):
    for model, voice in zip(models.keys(), ["onyx", "fable"]):
        print(f"=== {model} ===")
        last_response = models[model].invoke({"input": last_response["text"]})
        print(last_response["text"])
        speak(last_response["text"], voice)
        print("\n")

        # print("=== Claude ===")
        # response2 = chain2.invoke({"input": response1["text"]})
        # print(response2["text"])
        # speak(response2["text"], "echo")
        # print("\n")

# memory.load_memory_variables({})["chat_history"]

# judge_template = """
#     You are a professional chess arbiter, working on a LLM's Chess Competition.

#     Your job is to parse last player's move and ensure that all chess moves are valid and correctly formatted in 
#     Standard Algebraic Notation (SAN) for processing by the python-chess library.

#     ### Input:
#     - Last player's  move
#     - List of valid moves in SAN

#     ### Output:
#     - Return the corresponding move in the list of valid SAN moves.
#     - If the proposed move is not in the valid moves list, must respond with "None"

#     ### Your turn:
#     - Proposed move: {proposed_move}
#     - List of valid moves: {valid_moves}

#     You should only respond the valid move, without the move number, nothing more.
#     Your response:
#     """

# llm3 = ChatGroq(temperature=0, model_name="llama3-70b-8192")
# judge_prompt = PromptTemplate.from_template(template=judge_template)
# chain3 = judge_prompt | llm3
