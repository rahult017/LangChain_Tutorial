from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import( 
    RunnableParallel,
    RunnableBranch,
    RunnableLambda,
    RunnableSequence,
    RunnablePassthrough,
)

load_dotenv()

def word_count(text):
    return len(text.split())

prompt = PromptTemplate(
    template='Write an insanely funny joke about {topic} that is clever, unexpected, and so hilarious people can’t stop laughing. Use sharp humor, a surprising punchline, and modern comedic timing.',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

query = input("Please enter your topic for joke :")
result = final_chain.invoke({'topic':query})

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)