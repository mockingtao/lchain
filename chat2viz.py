# python lib
from datetime import date, datetime
from typing import Union, List
import sys
from functools import lru_cache

# project lib
from apikey import API_KEY, BASE_URL, DEPLOYMENT_NAME, PINECONE_API_KEY, PINECONE_ENV

# langchain lib
from langchain.chat_models import AzureChatOpenAI
from langchain import PromptTemplate
from langchain.prompts.chat import(
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
    )
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, APIChain, SequentialChain, SimpleSequentialChain

# Config
# data source
data_info = sys.argv[1]

# current datetime
today = date.today()
formatted_date = today.strftime("%Y-%m-%d")

today_dt = datetime.now()
formatted_dt = today_dt.strftime("%Y-%m-%d %H:%M:%S")

# parameter
question = sys.argv[1]
data_info = sys.argv[2]

# Model
chat_llm = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version="2023-05-15",
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type = "azure",
    temperature=0
)

# Output parser
card_type={"BASIC_COLUMN": "单柱图", 
           "GROUPED_COLUMN": "簇状图，柱图", 
           "STACKED_COLUMN": "堆积图，柱图", 
           "STACKED_SPLIT_COLUMN": "分组堆积图，柱图", 
           "BASIC_BAR": "单条图", 
           "BASIC_LINE": "单线图", 
           "PIE": "饼图", 
           "RISING_SUN": "旭日图", 
           "FUNNEL": "漏斗图", 
           "HORIZONTAL_FUNNEL": "水平漏斗图"}

description_chartType=f"""Recommended chart type from you. This is a string.
You can choose one from the python dictionary {card_type}, 
the dictionary key is chart type, and the dictionary value is the describe in Chinese, 
you should choose the dictionary key as the chart type.
"""

description_row="""The dimension fields. You can choose them from the Dataset Scheme.
This is a list, each element in the list is a python dictionary. There are one keys in the dictionary: name
The name indicates the dimension field, and you can choose from the Dataset Scheme.
For example:
```[{"name":"column_1"},
    {"name":"column_2"}]```
"""

description_filters = """Data filter. Based on the text entered by the user, if not returned as [],
if is, then output them as a comma separated Python list, each element in the list is a python dictionary.
There are three keys in the dictionary: name, filterType, filterValue.
1. The name indicates the filter field, you can choose from the Dataset Scheme.
2. The filterType indicates the filter type. This is an enumeration value, the optional values are BT,GT,GE,LT,LE,EQ,NE,IN,NI.
BT means between, GT means Greater Than. You can only select one from these optional values.
3. The filterValue indicates the filter value. This is a list, Two elements when filterType is BT. Multiple elements when filterType is IN or NI. One element when filterType is other.
For example:
```[{
"filterType": "BT"
"filterValue": ['2023-05-17','2023-06-17'],
"name": "date"
}]```
"""

description_metric = """Data metric field. This is a list, each element in the list is a python dictionary.
There are two keys in the dictionary: name, aggrType.
1. The name indicates the metric field, you can choose from the Dataset Scheme.
2. The aggrType indicates the Aggregation mode, This is an enumeration value, the optional values are SUM,CNT,MIN,MAX,AVG,STDDEV,VAR,CNT_DISTINCT,NUL,MED,PERCENTILE
For example:
```[{
"aggrType": "SUM"
"name": "column_1"
}]```
"""

description_advices = """The other three relevant content or questions adviced by the AI from user's input. The presentation should be in the user's intonation style.
This is a list, each element in the list is a String.
For example:
```['将销量与去年同期做对比', '用品类对销量进行进一步分析']```
"""


chart_type_schema = ResponseSchema(name="chartType",
                             description=description_chartType)

row_schema = ResponseSchema(name="row",
                            type = 'array',
                            description=description_row)

filters_schema = ResponseSchema(name="filters",
                                type = 'array',
                                description=description_filters)

metric_schema = ResponseSchema(name="metric",
                                type = 'array',
                                description=description_metric)

explanation_schema = ResponseSchema(name="explanation",
                            description="The explanation of the chart, expressed in Chinese,")

title_schema = ResponseSchema(name="title",
                            description="The title of the chart, within 15 characters, expressed in Chinese,")

description_schema = ResponseSchema(name="description",
                            description="The description of the chart, within 50 characters, expressed in Chinese")

advices_schema = ResponseSchema(name="advices",
                            description=description_advices)

response_schemas = [
    chart_type_schema,
    row_schema,
    filters_schema,
    metric_schema,
    explanation_schema,
    title_schema,
    description_schema,
    advices_schema
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# Promp template
api_parameters = """
{
    "chartType":"",
    "row":[{"name":"A"},],
    "filters":[{"name":"A", "filterType":"IN", "filterValue":["",""]},
        {"name":"A", "filterType":"GT", "filterValue":[""]}],
    "metric":[{"name":"A", "aggrType":"SUM"},],
    "sorting":[{"name":"A", "aggrType":"SUM", "ordering":"desc"},],
    "explanation":"",
    "description":"",
    "title":"",
    "advices": ["advice1", "advice2"]

}
"""

system_template = """You are Guandata-GPT, an AI assistant designed to help data analysts do their daily work.
Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications.

Current Time: {formatted_dt}

GOALS:
1. Check the Dataset Scheme meets user's inputs.
2. Generate the API Core Parameters for charting.
3. Check the API Core Parameters are correct.
4. Generate three other related questions or contents from user's inputs, using the user's intonation style change words into declarative sentences. Put them in the advices of API Parameters.

Dataset Scheme:
{data_info}

API Core Parameters:
{api_parameters}

Response Format: 
{format_instructions}
"""

human_template = """
{chat_history}

{text}
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(template=system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(template=human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="text")

# LLMChain
chain = LLMChain(llm=chat_llm, prompt=chat_prompt, memory=memory, verbose=True)

@lru_cache()
def main(question, data_info, format_instructions, formatted_dt, api_parameters):
    
    chain_response = chain.run({'text':question, 
                                'data_info':data_info, 
                                'format_instructions':format_instructions, 
                                'formatted_dt': formatted_dt,
                                'api_parameters': api_parameters,
                                })
    output_dict = {}
    try:
        output_dict = output_parser.parse(chain_response)
    except ValueError as e:
        output_dict['explanation'] = chain_response
    return output_dict


if __name__ == '__main__':
    output_dict = main(question, data_info, format_instructions, formatted_dt, api_parameters)
    print(output_dict)

